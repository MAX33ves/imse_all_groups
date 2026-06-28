from __future__ import annotations

import math
import pickle
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from project_config import BIKES

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_pipeline import make_feature_matrix
from labels import bike_pressure_levels

warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    space_name: str
    columns: tuple[str, ...]
    pca_components: int | None


@dataclass(frozen=True)
class Candidate:
    name: str
    stage: str
    candidate_type: str
    feature_config: FeatureConfig | None
    hidden_layers: tuple[int, ...] = ()
    activation: str = ""
    alpha: float = 0.0
    aggregate_method: str = "median"
    seeds: tuple[int, ...] = (42,)
    use_run_balanced_weights: bool = True


def compact_signal_columns(X_full: pd.DataFrame) -> list[str]:
    """Keep a compact, interpretable subset of signal feature families."""
    keep_tokens = [
        "_rms_mean",
        "_std_mean",
        "_energy_per_s_mean",
        "_p95_abs_mean",
        "_ptp_mean",
        "_dom_freq_mean",
        "_spectral_centroid_mean",
        "_spectral_entropy_mean",
        "_band_0p5_3_power_mean",
        "_band_3_8_power_mean",
        "_band_8_15_power_mean",
        "_band_15_30_power_mean",
    ]
    return [
        col
        for col in X_full.columns
        if (col.startswith("acc_") or col.startswith("gyro_")) and any(token in col for token in keep_tokens)
    ]


def clean_feature_space(frame: pd.DataFrame) -> pd.DataFrame:
    """Make feature spaces numeric and finite before sklearn pipelines."""
    out = frame.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(out.median(numeric_only=True))
    return out.astype(float)


def add_bike_interactions(frame: pd.DataFrame, signal_cols: list[str], bike_cols: list[str]) -> pd.DataFrame:
    """Create bike-specific signal interactions for candidate feature spaces."""
    pieces = [frame[signal_cols].copy(), frame[bike_cols].copy()]
    for bike_col in bike_cols:
        pieces.append(frame[signal_cols].multiply(frame[bike_col], axis=0).add_suffix(f"__x__{bike_col}"))
    return pd.concat(pieces, axis=1)


def add_run_medians(features: pd.DataFrame, frame: pd.DataFrame, signal_cols: list[str]) -> pd.DataFrame:
    """Add run-level median context, repeated for every window in the run."""
    return frame[signal_cols].groupby(features["run_id"]).transform("median").add_prefix("runmed_")


def build_feature_spaces(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, pd.DataFrame]]:
    """Build the base X matrix and alternative feature spaces for model selection."""
    X_full, feature_names = make_feature_matrix(features, include_weight=True)
    bike_cols = [f"bike_{bike}" for bike in BIKES if f"bike_{bike}" in X_full.columns]
    signal_cols = [col for col in X_full.columns if col not in bike_cols and col != "rider_weight_kg"]
    compact_cols = compact_signal_columns(X_full) or signal_cols[: min(32, len(signal_cols))]

    compact_runmed = add_run_medians(features, X_full, compact_cols)
    weight = features[["rider_weight_kg"]].astype(float).reset_index(drop=True)
    spaces = {
        "full": X_full.copy(),
        "compact": pd.concat([X_full[compact_cols].copy(), X_full[bike_cols].copy()], axis=1),
        "compact_interact": add_bike_interactions(X_full, compact_cols, bike_cols),
        "runmed_compact": pd.concat([compact_runmed, X_full[bike_cols].copy()], axis=1),
        "runcontext_compact": pd.concat([X_full[compact_cols].copy(), compact_runmed, X_full[bike_cols].copy()], axis=1),
        "compact_weight": pd.concat([X_full[compact_cols].copy(), X_full[bike_cols].copy(), weight], axis=1),
        "runcontext_compact_weight": pd.concat([X_full[compact_cols].copy(), compact_runmed, X_full[bike_cols].copy(), weight], axis=1),
    }
    return X_full, feature_names, {name: clean_feature_space(frame) for name, frame in spaces.items()}


def build_feature_configs(spaces: dict[str, pd.DataFrame], n_train_rows: int) -> list[FeatureConfig]:
    specs = [
        ("compact_pca6", "compact", 6),
        ("full_pca8", "full", 8),
        ("compact_interact_pca6", "compact_interact", 6),
        ("runmed_compact_pca6", "runmed_compact", 6),
        ("runcontext_compact_pca8", "runcontext_compact", 8),
        ("compact_weight_pca6", "compact_weight", 6),
        ("runcontext_compact_weight_pca8", "runcontext_compact_weight", 8),
    ]
    configs = []
    for name, space_name, requested in specs:
        frame = spaces[space_name]
        components = min(requested, frame.shape[1], max(2, n_train_rows - 2))
        if components >= 2:
            configs.append(FeatureConfig(name=name, space_name=space_name, columns=tuple(frame.columns), pca_components=components))
    return configs


def hidden_label(hidden_layers: tuple[int, ...]) -> str:
    return "x".join(str(value) for value in hidden_layers)


def candidate_name(stage: str, config: FeatureConfig, hidden_layers: tuple[int, ...], activation: str, alpha: float, aggregate_method: str, seeds: tuple[int, ...]) -> str:
    alpha_label = f"{alpha:g}".replace(".", "p")
    seed_label = f"ens{len(seeds)}" if len(seeds) > 1 else f"s{seeds[0]}"
    return f"ffnn_training_pool_{stage}_{seed_label}_{config.name}_{activation}_h{hidden_label(hidden_layers)}_a{alpha_label}_{aggregate_method}"


def build_stage1_candidates(configs: list[FeatureConfig]) -> list[Candidate]:
    """Small FFNN grid, intentionally conservative for 72 run-level labels."""
    network_grid = [
        ((4,), "tanh", 1.0),
        ((8,), "tanh", 0.1),
        ((8,), "tanh", 1.0),
        ((8, 4), "tanh", 1.0),
    ]
    out = []
    for config in configs:
        for hidden_layers, activation, alpha in network_grid:
            seeds = (42,)
            out.append(
                Candidate(
                    name=candidate_name("stage1", config, hidden_layers, activation, alpha, "median", seeds),
                    stage="stage1",
                    candidate_type="ffnn",
                    feature_config=config,
                    hidden_layers=hidden_layers,
                    activation=activation,
                    alpha=alpha,
                    aggregate_method="median",
                    seeds=seeds,
                )
            )
    return out


def build_reference_candidates() -> list[Candidate]:
    """Reference baselines help interpret whether the FFNN learns signal."""
    return [
        Candidate(name="reference_global_mean", stage="reference", candidate_type="global_mean", feature_config=None),
        Candidate(name="reference_bike_mean", stage="reference", candidate_type="bike_mean", feature_config=None),
    ]


def build_ensemble_candidates(top_candidates: list[Candidate]) -> list[Candidate]:
    """Re-train near-best candidates with multiple random seeds."""
    seeds = (11, 42, 91)
    out = []
    seen = set()
    for base in top_candidates:
        assert base.feature_config is not None
        key = (base.feature_config.name, base.hidden_layers, base.activation, base.alpha, base.aggregate_method)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            Candidate(
                name=candidate_name("ensemble", base.feature_config, base.hidden_layers, base.activation, base.alpha, base.aggregate_method, seeds),
                stage="ensemble",
                candidate_type="ffnn",
                feature_config=base.feature_config,
                hidden_layers=base.hidden_layers,
                activation=base.activation,
                alpha=base.alpha,
                aggregate_method=base.aggregate_method,
                seeds=seeds,
            )
        )
    return out


def build_pipeline(candidate: Candidate, seed: int) -> Pipeline:
    """Sklearn pipeline: StandardScaler -> optional PCA -> MLPRegressor."""
    assert candidate.feature_config is not None
    steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
    if candidate.feature_config.pca_components is not None:
        steps.append(("pca", PCA(n_components=candidate.feature_config.pca_components, random_state=seed)))
    steps.append(
        (
            "mlp",
            MLPRegressor(
                hidden_layer_sizes=candidate.hidden_layers,
                activation=candidate.activation,
                solver="lbfgs",
                alpha=candidate.alpha,
                max_iter=3000,
                max_fun=40000,
                random_state=seed,
            ),
        )
    )
    return Pipeline(steps)


def run_balanced_weights(meta_train: pd.DataFrame) -> np.ndarray:
    """Give each run equal total weight despite different window counts."""
    counts = meta_train["run_id"].map(meta_train["run_id"].value_counts()).astype(float)
    weights = 1.0 / counts.to_numpy(dtype=float)
    return weights / weights.mean()


def fit_predict_reference(candidate: Candidate, meta_train: pd.DataFrame, y_train: pd.Series, meta_pred: pd.DataFrame) -> np.ndarray:
    if candidate.candidate_type == "global_mean":
        return np.full(len(meta_pred), float(y_train.mean()))
    if candidate.candidate_type == "bike_mean":
        train = meta_train[["bike"]].copy()
        train["pressure_bar"] = y_train.to_numpy(dtype=float)
        global_mean = float(train["pressure_bar"].mean())
        bike_means = train.groupby("bike")["pressure_bar"].mean().to_dict()
        return meta_pred["bike"].map(bike_means).fillna(global_mean).to_numpy(dtype=float)
    raise ValueError(f"Unsupported reference candidate: {candidate.candidate_type}")


def fit_predict_ffnn(candidate: Candidate, X_train: pd.DataFrame, y_train: pd.Series, meta_train: pd.DataFrame, X_pred: pd.DataFrame) -> np.ndarray:
    predictions = []
    sample_weight = run_balanced_weights(meta_train) if candidate.use_run_balanced_weights else None
    for seed in candidate.seeds:
        model = clone(build_pipeline(candidate, seed))
        fit_kwargs = {"mlp__sample_weight": sample_weight} if sample_weight is not None else {}
        model.fit(X_train, y_train, **fit_kwargs)
        predictions.append(np.ravel(model.predict(X_pred)))
    return np.mean(np.vstack(predictions), axis=0)


def predict_candidate(candidate: Candidate, spaces: dict[str, pd.DataFrame], train_features: pd.DataFrame, pred_features: pd.DataFrame, train_index: pd.Index, pred_index: pd.Index) -> np.ndarray:
    y_train = train_features["pressure_bar"].reset_index(drop=True)
    if candidate.candidate_type == "ffnn":
        assert candidate.feature_config is not None
        frame = spaces[candidate.feature_config.space_name]
        cols = list(candidate.feature_config.columns)
        return fit_predict_ffnn(
            candidate,
            frame.loc[train_index, cols].reset_index(drop=True),
            y_train,
            train_features.reset_index(drop=True),
            frame.loc[pred_index, cols].reset_index(drop=True),
        )
    return fit_predict_reference(candidate, train_features.reset_index(drop=True), y_train, pred_features.reset_index(drop=True))


def aggregate_values(values: pd.Series, method: str) -> float:
    arr = values.to_numpy(dtype=float)
    if method == "trimmed_mean" and len(arr) >= 5:
        low, high = np.quantile(arr, [0.10, 0.90])
        trimmed = arr[(arr >= low) & (arr <= high)]
        return float(np.mean(trimmed if len(trimmed) else arr))
    return float(np.median(arr))


def aggregate_predictions(window_meta: pd.DataFrame, predictions: np.ndarray, candidate: Candidate, phase: str, fold: str = "") -> pd.DataFrame:
    pred_df = window_meta[["group", "run_id", "bike", "p_number", "dataset_role", "pressure_bar"]].copy()
    pred_df["window_pred_bar"] = np.asarray(predictions, dtype=float)
    rows = []
    for run_id, part in pred_df.groupby("run_id", sort=False):
        rows.append(
            {
                "phase": phase,
                "fold": fold,
                "model_name": candidate.name,
                "stage": candidate.stage,
                "candidate_type": candidate.candidate_type,
                "group": part["group"].iloc[0],
                "run_id": run_id,
                "bike": part["bike"].iloc[0],
                "p_number": part["p_number"].iloc[0],
                "dataset_role": part["dataset_role"].iloc[0],
                "actual_pressure_bar": float(part["pressure_bar"].iloc[0]),
                "pred_pressure_bar": aggregate_values(part["window_pred_bar"], candidate.aggregate_method),
                "pred_pressure_mean_bar": float(part["window_pred_bar"].mean()),
                "pred_pressure_window_std": float(part["window_pred_bar"].std(ddof=0)),
                "n_windows": int(len(part)),
            }
        )
    out = pd.DataFrame(rows)
    out["signed_error_bar"] = out["pred_pressure_bar"] - out["actual_pressure_bar"]
    out["abs_error_bar"] = out["signed_error_bar"].abs()
    return out


def nearest_level_by_bike(preds: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    levels_by_bike = {bike: np.asarray(levels, dtype=float) for bike, levels in bike_pressure_levels(labels).items()}
    out = preds.copy()
    actual_levels = []
    pred_levels = []
    for row in out.itertuples(index=False):
        levels = levels_by_bike[row.bike]
        actual_levels.append(levels[np.abs(levels - row.actual_pressure_bar).argmin()])
        pred_levels.append(levels[np.abs(levels - row.pred_pressure_bar).argmin()])
    out["nearest_level_actual_bar"] = np.asarray(actual_levels, dtype=float)
    out["nearest_level_pred_bar"] = np.asarray(pred_levels, dtype=float)
    return out


def pressure_text(value: float) -> str:
    return f"{value:g}"


def metrics_from_run_predictions(run_preds: pd.DataFrame, labels: pd.DataFrame) -> dict[str, float]:
    y_true = run_preds["actual_pressure_bar"].to_numpy(dtype=float)
    y_pred = run_preds["pred_pressure_bar"].to_numpy(dtype=float)
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    group_mae = run_preds.groupby("group")["abs_error_bar"].mean()
    bike_mae = run_preds.groupby("bike")["abs_error_bar"].mean()
    with_levels = nearest_level_by_bike(run_preds, labels)
    level_labels = [pressure_text(value) for value in sorted(labels["pressure_bar"].unique())]
    actual_level_labels = [pressure_text(value) for value in with_levels["nearest_level_actual_bar"].to_numpy(dtype=float)]
    pred_level_labels = [pressure_text(value) for value in with_levels["nearest_level_pred_bar"].to_numpy(dtype=float)]
    return {
        "run_mae_bar": float(np.mean(abs_errors)),
        "run_rmse_bar": math.sqrt(float(np.mean(errors**2))),
        "run_bias_bar": float(np.mean(errors)),
        "run_max_abs_error_bar": float(np.max(abs_errors)),
        "run_error_std_bar": float(np.std(abs_errors, ddof=0)),
        "group_mae_std_bar": float(group_mae.std(ddof=0)) if len(group_mae) > 1 else 0.0,
        "bike_mae_std_bar": float(bike_mae.std(ddof=0)) if len(bike_mae) > 1 else 0.0,
        "nearest_level_accuracy": float(np.mean(with_levels["nearest_level_actual_bar"].to_numpy() == with_levels["nearest_level_pred_bar"].to_numpy())),
        "nearest_level_macro_f1": float(f1_score(actual_level_labels, pred_level_labels, labels=level_labels, average="macro", zero_division=0)),
        "n_runs": int(len(run_preds)),
        "n_windows": int(run_preds["n_windows"].sum()),
    }


def candidate_columns(candidate: Candidate) -> dict[str, Any]:
    config = candidate.feature_config
    return {
        "model_name": candidate.name,
        "stage": candidate.stage,
        "candidate_type": candidate.candidate_type,
        "feature_config": config.name if config else "",
        "feature_space": config.space_name if config else "",
        "n_features_in": len(config.columns) if config else 0,
        "pca_components": config.pca_components if config else np.nan,
        "hidden_layers": str(candidate.hidden_layers) if candidate.hidden_layers else "",
        "hidden_units": sum(candidate.hidden_layers),
        "activation": candidate.activation,
        "alpha": candidate.alpha,
        "aggregate_method": candidate.aggregate_method,
        "n_seeds": len(candidate.seeds),
        "run_balanced_weights": candidate.use_run_balanced_weights if candidate.candidate_type == "ffnn" else False,
    }


def evaluate_group_cv(features: pd.DataFrame, labels: pd.DataFrame, spaces: dict[str, pd.DataFrame], candidates: list[Candidate]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-group-out CV: 5 groups train, 1 group validation."""
    groups = sorted(features["group"].unique())
    metric_rows = []
    pred_rows = []
    for i, candidate in enumerate(candidates, start=1):
        fold_preds = []
        for holdout_group in groups:
            fit_index = features.index[~features["group"].eq(holdout_group)]
            pred_index = features.index[features["group"].eq(holdout_group)]
            fit_features = features.loc[fit_index].reset_index(drop=True)
            pred_features = features.loc[pred_index].reset_index(drop=True)
            pred = predict_candidate(candidate, spaces, fit_features, pred_features, fit_index, pred_index)
            fold_preds.append(aggregate_predictions(pred_features, pred, candidate, "leave_one_group_out_cv", holdout_group))
        run_pred_df = nearest_level_by_bike(pd.concat(fold_preds, ignore_index=True), labels)
        metrics = metrics_from_run_predictions(run_pred_df, labels)
        row = candidate_columns(candidate)
        row.update({f"cv_{key}": value for key, value in metrics.items()})
        metric_rows.append(row)
        pred_rows.append(run_pred_df)
        if i % 5 == 0 or i == len(candidates):
            print(f"Group CV evaluated {i}/{len(candidates)} candidates", flush=True)
    return pd.DataFrame(metric_rows), pd.concat(pred_rows, ignore_index=True)


def add_selection_columns(comparison: pd.DataFrame) -> pd.DataFrame:
    out = comparison.copy()
    out["cv_selection_score"] = (
        out["cv_run_mae_bar"]
        + 0.10 * out["cv_run_rmse_bar"]
        + 0.10 * out["cv_run_bias_bar"].abs()
        + 0.05 * out["cv_run_max_abs_error_bar"]
        + 0.03 * out["cv_group_mae_std_bar"]
    )
    out["model_complexity"] = out["pca_components"].fillna(out["n_features_in"]).astype(float) + out["hidden_units"].astype(float)
    ffnn = out[out["candidate_type"].eq("ffnn")]
    cutoff = float(ffnn["cv_selection_score"].min() * 1.05)
    out["near_best_ffnn_cutoff"] = cutoff
    out["is_near_best_ffnn"] = out["candidate_type"].eq("ffnn") & (out["cv_selection_score"] <= cutoff)
    return out


def model_row_uses_rider_weight(rows: pd.DataFrame) -> pd.Series:
    """Return True for comparison rows whose feature space includes rider weight."""
    return rows["feature_space"].fillna("").str.contains("weight", case=False)


def top_stage1_names_for_ensembles(comparison: pd.DataFrame, n_overall: int = 4, n_weight: int = 4) -> list[str]:
    """Select stage-1 candidates for ensemble reruns, including strong weight models."""
    candidates = comparison[comparison["candidate_type"].eq("ffnn")].copy()
    if "stage" in candidates.columns:
        candidates = candidates[candidates["stage"].eq("stage1")]
    candidates = candidates.sort_values(["cv_selection_score", "cv_run_mae_bar", "model_name"])
    names = candidates.head(n_overall)["model_name"].tolist()
    names.extend(candidates[model_row_uses_rider_weight(candidates)].head(n_weight)["model_name"].tolist())
    return list(dict.fromkeys(names))


def select_model(comparison: pd.DataFrame, require_rider_weight: bool = True) -> str:
    candidates = comparison[comparison["is_near_best_ffnn"]].copy()
    if require_rider_weight:
        weight_candidates = candidates[model_row_uses_rider_weight(candidates)].copy()
        if weight_candidates.empty:
            weight_candidates = comparison[
                comparison["candidate_type"].eq("ffnn") & model_row_uses_rider_weight(comparison)
            ].copy()
        if weight_candidates.empty:
            raise ValueError("No FFNN candidate with rider_weight_kg was available for selection.")
        candidates = weight_candidates
    selected = candidates.sort_values(["model_complexity", "cv_selection_score", "cv_run_mae_bar", "model_name"]).iloc[0]
    return str(selected["model_name"])


def candidate_uses_rider_weight(candidate: Candidate) -> bool:
    return candidate.feature_config is not None and "rider_weight_kg" in candidate.feature_config.columns


def fit_final_model(candidate: Candidate, spaces: dict[str, pd.DataFrame], features: pd.DataFrame) -> tuple[list[Pipeline], pd.DataFrame]:
    """Fit the selected FFNN on all local P1-P4 data for teacher hidden testing."""
    assert candidate.feature_config is not None
    cols = list(candidate.feature_config.columns)
    X_train = spaces[candidate.feature_config.space_name][cols].reset_index(drop=True)
    y_train = features["pressure_bar"].reset_index(drop=True)
    sample_weight = run_balanced_weights(features.reset_index(drop=True)) if candidate.use_run_balanced_weights else None
    pipelines: list[Pipeline] = []
    preds = []
    for seed in candidate.seeds:
        model = build_pipeline(candidate, seed)
        fit_kwargs = {"mlp__sample_weight": sample_weight} if sample_weight is not None else {}
        model.fit(X_train, y_train, **fit_kwargs)
        pipelines.append(model)
        preds.append(np.ravel(model.predict(X_train)))
    pred = np.mean(np.vstack(preds), axis=0)
    training_fit = aggregate_predictions(features.reset_index(drop=True), pred, candidate, "training_pool_fit", "all_groups")
    return pipelines, training_fit


def candidate_by_name(candidates: list[Candidate], name: str) -> Candidate:
    for candidate in candidates:
        if candidate.name == name:
            return candidate
    raise KeyError(f"Candidate not found: {name}")


def build_confusion_rows(preds: pd.DataFrame, labels: pd.DataFrame, selected_model: str) -> pd.DataFrame:
    with_levels = nearest_level_by_bike(preds, labels)
    level_labels = [pressure_text(value) for value in sorted(labels["pressure_bar"].unique())]
    actual_level_labels = [pressure_text(value) for value in with_levels["nearest_level_actual_bar"].to_numpy(dtype=float)]
    pred_level_labels = [pressure_text(value) for value in with_levels["nearest_level_pred_bar"].to_numpy(dtype=float)]
    matrix = confusion_matrix(actual_level_labels, pred_level_labels, labels=level_labels)
    rows = []
    for i, actual in enumerate(level_labels):
        for j, predicted in enumerate(level_labels):
            rows.append({"model_name": selected_model, "actual_level_bar": actual, "predicted_level_bar": predicted, "n_runs": int(matrix[i, j])})
    return pd.DataFrame(rows)


def save_final_model(path: Path, candidate: Candidate, pipelines: list[Pipeline]) -> None:
    assert candidate.feature_config is not None
    payload = {
        "candidate": asdict(candidate),
        "feature_config": asdict(candidate.feature_config),
        "pipelines": pipelines,
        "prediction_note": "Apply the same raw-data processing and feature-space construction before using these pipelines.",
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
