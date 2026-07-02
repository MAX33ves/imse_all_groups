from __future__ import annotations

import pickle
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from project_config import BIKES
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_pipeline import signal_feature_columns

warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass(frozen=True)
class BikeFeatureConfig:
    name: str
    space_name: str
    columns: tuple[str, ...]
    pca_components: int | None


@dataclass(frozen=True)
class BikeCandidate:
    name: str
    stage: str
    feature_config: BikeFeatureConfig
    hidden_layers: tuple[int, ...]
    activation: str
    alpha: float
    seeds: tuple[int, ...] = (42,)
    aggregate_method: str = "mean_proba"
    use_run_balanced_weights: bool = True


def compact_signal_columns_for_bike(features: pd.DataFrame) -> list[str]:
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
        for col in signal_feature_columns(features)
        if (col.startswith("acc_") or col.startswith("gyro_")) and any(token in col for token in keep_tokens)
    ]


def clean_feature_space(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.replace([np.inf, -np.inf], np.nan)
    return out.astype(float)


def add_run_medians(features: pd.DataFrame, frame: pd.DataFrame, signal_cols: list[str]) -> pd.DataFrame:
    return frame[signal_cols].groupby(features["run_id"]).transform("median").add_prefix("runmed_")


def build_bike_feature_spaces(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build leakage-safe feature spaces for bike-type classification.

    Deliberately excludes bike label, pressure_bar, p_number, group, run_id,
    file names, and rider_weight_kg. The classifier must learn from signal
    behavior rather than metadata shortcuts.
    """
    signal_cols = signal_feature_columns(features)
    X_signal = features[signal_cols].copy()
    compact_cols = compact_signal_columns_for_bike(features) or signal_cols[: min(32, len(signal_cols))]
    compact_runmed = add_run_medians(features, X_signal, compact_cols)
    spaces = {
        "signal_compact": X_signal[compact_cols].copy(),
        "signal_compact_runmed": pd.concat([X_signal[compact_cols].copy(), compact_runmed], axis=1),
        "signal_full": X_signal.copy(),
    }
    return {name: clean_feature_space(frame) for name, frame in spaces.items()}


def build_bike_feature_configs(spaces: dict[str, pd.DataFrame], n_train_rows: int) -> list[BikeFeatureConfig]:
    specs = [
        ("signal_compact_pca6", "signal_compact", 6),
        ("signal_compact_pca10", "signal_compact", 10),
        ("signal_compact_runmed_pca8", "signal_compact_runmed", 8),
        ("signal_full_pca10", "signal_full", 10),
    ]
    configs: list[BikeFeatureConfig] = []
    for name, space_name, requested in specs:
        frame = spaces[space_name]
        components = min(requested, frame.shape[1], max(2, n_train_rows - 2))
        if components >= 2:
            configs.append(BikeFeatureConfig(name=name, space_name=space_name, columns=tuple(frame.columns), pca_components=components))
    return configs


def hidden_label(hidden_layers: tuple[int, ...]) -> str:
    return "x".join(str(value) for value in hidden_layers)


def bike_candidate_name(stage: str, config: BikeFeatureConfig, hidden_layers: tuple[int, ...], activation: str, alpha: float, seeds: tuple[int, ...]) -> str:
    alpha_label = f"{alpha:g}".replace(".", "p")
    seed_label = f"ens{len(seeds)}" if len(seeds) > 1 else f"s{seeds[0]}"
    return f"bike_type_ffnn_{stage}_{seed_label}_{config.name}_{activation}_h{hidden_label(hidden_layers)}_a{alpha_label}"


def build_bike_stage1_candidates(configs: list[BikeFeatureConfig]) -> list[BikeCandidate]:
    network_grid = [
        ((6,), "tanh", 1.0),
        ((8, 4), "tanh", 1.0),
        ((12, 6), "tanh", 1.0),
    ]
    out: list[BikeCandidate] = []
    for config in configs:
        for hidden_layers, activation, alpha in network_grid:
            seeds = (42,)
            out.append(
                BikeCandidate(
                    name=bike_candidate_name("stage1", config, hidden_layers, activation, alpha, seeds),
                    stage="stage1",
                    feature_config=config,
                    hidden_layers=hidden_layers,
                    activation=activation,
                    alpha=alpha,
                    seeds=seeds,
                )
            )
    return out


def build_bike_ensemble_candidates(top_candidates: list[BikeCandidate]) -> list[BikeCandidate]:
    seeds = (11, 42, 91)
    out: list[BikeCandidate] = []
    seen = set()
    for base in top_candidates:
        key = (base.feature_config.name, base.hidden_layers, base.activation, base.alpha)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            BikeCandidate(
                name=bike_candidate_name("ensemble", base.feature_config, base.hidden_layers, base.activation, base.alpha, seeds),
                stage="ensemble",
                feature_config=base.feature_config,
                hidden_layers=base.hidden_layers,
                activation=base.activation,
                alpha=base.alpha,
                seeds=seeds,
            )
        )
    return out


def build_bike_pipeline(candidate: BikeCandidate, seed: int) -> Pipeline:
    steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if candidate.feature_config.pca_components is not None:
        steps.append(("pca", PCA(n_components=candidate.feature_config.pca_components, random_state=seed)))
    steps.append(
        (
            "mlp",
            MLPClassifier(
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
    counts = meta_train["run_id"].map(meta_train["run_id"].value_counts()).astype(float)
    weights = 1.0 / counts.to_numpy(dtype=float)
    return weights / weights.mean()


def _aligned_proba(model: Pipeline, X_pred: pd.DataFrame, class_labels: list[str]) -> np.ndarray:
    proba = model.predict_proba(X_pred)
    model_labels = list(model.named_steps["mlp"].classes_)
    out = np.zeros((len(X_pred), len(class_labels)), dtype=float)
    for source_idx, label in enumerate(model_labels):
        target_idx = class_labels.index(label)
        out[:, target_idx] = proba[:, source_idx]
    return out


def fit_predict_bike(candidate: BikeCandidate, X_train: pd.DataFrame, y_train: pd.Series, meta_train: pd.DataFrame, X_pred: pd.DataFrame, class_labels: list[str]) -> np.ndarray:
    predictions = []
    sample_weight = run_balanced_weights(meta_train) if candidate.use_run_balanced_weights else None
    for seed in candidate.seeds:
        model = clone(build_bike_pipeline(candidate, seed))
        fit_kwargs = {"mlp__sample_weight": sample_weight} if sample_weight is not None else {}
        model.fit(X_train, y_train, **fit_kwargs)
        predictions.append(_aligned_proba(model, X_pred, class_labels))
    return np.mean(np.stack(predictions, axis=0), axis=0)


def aggregate_bike_predictions(window_meta: pd.DataFrame, probabilities: np.ndarray, candidate: BikeCandidate, class_labels: list[str], phase: str, fold: str = "") -> pd.DataFrame:
    pred_df = window_meta[["group", "run_id", "bike", "p_number", "dataset_role"]].copy()
    for idx, label in enumerate(class_labels):
        pred_df[f"window_proba_{label}"] = probabilities[:, idx]
    rows = []
    for run_id, part in pred_df.groupby("run_id", sort=False):
        class_probas = np.asarray([part[f"window_proba_{label}"].mean() for label in class_labels], dtype=float)
        pred_idx = int(np.argmax(class_probas))
        row = {
            "phase": phase,
            "fold": fold,
            "model_name": candidate.name,
            "stage": candidate.stage,
            "group": part["group"].iloc[0],
            "run_id": run_id,
            "p_number": part["p_number"].iloc[0],
            "dataset_role": part["dataset_role"].iloc[0],
            "actual_bike": part["bike"].iloc[0],
            "pred_bike": class_labels[pred_idx],
            "pred_confidence": float(class_probas[pred_idx]),
            "n_windows": int(len(part)),
        }
        for idx, label in enumerate(class_labels):
            row[f"proba_{label}"] = float(class_probas[idx])
        rows.append(row)
    out = pd.DataFrame(rows)
    out["is_correct"] = out["actual_bike"].eq(out["pred_bike"])
    return out


def bike_metrics(run_preds: pd.DataFrame, class_labels: list[str]) -> dict[str, float]:
    y_true = run_preds["actual_bike"].astype(str).to_numpy()
    y_pred = run_preds["pred_bike"].astype(str).to_numpy()
    group_acc = run_preds.groupby("group")["is_correct"].mean()
    return {
        "run_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0)),
        "min_group_accuracy": float(group_acc.min()),
        "group_accuracy_std": float(group_acc.std(ddof=0)) if len(group_acc) > 1 else 0.0,
        "mean_confidence": float(run_preds["pred_confidence"].mean()),
        "n_runs": int(len(run_preds)),
        "n_windows": int(run_preds["n_windows"].sum()),
    }


def bike_candidate_columns(candidate: BikeCandidate) -> dict[str, Any]:
    config = candidate.feature_config
    return {
        "model_name": candidate.name,
        "stage": candidate.stage,
        "candidate_type": "ffnn_classifier",
        "feature_config": config.name,
        "feature_space": config.space_name,
        "n_features_in": len(config.columns),
        "pca_components": config.pca_components if config.pca_components is not None else np.nan,
        "hidden_layers": str(candidate.hidden_layers),
        "hidden_units": sum(candidate.hidden_layers),
        "activation": candidate.activation,
        "alpha": candidate.alpha,
        "aggregate_method": candidate.aggregate_method,
        "n_seeds": len(candidate.seeds),
        "run_balanced_weights": candidate.use_run_balanced_weights,
    }


def evaluate_bike_group_cv(features: pd.DataFrame, spaces: dict[str, pd.DataFrame], candidates: list[BikeCandidate], class_labels: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_labels = class_labels or list(BIKES)
    groups = sorted(features["group"].unique())
    metric_rows = []
    pred_rows = []
    for i, candidate in enumerate(candidates, start=1):
        fold_preds = []
        frame = spaces[candidate.feature_config.space_name]
        cols = list(candidate.feature_config.columns)
        for holdout_group in groups:
            fit_index = features.index[~features["group"].eq(holdout_group)]
            pred_index = features.index[features["group"].eq(holdout_group)]
            fit_features = features.loc[fit_index].reset_index(drop=True)
            pred_features = features.loc[pred_index].reset_index(drop=True)
            probabilities = fit_predict_bike(
                candidate,
                frame.loc[fit_index, cols].reset_index(drop=True),
                fit_features["bike"].reset_index(drop=True),
                fit_features,
                frame.loc[pred_index, cols].reset_index(drop=True),
                class_labels,
            )
            fold_preds.append(aggregate_bike_predictions(pred_features, probabilities, candidate, class_labels, "leave_one_group_out_cv", holdout_group))
        run_pred_df = pd.concat(fold_preds, ignore_index=True)
        metrics = bike_metrics(run_pred_df, class_labels)
        row = bike_candidate_columns(candidate)
        row.update({f"cv_{key}": value for key, value in metrics.items()})
        metric_rows.append(row)
        pred_rows.append(run_pred_df)
        if i % 5 == 0 or i == len(candidates):
            print(f"Bike-type CV evaluated {i}/{len(candidates)} candidates", flush=True)
    return pd.DataFrame(metric_rows), pd.concat(pred_rows, ignore_index=True)


def add_bike_selection_columns(comparison: pd.DataFrame) -> pd.DataFrame:
    out = comparison.copy()
    out["cv_selection_score"] = (
        (1.0 - out["cv_macro_f1"])
        + 0.10 * (1.0 - out["cv_run_accuracy"])
        + 0.05 * (1.0 - out["cv_min_group_accuracy"])
        + 0.01 * out["cv_group_accuracy_std"]
    )
    out["model_complexity"] = out["pca_components"].fillna(out["n_features_in"]).astype(float) + out["hidden_units"].astype(float)
    best = float(out["cv_selection_score"].min())
    out["near_best_cutoff"] = best + 0.02
    out["is_near_best"] = out["cv_selection_score"] <= out["near_best_cutoff"]
    return out


def top_bike_stage1_names_for_ensembles(comparison: pd.DataFrame, top_n: int = 4) -> list[str]:
    stage1 = comparison[comparison["stage"].eq("stage1")].copy()
    stage1 = stage1.sort_values(["cv_selection_score", "model_complexity", "model_name"])
    return stage1.head(top_n)["model_name"].tolist()


def select_bike_model(comparison: pd.DataFrame) -> str:
    candidates = comparison[comparison["is_near_best"]].copy()
    selected = candidates.sort_values(["model_complexity", "cv_selection_score", "model_name"]).iloc[0]
    return str(selected["model_name"])


def bike_candidate_by_name(candidates: list[BikeCandidate], name: str) -> BikeCandidate:
    for candidate in candidates:
        if candidate.name == name:
            return candidate
    raise KeyError(f"Bike candidate not found: {name}")


def fit_final_bike_model(candidate: BikeCandidate, spaces: dict[str, pd.DataFrame], features: pd.DataFrame, class_labels: list[str] | None = None) -> tuple[list[Pipeline], pd.DataFrame]:
    class_labels = class_labels or list(BIKES)
    cols = list(candidate.feature_config.columns)
    frame = spaces[candidate.feature_config.space_name]
    X_train = frame[cols].reset_index(drop=True)
    y_train = features["bike"].reset_index(drop=True)
    sample_weight = run_balanced_weights(features.reset_index(drop=True)) if candidate.use_run_balanced_weights else None
    pipelines: list[Pipeline] = []
    probs = []
    for seed in candidate.seeds:
        model = build_bike_pipeline(candidate, seed)
        fit_kwargs = {"mlp__sample_weight": sample_weight} if sample_weight is not None else {}
        model.fit(X_train, y_train, **fit_kwargs)
        pipelines.append(model)
        probs.append(_aligned_proba(model, X_train, class_labels))
    probabilities = np.mean(np.stack(probs, axis=0), axis=0)
    training_fit = aggregate_bike_predictions(features.reset_index(drop=True), probabilities, candidate, class_labels, "training_pool_fit", "all_groups")
    return pipelines, training_fit


def build_bike_confusion_rows(preds: pd.DataFrame, selected_model: str, class_labels: list[str] | None = None) -> pd.DataFrame:
    class_labels = class_labels or list(BIKES)
    matrix = confusion_matrix(preds["actual_bike"], preds["pred_bike"], labels=class_labels)
    rows = []
    for i, actual in enumerate(class_labels):
        for j, predicted in enumerate(class_labels):
            rows.append({"model_name": selected_model, "actual_bike": actual, "pred_bike": predicted, "n_runs": int(matrix[i, j])})
    return pd.DataFrame(rows)


def save_final_bike_model(path: Path, candidate: BikeCandidate, pipelines: list[Pipeline], class_labels: list[str] | None = None) -> None:
    payload = {
        "candidate": asdict(candidate),
        "feature_config": asdict(candidate.feature_config),
        "class_labels": class_labels or list(BIKES),
        "forbidden_inputs": ["bike", "pressure_bar", "p_number", "group", "run_id", "file name", "rider_weight_kg"],
        "pipelines": pipelines,
        "prediction_note": "Apply the same raw-data processing and leakage-safe signal feature-space construction before using these pipelines.",
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
