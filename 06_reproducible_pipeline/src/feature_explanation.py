from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from feature_pipeline import make_feature_matrix
from modeling import build_feature_spaces


META_COLS = ["group", "run_id", "bike", "p_number", "pressure_bar"]
FINAL_SPACE_NAME = "compact_weight"


def _safe_corr(left: pd.Series, right: pd.Series, method: str) -> float:
    if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return np.nan
    return float(left.corr(right, method=method))


def _feature_family(feature_name: str) -> str:
    if feature_name.startswith("bike_"):
        return "bike_type"
    if feature_name == "rider_weight_kg":
        return "rider_weight"
    for prefix in ("acc_", "gyro_"):
        if feature_name.startswith(prefix):
            body = feature_name.removeprefix(prefix)
            for suffix in ("_mean", "_max", "_min", "_absdiff"):
                if body.endswith(suffix):
                    return body.removesuffix(suffix)
            return body
    return "other"


def _feature_source(feature_name: str) -> str:
    if feature_name.startswith("acc_"):
        return "acceleration_signal"
    if feature_name.startswith("gyro_"):
        return "gyro_signal"
    if feature_name.startswith("bike_"):
        return "bike_context"
    if feature_name == "rider_weight_kg":
        return "rider_context"
    return "other"


def _selection_reason(feature_name: str) -> str:
    family = _feature_family(feature_name)
    if feature_name.startswith("bike_"):
        return "Bike one-hot captures structural differences and pressure ranges across FAT, ISY, and MTB."
    if feature_name == "rider_weight_kg":
        return "Rider weight changes tire load, deformation, and vibration response, so it is a physically meaningful context input."
    if family in {"rms", "std", "energy_per_s", "p95_abs", "ptp"}:
        return "Amplitude or energy feature: tire pressure can change damping and impact intensity."
    if family in {"dom_freq", "spectral_centroid"}:
        return "Frequency-location feature: tire stiffness can shift dominant vibration frequencies."
    if family in {"spectral_entropy", "band_0p5_3_power", "band_3_8_power", "band_8_15_power", "band_15_30_power"}:
        return "Frequency-distribution feature: tire pressure can change how vibration energy spreads across bands."
    return "Candidate feature retained for model comparison and checked by cross-validation."


def build_run_level_input_frame(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return one row per run with candidate and final input columns."""
    X_all, candidate_cols = make_feature_matrix(features, include_weight=True)
    X_all = pd.DataFrame(X_all.to_numpy(dtype=float), columns=candidate_cols)
    _x_full, _feature_names, spaces = build_feature_spaces(features)
    final_cols = list(spaces[FINAL_SPACE_NAME].columns)

    frame = pd.concat([features[META_COLS].reset_index(drop=True), X_all.reset_index(drop=True)], axis=1).copy()
    agg = {col: "first" for col in META_COLS if col != "run_id"}
    agg.update({col: "median" for col in candidate_cols})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        run_level = frame.groupby("run_id", as_index=False).agg(agg)
    return run_level, candidate_cols, final_cols


def target_correlation_table(run_level: pd.DataFrame, candidate_cols: list[str], final_cols: list[str]) -> pd.DataFrame:
    """Compute target correlations at run level for all candidate inputs."""
    rows: list[dict[str, Any]] = []
    target = run_level["pressure_bar"].astype(float)
    for col in candidate_cols:
        series = run_level[col].astype(float)
        within_values = []
        signed_within = []
        for _bike, part in run_level.groupby("bike"):
            value = _safe_corr(part[col].astype(float), part["pressure_bar"].astype(float), "spearman")
            if np.isfinite(value):
                within_values.append(abs(value))
                signed_within.append(value)
        rows.append(
            {
                "feature_name": col,
                "source_type": _feature_source(col),
                "feature_family": _feature_family(col),
                "selected_final_input": col in final_cols,
                "pearson_to_pressure": _safe_corr(series, target, "pearson"),
                "spearman_to_pressure": _safe_corr(series, target, "spearman"),
                "within_bike_mean_abs_spearman": float(np.mean(within_values)) if within_values else np.nan,
                "within_bike_mean_signed_spearman": float(np.mean(signed_within)) if signed_within else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["abs_pearson_to_pressure"] = out["pearson_to_pressure"].abs()
    out["abs_spearman_to_pressure"] = out["spearman_to_pressure"].abs()
    out = out.sort_values(["selected_final_input", "abs_spearman_to_pressure", "abs_pearson_to_pressure"], ascending=[False, False, False]).reset_index(drop=True)
    out["abs_spearman_rank"] = out["abs_spearman_to_pressure"].rank(method="dense", ascending=False).astype("Int64")
    return out


def final_input_correlation_matrix(run_level: pd.DataFrame, final_cols: list[str]) -> pd.DataFrame:
    return run_level[final_cols].corr(method="spearman").fillna(0.0)


def redundancy_pairs(corr_matrix: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cols = list(corr_matrix.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = float(corr_matrix.loc[left, right])
            if abs(value) >= threshold:
                rows.append(
                    {
                        "feature_left": left,
                        "feature_right": right,
                        "spearman_corr": value,
                        "abs_spearman_corr": abs(value),
                    }
                )
    return pd.DataFrame(rows).sort_values("abs_spearman_corr", ascending=False).reset_index(drop=True)


def final_feature_rationale_table(corr_table: pd.DataFrame, corr_matrix: pd.DataFrame, final_cols: list[str]) -> pd.DataFrame:
    rows = []
    for feature in final_cols:
        row = corr_table[corr_table["feature_name"].eq(feature)].iloc[0].to_dict()
        peer_corrs = corr_matrix.loc[feature].drop(labels=[feature], errors="ignore").abs()
        row["max_abs_peer_spearman"] = float(peer_corrs.max()) if len(peer_corrs) else np.nan
        row["selection_reason"] = _selection_reason(feature)
        rows.append(row)
    out = pd.DataFrame(rows)
    return out[
        [
            "feature_name",
            "source_type",
            "feature_family",
            "selection_reason",
            "pearson_to_pressure",
            "spearman_to_pressure",
            "within_bike_mean_abs_spearman",
            "max_abs_peer_spearman",
            "abs_spearman_rank",
        ]
    ]


def write_feature_explanation_artifacts(features: pd.DataFrame, table_dir: Path, model_dir: Path) -> dict[str, Any]:
    run_level, candidate_cols, final_cols = build_run_level_input_frame(features)
    corr_table = target_correlation_table(run_level, candidate_cols, final_cols)
    matrix = final_input_correlation_matrix(run_level, final_cols)
    pairs = redundancy_pairs(matrix)
    rationale = final_feature_rationale_table(corr_table, matrix, final_cols)

    run_level.to_csv(table_dir / "training_pool_run_level_model_inputs.csv", index=False)
    corr_table.to_csv(table_dir / "training_pool_candidate_feature_target_correlations.csv", index=False)
    matrix.to_csv(table_dir / "training_pool_final_input_correlation_matrix.csv")
    pairs.to_csv(table_dir / "training_pool_high_redundancy_feature_pairs.csv", index=False)
    rationale.to_csv(table_dir / "training_pool_final_input_feature_rationale.csv", index=False)

    summary = {
        "analysis_grain": "run_level_median",
        "n_runs": int(run_level["run_id"].nunique()),
        "n_candidate_inputs": int(len(candidate_cols)),
        "n_final_inputs": int(len(final_cols)),
        "final_feature_space": FINAL_SPACE_NAME,
        "uses_rider_weight": "rider_weight_kg" in final_cols,
        "top_final_inputs_by_abs_spearman": rationale.sort_values("abs_spearman_rank").head(8)["feature_name"].tolist(),
        "n_high_redundancy_pairs_abs_spearman_ge_0p85": int(len(pairs)),
        "caveat": "Correlation is descriptive evidence only; final model choice is still judged by leave-one-group-out CV.",
    }
    (model_dir / "training_pool_feature_explanation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "run_level": run_level,
        "correlations": corr_table,
        "final_correlation_matrix": matrix,
        "redundancy_pairs": pairs,
        "rationale": rationale,
        "summary": summary,
    }
