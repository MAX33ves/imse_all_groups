from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from data_io import read_sagemotion_csv
from feature_pipeline import signal_feature_columns
from modeling import aggregate_values, build_feature_spaces
from project_config import BIKES, MODEL_DIR, REPORT_DIR, SUSPENSION_BY_BIKE, TABLE_DIR, WORKSPACE_ROOT, ensure_dirs
from signal_features import (
    acc_magnitude,
    aggregate_sensor_pair,
    crop_active_window,
    gyro_magnitude,
    sensor_features,
    window_starts,
)
from step_logging import run_logged_action
from suspension_modeling import bike_for_suspension_label, proba_column


TEST_DATA_DIR = WORKSPACE_ROOT / "Test data" / "Grp_2"
TEST_CASE_INFO = {
    "Case 1": {"rider_weight_kg": 75.0, "table_ride_time_s": 9.36},
    "Case 2": {"rider_weight_kg": 100.0, "table_ride_time_s": 8.61},
    "Case 3": {"rider_weight_kg": 89.0, "table_ride_time_s": 4.97},
}
DATASET_ROLE = "teacher_test"
CLASS_LABELS = list(SUSPENSION_BY_BIKE.values())


def extract_test_case_features(window_s: float = 1.0, overlap: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []

    for case_id, info in TEST_CASE_INFO.items():
        csv_path = TEST_DATA_DIR / "Sagemotion" / f"{case_id}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing Sagemotion test file: {csv_path}")

        df = read_sagemotion_csv(csv_path)
        ride_time_s = float(info["table_ride_time_s"])
        start, end, fs, _energy = crop_active_window(df, ride_time_s)
        active = df.iloc[start:end].reset_index(drop=True)
        window_n = max(12, int(round(window_s * fs))) if np.isfinite(fs) and fs > 0 else len(active)
        step_n = max(1, int(round(window_n * (1.0 - overlap))))
        run_id = f"TEST_{case_id.replace(' ', '_').upper()}"

        crop_rows.append(
            {
                "case_id": case_id,
                "run_id": run_id,
                "source_file": str(csv_path.relative_to(WORKSPACE_ROOT)),
                "rider_weight_kg": float(info["rider_weight_kg"]),
                "table_ride_time_s": ride_time_s,
                "sample_rate_hz": fs,
                "raw_n_rows": int(len(df)),
                "active_start_idx": start,
                "active_end_idx": end,
                "active_n_rows": end - start,
                "active_duration_s": float((end - start) / fs) if np.isfinite(fs) and fs > 0 else np.nan,
            }
        )

        for window_idx, w_start in enumerate(window_starts(len(active), window_n, step_n)):
            w_end = min(len(active), w_start + window_n)
            part = active.iloc[w_start:w_end].reset_index(drop=True)
            acc1 = sensor_features(acc_magnitude(part, "1"), fs, "acc1")
            acc2 = sensor_features(acc_magnitude(part, "2"), fs, "acc2")
            gyro1 = sensor_features(gyro_magnitude(part, "1"), fs, "gyro1")
            gyro2 = sensor_features(gyro_magnitude(part, "2"), fs, "gyro2")
            feature_values = aggregate_sensor_pair([acc1, acc2])
            feature_values.update(aggregate_sensor_pair([gyro1, gyro2]))

            feature_rows.append(
                {
                    "case_id": case_id,
                    "group": "TEST",
                    "group_id": 0,
                    "run_id": run_id,
                    "window_id": f"{run_id}_w{window_idx:03d}",
                    "bike": "UNKNOWN",
                    "p_number": case_id.replace(" ", "_"),
                    "p_order": int(case_id.split()[-1]),
                    "dataset_role": DATASET_ROLE,
                    "pressure_bar": np.nan,
                    "rider_weight_kg": float(info["rider_weight_kg"]),
                    "table_ride_time_s": ride_time_s,
                    "sample_rate_hz": fs,
                    "window_start_idx": int(start + w_start),
                    "window_end_idx": int(start + w_end),
                    "window_s": float((w_end - w_start) / fs) if np.isfinite(fs) and fs > 0 else np.nan,
                    **feature_values,
                }
            )

    return pd.DataFrame(feature_rows), pd.DataFrame(crop_rows)


def load_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def align_numeric_frame(frame: pd.DataFrame, columns: list[str], fill_values: pd.Series | None = None) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    out = out[columns].replace([np.inf, -np.inf], np.nan)
    if fill_values is not None:
        out = out.fillna(fill_values.reindex(columns))
    out = out.fillna(out.median(numeric_only=True))
    out = out.fillna(0.0)
    return out.astype(float)


def predict_suspension(features: pd.DataFrame, payload: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = list(payload["feature_config"]["columns"])
    class_labels = list(payload["class_labels"])
    X = align_numeric_frame(features[signal_feature_columns(features)], columns)

    probs = []
    for model in payload["pipelines"]:
        raw = model.predict_proba(X)
        model_labels = list(model.named_steps["mlp"].classes_)
        aligned = np.zeros((len(X), len(class_labels)), dtype=float)
        for source_idx, label in enumerate(model_labels):
            aligned[:, class_labels.index(label)] = raw[:, source_idx]
        probs.append(aligned)
    probabilities = np.mean(np.stack(probs, axis=0), axis=0)

    window_out = features[["case_id", "run_id", "window_id"]].copy()
    for idx, label in enumerate(class_labels):
        window_out[proba_column(label)] = probabilities[:, idx]

    rows = []
    for run_id, part in window_out.groupby("run_id", sort=False):
        class_probas = np.asarray([part[proba_column(label)].mean() for label in class_labels], dtype=float)
        pred_idx = int(np.argmax(class_probas))
        pred_suspension = class_labels[pred_idx]
        rows.append(
            {
                "case_id": part["case_id"].iloc[0],
                "run_id": run_id,
                "pred_suspension_type": pred_suspension,
                "pred_bike_context": bike_for_suspension_label(pred_suspension),
                "suspension_confidence": float(class_probas[pred_idx]),
                **{proba_column(label): float(class_probas[idx]) for idx, label in enumerate(class_labels)},
                "n_windows": int(len(part)),
            }
        )
    return pd.DataFrame(rows), window_out


def transform_before_classifier(model, X: pd.DataFrame) -> np.ndarray:
    Xt: Any = X
    for _name, step in model.steps[:-1]:
        Xt = step.transform(Xt)
    return np.asarray(Xt, dtype=float)


def build_signal_similarity_tables(
    training_features: pd.DataFrame,
    test_features: pd.DataFrame,
    suspension_payload: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare test cases with training runs in the fitted suspension PCA space."""
    columns = list(suspension_payload["feature_config"]["columns"])
    model = suspension_payload["pipelines"][0]

    X_train = align_numeric_frame(training_features[signal_feature_columns(training_features)], columns)
    X_test = align_numeric_frame(test_features[signal_feature_columns(test_features)], columns)
    train_emb = transform_before_classifier(model, X_train)
    test_emb = transform_before_classifier(model, X_test)
    emb_cols = [f"suspension_pc{i + 1}" for i in range(train_emb.shape[1])]

    train_frame = pd.concat(
        [
            training_features[["group", "run_id", "bike", "p_number"]].reset_index(drop=True),
            pd.DataFrame(train_emb, columns=emb_cols),
        ],
        axis=1,
    )
    train_frame["suspension_type"] = train_frame["bike"].map(SUSPENSION_BY_BIKE)
    train_runs = (
        train_frame.groupby(["group", "run_id", "bike", "p_number", "suspension_type"], as_index=False)[emb_cols]
        .median()
        .reset_index(drop=True)
    )

    test_frame = pd.concat(
        [
            test_features[["case_id", "run_id"]].reset_index(drop=True),
            pd.DataFrame(test_emb, columns=emb_cols),
        ],
        axis=1,
    )
    test_runs = test_frame.groupby(["case_id", "run_id"], as_index=False)[emb_cols].median().reset_index(drop=True)

    nearest_rows = []
    centroid_rows = []
    centroids = train_runs.groupby(["bike", "suspension_type"], as_index=False)[emb_cols].mean()
    for test_row in test_runs.itertuples(index=False):
        test_vec = np.asarray([getattr(test_row, col) for col in emb_cols], dtype=float)
        train_dist = train_runs.copy()
        train_matrix = train_dist[emb_cols].to_numpy(dtype=float)
        train_dist["distance"] = np.linalg.norm(train_matrix - test_vec, axis=1)
        for rank, near in enumerate(train_dist.sort_values("distance").head(5).itertuples(index=False), start=1):
            nearest_rows.append(
                {
                    "case_id": test_row.case_id,
                    "rank": rank,
                    "nearest_training_run": near.run_id,
                    "nearest_group": near.group,
                    "nearest_bike": near.bike,
                    "nearest_p_number": near.p_number,
                    "nearest_suspension_type": near.suspension_type,
                    "pca_distance": float(near.distance),
                }
            )
        centroid_dist = centroids.copy()
        centroid_matrix = centroid_dist[emb_cols].to_numpy(dtype=float)
        centroid_dist["centroid_distance"] = np.linalg.norm(centroid_matrix - test_vec, axis=1)
        for row in centroid_dist.sort_values("centroid_distance").itertuples(index=False):
            centroid_rows.append(
                {
                    "case_id": test_row.case_id,
                    "bike_centroid": row.bike,
                    "suspension_centroid": row.suspension_type,
                    "centroid_distance": float(row.centroid_distance),
                }
            )

    return pd.DataFrame(nearest_rows), pd.DataFrame(centroid_rows)


def build_pressure_frame(features: pd.DataFrame, bike_context: str) -> pd.DataFrame:
    out = features[signal_feature_columns(features)].copy()
    for bike in BIKES:
        out[f"bike_{bike}"] = 1.0 if bike == bike_context else 0.0
    out["rider_weight_kg"] = features["rider_weight_kg"].astype(float)
    return out


def predict_pressure_scenarios(features: pd.DataFrame, payload: dict[str, Any], training_space_medians: pd.Series) -> pd.DataFrame:
    columns = list(payload["feature_config"]["columns"])
    aggregate_method = str(payload["candidate"].get("aggregate_method", "median"))
    rows = []

    for case_id, case_part in features.groupby("case_id", sort=False):
        for bike_context in BIKES:
            frame = build_pressure_frame(case_part, bike_context)
            X = align_numeric_frame(frame, columns, training_space_medians)
            preds = [np.ravel(model.predict(X)) for model in payload["pipelines"]]
            window_pred = np.mean(np.vstack(preds), axis=0)
            rows.append(
                {
                    "case_id": case_id,
                    "run_id": case_part["run_id"].iloc[0],
                    "bike_context_for_pressure_model": bike_context,
                    "pred_pressure_bar": aggregate_values(pd.Series(window_pred), aggregate_method),
                    "pred_pressure_mean_bar": float(np.mean(window_pred)),
                    "pred_pressure_window_std": float(np.std(window_pred, ddof=0)),
                    "n_windows": int(len(window_pred)),
                }
            )
    return pd.DataFrame(rows)


def nearest_known_pressure_level(bike_context: str, pressure: float) -> float:
    levels = {
        "FAT": np.asarray([0.4, 0.6, 0.8], dtype=float),
        "ISY": np.asarray([1.0, 2.0, 3.0], dtype=float),
        "MTB": np.asarray([1.0, 2.0, 3.0], dtype=float),
    }[bike_context]
    return float(levels[np.abs(levels - float(pressure)).argmin()])


def build_final_predictions(suspension_preds: pd.DataFrame, pressure_scenarios: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scenario_lookup = {
        (row.case_id, row.bike_context_for_pressure_model): row
        for row in pressure_scenarios.itertuples(index=False)
    }
    for row in suspension_preds.itertuples(index=False):
        pressure_row = scenario_lookup[(row.case_id, row.pred_bike_context)]
        pred_pressure = float(pressure_row.pred_pressure_bar)
        rows.append(
            {
                "case_id": row.case_id,
                "rider_weight_kg": TEST_CASE_INFO[row.case_id]["rider_weight_kg"],
                "table_ride_time_s": TEST_CASE_INFO[row.case_id]["table_ride_time_s"],
                "pred_pressure_bar": pred_pressure,
                "nearest_known_pressure_level_bar": nearest_known_pressure_level(row.pred_bike_context, pred_pressure),
                "pred_suspension_type": row.pred_suspension_type,
                "pred_bike_context_for_pressure_model": row.pred_bike_context,
                "suspension_confidence": row.suspension_confidence,
                "n_windows": row.n_windows,
                "pressure_note": "Pressure is predicted using the suspension-classifier bike context because the hidden test sheet does not provide bike labels.",
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))
    header = "| " + " | ".join(display.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, sep, *body])


def write_report(
    final_preds: pd.DataFrame,
    pressure_scenarios: pd.DataFrame,
    crop_summary: pd.DataFrame,
    nearest_runs: pd.DataFrame,
    centroid_distances: pd.DataFrame,
    report_path: Path,
) -> None:
    scenario_cols = ["case_id", "bike_context_for_pressure_model", "pred_pressure_bar", "pred_pressure_window_std", "n_windows"]
    final_cols = [
        "case_id",
        "rider_weight_kg",
        "table_ride_time_s",
        "pred_pressure_bar",
        "nearest_known_pressure_level_bar",
        "pred_suspension_type",
        "pred_bike_context_for_pressure_model",
        "suspension_confidence",
        "n_windows",
    ]
    lines = [
        "# Teacher Test Case Predictions / 老师测试集预测结果",
        "",
        "## Inputs / 输入信息",
        "",
        "The instructor supplied rider weight and ride-time metadata on 2026-07-06. The saved final models were not retrained; they were only applied to the teacher test cases.",
        "老师在 2026-07-06 提供了测试集的 rider weight 和 ride time。这里没有重新训练模型，只是把已经保存的最终模型应用到老师测试数据上。",
        "",
        markdown_table(crop_summary[["case_id", "rider_weight_kg", "table_ride_time_s", "sample_rate_hz", "raw_n_rows", "active_start_idx", "active_end_idx", "active_duration_s"]]),
        "",
        "## Final Predictions / 最终预测",
        "",
        markdown_table(final_preds[final_cols]),
        "",
        "## Pressure Sensitivity By Bike Context / 不同车型上下文下的胎压敏感性",
        "",
        "The pressure FFNN requires bike one-hot inputs. Because the hidden test sheet does not provide bike labels, the final pressure column uses the bike context implied by the suspension classifier. The table below keeps all three context scenarios for auditability.",
        "胎压 FFNN 需要 bike one-hot 输入。由于隐藏测试表没有给出 bike label，最终胎压使用悬挂分类器推断出的 bike context。下表保留三种 context 的胎压情景，方便审查。",
        "",
        markdown_table(pressure_scenarios[scenario_cols]),
        "",
        "## Case 3 Comment / Case 3 说明",
        "",
        "Case 3 is treated as a new bicycle. The model can compare its signal pattern with the learned FAT/ISY/MTB suspension categories, but the pressure result remains an approximation because the pressure regressor was trained with known bike one-hot context.",
        "Case 3 是新单车。模型可以把它的信号模式与已学习的 FAT/ISY/MTB 悬挂类别比较，但胎压结果仍是近似值，因为胎压回归模型训练时使用了已知 bike one-hot 上下文。",
        "",
        "Nearest training runs in the fitted suspension-model PCA space:",
        "",
        markdown_table(nearest_runs[nearest_runs["case_id"].eq("Case 3")].head(5)),
        "",
        "Distances to training bike/suspension centroids:",
        "",
        markdown_table(centroid_distances[centroid_distances["case_id"].eq("Case 3")]),
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    def action(emit):
        ensure_dirs(TABLE_DIR, MODEL_DIR, REPORT_DIR)

        features, crop_summary = extract_test_case_features()
        features.to_csv(TABLE_DIR / "teacher_test_case_window_features.csv", index=False)
        crop_summary.to_csv(TABLE_DIR / "teacher_test_case_active_window_summary.csv", index=False)

        suspension_payload = load_payload(MODEL_DIR / "training_pool_suspension_final_model.pkl")
        pressure_payload = load_payload(MODEL_DIR / "training_pool_ffnn_final_model.pkl")

        suspension_preds, suspension_window_probs = predict_suspension(features, suspension_payload)
        suspension_preds.to_csv(TABLE_DIR / "teacher_test_case_suspension_predictions.csv", index=False)
        suspension_window_probs.to_csv(TABLE_DIR / "teacher_test_case_suspension_window_probabilities.csv", index=False)

        training_features = pd.read_csv(TABLE_DIR / "training_pool_window_features.csv")
        nearest_runs, centroid_distances = build_signal_similarity_tables(training_features, features, suspension_payload)
        nearest_runs.to_csv(TABLE_DIR / "teacher_test_case_nearest_training_runs.csv", index=False)
        centroid_distances.to_csv(TABLE_DIR / "teacher_test_case_suspension_centroid_distances.csv", index=False)

        _, _, training_spaces = build_feature_spaces(training_features)
        selected_pressure_space = pressure_payload["feature_config"]["space_name"]
        training_space_medians = training_spaces[selected_pressure_space].median(numeric_only=True)

        pressure_scenarios = predict_pressure_scenarios(features, pressure_payload, training_space_medians)
        pressure_scenarios.to_csv(TABLE_DIR / "teacher_test_case_pressure_scenarios.csv", index=False)

        final_preds = build_final_predictions(suspension_preds, pressure_scenarios)
        final_preds.to_csv(TABLE_DIR / "teacher_test_case_final_predictions.csv", index=False)

        summary = {
            "metadata_source": "Instructor email screenshot supplied by user on 2026-07-06.",
            "test_case_info": TEST_CASE_INFO,
            "pressure_model": pressure_payload["candidate"]["name"],
            "suspension_model": suspension_payload["candidate"]["name"],
            "important_caveat": "Pressure predictions use the suspension-classifier bike context because the hidden test sheet does not provide bike labels.",
            "outputs": {
                "final_predictions_csv": str(TABLE_DIR / "teacher_test_case_final_predictions.csv"),
                "pressure_scenarios_csv": str(TABLE_DIR / "teacher_test_case_pressure_scenarios.csv"),
                "suspension_predictions_csv": str(TABLE_DIR / "teacher_test_case_suspension_predictions.csv"),
                "nearest_training_runs_csv": str(TABLE_DIR / "teacher_test_case_nearest_training_runs.csv"),
                "suspension_centroid_distances_csv": str(TABLE_DIR / "teacher_test_case_suspension_centroid_distances.csv"),
                "report_md": str(REPORT_DIR / "teacher_test_case_predictions_bilingual.md"),
            },
        }
        (MODEL_DIR / "teacher_test_case_prediction_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        write_report(
            final_preds,
            pressure_scenarios,
            crop_summary,
            nearest_runs,
            centroid_distances,
            REPORT_DIR / "teacher_test_case_predictions_bilingual.md",
        )

        emit("Final predictions:")
        for row in final_preds.itertuples(index=False):
            emit(
                f"{row.case_id}: pressure={row.pred_pressure_bar:.3f} bar, "
                f"suspension={row.pred_suspension_type}, context={row.pred_bike_context_for_pressure_model}, "
                f"confidence={row.suspension_confidence:.3f}"
            )

    run_logged_action("08_predict_teacher_test_cases", action)


if __name__ == "__main__":
    main()
