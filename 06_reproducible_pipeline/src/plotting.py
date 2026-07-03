from __future__ import annotations

from pathlib import Path

from project_config import BIKES, BLUE, FIG_EDA_DIR, FIG_MODEL_DIR, GOLD, OLIVE, ORANGE, SUSPENSION_TYPES, TOKENS

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from feature_pipeline import make_feature_matrix


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        },
    )


def savefig(fig, path: Path, *, rect=(0, 0, 1, 0.88), dpi: int = 220) -> Path:
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def add_header(fig, ax, title: str, subtitle: str, *, top: float = 0.82) -> None:
    ax.set_title("")
    sns.despine(ax=ax)
    fig.subplots_adjust(top=top)
    left = ax.get_position().x0
    fig.text(left, 0.97, title, ha="left", va="top", fontsize=14, fontweight="bold", color=TOKENS["ink"])
    fig.text(left, 0.91, subtitle, ha="left", va="top", fontsize=9, color=TOKENS["muted"])


def plot_pressure_design(labels: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    plot_df = labels.copy()
    plot_df["run_label"] = plot_df["bike"] + " " + plot_df["p_number"]
    pivot = plot_df.pivot_table(index="group", columns="run_label", values="pressure_bar", aggfunc="first")
    pivot = pivot[[f"{bike} P{i}" for bike in BIKES for i in range(1, 5)]]
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap=sns.light_palette(BLUE["mid"], as_cmap=True), cbar_kws={"label": "bar"}, ax=ax)
    ax.set_xlabel("Bike and P number")
    ax.set_ylabel("Group")
    add_header(fig, ax, "Observed pressure design in the local training pool", "All P1-P4 runs are now usable for training; teacher hidden data is the external test set.", top=0.80)
    return savefig(fig, FIG_EDA_DIR / "training_pool_01_pressure_design.png")


def plot_counts(labels: pd.DataFrame, features: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    sns.countplot(data=labels, x="group", color=BLUE["base"], ax=axes[0])
    axes[0].set_title("Run count by group")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("runs")
    window_counts = features.groupby(["group"], as_index=False).size()
    sns.barplot(data=window_counts, x="group", y="size", color=ORANGE["base"], ax=axes[1])
    axes[1].set_title("Window count by group")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("windows")
    return savefig(fig, FIG_EDA_DIR / "training_pool_02_counts.png", dpi=180)


def plot_pca(features: pd.DataFrame) -> Path:
    X, _ = make_feature_matrix(features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    plot_df = features[["group", "run_id", "bike", "p_number", "pressure_bar"]].copy()
    plot_df["PC1"] = coords[:, 0]
    plot_df["PC2"] = coords[:, 1]
    run_df = plot_df.groupby(["group", "run_id", "bike", "p_number", "pressure_bar"], as_index=False)[["PC1", "PC2"]].median()
    fig, ax = plt.subplots(figsize=(9.0, 6.5))
    sns.scatterplot(data=run_df, x="PC1", y="PC2", hue="pressure_bar", style="bike", size="p_number", sizes=(50, 140), palette="viridis", ax=ax)
    ax.axhline(0, color=TOKENS["axis"], linewidth=0.8)
    ax.axvline(0, color=TOKENS["axis"], linewidth=0.8)
    add_header(fig, ax, f"PCA feature space fitted on all local P1-P4 data ({pca.explained_variance_ratio_.sum()*100:.1f}% variance)", "This is an EDA view only; model validation still uses leave-one-group-out CV.", top=0.80)
    return savefig(fig, FIG_EDA_DIR / "training_pool_03_pca_all_local_data.png")


def plot_feature_relationship(run_summary: pd.DataFrame) -> Path:
    candidates = [col for col in run_summary.columns if col.endswith("acc_rms_mean_median")]
    y_col = candidates[0] if candidates else next((col for col in run_summary.columns if col.endswith("_median")), None)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    if y_col is not None:
        sns.scatterplot(data=run_summary, x="pressure_bar", y=y_col, hue="bike", style="p_number", palette=[BLUE["mid"], ORANGE["base"], OLIVE["mid"]], s=90, ax=ax)
        ax.set_ylabel(y_col)
    ax.set_xlabel("Pressure (bar)")
    add_header(fig, ax, "Run-level signal feature versus pressure", "A quick check of whether extracted vibration features move with pressure.", top=0.80)
    return savefig(fig, FIG_EDA_DIR / "training_pool_04_feature_vs_pressure.png")


def short_feature_label(feature_name: str) -> str:
    label = feature_name
    label = label.replace("acc_", "acc ")
    label = label.replace("gyro_", "gyro ")
    label = label.replace("bike_", "bike ")
    label = label.replace("_weight_kg", " weight")
    label = label.replace("_", " ")
    return label


def plot_target_correlation_ranking(correlations: pd.DataFrame) -> Path:
    plot_df = correlations[correlations["selected_final_input"]].copy()
    plot_df = plot_df.sort_values("abs_spearman_to_pressure", ascending=False).head(18)
    plot_df = plot_df.sort_values("abs_spearman_to_pressure", ascending=True)
    plot_df["label"] = plot_df["feature_name"].map(short_feature_label)
    values = plot_df["spearman_to_pressure"].fillna(0.0).to_numpy(dtype=float)
    colors = np.where(values >= 0, OLIVE["base"], ORANGE["base"])
    edges = np.where(values >= 0, OLIVE["dark"], ORANGE["dark"])
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    bars = ax.barh(plot_df["label"], values, color=colors, edgecolor=edges, linewidth=1.0)
    ax.axvline(0, color=TOKENS["ink"], linewidth=1.0)
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Spearman correlation with pressure_bar")
    ax.set_ylabel("")
    for bar, value in zip(bars, values):
        x = value + (0.035 if value >= 0 else -0.035)
        ha = "left" if value >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{value:+.2f}", ha=ha, va="center", fontsize=8, color=TOKENS["ink"])
    add_header(
        fig,
        ax,
        "Final input features show different degrees of monotonic relationship with pressure",
        "Run-level median inputs, n=72 runs. Correlation supports interpretation; final choice is still validated by leave-one-group-out CV.",
        top=0.76,
    )
    return savefig(fig, FIG_EDA_DIR / "training_pool_05_feature_target_correlation.png")


def plot_final_input_correlation_matrix(corr_matrix: pd.DataFrame) -> Path:
    labels = [short_feature_label(col) for col in corr_matrix.columns]
    matrix = corr_matrix.copy()
    fig, ax = plt.subplots(figsize=(14.0, 11.5))
    cmap = sns.diverging_palette(27, 220, s=75, l=65, as_cmap=True)
    sns.heatmap(
        matrix,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.35,
        linecolor=TOKENS["panel"],
        cbar_kws={"label": "Spearman correlation"},
        ax=ax,
    )
    ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=7)
    ax.set_yticklabels(labels, rotation=0, fontsize=7)
    ax.set_xlabel("")
    ax.set_ylabel("")
    add_header(
        fig,
        ax,
        "Final input correlation matrix checks redundancy before modeling",
        "Run-level median compact_weight inputs, n=72 runs. Blocks of high correlation indicate related signal families, not separate proof of causality.",
        top=0.82,
    )
    return savefig(fig, FIG_EDA_DIR / "training_pool_06_final_input_correlation_matrix.png", rect=(0, 0, 1, 0.90), dpi=220)


def plot_cv_selection(comparison: pd.DataFrame, selected_model: str) -> Path:
    plot_df = comparison.sort_values(["cv_selection_score", "cv_run_mae_bar"]).head(14).copy()
    plot_df = plot_df.sort_values("cv_selection_score", ascending=False)
    colors = np.where(plot_df["model_name"].eq(selected_model), ORANGE["base"], np.where(plot_df["candidate_type"].eq("ffnn"), BLUE["base"], GOLD["base"]))
    edges = np.where(plot_df["model_name"].eq(selected_model), ORANGE["dark"], BLUE["dark"])
    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    bars = ax.barh(plot_df["model_name"], plot_df["cv_selection_score"], color=colors, edgecolor=edges, linewidth=1.0)
    ax.axvline(float(plot_df["near_best_ffnn_cutoff"].iloc[0]), color=ORANGE["dark"], linestyle=":", linewidth=1.1)
    ax.set_xlabel("Group-CV selection score (lower is better)")
    ax.set_ylabel("")
    for bar, score, mae in zip(bars, plot_df["cv_selection_score"], plot_df["cv_run_mae_bar"]):
        ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2, f"{score:.2f} | MAE {mae:.2f}", va="center", fontsize=7.4)
    add_header(fig, ax, "FFNN selection uses leave-one-group-out validation", "Orange is selected. Gold rows are simple baselines.", top=0.78)
    return savefig(fig, FIG_MODEL_DIR / "training_pool_ffnn_01_cv_selection.png")


def plot_cv_predictions(selected_cv_preds: pd.DataFrame, selected_model: str) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    ax.plot([0, 3.2], [0, 3.2], color=TOKENS["ink"], linestyle=":", linewidth=1.0, label="perfect")
    palette = {"FAT": BLUE["base"], "ISY": ORANGE["base"], "MTB": OLIVE["base"]}
    edge = {"FAT": BLUE["dark"], "ISY": ORANGE["dark"], "MTB": OLIVE["dark"]}
    for bike, part in selected_cv_preds.groupby("bike", sort=False):
        ax.scatter(part["actual_pressure_bar"], part["pred_pressure_bar"], s=95, color=palette[bike], edgecolor=edge[bike], linewidth=1.0, label=bike)
    ax.set_xlim(0, 3.25)
    ax.set_ylim(0, 3.25)
    ax.set_xlabel("Actual pressure (bar)")
    ax.set_ylabel("CV predicted pressure (bar)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=4, borderaxespad=0)
    add_header(fig, ax, "Leave-one-group-out CV predictions", f"Selected model: {selected_model}. Each point is one held-out run.", top=0.78)
    return savefig(fig, FIG_MODEL_DIR / "training_pool_ffnn_02_cv_predicted_vs_actual.png")


def plot_cv_error_heatmap(selected_cv_preds: pd.DataFrame) -> Path:
    plot_df = selected_cv_preds.copy()
    plot_df["run_slot"] = plot_df["bike"] + " " + plot_df["p_number"]
    pivot = plot_df.pivot(index="group", columns="run_slot", values="signed_error_bar")
    ordered_cols = [f"{bike} P{i}" for bike in BIKES for i in range(1, 5)]
    pivot = pivot[[col for col in ordered_cols if col in pivot.columns]]
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    cmap = sns.diverging_palette(27, 220, s=75, l=65, as_cmap=True)
    sns.heatmap(pivot, annot=True, fmt="+.2f", center=0, cmap=cmap, linewidths=1.0, linecolor=TOKENS["panel"], cbar_kws={"label": "predicted - actual (bar)"}, ax=ax)
    ax.set_xlabel("Bike and P number")
    ax.set_ylabel("Held-out group")
    add_header(fig, ax, "CV signed error by held-out group and run", "Positive cells are over-predictions; negative cells are under-predictions.", top=0.80)
    return savefig(fig, FIG_MODEL_DIR / "training_pool_ffnn_03_cv_error_heatmap.png")


def plot_abs_error_by_bike(selected_cv_preds: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    sns.boxplot(data=selected_cv_preds, x="bike", y="abs_error_bar", color=BLUE["light"], linewidth=1.0, ax=ax)
    sns.stripplot(data=selected_cv_preds, x="bike", y="abs_error_bar", hue="p_number", palette="viridis", dodge=True, size=5, edgecolor=TOKENS["ink"], linewidth=0.4, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Absolute CV error (bar)")
    ax.legend(title="P number", frameon=False, loc="upper right")
    add_header(fig, ax, "CV error spread by bike type", "Each dot is one held-out run from leave-one-group-out CV.", top=0.80)
    return savefig(fig, FIG_MODEL_DIR / "training_pool_ffnn_04_cv_abs_error_by_bike.png")


def plot_confusion(confusion_rows: pd.DataFrame, selected_model: str) -> Path:
    part = confusion_rows[confusion_rows["model_name"].eq(selected_model)].copy()
    matrix = part.pivot(index="actual_level_bar", columns="predicted_level_bar", values="n_runs").fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    cmap = sns.blend_palette([TOKENS["panel"], BLUE["xlight"], BLUE["base"], ORANGE["base"]], as_cmap=True)
    sns.heatmap(matrix, annot=True, fmt="d", cmap=cmap, linewidths=1.0, linecolor=TOKENS["panel"], cbar=False, ax=ax)
    ax.set_xlabel("Nearest predicted pressure level (bar)")
    ax.set_ylabel("Actual pressure level (bar)")
    add_header(fig, ax, "Nearest-level confusion from CV regression outputs", "This is an auxiliary classification view; the main task is continuous regression.", top=0.80)
    return savefig(fig, FIG_MODEL_DIR / "training_pool_ffnn_05_cv_nearest_level_confusion.png")


def plot_suspension_cv_selection(comparison: pd.DataFrame, selected_model: str) -> Path:
    plot_df = comparison.sort_values(["cv_selection_score", "model_complexity", "model_name"]).head(12).copy()
    plot_df = plot_df.sort_values("cv_macro_f1", ascending=True)
    colors = np.where(plot_df["model_name"].eq(selected_model), ORANGE["base"], BLUE["base"])
    edges = np.where(plot_df["model_name"].eq(selected_model), ORANGE["dark"], BLUE["dark"])
    fig, ax = plt.subplots(figsize=(12.0, 6.8))
    bars = ax.barh(plot_df["model_name"], plot_df["cv_macro_f1"], color=colors, edgecolor=edges, linewidth=1.0)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Leave-one-group-out CV macro-F1 (higher is better)")
    ax.set_ylabel("")
    for bar, f1_value, acc in zip(bars, plot_df["cv_macro_f1"], plot_df["cv_run_accuracy"]):
        ax.text(min(f1_value + 0.02, 0.98), bar.get_y() + bar.get_height() / 2, f"F1 {f1_value:.2f} | acc {acc:.2f}", va="center", fontsize=7.4)
    add_header(
        fig,
        ax,
        "Suspension classifier selection uses leakage-safe signal features",
        "Orange is selected. No bike label, suspension label, pressure, p-number, group, run id, file name, or rider weight is used as input.",
        top=0.76,
    )
    return savefig(fig, FIG_MODEL_DIR / "training_pool_suspension_01_cv_selection.png")


def plot_suspension_confusion(confusion_rows: pd.DataFrame, selected_model: str) -> Path:
    part = confusion_rows[confusion_rows["model_name"].eq(selected_model)].copy()
    matrix = part.pivot(index="actual_suspension_type", columns="pred_suspension_type", values="n_runs").reindex(index=SUSPENSION_TYPES, columns=SUSPENSION_TYPES).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    cmap = sns.blend_palette([TOKENS["panel"], BLUE["xlight"], BLUE["base"], ORANGE["base"]], as_cmap=True)
    sns.heatmap(matrix, annot=True, fmt="d", cmap=cmap, linewidths=1.0, linecolor=TOKENS["panel"], cbar=False, ax=ax)
    ax.set_xlabel("Predicted suspension type")
    ax.set_ylabel("Actual suspension type")
    add_header(fig, ax, "Suspension classification confusion matrix", "Run-level predictions from leave-one-group-out CV.", top=0.80)
    return savefig(fig, FIG_MODEL_DIR / "training_pool_suspension_02_cv_confusion.png")


def plot_suspension_confidence(selected_cv_preds: pd.DataFrame) -> Path:
    plot_df = selected_cv_preds.copy()
    plot_df["status"] = np.where(plot_df["is_correct"], "correct", "wrong")
    fig, ax = plt.subplots(figsize=(9.5, 6.3))
    sns.stripplot(
        data=plot_df,
        x="actual_suspension_type",
        y="pred_confidence",
        hue="status",
        order=list(SUSPENSION_TYPES),
        palette={"correct": OLIVE["base"], "wrong": ORANGE["base"]},
        size=8,
        jitter=0.18,
        edgecolor=TOKENS["ink"],
        linewidth=0.5,
        ax=ax,
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Actual suspension type")
    ax.set_ylabel("Run-level predicted confidence")
    ax.legend(title="", frameon=False, loc="lower right")
    add_header(fig, ax, "Suspension classifier confidence by actual class", "Each point is one held-out run; confidence is the averaged window probability of the predicted class.", top=0.80)
    return savefig(fig, FIG_MODEL_DIR / "training_pool_suspension_03_cv_confidence.png")
