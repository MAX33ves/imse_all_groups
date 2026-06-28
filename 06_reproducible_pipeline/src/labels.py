from __future__ import annotations

import pandas as pd

from project_config import BIKES, GROUP_DATA, LOCAL_DATA_ROLE


def pressure_label(value: float) -> str:
    """Format pressure for human-readable tables without trailing .0."""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def labels_frame() -> pd.DataFrame:
    """Build the canonical run-level label table from Measurement Details."""
    rows = []
    for group, bikes in GROUP_DATA.items():
        for bike, records in bikes.items():
            for p_number, pressure_bar, weight_kg, ride_time_s in records:
                rows.append(
                    {
                        "group": group,
                        "group_id": int(group[1:]),
                        "bike": bike,
                        "p_number": p_number,
                        "p_order": int(p_number[1:]),
                        "run_id": f"{group}_{bike}_{p_number}",
                        "pressure_bar": float(pressure_bar),
                        "pressure_label": pressure_label(float(pressure_bar)),
                        "rider_weight_kg": float(weight_kg),
                        "table_ride_time_s": float(ride_time_s),
                        "dataset_role": LOCAL_DATA_ROLE,
                    }
                )
    labels = pd.DataFrame(rows).sort_values(["group_id", "bike", "p_order"]).reset_index(drop=True)

    # These columns document pressure coverage now that P1-P4 are all available
    # for local training. They are not model inputs.
    pressure_counts = labels.groupby(["bike", "pressure_bar"])["run_id"].transform("count")
    labels["pressure_count_by_bike"] = pressure_counts.astype(int)
    labels["pressure_seen_in_training_pool"] = True
    labels["validation_plan"] = "held out only when its group is the validation fold"
    return labels


def bike_pressure_levels(labels: pd.DataFrame) -> dict[str, list[float]]:
    """Observed pressure levels per bike, used for auxiliary nearest-level metrics."""
    return {
        bike: sorted(labels.loc[labels["bike"].eq(bike), "pressure_bar"].unique().astype(float).tolist())
        for bike in BIKES
    }
