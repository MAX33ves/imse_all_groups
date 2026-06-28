from __future__ import annotations

import math
import re

from data_io import median_sample_rate

import numpy as np
import pandas as pd
from scipy import signal, stats


def vector_columns(df: pd.DataFrame, prefixes: list[str], suffix: str) -> list[str]:
    """Find X/Y/Z vector columns, accepting both old and new column prefixes."""
    for prefix in prefixes:
        cols = [f"{prefix}X_{suffix}", f"{prefix}Y_{suffix}", f"{prefix}Z_{suffix}"]
        if all(col in df.columns for col in cols):
            return cols
    raise KeyError(f"Cannot find vector columns for suffix {suffix}: {prefixes}")


def acc_magnitude(df: pd.DataFrame, suffix: str) -> np.ndarray:
    """Convert 3-axis acceleration to magnitude to reduce orientation dependence."""
    cols = vector_columns(df, ["Acc", "Accel"], suffix)
    arr = df[cols].to_numpy(dtype=float)
    return np.sqrt(np.nansum(arr * arr, axis=1))


def gyro_magnitude(df: pd.DataFrame, suffix: str) -> np.ndarray:
    """Convert 3-axis angular velocity to magnitude."""
    cols = vector_columns(df, ["As", "Gyro"], suffix)
    arr = df[cols].to_numpy(dtype=float)
    return np.sqrt(np.nansum(arr * arr, axis=1))


def clean_signal(values: np.ndarray) -> np.ndarray:
    """Replace infinities, interpolate missing values, and fill remaining gaps."""
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan)
    if series.notna().sum() == 0:
        return np.zeros(len(series), dtype=float)
    series = series.interpolate(limit_direction="both").fillna(series.median())
    return series.to_numpy(dtype=float)


def bandpass_filter(values: np.ndarray, fs: float, low: float = 0.5, high: float = 30.0) -> np.ndarray:
    """Band-pass filter the riding signal when sample rate is sufficient."""
    values = clean_signal(values)
    centered = values - np.nanmedian(values)
    if not np.isfinite(fs) or fs <= high * 2.5 or len(centered) < 24:
        return centered
    nyq = fs / 2.0
    high_adj = min(high, nyq * 0.8)
    if low >= high_adj:
        return centered
    try:
        b, a = signal.butter(4, [low / nyq, high_adj / nyq], btype="bandpass")
        return signal.filtfilt(b, a, centered)
    except Exception:
        return centered


def crop_active_window(df: pd.DataFrame, ride_time_s: float) -> tuple[int, int, float, np.ndarray]:
    """Find the most energetic riding segment with duration from Measurement Details."""
    fs = median_sample_rate(df["Sampletime_1"])
    if not np.isfinite(fs) or fs <= 0:
        return 0, len(df), fs, np.zeros(len(df), dtype=float)

    mag1 = clean_signal(acc_magnitude(df, "1"))
    mag2 = clean_signal(acc_magnitude(df, "2"))
    combined = (mag1 - np.nanmedian(mag1)) ** 2 + (mag2 - np.nanmedian(mag2)) ** 2
    smooth_n = max(5, int(round(fs * 0.25)))
    kernel = np.ones(smooth_n) / smooth_n
    energy = np.convolve(combined, kernel, mode="same")

    target_n = max(20, min(len(df), int(round(ride_time_s * fs))))
    if target_n >= len(df):
        return 0, len(df), fs, energy
    cumsum = np.cumsum(np.insert(energy, 0, 0.0))
    window_energy = cumsum[target_n:] - cumsum[:-target_n]
    start = int(np.argmax(window_energy))
    return start, start + target_n, fs, energy


def time_features(values: np.ndarray, fs: float, prefix: str) -> dict[str, float]:
    """Extract robust time-domain features from a cleaned window."""
    values = clean_signal(values)
    centered = values - np.nanmedian(values)
    return {
        f"{prefix}_mean": float(np.nanmean(centered)),
        f"{prefix}_std": float(np.nanstd(centered)),
        f"{prefix}_rms": float(np.sqrt(np.nanmean(centered**2))),
        f"{prefix}_max_abs": float(np.nanmax(np.abs(centered))),
        f"{prefix}_p95_abs": float(np.nanpercentile(np.abs(centered), 95)),
        f"{prefix}_ptp": float(np.nanmax(centered) - np.nanmin(centered)),
        f"{prefix}_energy_per_s": float(np.nansum(centered**2) / max(len(centered) / fs, 1e-9)) if np.isfinite(fs) and fs > 0 else np.nan,
        f"{prefix}_skew": float(stats.skew(centered, nan_policy="omit")) if len(centered) > 2 else np.nan,
        f"{prefix}_kurtosis": float(stats.kurtosis(centered, nan_policy="omit")) if len(centered) > 3 else np.nan,
    }


def fft_features(values: np.ndarray, fs: float, prefix: str) -> dict[str, float]:
    """Extract FFT features: dominant frequency, centroid, entropy, band powers."""
    values = clean_signal(values)
    if len(values) < 8 or not np.isfinite(fs) or fs <= 0:
        return {f"{prefix}_dom_freq": np.nan, f"{prefix}_spectral_centroid": np.nan, f"{prefix}_spectral_entropy": np.nan}
    centered = values - np.mean(values)
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / fs)
    power = np.abs(np.fft.rfft(centered)) ** 2
    if len(power) <= 1 or np.nansum(power) <= 0:
        return {f"{prefix}_dom_freq": 0.0, f"{prefix}_spectral_centroid": 0.0, f"{prefix}_spectral_entropy": 0.0}

    power[0] = 0.0
    total = float(np.sum(power))
    probs = power / total if total > 0 else np.zeros_like(power)
    entropy = -float(np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))) / math.log2(len(probs)) if len(probs) > 1 else 0.0
    out = {
        f"{prefix}_dom_freq": float(freqs[int(np.argmax(power))]),
        f"{prefix}_spectral_centroid": float(np.sum(freqs * power) / total) if total > 0 else 0.0,
        f"{prefix}_spectral_entropy": entropy,
    }
    for low, high, name in [(0.5, 3, "band_0p5_3"), (3, 8, "band_3_8"), (8, 15, "band_8_15"), (15, 30, "band_15_30")]:
        mask = (freqs >= low) & (freqs < high)
        out[f"{prefix}_{name}_power"] = float(np.sum(power[mask]) / total) if total > 0 and mask.any() else 0.0
    return out


def sensor_features(values: np.ndarray, fs: float, prefix: str) -> dict[str, float]:
    """Apply cleaning/filtering, then concatenate time and frequency features."""
    filtered = bandpass_filter(values, fs)
    out = time_features(filtered, fs, prefix)
    out.update(fft_features(filtered, fs, prefix))
    return out


def aggregate_sensor_pair(feature_dicts: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate sensor 1 and sensor 2 features into mean/max/min/difference."""
    grouped: dict[str, list[float]] = {}
    for item in feature_dicts:
        for key, value in item.items():
            stem = re.sub(r"^(acc|gyro)[12]_", r"\1_", key)
            grouped.setdefault(stem, []).append(float(value))

    out: dict[str, float] = {}
    for stem, values in sorted(grouped.items()):
        vals = np.asarray(values, dtype=float)
        out[f"{stem}_mean"] = float(np.nanmean(vals))
        out[f"{stem}_max"] = float(np.nanmax(vals))
        out[f"{stem}_min"] = float(np.nanmin(vals))
        out[f"{stem}_absdiff"] = float(abs(vals[0] - vals[1])) if vals.size == 2 else np.nan
    return out


def window_starts(n_rows: int, window_n: int, step_n: int) -> list[int]:
    """Return start indices for fixed windows, always including the last window."""
    if n_rows < window_n:
        return [0] if n_rows > 0 else []
    starts = list(range(0, n_rows - window_n + 1, step_n))
    if starts and starts[-1] != n_rows - window_n:
        starts.append(n_rows - window_n)
    return starts
