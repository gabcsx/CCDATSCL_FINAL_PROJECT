"""
apple_health_pipeline.py
========================
Importable pipeline for building daily Apple Health datasets from:
  1. Apple Health export.xml  (XML)
  2. Health AutoExport CSV    (AutoExport)

Produces two Gold-level datasets:
  - gold_core  : canonical daily activity & energy features
  - gold_ext   : core + extended physiology + audit columns + missingness flags

Design decisions
----------------
* **Steps & distance** come exclusively from AutoExport (canonical source).
  XML step/distance sums are retained as `steps_xml_rawsum` / `distance_km_xml_rawsum`
  for audit only — they can be inflated by overlapping Apple Health records.
* Stand hours column is consistently named `stand_hours` (plural) everywhere.
* One row per calendar day between START_DATE and END_DATE inclusive (strict spine).
* Suspicious zero-energy days (both active & resting == 0) are converted to NaN.

Fixes applied (relative to original data.ipynb)
------------------------------------------------
1.1  Outputs go to data_processed/ and reports/ via pathlib.
1.2  HKCategoryTypeIdentifierAppleStandHour removed from TYPE_MAP; stand parsing
     is handled by dedicated logic, and the canonical column is `stand_hours`.
1.3  XML step/distance kept as audit-only columns; never used in modeling.
1.4  assert_daily_spine_integrity (renamed) enforces strict 1-row-per-day.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# TYPE_MAP — Apple Health XML record types to extract
# ============================================================
# NOTE: HKCategoryTypeIdentifierAppleStandHour is intentionally excluded.
#       Stand hours are parsed via dedicated logic (Record + Category tags)
#       and always written as `stand_hours` (plural).
TYPE_MAP: dict[str, str] = {
    "HKQuantityTypeIdentifierStepCount": "steps",
    "HKQuantityTypeIdentifierDistanceWalkingRunning": "distance_km",
    "HKQuantityTypeIdentifierFlightsClimbed": "flights_climbed",
    "HKQuantityTypeIdentifierActiveEnergyBurned": "active_energy_kcal",
    "HKQuantityTypeIdentifierAppleExerciseTime": "exercise_min",
    "HKQuantityTypeIdentifierHeartRate": "heart_rate_bpm",
}


# ============================================================
# Helpers
# ============================================================
def parse_apple_datetime(s: str) -> datetime:
    """Parse Apple Health datetime string, e.g. '2025-11-24 19:03:12 -0500'."""
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S %z")


def _hour_key(dt: datetime) -> datetime:
    """Truncate a datetime to the hour (used for stand-hour deduplication)."""
    return dt.replace(minute=0, second=0, microsecond=0)


def kj_to_kcal(x):
    """Convert kilojoules to kilocalories."""
    return x / 4.184


def normalize_date_col(s: pd.Series) -> pd.Series:
    """Coerce a Series to datetime and normalize to midnight."""
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def assert_daily_spine_integrity(
    df: pd.DataFrame,
    start_date: date,
    end_date: date,
    name: str = "df",
) -> None:
    """
    Hard check: exactly one row per calendar day in [start_date, end_date].
    Raises ValueError on nulls, duplicates, missing days, or extra days.
    """
    df = df.copy()
    df["date"] = normalize_date_col(df["date"])
    df = df.sort_values("date")

    expected = pd.date_range(start_date, end_date, freq="D")
    have = df["date"].dropna().unique()

    if len(have) != len(df):
        raise ValueError(f"{name}: null or non-normalized dates detected")

    if df["date"].duplicated().any():
        dupes = df.loc[df["date"].duplicated(), "date"].head(5).tolist()
        raise ValueError(f"{name}: duplicate dates found (e.g. {dupes})")

    missing = sorted(set(expected) - set(have))
    extra = sorted(set(have) - set(expected))
    if missing:
        raise ValueError(
            f"{name}: missing dates (e.g. {missing[:5]} … total {len(missing)})"
        )
    if extra:
        raise ValueError(
            f"{name}: extra dates outside range (e.g. {extra[:5]} … total {len(extra)})"
        )


def add_missing_flags(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Add binary `<col>_missing` flags for each column in *cols*."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[f"{c}_missing"] = df[c].isna().astype(int)
    return df


def zero_to_nan_when_suspicious(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert "impossible zeros" to NaN when both active_energy_kcal and
    resting_energy_kcal are exactly 0 — a common signature of a missing
    AutoExport record for that day.
    """
    df = df.copy()
    if "active_energy_kcal" not in df.columns or "resting_energy_kcal" not in df.columns:
        return df

    suspicious = (df["active_energy_kcal"] == 0) & (df["resting_energy_kcal"] == 0)

    for c in [
        "active_energy_kcal",
        "resting_energy_kcal",
        "exercise_min",
        "stand_hours",
        "hr_mean_bpm",
        "hr_min_bpm",
        "hr_max_bpm",
    ]:
        if c in df.columns:
            df.loc[suspicious, c] = np.nan

    return df


# ============================================================
# 1) Parse export.xml → daily dataset
# ============================================================
def build_daily_from_xml(
    xml_path: str | Path,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Stream-parse Apple Health export.xml and aggregate to daily rows.

    Steps and distance are summed as `steps_xml_rawsum` and
    `distance_km_xml_rawsum` (audit columns only — XML sums may be
    inflated by overlapping records).
    """
    xml_path = str(xml_path)
    daily_sum: dict = defaultdict(lambda: defaultdict(float))
    daily_hr: dict = defaultdict(lambda: {"sum": 0.0, "min": None, "max": None, "n": 0})
    daily_stand: dict[date, set] = defaultdict(set)

    context = ET.iterparse(xml_path, events=("end",))

    for _event, elem in context:
        tag = elem.tag

        # --- Record ---
        if tag == "Record":
            t = elem.attrib.get("type")

            # Handle stand-hour Records (may appear as Record in some exports)
            if t == "HKCategoryTypeIdentifierAppleStandHour":
                v = elem.attrib.get("value", "")
                if "Stood" in v:
                    try:
                        start_dt = parse_apple_datetime(elem.attrib["startDate"])
                    except Exception:
                        elem.clear()
                        continue
                    end_str = elem.attrib.get("endDate")
                    try:
                        end_dt = parse_apple_datetime(end_str) if end_str else start_dt
                    except Exception:
                        end_dt = start_dt
                    dd = end_dt.date()
                    if start_date <= dd <= end_date:
                        daily_stand[dd].add(_hour_key(end_dt))
                elem.clear()
                continue

            if t not in TYPE_MAP:
                elem.clear()
                continue

            try:
                start_dt = parse_apple_datetime(elem.attrib["startDate"])
                d = start_dt.date()
            except Exception:
                elem.clear()
                continue

            if d < start_date or d > end_date:
                elem.clear()
                continue

            mapped = TYPE_MAP[t]

            val_str = elem.attrib.get("value")
            unit = elem.attrib.get("unit")
            if val_str is None:
                elem.clear()
                continue
            try:
                val = float(val_str)
            except ValueError:
                elem.clear()
                continue

            if mapped == "distance_km":
                if unit == "mi":
                    val *= 1.609344
                daily_sum[d]["distance_km"] += val
            elif mapped == "heart_rate_bpm":
                hr = daily_hr[d]
                hr["sum"] += val
                hr["n"] += 1
                hr["min"] = val if hr["min"] is None else min(hr["min"], val)
                hr["max"] = val if hr["max"] is None else max(hr["max"], val)
            else:
                daily_sum[d][mapped] += val

            elem.clear()
            continue

        # --- Category (stand hour) ---
        if tag == "Category":
            t = elem.attrib.get("type")
            if t != "HKCategoryTypeIdentifierAppleStandHour":
                elem.clear()
                continue
            v = elem.attrib.get("value", "")
            if "Stood" not in v:
                elem.clear()
                continue
            try:
                start_dt = parse_apple_datetime(elem.attrib["startDate"])
            except Exception:
                elem.clear()
                continue
            end_str = elem.attrib.get("endDate")
            try:
                end_dt = parse_apple_datetime(end_str) if end_str else start_dt
            except Exception:
                end_dt = start_dt
            dd = end_dt.date()
            if start_date <= dd <= end_date:
                daily_stand[dd].add(_hour_key(end_dt))
            elem.clear()
            continue

        elem.clear()

    # --- Build daily DataFrame ---
    all_days = pd.date_range(start_date, end_date, freq="D")
    rows = []
    for dts in all_days:
        d = dts.date()
        s = daily_sum[d]
        hr = daily_hr[d]
        stand_hours = len(daily_stand[d])
        hr_mean = (hr["sum"] / hr["n"]) if hr["n"] > 0 else np.nan

        rows.append(
            {
                "date": pd.to_datetime(dts).normalize(),
                # Audit-only sums (may be inflated by overlapping records)
                "steps_xml_rawsum": s.get("steps", 0.0),
                "distance_km_xml_rawsum": s.get("distance_km", 0.0),
                # XML signals usable as fallback
                "flights_climbed": s.get("flights_climbed", 0.0),
                "active_energy_kcal": s.get("active_energy_kcal", 0.0),
                "exercise_min": s.get("exercise_min", 0.0),
                "stand_hours": float(stand_hours),
                # Heart rate from XML (fallback / debug)
                "hr_mean_bpm": hr_mean,
                "hr_min_bpm": hr["min"] if hr["min"] is not None else np.nan,
                "hr_max_bpm": hr["max"] if hr["max"] is not None else np.nan,
                "hr_samples": float(hr["n"]),
            }
        )

    df_xml = pd.DataFrame(rows)
    num_cols = [c for c in df_xml.columns if c != "date"]
    df_xml[num_cols] = df_xml[num_cols].apply(pd.to_numeric, errors="coerce")
    return df_xml


# ============================================================
# 2) Parse AutoExport CSV → daily dataset
# ============================================================
def build_daily_from_autoexport(
    csv_path: str | Path,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Read Health AutoExport CSV and aggregate to daily rows.

    AutoExport is the canonical source for steps and distance (avoids
    double-counting that can occur in raw XML).
    """
    raw = pd.read_csv(str(csv_path))

    raw["Date/Time"] = pd.to_datetime(raw["Date/Time"], errors="coerce")
    raw["date"] = raw["Date/Time"].dt.normalize()

    rename_map = {
        "Active Energy (kJ)": "active_energy_kj",
        "Resting Energy (kJ)": "resting_energy_kj",
        "Apple Exercise Time (min)": "exercise_min",
        "Apple Stand Hour (count)": "stand_hours",
        "Apple Stand Time (min)": "stand_min",
        "Step Count (count)": "steps",
        "Walking + Running Distance (km)": "distance_km",
        "Heart Rate [Min] (count/min)": "hr_min_bpm",
        "Heart Rate [Max] (count/min)": "hr_max_bpm",
        "Heart Rate [Avg] (count/min)": "hr_mean_bpm",
        "Heart Rate Variability (ms)": "hrv_ms",
        "Respiratory Rate (count/min)": "resp_rate",
        "Resting Heart Rate (count/min)": "resting_hr_bpm",
        "Walking Speed (km/hr)": "walking_speed_kmh",
        "Walking Step Length (cm)": "walking_step_length_cm",
        "Flights Climbed (count)": "flights_climbed",
    }
    raw = raw.rename(columns={k: v for k, v in rename_map.items() if k in raw.columns})

    for c in rename_map.values():
        if c in raw.columns and c != "date":
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    if "active_energy_kj" in raw.columns:
        raw["active_energy_kcal"] = kj_to_kcal(raw["active_energy_kj"])
    if "resting_energy_kj" in raw.columns:
        raw["resting_energy_kcal"] = kj_to_kcal(raw["resting_energy_kj"])

    mask = (raw["date"].dt.date >= start_date) & (raw["date"].dt.date <= end_date)
    raw = raw.loc[mask].copy()

    # Robust daily aggregation
    sum_cols = [
        "steps", "distance_km", "exercise_min", "stand_min",
        "active_energy_kcal", "resting_energy_kcal", "flights_climbed",
    ]
    mean_cols = [
        "stand_hours", "hr_min_bpm", "hr_max_bpm", "hr_mean_bpm",
        "hrv_ms", "resp_rate", "resting_hr_bpm",
        "walking_speed_kmh", "walking_step_length_cm",
    ]

    agg_dict = {}
    for c in sum_cols:
        if c in raw.columns:
            agg_dict[c] = "sum"
    for c in mean_cols:
        if c in raw.columns:
            agg_dict[c] = "mean"

    daily = (
        raw.groupby("date", as_index=False)
        .agg(agg_dict)
        .sort_values("date")
        .reset_index(drop=True)
    )
    return daily


# ============================================================
# 3) Build GOLD datasets (core + extended)
# ============================================================
def build_gold(
    xml_daily: pd.DataFrame,
    auto_daily: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Merge XML and AutoExport daily datasets onto a strict calendar spine
    and produce Gold Core + Gold Extended + a quality report string.

    Returns
    -------
    gold_core : pd.DataFrame
    gold_ext  : pd.DataFrame
    report    : str   (human-readable quality report)
    """
    xml = xml_daily.copy()
    auto = auto_daily.copy()
    xml["date"] = normalize_date_col(xml["date"])
    auto["date"] = normalize_date_col(auto["date"])

    spine = pd.DataFrame(
        {"date": pd.date_range(start_date, end_date, freq="D").normalize()}
    )

    df = spine.merge(auto, on="date", how="left")
    df = df.merge(xml, on="date", how="left", suffixes=("", "_xml"))

    # --- Canonical feature selection ---

    # Steps & distance: AutoExport only (never XML)
    for c in ["steps", "distance_km"]:
        if c not in df.columns:
            df[c] = np.nan

    # Prefer AutoExport, fallback to XML
    prefer_auto_fallback_xml = [
        "active_energy_kcal", "exercise_min", "stand_hours",
        "hr_mean_bpm", "hr_min_bpm", "hr_max_bpm",
        "flights_climbed", "resting_energy_kcal",
    ]
    for c in prefer_auto_fallback_xml:
        if c not in df.columns:
            df[c] = np.nan
        c_xml = f"{c}_xml"
        if c_xml in df.columns:
            df[c] = df[c].combine_first(df[c_xml])

    # XML-only audit columns
    if "steps_xml_rawsum" in df.columns:
        df["steps_xml_rawsum"] = pd.to_numeric(df["steps_xml_rawsum"], errors="coerce")
    if "distance_km_xml_rawsum" in df.columns:
        df["distance_km_xml_rawsum"] = pd.to_numeric(
            df["distance_km_xml_rawsum"], errors="coerce"
        )

    # Extended-only features
    extended_cols = [
        "hrv_ms", "resp_rate", "resting_hr_bpm",
        "walking_speed_kmh", "walking_step_length_cm",
        "stand_min",
    ]
    for c in extended_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Ensure numeric
    numeric_cols = [c for c in df.columns if c != "date"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # --- Quality gates (hard) ---
    assert_daily_spine_integrity(df[["date"]], start_date, end_date, name="gold_spine")

    if (df["steps"] < 0).any():
        raise ValueError("Negative steps found")
    if (df["distance_km"] < 0).any():
        raise ValueError("Negative distance_km found")

    # --- Build Core / Extended ---
    core_cols = [
        "date",
        "steps", "distance_km",
        "active_energy_kcal", "resting_energy_kcal",
        "exercise_min", "stand_hours",
        "flights_climbed",
        "hr_mean_bpm", "hr_min_bpm", "hr_max_bpm",
        "hr_samples",
    ]
    core_cols = [c for c in core_cols if c in df.columns]
    gold_core = df[core_cols].copy()

    extended_keep = core_cols + extended_cols + [
        "steps_xml_rawsum", "distance_km_xml_rawsum",
    ]
    extended_keep = [c for c in extended_keep if c in df.columns]
    gold_ext = df[extended_keep].copy()

    # Convert suspicious zeros → NaN (model-friendly)
    gold_core = zero_to_nan_when_suspicious(gold_core)
    gold_ext = zero_to_nan_when_suspicious(gold_ext)

    # Missingness flags for sparse physiology columns
    gold_ext = add_missing_flags(gold_ext, extended_cols)

    gold_core = gold_core.sort_values("date").reset_index(drop=True)
    gold_ext = gold_ext.sort_values("date").reset_index(drop=True)

    # --- Report + soft checks (informational) ---
    report_lines: list[str] = []
    report_lines.append("GOLD DATA QUALITY REPORT\n")
    report_lines.append(f"Range: {start_date} to {end_date} (inclusive)")
    report_lines.append(f"Rows: {len(df)}\n")

    def _miss(df_: pd.DataFrame, name: str) -> str:
        s = (df_.isna().mean() * 100).round(2).sort_values(ascending=False)
        return f"{name} missingness (% top 20):\n{s.head(20).to_string()}\n"

    report_lines.append(_miss(gold_core, "gold_core"))
    report_lines.append(_miss(gold_ext, "gold_extended"))

    # Meters per step sanity
    valid = (gold_core["steps"] > 0) & (gold_core["distance_km"] > 0)
    mps = (gold_core.loc[valid, "distance_km"] * 1000) / gold_core.loc[valid, "steps"]
    report_lines.append("Meters per step (valid days) describe:\n")
    report_lines.append(
        mps.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_string() + "\n"
    )

    outliers = gold_core.loc[valid].loc[
        (mps < 0.3) | (mps > 1.5), ["date", "steps", "distance_km"]
    ]
    report_lines.append(f"Meters-per-step outliers (<0.3 or >1.5): {len(outliers)}\n")
    if len(outliers) > 0:
        tmp = outliers.head(10).copy()
        tmp["meters_per_step"] = ((tmp["distance_km"] * 1000) / tmp["steps"]).round(3)
        report_lines.append(
            "Showing up to 10:\n" + tmp.to_string(index=False) + "\n"
        )

    # Suspicious zero-energy day count (pre-conversion)
    suspicious_before = (
        (df.get("active_energy_kcal", pd.Series(dtype=float)) == 0)
        & (df.get("resting_energy_kcal", pd.Series(dtype=float)) == 0)
    ).sum()
    report_lines.append(
        f"Suspicious zero-energy days detected (pre-conversion): {int(suspicious_before)}\n"
    )

    # XML vs Auto inflation (audit)
    if "steps_xml_rawsum" in gold_ext.columns and "steps" in gold_ext.columns:
        ratio = (gold_ext["steps_xml_rawsum"] / gold_ext["steps"]).replace(
            [np.inf, -np.inf], np.nan
        )
        report_lines.append("XML rawsum steps / Auto steps ratio (describe):\n")
        report_lines.append(
            ratio.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_string()
            + "\n"
        )

    return gold_core, gold_ext, "\n".join(report_lines)


# ============================================================
# 4) Console sanity checks (prints after build)
# ============================================================
def print_sanity_checks(
    gold_core: pd.DataFrame, gold_ext: pd.DataFrame
) -> None:
    """Print human-readable sanity checks to stdout."""
    print("\n================ SANITY CHECKS ================")
    print("Rows (core, ext):", gold_core.shape[0], gold_ext.shape[0])
    print("Date range:", gold_core["date"].min(), "to", gold_core["date"].max())

    # Meters per step
    valid = (gold_core["steps"] > 0) & (gold_core["distance_km"] > 0)
    mps = (gold_core.loc[valid, "distance_km"] * 1000) / gold_core.loc[valid, "steps"]
    print("\nMeters per step (valid days) describe:")
    print(mps.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_string())

    # Negative / impossible checks
    neg_steps = int((gold_core["steps"] < 0).sum())
    neg_dist = int((gold_core["distance_km"] < 0).sum())
    print("\nNegative values:", {"steps": neg_steps, "distance_km": neg_dist})

    # Missingness top
    print("\nGold core missingness (top 12):")
    print(gold_core.isna().sum().sort_values(ascending=False).head(12).to_string())

    print("\nGold extended missingness (top 12):")
    print(gold_ext.isna().sum().sort_values(ascending=False).head(12).to_string())

    # Zero-energy days
    if (
        "active_energy_kcal" in gold_core.columns
        and "resting_energy_kcal" in gold_core.columns
    ):
        zero_energy = int(
            (
                (gold_core["active_energy_kcal"] == 0)
                & (gold_core["resting_energy_kcal"] == 0)
            ).sum()
        )
        print("\nZero-energy days in gold_core AFTER conversion:", zero_energy)

    # XML inflation spot-check
    if "steps_xml_rawsum" in gold_ext.columns:
        ratio = (gold_ext["steps_xml_rawsum"] / gold_ext["steps"]).replace(
            [np.inf, -np.inf], np.nan
        )
        print("\nXML steps rawsum / Auto steps ratio (describe):")
        print(ratio.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_string())

    print("===============================================")
