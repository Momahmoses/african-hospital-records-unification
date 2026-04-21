"""
Full ETL pipeline: ingest → validate → clean → deduplicate → feature engineer → output master record.
"""
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime

from src.cleaning.imputer import ClinicalImputer, missing_data_report
from src.dedup.matcher import PatientDeduplicator
from src.validation.validator import RecordValidator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    now = pd.Timestamp.now()

    if "last_visit_date" in df.columns:
        df["last_visit_date"] = pd.to_datetime(df["last_visit_date"], errors="coerce")
        df["days_since_last_visit"] = (now - df["last_visit_date"]).dt.days.clip(0)

    if "first_visit_date" in df.columns:
        df["first_visit_date"] = pd.to_datetime(df["first_visit_date"], errors="coerce")
        df["patient_tenure_days"] = (now - df["first_visit_date"]).dt.days.clip(0)

    chronic_conditions = ["hypertension", "diabetes", "asthma", "chronic_kidney_disease", "hiv"]
    chronic_cols = [c for c in chronic_conditions if c in df.columns]
    if chronic_cols:
        df["chronic_condition_score"] = df[chronic_cols].apply(
            lambda row: sum(1 for v in row if v == 1 or v is True), axis=1
        )

    if "appointment_count" in df.columns and "missed_appointment_count" in df.columns:
        df["medication_adherence_rate"] = np.where(
            df["appointment_count"] > 0,
            1 - (df["missed_appointment_count"] / df["appointment_count"]),
            np.nan,
        ).clip(0, 1)

    if "age" in df.columns:
        bins = [0, 12, 17, 35, 60, 120]
        labels = ["child", "adolescent", "young_adult", "adult", "elderly"]
        df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)

    if "weight_kg" in df.columns and "height_cm" in df.columns:
        df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2
        df["bmi"] = df["bmi"].clip(10, 70).round(1)

    return df


def load_system_records(data_dir: str = "data/raw") -> pd.DataFrame:
    dfs = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".csv"):
            path = os.path.join(data_dir, fname)
            df = pd.read_csv(path)
            df["source_system"] = fname.replace(".csv", "")
            dfs.append(df)
            logger.info(f"Loaded {len(df)} records from {fname}")

    if not dfs:
        logger.warning("No CSV files found in data/raw — using synthetic data")
        from src.etl.generate_sample_data import generate_patient_data
        return generate_patient_data(1000)

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined {len(combined)} records from {len(dfs)} systems")
    return combined


def run_pipeline(data_dir: str = "data/raw", output_path: str = "data/clean/patient_master.csv"):
    logger.info("=" * 50)
    logger.info("HOSPITAL RECORDS UNIFICATION PIPELINE")
    logger.info("=" * 50)

    raw = load_system_records(data_dir)
    logger.info(f"Step 1: Loaded {len(raw)} raw records")

    validator = RecordValidator()
    validation_report = validator.validate(raw)
    logger.info(f"Step 2: Validation — {validation_report['errors']} errors, {validation_report['warnings']} warnings")

    imputer = ClinicalImputer()
    cleaned = imputer.fit_transform(raw)
    logger.info(f"Step 3: Imputation complete")

    deduplicator = PatientDeduplicator(threshold=0.85)
    duplicates = deduplicator.find_duplicates(cleaned)
    master = deduplicator.merge_records(cleaned, duplicates)
    logger.info(f"Step 4: Deduplication — {len(duplicates)} pairs merged, {len(master)} unique patients")

    master = engineer_features(master)
    logger.info(f"Step 5: Feature engineering complete")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    master.to_csv(output_path, index=False)
    master.to_parquet(output_path.replace(".csv", ".parquet"), index=False)
    logger.info(f"Step 6: Master record saved → {output_path}")

    print(f"\n{'='*50}")
    print(f"PIPELINE COMPLETE")
    print(f"  Input records:    {len(raw):,}")
    print(f"  Duplicate pairs:  {len(duplicates):,}")
    print(f"  Unique patients:  {len(master):,}")
    print(f"  Output: {output_path}")
    print(f"{'='*50}")
    return master


if __name__ == "__main__":
    master = run_pipeline()
    print("\nSample master records:")
    cols = ["patient_id", "full_name", "age", "chronic_condition_score",
            "days_since_last_visit", "medication_adherence_rate"]
    available = [c for c in cols if c in master.columns]
    print(master[available].head(10).to_string())
