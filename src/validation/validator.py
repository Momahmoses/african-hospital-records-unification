"""
Clinical data validation: checks completeness, consistency, and clinical plausibility.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = ["patient_id", "full_name", "date_of_birth"]

CLINICAL_RULES = [
    ("systolic_bp > diastolic_bp", lambda df: df["systolic_bp"] > df["diastolic_bp"]),
    ("heart_rate 30-200", lambda df: df["heart_rate"].between(30, 200)),
    ("temperature_c 34-42", lambda df: df["temperature_c"].between(34.0, 42.0)),
    ("age 0-120", lambda df: df["age"].between(0, 120)),
    ("bmi 10-70", lambda df: df["bmi"].between(10, 70) if "bmi" in df.columns else pd.Series([True] * len(df))),
]


class RecordValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        self.errors = []
        self.warnings = []

        for field in REQUIRED_FIELDS:
            if field not in df.columns:
                self.errors.append(f"Missing required column: {field}")
            elif df[field].isnull().any():
                missing = df[field].isnull().sum()
                self.warnings.append(f"{field}: {missing} null values")

        for rule_name, rule_fn in CLINICAL_RULES:
            cols_needed = [c.split()[0] for c in rule_name.split() if not c[0].isdigit() and "-" not in c]
            present_cols = [c for c in cols_needed if c in df.columns]
            if not present_cols:
                continue
            try:
                mask = rule_fn(df)
                violations = (~mask).sum()
                if violations > 0:
                    self.warnings.append(f"Rule '{rule_name}': {violations} violations")
            except Exception:
                pass

        dup_ids = df["patient_id"].duplicated().sum() if "patient_id" in df.columns else 0
        if dup_ids > 0:
            self.warnings.append(f"Duplicate patient_ids: {dup_ids}")

        return {
            "total_records": len(df),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "error_details": self.errors,
            "warning_details": self.warnings,
        }
