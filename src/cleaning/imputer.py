"""
Clinical missing data imputation using MICE (Multiple Imputation by Chained Equations).
Handles lab results, vitals, and demographic fields with domain-specific rules.
"""
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

CLINICAL_BOUNDS = {
    "age": (0, 120),
    "systolic_bp": (60, 250),
    "diastolic_bp": (40, 150),
    "heart_rate": (30, 200),
    "temperature_c": (34.0, 42.0),
    "weight_kg": (1.0, 300.0),
    "height_cm": (30.0, 250.0),
    "hemoglobin": (4.0, 20.0),
    "glucose_fasting": (2.0, 30.0),
    "creatinine": (0.3, 15.0),
}

CATEGORICAL_DEFAULTS = {
    "blood_type": "Unknown",
    "gender": "Unknown",
    "marital_status": "Unknown",
}


class ClinicalImputer:
    def __init__(self, max_iter: int = 10, random_state: int = 42):
        self.max_iter = max_iter
        self.random_state = random_state
        self.mice = IterativeImputer(
            max_iter=max_iter,
            random_state=random_state,
            min_value=0,
            sample_posterior=True,
        )
        self.scaler = StandardScaler()
        self.numeric_cols = []
        self.cat_cols = []

    def _validate_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, (lo, hi) in CLINICAL_BOUNDS.items():
            if col in df.columns:
                mask = (df[col] < lo) | (df[col] > hi)
                if mask.any():
                    logger.warning(f"{col}: {mask.sum()} out-of-bound values set to NaN")
                    df.loc[mask, col] = np.nan
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._validate_bounds(df)

        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = [c for c in df.columns if c not in self.numeric_cols]

        for col, default in CATEGORICAL_DEFAULTS.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)

        missing_before = df[self.numeric_cols].isnull().sum().sum()
        logger.info(f"Missing numeric values before imputation: {missing_before}")

        if self.numeric_cols and missing_before > 0:
            scaled = self.scaler.fit_transform(df[self.numeric_cols])
            imputed_scaled = self.mice.fit_transform(scaled)
            imputed = self.scaler.inverse_transform(imputed_scaled)
            df[self.numeric_cols] = imputed

        for col, (lo, hi) in CLINICAL_BOUNDS.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)

        missing_after = df[self.numeric_cols].isnull().sum().sum()
        logger.info(f"Missing numeric values after imputation: {missing_after}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._validate_bounds(df)
        for col, default in CATEGORICAL_DEFAULTS.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)
        if self.numeric_cols:
            scaled = self.scaler.transform(df[self.numeric_cols])
            imputed_scaled = self.mice.transform(scaled)
            imputed = self.scaler.inverse_transform(imputed_scaled)
            df[self.numeric_cols] = imputed
        return df


def missing_data_report(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    report = []
    for col in df.columns:
        missing = df[col].isnull().sum()
        report.append({
            "column": col,
            "missing_count": missing,
            "missing_pct": round(missing / total * 100, 2),
            "dtype": str(df[col].dtype),
        })
    return pd.DataFrame(report).sort_values("missing_pct", ascending=False)
