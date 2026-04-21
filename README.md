# African Hospital Patient Records Unification System

Cleans, deduplicates, and unifies patient records from multiple hospital information systems into a single master patient record. Built for hospital networks operating across multiple states with incompatible EHR systems.

## Problem
A hospital network across 3 states has patient records in 5 different systems. The same patient appears 14 different ways. Drug names are misspelled. Dates are in 3 formats. Diagnoses use both ICD-9 and ICD-10 codes. Doctors can't see a patient's full history.

## Pipeline
```
Raw CSVs (multiple systems)
    → Validation (clinical bounds, required fields)
    → Imputation (MICE algorithm for missing lab values)
    → Deduplication (fuzzy name matching + DOB + phone)
    → Feature Engineering (BMI, chronic score, adherence rate)
    → Patient Master Record (CSV + Parquet)
```

## Quick Start

```bash
pip install -r requirements.txt

# Generate 2,000 synthetic records across 3 hospital systems
python src/etl/generate_sample_data.py

# Run the full ETL pipeline
python src/etl/pipeline.py
```

## Key Components

### Missing Data Imputation (`src/cleaning/imputer.py`)
- MICE (Multiple Imputation by Chained Equations) for lab results
- Clinical domain bounds validation (e.g., BP 60-250, age 0-120)
- Out-of-range values flagged and set to NaN before imputation

### Patient Deduplication (`src/dedup/matcher.py`)
- Blocking by first initial + birth year (efficient)
- Fuzzy name matching (RapidFuzz token sort ratio)
- Composite score: 50% name + 35% DOB + 15% phone
- Threshold: 0.85 composite score → marked as duplicate
- SHA-256 based canonical Master Record Number (MRN)

### Feature Engineering (`src/etl/pipeline.py`)
- `days_since_last_visit` — recency of care
- `chronic_condition_score` — count of chronic conditions
- `medication_adherence_rate` — appointments kept ratio
- `bmi` — calculated from weight + height
- `age_group` — child/adolescent/adult/elderly

### Validation (`src/validation/validator.py`)
- Required fields presence check
- Clinical plausibility rules (systolic > diastolic, etc.)
- Duplicate patient ID detection

## Sample Output

```
PIPELINE COMPLETE
  Input records:    2,847
  Duplicate pairs:  412
  Unique patients:  2,106
  Output: data/clean/patient_master.csv
```

## Real Impact
- Reduces duplicate prescriptions and adverse drug interactions
- Enables accurate chronic disease management
- Provides complete patient history to any doctor in the network
