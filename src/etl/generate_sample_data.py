"""
Generates synthetic hospital patient records from multiple systems with intentional
duplicates, missing values, inconsistencies, and typos — to demonstrate cleaning pipeline.
"""
import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta
from faker import Faker

fake = Faker(["en_NG", "en_GB"])
random.seed(42)
np.random.seed(42)

DIAGNOSES = [
    "Malaria", "Typhoid Fever", "Hypertension", "Type 2 Diabetes", "Asthma",
    "Tuberculosis", "HIV/AIDS", "Sickle Cell Disease", "Peptic Ulcer Disease",
    "Chronic Kidney Disease", "Heart Failure", "Stroke", "Appendicitis",
    "Pneumonia", "Dengue Fever", "Hepatitis B", "Anaemia",
]

ICD10_MAP = {
    "Malaria": "B54", "Typhoid Fever": "A01.0", "Hypertension": "I10",
    "Type 2 Diabetes": "E11", "Asthma": "J45", "Tuberculosis": "A15",
    "HIV/AIDS": "B20", "Sickle Cell Disease": "D57", "Peptic Ulcer Disease": "K27",
    "Chronic Kidney Disease": "N18", "Heart Failure": "I50", "Stroke": "I64",
    "Appendicitis": "K35", "Pneumonia": "J18", "Dengue Fever": "A90",
    "Hepatitis B": "B16", "Anaemia": "D64",
}

WARDS = ["General Ward", "ICU", "Maternity", "Paediatrics", "Outpatient", "Emergency", "Surgical"]
BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]


def random_typo(name: str) -> str:
    if not name or random.random() > 0.3:
        return name
    name = list(name)
    idx = random.randint(0, len(name) - 1)
    operations = ["swap", "delete", "duplicate"]
    op = random.choice(operations)
    if op == "swap" and idx < len(name) - 1:
        name[idx], name[idx + 1] = name[idx + 1], name[idx]
    elif op == "delete":
        name.pop(idx)
    elif op == "duplicate":
        name.insert(idx, name[idx])
    return "".join(name)


def generate_patient_data(n: int = 2000, n_sources: int = 3) -> pd.DataFrame:
    base_patients = []
    n_unique = int(n * 0.75)

    for i in range(n_unique):
        dob = fake.date_of_birth(minimum_age=1, maximum_age=90)
        base_patients.append({
            "true_id": f"TRUE-{i:05d}",
            "full_name": fake.name(),
            "date_of_birth": dob.isoformat(),
            "gender": random.choice(["Male", "Female"]),
            "phone": fake.phone_number(),
            "blood_type": random.choice(BLOOD_TYPES),
            "age": (datetime.now().date() - dob).days // 365,
        })

    all_records = []
    for i, patient in enumerate(base_patients):
        n_appearances = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        for j in range(n_appearances):
            source = f"system_{random.randint(1, n_sources)}"
            record = {
                "patient_id": f"{source.upper()}-{i:05d}-{j}",
                "source_system": source,
                "full_name": random_typo(patient["full_name"]) if j > 0 else patient["full_name"],
                "date_of_birth": patient["date_of_birth"] if random.random() > 0.05 else None,
                "gender": patient["gender"],
                "phone": patient["phone"] if random.random() > 0.1 else None,
                "blood_type": patient["blood_type"] if random.random() > 0.15 else None,
                "age": patient["age"] + random.randint(-1, 1),
                "systolic_bp": random.gauss(120, 20) if random.random() > 0.1 else None,
                "diastolic_bp": random.gauss(80, 12) if random.random() > 0.1 else None,
                "heart_rate": random.gauss(75, 12) if random.random() > 0.08 else None,
                "temperature_c": random.gauss(36.8, 0.5) if random.random() > 0.12 else None,
                "weight_kg": random.gauss(70, 15) if random.random() > 0.08 else None,
                "height_cm": random.gauss(168, 10) if random.random() > 0.08 else None,
                "hemoglobin": random.gauss(12.5, 2) if random.random() > 0.2 else None,
                "glucose_fasting": random.gauss(5.5, 1.5) if random.random() > 0.25 else None,
                "diagnosis": random.choice(DIAGNOSES),
                "ward": random.choice(WARDS),
                "hypertension": random.choice([0, 0, 0, 1]),
                "diabetes": random.choice([0, 0, 0, 1]),
                "appointment_count": random.randint(1, 20),
                "missed_appointment_count": random.randint(0, 5),
                "last_visit_date": (datetime.now() - timedelta(days=random.randint(0, 730))).isoformat(),
                "first_visit_date": (datetime.now() - timedelta(days=random.randint(365, 2555))).isoformat(),
                "created_at": (datetime.now() - timedelta(days=random.randint(0, 1000))).isoformat(),
            }
            record["icd10_code"] = ICD10_MAP.get(record["diagnosis"], "Z00")
            if random.random() < 0.02:
                record["systolic_bp"] = random.choice([500, -10, 999])
            all_records.append(record)

    df = pd.DataFrame(all_records)
    os.makedirs("data/raw", exist_ok=True)

    n_per_system = len(df) // n_sources
    for i in range(1, n_sources + 1):
        subset = df[df["source_system"] == f"system_{i}"].copy()
        subset.to_csv(f"data/raw/system_{i}_records.csv", index=False)
    print(f"Generated {len(df)} records across {n_sources} source systems → data/raw/")
    return df


if __name__ == "__main__":
    df = generate_patient_data(2000)
    print(f"\nTotal records: {len(df)}")
    print(f"Records per system:\n{df['source_system'].value_counts().to_string()}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")
