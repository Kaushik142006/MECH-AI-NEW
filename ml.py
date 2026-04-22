"""ML models for simulation  trained on real manufacturing data."""

import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

from Sim import MATERIALS

# ================= PATH =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_MODELS_DIR = os.path.join(BASE_DIR, "ML_models")
os.makedirs(ML_MODELS_DIR, exist_ok=True)

_ml_models_cache = {}


# ================= LOAD DATASET =================
def _load_dataset() -> pd.DataFrame:
    """Load the manufacturing dataset from the data_sets folder."""
    csv_path = os.path.join(BASE_DIR, "data_sets", "updated_manufacturing_dataset.csv")
    if not os.path.exists(csv_path):
        import glob
        files = glob.glob(os.path.join(BASE_DIR, "data_sets", "*.csv"))
        if files:
            csv_path = files[0]
        else:
            raise FileNotFoundError(
                "No dataset found in 'data_sets/'. Please provide the CSV file."
            )
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        'load_N': 'load_n',
        'temperature_C': 'temperature_c',
        'material_type': 'material',
        'stress_Pa': 'stress_pa',
        'strain': 'strain',
        'yield_point_Pa': 'yield_pa',
        'safety_factor': 'safety',
        'manufacturing_method': 'manufacturing_method',
        'estimation_cost': 'estimation_cost',
    })
    required = [
        'load_n', 'temperature_c', 'component', 'material',
        'stress_pa', 'strain', 'yield_pa', 'safety',
        'manufacturing_method', 'estimation_cost',
    ]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' missing from dataset.")
    return df[required]


# ================= TRAIN MODELS =================
def _train_and_save_models():
    """Train all ML models using the real dataset and save them to ML_models/."""
    print("🔄 Training ML models on real manufacturing data...")
    df = _load_dataset()

    X = df[['load_n', 'temperature_c', 'component']]
    y_targets = {
        'material':             df['material'],
        'stress':               df['stress_pa'],
        'strain':               df['strain'],
        'yield':                df['yield_pa'],
        'safety':               df['safety'],
        'manufacturing_method': df['manufacturing_method'],
        'estimation_cost':      df['estimation_cost'],
    }

    numeric_features    = ['load_n', 'temperature_c']
    categorical_features = ['component']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ]
    )

    models = {
        'material':             RandomForestClassifier(n_estimators=100, random_state=42),
        'stress':               RandomForestRegressor(n_estimators=100, random_state=42),
        'strain':               RandomForestRegressor(n_estimators=100, random_state=42),
        'yield':                RandomForestRegressor(n_estimators=100, random_state=42),
        'safety':               RandomForestRegressor(n_estimators=100, random_state=42),
        'manufacturing_method': RandomForestClassifier(n_estimators=100, random_state=42),
        'estimation_cost':      RandomForestRegressor(n_estimators=100, random_state=42),
    }

    filename_map = {
        'material':             'Material_model.pkl',
        'stress':               'stress_ml_gen.pkl',
        'strain':               'Strain_model.pkl',
        'yield':                'Yield_model.pkl',
        'safety':               'Safety_model.pkl',
        'manufacturing_method': 'ManufacturingMethod_model.pkl',
        'estimation_cost':      'EstimationCost_model.pkl',
    }

    for name, model in models.items():
        y = y_targets[name]
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model),
        ])
        pipeline.fit(X, y)

        save_path = os.path.join(ML_MODELS_DIR, filename_map[name])
        with open(save_path, 'wb') as f:
            pickle.dump((preprocessor, pipeline), f)
        print(f"✅ Saved {filename_map[name]}")

    print("🎉 All models trained and saved.")


# ================= LOAD MODEL (with auto‑training if missing) =================
def _load_ml_model(model_file: str):
    full_path = os.path.join(ML_MODELS_DIR, model_file)

    if not os.path.exists(full_path):
        _train_and_save_models()
        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"❌ Could not create model: {model_file}\n"
                f"Expected at: {full_path}\n"
            )

    if model_file not in _ml_models_cache:
        with open(full_path, "rb") as f:
            _ml_models_cache[model_file] = pickle.load(f)

    return _ml_models_cache[model_file]


# ================= PREDICTION =================
def _predict_from_ml_models(load: float, temperature: float, component_type: str) -> Dict[str, float]:
    model_map = {
        "material":             "Material_model.pkl",
        "stress":               "stress_ml_gen.pkl",
        "strain":               "Strain_model.pkl",
        "yield":                "Yield_model.pkl",
        "safety":               "Safety_model.pkl",
        "manufacturing_method": "ManufacturingMethod_model.pkl",
        "estimation_cost":      "EstimationCost_model.pkl",
    }

    x_input = pd.DataFrame([{
        "load_n":       float(load),
        "temperature_c": float(temperature),
        "component":    str(component_type).strip().lower(),
    }])

    predictions = {}
    for key, file_name in model_map.items():
        preprocessor, pipeline = _load_ml_model(file_name)
        X_processed = preprocessor.transform(x_input)
        predictions[key] = pipeline.named_steps['model'].predict(X_processed)[0]

    # Post‑process existing predictions
    stress_mpa   = float(predictions["stress"]) / 1_000_000.0
    yield_mpa    = max(1e-6, float(predictions["yield"]) / 1_000_000.0)
    stress_ratio = max(0.0, stress_mpa / yield_mpa)
    safety_factor = max(0.0, float(predictions["safety"]))
    strain_value  = max(0.0, float(predictions["strain"]))

    material_raw   = str(predictions["material"]).strip().lower()
    material_key   = material_raw if material_raw in MATERIALS else "steel"
    material_label = MATERIALS[material_key]["label"]

    # Post‑process new predictions
    manufacturing_method = str(predictions["manufacturing_method"]).strip()
    estimation_cost      = round(max(0.0, float(predictions["estimation_cost"])), 2)

    return {
        "material_key":          material_key,
        "material_label":        material_label,
        "stress_mpa":            stress_mpa,
        "strain":                strain_value,
        "yield_mpa":             yield_mpa,
        "safety_factor":         safety_factor,
        "stress_ratio":          stress_ratio,
        "manufacturing_method":  manufacturing_method,
        "estimation_cost":       estimation_cost,
    }


# ================= DIRECT EXECUTION (for training only) =================
if __name__ == "__main__":
    _train_and_save_models()