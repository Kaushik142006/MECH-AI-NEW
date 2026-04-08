import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import math 
import joblib

pd.set_option('display.float_format', '{:.6f}'.format)

# ================= LOAD DATA =================
df = pd.read_csv('data_sets/data_set.csv')
df.columns = df.columns.str.lower().str.strip()

df['temperature_c'] = round(df['temperature_c'], 2)
df['safety_factor'] = round(df['safety_factor'], 2)
df['load_n'] = round(df['load_n'], 2)

# ================= FEATURES =================
num_features = [
    "load_n",
    "temperature_c",
]

cat_features = [
    "component"
]
# -------------------------------------------------------------------- material_type model_1
target_1 = "material_type"

X = df[num_features + cat_features]
y = df[target_1]

# ================= PREPROCESSOR =================
def create_preprocessor():
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features)
    ])

    return preprocessor


preprocessor = create_preprocessor()

# Transform data
X_processed = preprocessor.fit_transform(X)

# Model
model_material = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train model
model_material.fit(X_processed, y)

os.makedirs("ML_models", exist_ok=True)

with open("ML_models/Material_model.pkl", "wb") as f:
    pickle.dump((preprocessor, model_material), f)

print("Material Model trained successfully")
print("Saved at: ML_models/Material_model.pkl")

# ------------------------------------------------------------------------------stress model_2

target_2 = "stress_pa"
X = df[num_features + cat_features]
y = df[target_2]

model_stress = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model_stress.fit(X_processed, y)

os.makedirs("ML_models", exist_ok=True)

with open("ML_models/Stress_model.pkl", "wb") as f:
    pickle.dump((preprocessor, model_stress), f)

print("Stress Model trained successfully")
print("Saved at: ML_models/Stress_model.pkl")

# ------------------------------------------------------------------------------strain model_3

target_3 = "strain"
X = df[num_features + cat_features]
y = df[target_3]

model_strain = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model_strain.fit(X_processed, y)

os.makedirs("ML_models", exist_ok=True)

with open("ML_models/Strain_model.pkl", "wb") as f:
    pickle.dump((preprocessor, model_strain), f)

print("Strain Model trained successfully")
print("Saved at: ML_models/Strain_model.pkl")

# ------------------------------------------------------- yield_point_Pa model_4


target_4 = "yield_point_pa"
X = df[num_features + cat_features]
y = df[target_4]

model_yield = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model_yield.fit(X_processed, y)

os.makedirs("ML_models", exist_ok=True)

with open("ML_models/Yield_model.pkl", "wb") as f:
    pickle.dump((preprocessor, model_yield), f)

print("Yield Model trained successfully")
print("Saved at: ML_models/Yield_model.pkl")

# ------------------------------------------------------- safety_factor model_5

target_5 = "safety_factor"
X = df[num_features + cat_features]
y = df[target_5]

model_safety = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model_safety.fit(X_processed, y)

os.makedirs("ML_models", exist_ok=True)

with open("ML_models/Safety_model.pkl", "wb") as f:
    pickle.dump((preprocessor, model_safety), f)

print("Safety Model trained successfully")
print("Saved at: ML_models/Safety_model.pkl")

