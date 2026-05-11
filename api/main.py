# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np

class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sexe: str = Field(...)
    temperature: float = Field(..., ge=35.0, le=42.0)
    tension_sys: int = Field(..., ge=60, le=250)
    toux: bool = Field(...)
    fatigue: bool = Field(...)
    maux_tete: bool = Field(...)
    region: str = Field(...)

class DiagnosticOutput(BaseModel):
    diagnostic: str
    probabilite: float
    confiance: str
    message: str

app = FastAPI(
    title="SenSante API",
    description="Assistant pré-diagnostic médical pour le Sénégal",
    version="0.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Chargement du modèle...")
model        = joblib.load("models/model.pkl")
le_sexe      = joblib.load("models/encoder_sexe.pkl")
le_region    = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")
print(f"Modèle chargé : {list(model.classes_)}")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    sexe_enc   = le_sexe.transform([patient.sexe])[0]
    region_enc = le_region.transform([patient.region])[0]

    features = [
        patient.age,
        sexe_enc,
        patient.temperature,
        patient.tension_sys,
        int(patient.toux),
        int(patient.fatigue),
        int(patient.maux_tete),
        region_enc
    ]

    X = np.array([features])
    prediction  = model.predict(X)[0]
    probas      = model.predict_proba(X)[0]
    probabilite = float(max(probas))

    if probabilite >= 0.75:
        confiance = "haute"
    elif probabilite >= 0.5:
        confiance = "moyenne"
    else:
        confiance = "faible"

    messages = {
        "paludisme": "Consulter un médecin rapidement.",
        "grippe":    "Repos et hydratation recommandés.",
        "typhoide":  "Consultation médicale conseillée.",
        "sain":      "Aucun signe critique détecté."
    }

    return DiagnosticOutput(
        diagnostic=prediction,
        probabilite=probabilite,
        confiance=confiance,
        message=messages.get(prediction, "Consultez un professionnel de santé.")
    )
