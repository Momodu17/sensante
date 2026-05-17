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

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Client Groq initialise.")
else:
    print("ATTENTION : GROQ_API_KEY non trouvee. /explain sera desactive.")

class ExplainInput(BaseModel):
    diagnostic: str = Field(...)
    probabilite: float = Field(...)
    age: int = Field(...)
    sexe: str = Field(...)
    temperature: float = Field(...)
    region: str = Field(...)

class ExplainOutput(BaseModel):
    explication: str = Field(...)
    modele_llm: str = Field(default="llama-3.1-8b-instant")

SYSTEM_PROMPT = """Tu es un assistant medical senegalais.
Tu recois un diagnostic et des donnees patient.
Explique le resultat en francais simple,
comme un medecin parlerait a son patient.
Sois rassurant mais recommande toujours une consultation medicale.
Maximum 3 phrases.
Ne fais JAMAIS de diagnostic toi-meme.
Tu expliques uniquement le diagnostic fourni."""

@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):
    if not groq_client:
        return ExplainOutput(
            explication="Service d'explication indisponible. Cle API non configuree.",
            modele_llm="aucun"
        )
    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, region {data.region}\n"
        f"Temperature : {data.temperature} C\n"
        f"Diagnostic du modele : {data.diagnostic} (probabilite {data.probabilite:.0%})\n"
        f"Explique ce resultat au patient."
    )
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        explication = response.choices[0].message.content
    except Exception as e:
        explication = f"Erreur lors de l'appel au LLM : {str(e)}"
    return ExplainOutput(explication=explication)
