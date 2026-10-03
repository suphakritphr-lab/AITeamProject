from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "posture_model.pkl"

# HTTPS certificate and private key
CERT_FILE = BASE_DIR.parent / "certs" / "cert.pem"
KEY_FILE = BASE_DIR.parent / "certs" / "key.pem"

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {MODEL_FILE}. Put posture_model.pkl in the backend folder."
    )

if not CERT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {CERT_FILE}. Put the HTTPS certificate in the certs folder."
    )

if not KEY_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {KEY_FILE}. Put the HTTPS private key in the certs folder."
    )


bundle = joblib.load(MODEL_FILE)
model = bundle["model"]
FEATURES = bundle["features"]

app = FastAPI(title="Posture Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Posture Classification API is running",
        "features": FEATURES,
    }


@app.post("/predict")
async def predict(data: dict):
    missing = [feature for feature in FEATURES if feature not in data]

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {', '.join(missing)}"
        )

    try:
        row = {
            feature: float(data[feature])
            for feature in FEATURES
        }
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="All model features must be numeric."
        )

    X = pd.DataFrame([row], columns=FEATURES)

    prediction = model.predict(X)[0]

    response = {
        "posture": str(prediction),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]

        response["confidence"] = float(max(probabilities))

        response["probabilities"] = {
            str(label): float(prob)
            for label, prob in zip(
                model.classes_,
                probabilities
            )
        }

    return response


# ==============================
# Start HTTPS server
# ==============================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_certfile=str(CERT_FILE),
        ssl_keyfile=str(KEY_FILE),
    )