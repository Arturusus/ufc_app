\
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import joblib
import tensorflow as tf

from .ufc_infer import build_store, list_fights_between, make_features, get_card

app = FastAPI()

# ---- загрузка данных и модели ----
fighters_df = pd.read_csv("backend/fighters.csv")
fights_df = pd.read_csv("backend/Fights.csv")
store = build_store(fighters_df, fights_df)

model = tf.keras.models.load_model("backend/ufc_final_improved_model.h5")
scaler = joblib.load("backend/ufc_scaler.pkl")

class PredictBody(BaseModel):
    fighter1: str
    fighter2: str

@app.get("/api/fighters")
def get_fighters():
    return {"fighters": store["fighter_names"]}

@app.get("/api/fight/list")
def fight_list(f1: str, f2: str):
    return list_fights_between(store, f1, f2)

@app.post("/api/fight/predict")
def predict(body: PredictBody):
    if body.fighter1 == body.fighter2:
        raise HTTPException(status_code=400, detail="Нужно выбрать двух разных бойцов")

    try:
        x = make_features(store, body.fighter1, body.fighter2)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    x_scaled = scaler.transform([x])
    proba_f1 = float(model.predict(x_scaled, verbose=0)[0][0])
    pred = "Fighter_1" if proba_f1 > 0.5 else "Fighter_2"

    return {
        "predicted_winner": pred,
        "proba_fighter1_win": proba_f1,
        "fighter1": get_card(store, body.fighter1),
        "fighter2": get_card(store, body.fighter2),
    }

# UI (статика)
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")
