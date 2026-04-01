from fastapi import APIRouter, HTTPException
import joblib
import numpy as np
import pandas as pd
from app.data_loader import load_crop_reco, load_commodity
from app.models.model_utils import predict_price_sequence

router = APIRouter(prefix="/crop", tags=["Crop Recommendation"])

# Load trained models
crop_model = joblib.load("app/models/export/crop_suitability_xgb.pkl")
crop_scaler = joblib.load("app/models/export/crop_suitability_scaler.pkl")
crop_label_encoder = joblib.load("app/models/export/crop_label_encoder.pkl")

price_model = joblib.load("app/models/export/price_xgb.pkl")
price_scaler = joblib.load("app/models/export/price_scaler.pkl")

commodity_df = load_commodity()


# -----------------------------------------
# 1️⃣ Crop Recommendation (soil-based)
# -----------------------------------------
@router.post("/recommend")
def recommend_crop(N: float, P: float, K: float, temperature: float, humidity: float, ph: float, rainfall: float):
    try:
        X = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        Xs = crop_scaler.transform(X)
        preds = crop_model.predict_proba(Xs)[0]

        top3_idx = preds.argsort()[::-1][:3]
        labels = crop_label_encoder.inverse_transform(top3_idx)
        scores = preds[top3_idx]

        return {
            "recommendations": [
                {"crop": labels[i], "score": float(scores[i])}
                for i in range(3)
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------
# 2️⃣ Profit-based Recommendation
# -----------------------------------------
@router.post("/profit")
def profit_based_recommendation(state: str, district: str):
    df = commodity_df.copy()

    # Normalize input
    state = state.strip().lower()
    district = district.strip().lower()

    df["state_low"] = df["state"].str.lower().str.strip()
    df["district_low"] = df["district"].str.lower().str.strip()

    # Step 1: district-level data
    dist_df = df[(df["state_low"] == state) & (df["district_low"] == district)]

    # Step 2: fallback to state-level if district empty
    if dist_df.empty:
        dist_df = df[df["state_low"] == state]

    # Step 3: if still empty → fallback to all India commodity data
    if dist_df.empty:
        dist_df = df.copy()

    unique_crops = dist_df["commodity"].unique()
    results = []

    for crop in unique_crops:
        rows = dist_df[dist_df["commodity"] == crop]
        if rows.empty:
            continue

        prices = rows["modal_price"].tolist()

        # Ensure at least 3 values
        if len(prices) < 3:
            prices = [prices[0]] * (3 - len(prices)) + prices

        pred = predict_price_sequence(price_model, price_scaler, prices, months_ahead=1)
        expected_price = pred[0]

        results.append({
            "crop": crop,
            "expected_price": round(float(expected_price), 2)
        })

    # Sort + return top 5
    results = sorted(results, key=lambda x: x["expected_price"], reverse=True)[:5]

    return {"most_profitable_crops": results}
