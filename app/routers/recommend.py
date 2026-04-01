from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from pathlib import Path
from typing import List, Dict
from app.models.model_utils import fertilizer_suggestion_for_crop, predict_price_sequence
from app.data_loader import load_commodity, load_crop_timeseries, list_crop_timeseries_files
import pandas as pd

router = APIRouter()

EXPORT_DIR = Path(__file__).resolve().parents[1] / "models" / "export"

class RecommendRequest(BaseModel):
    state: str
    district: str
    month: int = Field(..., ge=1, le=12)
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class RecommendResponse(BaseModel):
    top_crops: List[Dict]
    profit_table: List[Dict]

def _load_models():
    try:
        model = joblib.load(EXPORT_DIR / "crop_suitability_xgb.pkl")
        scaler = joblib.load(EXPORT_DIR / "crop_suitability_scaler.pkl")
        le = joblib.load(EXPORT_DIR / "crop_label_encoder.pkl")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Crop model artifacts not found. Train models first.")
    # price model optional
    try:
        price_model = joblib.load(EXPORT_DIR / "price_xgb.pkl")
        price_scaler = joblib.load(EXPORT_DIR / "price_scaler.pkl")
    except Exception:
        price_model = None
        price_scaler = None
    return model, scaler, le, price_model, price_scaler

@router.post("/crop", response_model=RecommendResponse)
def recommend_crop(req: RecommendRequest):
    model, scaler, le, price_model, price_scaler = _load_models()

    # Prepare input
    X = np.array([[req.N, req.P, req.K, req.temperature, req.humidity, req.ph, req.rainfall]], dtype=float)
    Xs = scaler.transform(X)
    proba = model.predict_proba(Xs)[0]
    # classes returned may be fewer/more; use label encoder inverse
    classes = le.inverse_transform(np.arange(len(proba)))

    idx = np.argsort(proba)[::-1][:5]  # top 5 to give more info
    top = [{'crop': classes[i], 'suitability_score': float(proba[i])} for i in idx]

    # Profit table: try to get predicted future price for each crop in the district state
    profit_table = []
    if price_model is None:
        # can't predict prices, return placeholder
        for t in top:
            profit_table.append({'crop': t['crop'], 'suitability_score': t['suitability_score'], 'predicted_price_next_6_months': []})
        return {"top_crops": top[:3], "profit_table": profit_table}

    # to create last modal prices for the district/commodity, use commodity history
    commodity_df = load_commodity()
    # normalize state/district names (strip/upper)
    s = req.state.strip()
    d = req.district.strip()

    for t in top:
        commodity = t['crop']
        # filter historical modal prices for this state/district/crop
        hist = commodity_df[
            (commodity_df['state'].astype(str).str.strip().str.upper() == s.upper()) &
            (commodity_df['district'].astype(str).str.strip().str.upper() == d.upper()) &
            (commodity_df['commodity'].astype(str).str.strip().str.lower() == commodity.lower())
        ].sort_values('arrival_date')

        # if insufficient history, try to use larger district or national median
        if len(hist) >= 3:
            last_three = hist['modal_price'].astype(float).dropna().values[-3:]
            preds = predict_price_sequence(price_model, price_scaler, last_three, months_ahead=6)
            profit_table.append({'crop': commodity, 'suitability_score': t['suitability_score'], 'predicted_price_next_6_months': [float(round(p,2)) for p in preds]})
        else:
            # fallback: try other districts state-wide or use overall commodity median
            fallback = commodity_df[commodity_df['commodity'].astype(str).str.strip().str.lower() == commodity.lower()]
            if len(fallback) >= 3:
                agg = fallback.sort_values('arrival_date')
                last_three = agg['modal_price'].astype(float).dropna().values[-3:]
                preds = predict_price_sequence(price_model, price_scaler, last_three, months_ahead=6)
                profit_table.append({'crop': commodity, 'suitability_score': t['suitability_score'], 'predicted_price_next_6_months': [float(round(p,2)) for p in preds], 'note': 'used wider-area fallback data'})
            else:
                profit_table.append({'crop': commodity, 'suitability_score': t['suitability_score'], 'predicted_price_next_6_months': [], 'note': 'insufficient price history'})

    # Trim to top 3 in final response for main UI
    return {"top_crops": top[:3], "profit_table": profit_table}

@router.post("/fertilizer")
def fertilizer(req: dict):
    """
    Expects: {"crop": "rice", "soil_N": 80, "soil_P": 40, "soil_K": 30}
    """
    try:
        crop = req.get("crop")
        soil_N = float(req.get("soil_N", 0))
        soil_P = float(req.get("soil_P", 0))
        soil_K = float(req.get("soil_K", 0))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid input for fertilizer endpoint")
    return fertilizer_suggestion_for_crop(crop, soil_N, soil_P, soil_K)
