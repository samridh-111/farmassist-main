from fastapi import APIRouter, HTTPException, Query
from app.data_loader import load_commodity
from app.models.model_utils import predict_price_sequence
import joblib
import pandas as pd
from typing import Optional

router = APIRouter(prefix="/price", tags=["Price Forecasting"])

commodity_df = load_commodity()
price_model = joblib.load("app/models/export/price_xgb.pkl")
price_scaler = joblib.load("app/models/export/price_scaler.pkl")


def norm(x):
    return str(x).strip().lower() if pd.notna(x) else ""


# ---------- Diagnostic: list commodities available (optionally scoped to state/district)
@router.get("/list")
def list_commodities(state: Optional[str] = None, district: Optional[str] = None):
    df = commodity_df.copy()
    if state:
        df = df[df['state'].apply(lambda v: norm(v) == norm(state))]
    if district:
        df = df[df['district'].apply(lambda v: norm(v) == norm(district))]
    uniques = sorted(set([str(c).strip() for c in df['commodity'].dropna().unique()]))
    return {"count": len(uniques), "commodities_sample": uniques[:200]}


# ---------- 6-month forecast with fallback logic ----------
@router.post("/forecast")
def forecast_price(state: str, district: str, commodity: str):
    df = commodity_df.copy()
    s = norm(state)
    d = norm(district)
    c = norm(commodity)

    df['state_norm'] = df['state'].apply(norm)
    df['district_norm'] = df['district'].apply(norm)
    df['commodity_norm'] = df['commodity'].apply(norm)

    # 1) strict match (state + district + commodity contains)
    filt = df[
        (df['state_norm'] == s) &
        (df['district_norm'] == d) &
        (df['commodity_norm'].str.contains(c))
    ].sort_values('arrival_date')

    note = None

    # 2) if no strict match, try matching by state + commodity (any district)
    if filt.empty:
        filt = df[
            (df['state_norm'] == s) &
            (df['commodity_norm'].str.contains(c))
        ].sort_values('arrival_date')
        if not filt.empty:
            note = "No exact district match — used state-level commodity history"
    
    # 3) if still empty, fall back to commodity across whole country
    if filt.empty:
        filt = df[df['commodity_norm'].str.contains(c)].sort_values('arrival_date')
        if not filt.empty:
            note = "No state/district match — used country-level commodity history"
    
    if filt.empty:
        raise HTTPException(status_code=404, detail=f"No price history found for commodity '{commodity}' in {district}, {state} (even at country level).")

    # ensure modal_price is numeric
    prices = filt['modal_price'].astype(float).tolist()
    # ensure at least 3 values
    while len(prices) < 3:
        prices.insert(0, prices[0])

    last_three = [float(p) for p in prices[-3:]]
    preds = predict_price_sequence(price_model, price_scaler, last_three, months_ahead=6)

    return {
        "state_requested": state,
        "district_requested": district,
        "commodity_requested": commodity,
        "matched_rows_used": int(len(filt)),
        "note": note,
        "last_3_modal_prices_used": last_three,
        "forecast_next_6_months": [float(round(x,2)) for x in preds]
    }


# ---------- Best market finder (safe types) ----------
@router.get("/markets")
def best_market(commodity: str):
    df = commodity_df.copy()
    c = norm(commodity)
    df['commodity_norm'] = df['commodity'].apply(norm)
    filt = df[df['commodity_norm'].str.contains(c)]
    if filt.empty:
        raise HTTPException(status_code=404, detail="Commodity not found")
    best = filt.sort_values('modal_price', ascending=False).iloc[0]
    return {
        "commodity": str(best['commodity']),
        "best_state": str(best['state']),
        "best_district": str(best['district']),
        "best_market": str(best['market']),
        "highest_modal_price": float(best['modal_price'])
    }
