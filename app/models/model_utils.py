import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    class XGBClassifier: pass
    class XGBRegressor: pass
import joblib
from pathlib import Path
import math

EXPORT_DIR = Path(__file__).resolve().parent / "export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42
np.random.seed(RNG_SEED)

def _save(obj, name):
    joblib.dump(obj, EXPORT_DIR / name)
    print(f"Saved {name}")

# ============================================================
# 1) CROP SUITABILITY MODEL (XGBoost Classifier)
# ============================================================

def train_crop_suitability(df):
    required = ['N','P','K','temperature','humidity','ph','rainfall','label']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    X = df[['N','P','K','temperature','humidity','ph','rainfall']].astype(float)
    y = df['label'].astype(str)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y_enc, test_size=0.15, random_state=42, stratify=y_enc
    )

    model = XGBClassifier(
        eval_metric='mlogloss',
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print("Crop suitability accuracy:", acc)

    _save(model, "crop_suitability_xgb.pkl")
    _save(scaler, "crop_suitability_scaler.pkl")
    _save(le, "crop_label_encoder.pkl")

    return model, scaler, le


# ============================================================
# 2) SYNTHETIC PRICE SERIES GENERATOR
# ============================================================

def synthesize_series_from_snapshot(
    base_price, months=36, seasonal_strength=0.08, trend_pct=0.02, noise_pct=0.05
):
    """
    Generate synthetic 36-month series:
    - yearly seasonality
    - linear trend
    - gaussian noise
    """
    base_price = float(base_price)
    arr = np.arange(months)

    seasonal = seasonal_strength * np.sin(2 * math.pi * arr / 12)
    trend = (1 + trend_pct) ** (arr / 12) - 1
    noise = np.random.normal(0, noise_pct, months)

    series = base_price * (1 + seasonal + trend + noise)
    series = [max(1, round(float(v), 2)) for v in series]

    return series


# ============================================================
# 3) AGGREGATOR + SYNTHETIC DATA BUILDER
# ============================================================

def prepare_price_agg_with_synthesis(df, min_months=6, synth_months=36):
    df = df.copy()

    df['modal_price'] = pd.to_numeric(df['modal_price'], errors='coerce')
    df = df.dropna(subset=['modal_price'])

    df['year'] = df['arrival_date'].dt.year
    df['month'] = df['arrival_date'].dt.month

    # Real monthly aggregation
    agg = df.groupby(
        ['state','district','commodity','year','month'],
        as_index=False
    )['modal_price'].median()

    groups = agg.groupby(['state','district','commodity']).size().reset_index(name='count')

    final = []

    for _, row in groups.iterrows():
        s, d, c = row['state'], row['district'], row['commodity']

        gdf = agg[(agg['state']==s) & (agg['district']==d) & (agg['commodity']==c)]
        gdf = gdf.sort_values(['year','month'])

        # If enough real months, use real data only
        if len(gdf) >= min_months:
            final.append(gdf)
            continue

        # Otherwise synthesize
        if len(gdf) > 0:
            base = float(gdf['modal_price'].median())
            last_y = int(gdf['year'].iloc[-1])
            last_m = int(gdf['month'].iloc[-1])
        else:
            raw = df[(df['state']==s) & (df['district']==d) & (df['commodity']==c)]
            base = float(raw['modal_price'].median()) if len(raw) else float(df['modal_price'].median())
            last_y, last_m = 2019, 3

        synth_prices = synthesize_series_from_snapshot(base, months=synth_months)
        rows = []

        y, m = last_y, last_m
        for p in synth_prices:
            rows.append({
                "state": s, "district": d, "commodity": c,
                "year": y, "month": m, "modal_price": p
            })
            m += 1
            if m > 12:
                m = 1
                y += 1

        synth_df = pd.DataFrame(rows)

        combined = pd.concat([gdf, synth_df], ignore_index=True)
        combined = combined.groupby(
            ['state','district','commodity','year','month'], as_index=False
        )['modal_price'].median()

        final.append(combined)

    full = pd.concat(final, ignore_index=True)
    full = full.sort_values(['state','district','commodity','year','month'])

    # Lag features
    full['lag_1'] = full.groupby(['state','district','commodity'])['modal_price'].shift(1)
    full['lag_2'] = full.groupby(['state','district','commodity'])['modal_price'].shift(2)
    full['lag_3'] = full.groupby(['state','district','commodity'])['modal_price'].shift(3)

    full = full.dropna(subset=['lag_1','lag_2','lag_3']).reset_index(drop=True)
    return full


# ============================================================
# 4) PRICE PREDICTOR (REGRESSOR)
# ============================================================

def train_price_predictor(df):
    agg = prepare_price_agg_with_synthesis(df, min_months=6, synth_months=36)

    if len(agg) < 30:
        raise ValueError("After synthesis, not enough rows to train model.")

    X = agg[['lag_1','lag_2','lag_3']].values
    y = agg['modal_price'].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = XGBRegressor(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(Xs, y)

    _save(model, "price_xgb.pkl")
    _save(scaler, "price_scaler.pkl")

    print("Price predictor trained successfully. Rows:", len(agg))
    return model, scaler


# ============================================================
# 5) PREDICT NEXT 6 MONTH PRICES
# ============================================================

def predict_price_sequence(model, scaler, modal_values, months_ahead=6):
    last = modal_values[-3:]
    preds = []
    for _ in range(months_ahead):
        Xp = scaler.transform([last[-3:]])
        p = float(model.predict(Xp)[0])
        preds.append(p)
        last.append(p)
    return preds


# ============================================================
# 6) SIMPLE FERTILIZER RECOMMENDATION
# ============================================================

CROP_NPK = {
    "rice": {"N": 100, "P": 50, "K": 40},
    "wheat": {"N": 120, "P": 60, "K": 40},
    "cotton": {"N": 80, "P": 40, "K": 60},
    "bajra": {"N": 50, "P": 20, "K": 20}
}

def fertilizer_suggestion_for_crop(crop, N, P, K):
    crop = crop.lower()
    if crop not in CROP_NPK:
        return {"error": "No nutrient data for this crop"}

    rec = CROP_NPK[crop]
    return {
        "crop": crop,
        "needed": {
            "N": max(0, rec["N"] - N),
            "P": max(0, rec["P"] - P),
            "K": max(0, rec["K"] - K),
        }
    }
