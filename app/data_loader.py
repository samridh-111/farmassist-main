import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]  # project root/app/..
DATA_DIR = BASE / "data"

def load_crop_reco():
    path = DATA_DIR / "Crop_recommendation.csv"
    # Expect columns: N,P,K,temperature,humidity,ph,rainfall,label
    return pd.read_csv(path)

def load_rainfall():
    path = DATA_DIR / "district_wise_rainfall_normal.csv"
    # expects columns like STATE_UT_NAME,DISTRICT,JAN,...,ANNUAL
    return pd.read_csv(path)

def load_commodity():
    path = DATA_DIR / "Commodity_prices.txt"
    # commodity file may be quoted, but CSV parser should handle standard CSV
    # Expected headings: state,district,market,commodity,variety,arrival_date,min_price,max_price,modal_price
    return pd.read_csv(path, parse_dates=["arrival_date"], dayfirst=True, infer_datetime_format=True)

def list_crop_timeseries_files():
    folder = DATA_DIR / "crop_data"
    if not folder.exists():
        return []
    return [p.name for p in folder.glob("*.csv")]

def load_crop_timeseries(name):
    path = DATA_DIR / "crop_data" / name
    return pd.read_csv(path)
