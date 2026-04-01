from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.data_loader import load_commodity
import pandas as pd

router = APIRouter()

class MarketRequest(BaseModel):
    commodity: str
    month: int = Field(..., ge=1, le=12)
    year: int = None  # optional; if not provided use latest year in data

@router.post("/best_markets")
def best_markets(req: MarketRequest):
    df = load_commodity()
    if req.year:
        dfm = df[(df['arrival_date'].dt.month == req.month) & (df['arrival_date'].dt.year == req.year)]
    else:
        # use latest year available
        latest_year = int(df['arrival_date'].dt.year.max())
        dfm = df[(df['arrival_date'].dt.month == req.month) & (df['arrival_date'].dt.year == latest_year)]

    dfm = dfm[dfm['commodity'].astype(str).str.strip().str.lower() == req.commodity.strip().lower()]
    if dfm.empty:
        return {"best_markets": [], "note": "No market data found for given commodity/month/year"}
    top = dfm.groupby(['state','district','market'])['modal_price'].median().reset_index().sort_values('modal_price', ascending=False).head(10)
    return {"best_markets": top.to_dict(orient='records')}
