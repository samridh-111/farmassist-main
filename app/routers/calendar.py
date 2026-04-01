from fastapi import APIRouter
import pandas as pd

router = APIRouter(prefix="/crop", tags=["Crop Calendar"])

rainfall_df = pd.read_csv("data/district_wise_rainfall_normal.csv")

@router.get("/calendar")
def crop_calendar(state: str, district: str):
    df = rainfall_df.copy()

    # Normalize both input and dataset
    df["state_low"] = df["STATE_UT_NAME"].str.lower().str.strip()
    df["district_low"] = df["DISTRICT"].str.lower().str.strip()

    state = state.lower().strip()
    district = district.lower().strip()

    row = df[(df["state_low"] == state) & (df["district_low"] == district)]

    if row.empty:
        return {"error": "District not found"}

    row = row.iloc[0]

    # Month columns in order
    months = [
        ("JAN", "Jan"), ("FEB", "Feb"), ("MAR", "Mar"), ("APR", "Apr"),
        ("MAY", "May"), ("JUN", "Jun"), ("JUL", "Jul"), ("AUG", "Aug"),
        ("SEP", "Sep"), ("OCT", "Oct"), ("NOV", "Nov"), ("DEC", "Dec")
    ]

    # Crop season mapping
    season_crops = {
        "Rabi": ["wheat", "gram", "mustard", "barley"],
        "Kharif": ["rice", "maize", "cotton", "soybean", "bajra"],
        "Zaid": ["watermelon", "vegetables", "sunflower"]
    }

    calendar = []

    for col, name in months:
        rainfall = float(row[col])

        # Determine season
        if col in ["NOV", "DEC", "JAN", "FEB", "MAR"]:
            base_crops = season_crops["Rabi"]
        elif col in ["JUN", "JUL", "AUG", "SEP", "OCT"]:
            base_crops = season_crops["Kharif"]
        else:
            base_crops = season_crops["Zaid"]

        # Adjust crops based on rainfall
        if rainfall > 200:
            best = ["rice", "jute", "sugarcane"]
        elif rainfall > 80:
            best = ["maize", "cotton", "bajra"]
        else:
            best = ["wheat", "gram", "mustard"]

        # Merge season + rainfall crops
        final_crops = list(dict.fromkeys(best + base_crops))

        calendar.append({
            "month": name,
            "rainfall_mm": rainfall,
            "recommended_crops": final_crops
        })

    return {
        "state": state,
        "district": district,
        "calendar": calendar
    }
