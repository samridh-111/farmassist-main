from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import shap
import os
import csv
from datetime import datetime

router = APIRouter(
    prefix="/yield",
    tags=["Yield Variability Analysis Pattern"]
)

# Load the model
try:
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'yield_model.joblib')
    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
except Exception as e:
    print(f"Failed to load yield model: {e}")
    model = None
    explainer = None

class ZoneData(BaseModel):
    Time: str = ""
    FieldID: str
    ZoneID: str
    soil_nitrogen: float
    soil_phosphorus: float
    soil_potassium: float
    temperature: float
    humidity: float
    rainfall: float
    irrigation_hours: float

@router.post("/analyze")
def analyze_yield(data: ZoneData):
    global model, explainer
    
    # Try reloading if it was generated after startup
    if model is None or explainer is None:
        try:
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'yield_model.joblib')
            model = joblib.load(model_path)
            explainer = shap.TreeExplainer(model)
        except Exception:
            raise HTTPException(status_code=500, detail="Yield model not completely loaded yet.")
            
    # Prepare data for prediction
    features = ['soil_nitrogen', 'soil_phosphorus', 'soil_potassium', 'temperature', 'humidity', 'rainfall', 'irrigation_hours']
    input_df = pd.DataFrame([{f: getattr(data, f) for f in features}])
    
    # Predict
    pred_yield = model.predict(input_df)[0]
    
    # Explain using SHAP
    shap_values = explainer.shap_values(input_df)
    
    # Find the top absolute impact feature (shap_values[0] for single row)
    feature_impacts = dict(zip(features, shap_values[0]))
    
    top_feature = max(feature_impacts, key=lambda k: abs(feature_impacts[k]))
    impact_value = feature_impacts[top_feature]
    
    # Format the top factor string, e.g., LOW_SOIL_NITROGEN, HIGH_TEMPERATURE
    factor_prefix = "HIGH" if impact_value > 0 else "LOW"
    top_factor_str = f"{factor_prefix}_{top_feature.upper()}"
    
    predicted_yield_category = "HIGH" if pred_yield > 150 else ("LOW" if pred_yield < 100 else "MEDIUM")
    
    # Log evidence
    log_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'yield_evidence.csv')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Write header if doesn't exist
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Time', 'FieldID', 'ZoneID', 'PredictedYield', 'TopFactor'])
        
        current_time = data.Time if data.Time else datetime.now().strftime("%H%M")
        writer.writerow([
            current_time,
            data.FieldID,
            data.ZoneID,
            predicted_yield_category,
            top_factor_str
        ])
        
    return {
        "Time": data.Time if data.Time else datetime.now().strftime("%H%M"),
        "FieldID": data.FieldID,
        "ZoneID": data.ZoneID,
        "PredictedYield": predicted_yield_category,
        "PredictedValue": round(float(pred_yield), 2),
        "TopFactor": top_factor_str,
        "FeatureImpacts": {k: round(float(v), 2) for k, v in feature_impacts.items()}
    }
