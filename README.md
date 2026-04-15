# FarmAssist — Explainable Crop Yield Variability Analysis Pattern

FarmAssist is a Cloud Microservice-Based Machine-Learning Framework for Interpretable Precision Yield Management.

## Setup
1. Create venv & install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\Activate.ps1 on Windows
   pip install -r requirements.txt
   ```

2. Train the Explainable Yield Model:
   ```bash
   python app/models/train_yield_model.py
   ```

3. Run the FastAPI Application:
   ```bash
   uvicorn app.main:app --reload
   ```
   Open `frontend/index.html` in your browser.

## Features
- **Explainable Yield Variability Analysis Pattern**: Predicts crop yield and provides SHAP-based feature attribution to identify exact causes for yield variation per zone (e.g., `LOW_SOIL_NITROGEN`).
- **Zone-Wise Visualization Interface**: Evaluates and displays SHAP analysis.
- **Yield Evidence Logger**: Local logs of yield decisions to `/data/yield_evidence.csv`.
- Crop Recommendation, Profit Suggestor, Price Forecast, Fertilizer Analysis, and Crop Calendar.
