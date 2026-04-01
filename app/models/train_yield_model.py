import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
import joblib

def main():
    # Create synthetic agricultural yield data for different zones
    np.random.seed(42)

    n_samples = 1000
    data = {
        'soil_nitrogen': np.random.uniform(20, 100, n_samples),
        'soil_phosphorus': np.random.uniform(10, 60, n_samples),
        'soil_potassium': np.random.uniform(10, 60, n_samples),
        'temperature': np.random.uniform(15, 35, n_samples),
        'humidity': np.random.uniform(30, 90, n_samples),
        'rainfall': np.random.uniform(50, 300, n_samples),
        'irrigation_hours': np.random.uniform(0, 10, n_samples)
    }

    df = pd.DataFrame(data)

    # Yield formulation: A combination of factors with some optimums and penalties
    # Optimal ranges: Temp~25
    yield_base = 50.0

    # Using simple linear and quadratic terms for realistic ML learning
    df['yield'] = (
        yield_base 
        + 0.5 * df['soil_nitrogen'] 
        + 0.3 * df['soil_phosphorus'] 
        + 0.2 * df['soil_potassium']
        - 0.1 * (df['temperature'] - 25)**2 
        + 0.1 * df['rainfall']
        + 2.0 * df['irrigation_hours']
        + np.random.normal(0, 5, n_samples) # noise
    )

    X = df.drop(columns=['yield'])
    y = df['yield']

    print("Training RandomForest Yield Model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save the model
    model_path = os.path.join(os.path.dirname(__file__), 'yield_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
