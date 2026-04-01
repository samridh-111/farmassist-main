from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.routers.yield_analysis import router
import json
import os

app = FastAPI()
app.include_router(router)
client = TestClient(app)

def test_yield_analysis():
    print("Testing Yield Analysis Endpoint...")
    
    payload = {
        "Time": "1600",
        "FieldID": "F05",
        "ZoneID": "Z2",
        "soil_nitrogen": 10.0,
        "soil_phosphorus": 40.0,
        "soil_potassium": 40.0,
        "temperature": 25.0,
        "humidity": 60.0,
        "rainfall": 200.0,
        "irrigation_hours": 5.0
    }
    
    response = client.post("/yield/analyze", json=payload)
    
    print(f"Status Code: {response.status_code}")
    print("Response Body:")
    print(json.dumps(response.json(), indent=2))
    
    evidence_path = os.path.join(os.path.dirname(__file__), 'data', 'yield_evidence.csv')
    if os.path.exists(evidence_path):
        print(f"\nEvidence log found at: {evidence_path}")
        with open(evidence_path, 'r') as f:
            print("Recent log entries:")
            print("".join(f.readlines()[-2:]))
    else:
        print(f"Evidence log missing at: {evidence_path}")

if __name__ == "__main__":
    test_yield_analysis()
