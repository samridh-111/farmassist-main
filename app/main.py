from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.crop_recommendation import router as crop_router
from app.routers.price import router as price_router
from app.routers.fertilizer import router as fert_router
from app.routers.calendar import router as calendar_router
from app.routers.yield_analysis import router as yield_router

app = FastAPI(title="FarmAssist API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(crop_router)
app.include_router(price_router)
app.include_router(fert_router)
app.include_router(calendar_router)
app.include_router(yield_router)

@app.get("/")
def home():
    return {"message": "FarmAssist API running successfully!"}
