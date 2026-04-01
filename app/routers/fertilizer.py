from fastapi import APIRouter
from app.models.model_utils import fertilizer_suggestion_for_crop

router = APIRouter(prefix="/fertilizer", tags=["Fertilizer"])

@router.post("/npk")
def npk_recommendation(crop: str, N: float, P: float, K: float):
    return fertilizer_suggestion_for_crop(crop, N, P, K)
