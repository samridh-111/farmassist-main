from app.data_loader import load_crop_reco, load_commodity
from app.models.model_utils import train_crop_suitability, train_price_predictor
import traceback

def main():
    print("Loading crop recommendation dataset...")
    crop = load_crop_reco()
    try:
        print("Training crop suitability model...")
        train_crop_suitability(crop)
    except Exception as e:
        print("Failed to train crop suitability:", e)
        traceback.print_exc()

    print("Loading commodity prices dataset...")
    commodity = load_commodity()
    try:
        print("Training price predictor (with synthesis where needed)...")
        train_price_predictor(commodity)
    except Exception as e:
        print("Failed to train price predictor:", e)
        traceback.print_exc()

    print("Training complete.")

if __name__ == '__main__':
    main()
