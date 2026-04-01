from fastapi import HTTPException

def validate_float(value, name):
    try:
        return float(value)
    except:
        raise HTTPException(status_code=400, detail=f"Invalid value for {name}")

def validate_int(value, name):
    try:
        return int(value)
    except:
        raise HTTPException(status_code=400, detail=f"Invalid integer for {name}")

def get_last_3_prices(df, state, district, commodity):
    filt = df[
        (df['state']==state) &
        (df['district']==district) &
        (df['commodity']==commodity)
    ].sort_values("arrival_date")

    if len(filt) == 0:
        raise HTTPException(status_code=404, detail="No price data found for this combination")

    prices = filt["modal_price"].tolist()

    if len(prices) < 3:
        # duplicate last price until we reach length 3
        while len(prices) < 3:
            prices.insert(0, prices[0])

    return prices[-3:]
