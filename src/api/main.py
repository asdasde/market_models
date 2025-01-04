import pickle
from typing import List, Dict, Any, Optional

import numpy as np
from pydantic import BaseModel, Field, create_model
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
import traceback
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.load_utils import load_model, load_data_name_reference, load_on_top_file, load_data
from utilities.model_utils import predict, predict_on_top
from utilities.path_utils import get_model_name
from data.data_processors import make_processed_mubi_data
from config import config

# Constants
TP_MODEL_V = 'mubi_v5'
TP_KERNEL = [
    ('UNIQA-(OC),(NNW),(Assistance=75 km PL,After breakdown,Replacement vehicle)-price', 0.9),
    ('MTU24-(OC)-price', 0.85),
]

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def create_input_features_class(columns: List[str]):
    feature_dict = {col: (Optional[Any], Field(default=None)) for col in columns}
    return create_model("InputFeatures", **feature_dict)


InputFeatures = create_input_features_class(config.get_feature_columns())

app = FastAPI(title="Insurance Price Prediction API",
              description="API for predicting insurance prices across different providers",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data and models
real_datas = {x: load_data(x)[0] for x in config.MODEL_VERSIONS}
models = {
    model_v: {
        target_var: load_model(get_model_name(model_v, target_var))
        for target_var in config.TARGET_VARIABLES
    }
    for model_v in config.MODEL_VERSIONS
}
on_top = load_on_top_file('mubi')


def process_input_data(datas: List[InputFeatures]) -> pd.DataFrame:
    """Process input data into the required format."""
    features = [data.model_dump() for data in datas]
    df = pd.DataFrame(features)
    processed_data, _, _, fm, _ = make_processed_mubi_data([df], {}, "native")
    return processed_data


def prepare_categorical_data(processed_data: pd.DataFrame, model_v: str) -> pd.DataFrame:
    """Prepare categorical columns with correct categories."""
    for col in processed_data.columns:
        if processed_data[col].dtype == 'category':
            processed_data[col] = pd.Categorical(
                processed_data[col],
                categories=real_datas[model_v][col].cat.categories
            )
    return processed_data


def predict_competitors(processed_data: pd.DataFrame) -> Dict:
    """Get predictions for all competitors."""
    predictions = {}
    for model_v in config.MODEL_VERSIONS:
        predictions[model_v] = {}
        prepared_data = prepare_categorical_data(processed_data.copy(), model_v)

        for target_var in config.TARGET_VARIABLES:
            model = models[model_v][target_var]
            data_slice = prepared_data[model.feature_names]

            if config.DEBUG_MODE:
                print(f"Processing {model_v} - {target_var}")

            prediction = float(predict_on_top(model, data_slice, on_top, target_var)[0])
            predictions[model_v][target_var] = prediction

    return predictions


def calculate_technical_price(processed_data: pd.DataFrame) -> float:
    """Calculate technical price based on kernel models."""
    preds = []
    for tp_kernel, cost_estimate in TP_KERNEL:
        model = models[TP_MODEL_V][tp_kernel]
        data_slice = processed_data[model.feature_names]
        preds.append(predict_on_top(model, data_slice, on_top, tp_kernel)[0] * cost_estimate)
    return float(np.array(preds).mean().round(2))


def find_cheapest_competitor(predictions: Dict, model_v : str) -> tuple:
    """Find the cheapest competitor and their price."""
    min_price = float('inf')
    min_competitor = None

    for target_var, price in predictions[model_v].items():
        if price < min_price:
            min_price = price
            min_competitor = (model_v, target_var)

    return min_competitor, min_price


async def verify_api_key(api_key: str = Security(api_key_header)):
    if config.API_KEY and (not api_key or api_key != config.API_KEY):
        raise HTTPException(status_code=403, detail="Could not validate API Key")
    return api_key


@app.post("/predict/competitors", response_model=Dict)
async def predict_post(datas: List[InputFeatures], api_key: str = Depends(verify_api_key)):
    try:
        processed_data = process_input_data(datas)
        return predict_competitors(processed_data)
    except Exception as e:
        if config.DEBUG_MODE:
            raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail="An error occurred during prediction")


@app.post("/predict/technical_price", response_model=Dict)
async def predict_tp_post(datas: List[InputFeatures], api_key: str = Depends(verify_api_key)):
    try:
        processed_data = process_input_data(datas)
        tp = calculate_technical_price(processed_data)
        return {'tp': tp}
    except Exception as e:
        if config.DEBUG_MODE:
            raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail="An error occurred during prediction")


@app.post("/predict/optimal_price", response_model=Dict)
async def predict_optimal_price(datas: List[InputFeatures], api_key: str = Depends(verify_api_key)):
    try:
        processed_data = process_input_data(datas)

        competitor_predictions = predict_competitors(processed_data)
        technical_price = calculate_technical_price(processed_data)

        _, cheapest_price = find_cheapest_competitor(competitor_predictions, TP_MODEL_V)

        suggested_price = cheapest_price * 0.95

        if technical_price < suggested_price:
            optimal_price = suggested_price
        else:
            optimal_price = technical_price

        return {
            'optimal_price': round(optimal_price, 2),
            'technical_price': technical_price,
            'cheapest_competitor_price': cheapest_price,
            'suggested_competitive_price': round(suggested_price, 2)
        }
    except Exception as e:
        if config.DEBUG_MODE:
            raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail="An error occurred during prediction")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.DEBUG_MODE
    )