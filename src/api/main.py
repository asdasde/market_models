import pickle
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field, create_model
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware

import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.load_utils import load_model, load_data_name_reference, load_on_top_file, load_data
from utilities.model_utils import predict, predict_on_top
from utilities.path_utils import get_model_name
from data.data_processors import make_processed_mubi_data
from config import config

# API Key security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def create_input_features_class(columns: List[str]):
    """
    Create a Pydantic model dynamically based on column names.
    All fields are defined as Optional to allow partial data input.
    """
    feature_dict = {col: (Optional[Any], Field(default=None)) for col in columns}
    return create_model("InputFeatures", **feature_dict)


# Create the InputFeatures model
InputFeatures = create_input_features_class(config.get_feature_columns())

# Initialize FastAPI app
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

# Load the models
models = {
    model_v: {
        target_var: load_model(get_model_name(model_v, target_var))
        for target_var in config.TARGET_VARIABLES
    }
    for model_v in config.MODEL_VERSIONS
}


async def verify_api_key(api_key: str = Security(api_key_header)):
    if config.API_KEY and (not api_key or api_key != config.API_KEY):
        raise HTTPException(
            status_code=403,
            detail="Could not validate API Key"
        )
    return api_key


@app.post("/predict", response_model=Dict)
async def predict_post(
        datas: List[InputFeatures],
        api_key: str = Depends(verify_api_key)
):
    try:
        features = [data.model_dump() for data in datas]
        df = pd.DataFrame(features)

        # Process data
        processed_data, _, _, fm, _ = make_processed_mubi_data(
            [df],
            {},
            "native",
        )

        # Load real data for each model version
        real_datas = {}
        for x in config.MODEL_VERSIONS:
            real_data, _, _, fm = load_data(config.DEFAULT_MODEL_VERSION)
            real_data = real_data[
                real_data['contractor_personal_id'] == processed_data['contractor_personal_id'].values[0]]
            real_datas[x] = real_data

        on_top = load_on_top_file('mubi')
        processed_data['MTU24-vehicle_age_factor'] = 1
        processed_data['MTU24-contractor_age_factor'] = 1

        predictions = {}
        for model_v in config.MODEL_VERSIONS:
            predictions[model_v] = {}

            for col in processed_data.columns:
                if processed_data[col].dtype == 'category':
                    processed_data[col] = pd.Categorical(
                        processed_data[col],
                        categories=real_datas[model_v][col].cat.categories
                    )

            for target_var in config.TARGET_VARIABLES:
                model = models[model_v][target_var]
                data_slice = processed_data[model.feature_names]

                if config.DEBUG_MODE:
                    print(f"Processing {model_v} - {target_var}")

                prediction = float(predict_on_top(model, data_slice, on_top, target_var)[0])
                predictions[model_v][target_var] = prediction

        return predictions

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