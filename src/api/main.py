from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field, create_model
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.model_utils import *
from utilities.load_utils import *
import data.data_processors as data_processors

from config import ApiConfig

config = ApiConfig.load_from_json('mubi_cheapest_offers')

app = FastAPI(
    title=f"{config.config_name.title()} Prediction API",
    description=f"API for {config.config_name} predictions",
    version="1.0.0"
)


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def create_input_features_class(columns: List[str]):
    feature_dict = {col: (Optional[Any], Field(default=None)) for col in columns}
    return create_model("InputFeatures", **feature_dict)


InputFeatures = create_input_features_class(config.feature_columns)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


path_manager = None
load_manager = None

real_datas = {}
models = {}
data_name_reference = {}
on_top = None


def init_models():
    """Initialize model data and load models"""
    global real_datas, models, on_top, data_name_reference, path_manager, load_manager

    path_manager = PathManager(config.service)
    load_manager = LoadManager(path_manager)

    print("Loading models and data...")
    real_datas = {x: load_manager.load_data(x)[0] for x in config.model_versions}

    models = {
        model_v: {
            target_var: load_manager.load_model(model_v, path_manager.get_model_name(model_v, target_var))
            for target_var in config.target_variables
        }
        for model_v in config.model_versions
    }

    on_top = load_manager.load_on_top_file()
    data_name_reference = LoadManager.load_data_name_reference()
    print("Models and data loaded successfully")

def process_input_data(datas: List[InputFeatures]) -> pd.DataFrame:
    global data_name_reference
    features = [data.model_dump() for data in datas]
    df = pd.DataFrame(features)
    processor = getattr(data_processors, config.input_processor)
    processed_data, _, _, fm, _ = processor([df], data_name_reference, "native", path_manager, load_manager)
    return processed_data


def prepare_categorical_data(processed_data: pd.DataFrame, model_v: str) -> pd.DataFrame:
    for col in processed_data.columns:
        if processed_data[col].dtype == 'category':
            processed_data[col] = pd.Categorical(
                processed_data[col],
                categories=real_datas[model_v][col].cat.categories
            )
    return processed_data


def predict_competitors(processed_data: pd.DataFrame) -> Dict:
    predictions = {}
    for model_v in config.model_versions:
        predictions[model_v] = {}
        prepared_data = prepare_categorical_data(processed_data.copy(), model_v)

        for target_var in config.target_variables:
            model = models[model_v][target_var]
            data_slice = prepared_data[model.feature_names]

            if config.debug_mode:
                print(f"Processing {model_v} - {target_var}")


            prediction = float(predict_on_top(model, data_slice, on_top, target_var)[0])
            predictions[model_v][target_var] = prediction

    return predictions


def calculate_technical_price(processed_data: pd.DataFrame) -> float:
    preds = []
    for tp_kernel, cost_estimate in config.model_kernels:
        model = models[config.model_versions[0]][tp_kernel]
        data_slice = processed_data[model.feature_names]
        preds.append(predict_on_top(model, data_slice, on_top, tp_kernel)[0] * cost_estimate)
    return float(np.array(preds).mean().round(2))


def find_cheapest_competitor(predictions: Dict, model_v : str) -> tuple:
    min_price = float('inf')
    min_competitor = None

    for target_var, price in predictions[model_v].items():
        if price < min_price:
            min_price = price
            min_competitor = (model_v, target_var)

    return min_competitor, min_price


async def verify_api_key(api_key: str = Security(api_key_header)):
    if config.api_key and (not api_key or api_key != config.api_key):
        raise HTTPException(status_code=403, detail="Could not validate API Key")
    return api_key


@app.post("/predict/competitors", response_model=Dict)
async def predict_post(datas: List[InputFeatures], api_key: str = Depends(verify_api_key)):
    processed_data = process_input_data(datas)
    return predict_competitors(processed_data)


@app.post("/predict/technical_price", response_model=Dict)
async def predict_tp_post(datas: List[InputFeatures], api_key: str = Depends(verify_api_key)):

    processed_data = process_input_data(datas)
    tp = calculate_technical_price(processed_data)
    return {'tp': tp}
# except Exception as e:
#         if config.debug_mode:
#             raise HTTPException(status_code=500, detail=str(e))
#         raise HTTPException(status_code=500, detail="An error occurred during prediction")


@app.post("/predict/optimal_price", response_model=Dict)
async def predict_optimal_price(datas: List[InputFeatures], api_key: str = Depends(verify_api_key)):

    processed_data = process_input_data(datas)

    competitor_predictions = predict_competitors(processed_data)
    technical_price = calculate_technical_price(processed_data)

    _, cheapest_price = find_cheapest_competitor(competitor_predictions, config.model_versions[0])

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
    # except Exception as e:
    #     if config.debug_mode:
    #         raise HTTPException(status_code=500, detail=str(e))
    #     raise HTTPException(status_code=500, detail="An error occurred during prediction")

init_models()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.debug_mode
    )