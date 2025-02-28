import os
import sys

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contextlib import asynccontextmanager

from utilities.model_utils import *
from utilities.load_utils import *
import data.data_processors as data_processors
from config import ApiConfig
from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

config = ApiConfig.load_from_json('mubi_cheapest_offers')
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@asynccontextmanager
async def lifespan(app : FastAPI):
    init_models()
    yield

app = FastAPI(
    lifespan=lifespan,
    title=f"{config.config_name.title()} Prediction API",
    description=f"API for {config.config_name} predictions",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=config.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

path_manager = None
load_manager = None

real_data = None
models = {}
data_name_reference = {}
on_top = None

InputFeatures = config.feature_validator



def init_models():
    global real_data, models, on_top, data_name_reference, path_manager, load_manager

    path_manager = PathManager(config.service_name)
    load_manager = LoadManager(path_manager)

    real_data, _, _, _  = load_manager.load_data(config.train_data_name)

    model_names = {target_variable : path_manager.get_model_name(config.train_data_name, target_variable, model_config)
                   for target_variable, model_config in config.target_variables_and_model_config.items()}

    models = {
        target_variable : load_manager.load_model(config.train_data_name, model_name)
        for target_variable, model_name in model_names.items()
    }

    on_top = load_manager.load_on_top_file()
    data_name_reference = LoadManager.load_data_name_reference()

def process_input_data(datas) -> pd.DataFrame:
    global data_name_reference
    data_dict = datas.model_dump()
    data_dict = {
        key: (value.value if isinstance(value, Enum) else value)
        for key, value in data_dict.items()
    }
    df = pd.DataFrame([data_dict])
    processor = getattr(data_processors, config.input_processor)
    processed_data, _, _, fm, _ = processor([df], data_name_reference, "native", path_manager, load_manager)
    return processed_data



def prepare_categorical_data(processed_data: pd.DataFrame) -> pd.DataFrame:
    for col in processed_data.columns:
        if processed_data[col].dtype == 'category':
            processed_data[col] = pd.Categorical(
                processed_data[col],
                categories=real_data[col].cat.categories
            )
    return processed_data


def predict_competitors(processed_data: pd.DataFrame) -> Dict:
    predictions = {}
    prepared_data = prepare_categorical_data(processed_data.copy())
    for target_var in config.target_variables_and_model_config.keys():
        model = models[target_var]
        data_slice = prepared_data[model.feature_names]
        if config.debug_mode:
            print(f"Processing {target_var}")
        prediction = float(predict_on_top(model, data_slice, on_top, target_var)[0])
        predictions[target_var] = prediction
    return predictions


def calculate_technical_price(processed_data: pd.DataFrame) -> float:
    predictions = []
    for tp_target_variable, cost_estimate, weight in config.tp_kernel:
        model = models[tp_target_variable]
        data_slice = processed_data[model.feature_names]
        predictions.append(predict_on_top(model, data_slice, on_top, tp_target_variable)[0] * cost_estimate * weight)

    return float(np.sum(np.array(predictions)).round(2))


def find_cheapest_competitor(predictions: Dict) -> tuple:
    min_price = float('inf')
    min_competitor = None

    for target_variable, price in predictions.items():
        if price < min_price:
            min_price = price
            min_competitor = target_variable

    return min_competitor, min_price


async def verify_api_key(api_key: str = Security(api_key_header)):
    if config.api_key and (not api_key or api_key != config.api_key):
        raise HTTPException(status_code=403, detail="Could not validate API Key")
    return api_key


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_requests.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logging.info(body)
    error_details = []
    for error in exc.errors():
        error_details.append({
            'location': error.get('loc', []),
            'message': error.get('msg', ''),
            'type': error.get('type', '')
        })

    logger.error(f"""
    Validation Error:
    URL: {request.url}
    Method: {request.method}
    Client IP: {request.client.host}
    Time: {datetime.now().isoformat()}
    Details: {json.dumps(error_details, indent=2)}
    """)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": error_details,
            "message": "Input validation error"
        }
    )


@app.post("/predict/competitors", response_model=Dict)
async def predict_post(data : InputFeatures, api_key: str = Depends(verify_api_key)):
    try:
        processed_data = process_input_data(data)
        return predict_competitors(processed_data)
    except Exception as e:
        print(e)


@app.post("/predict/technical_price", response_model=Dict)
async def predict_tp_post(data : InputFeatures, api_key: str = Depends(verify_api_key)):
    try:
        processed_data = process_input_data(data)
        tp = calculate_technical_price(processed_data)
        return {'tp': tp}
    except Exception as e:
        if config.debug_mode:
            raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail="An error occurred during prediction")


@app.post("/predict/optimal_price", response_model=Dict)
async def predict_optimal_price(data : InputFeatures, api_key: str = Depends(verify_api_key)):

    try:
        processed_data = process_input_data(data)

        competitor_predictions = predict_competitors(processed_data)
        technical_price = calculate_technical_price(processed_data)

        _, cheapest_price = find_cheapest_competitor(competitor_predictions)

        suggested_price = cheapest_price * config.rank1_undercut_factor

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
         if config.debug_mode:
             raise HTTPException(status_code=500, detail=str(e))
         raise HTTPException(status_code=500, detail="An error occurred during prediction")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.debug_mode
    )
