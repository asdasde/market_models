import os
import pprint
import sys

import uvicorn
from contextlib import asynccontextmanager
from typing import Tuple, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Security, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data.data_processors as data_processors
from utilities.model_utils import *
from utilities.load_utils import *
import pricing.pricing_logic as pricing_logic

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_requests.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PricingManager:
    def __init__(self, pricing_config: PricingConfig, train_date_name : str, path_manager: "PathManager", load_manager: "LoadManager"):
        self.pricing_config = pricing_config
        self.train_data_name = train_date_name
        self.path_manager = path_manager
        self.load_manager = load_manager
        self.models = self.load_manager.load_pricing_config_models(train_date_name, pricing_config)

###############################################################################
# ModelManager
###############################################################################
class ModelManager:
    def __init__(self, config: ApiConfig):
        self.config = config
        self.path_manager = PathManager(config.service_name)
        self.load_manager = LoadManager(self.path_manager)
        self.data_name_reference = LoadManager.load_data_name_reference()
        self.train_data, _, _, _ = self.load_manager.load_data(self.config.train_data_name)
        self.on_top = self.load_manager.load_on_top_file()

        self.pricing_managers: Dict[str, PricingManager] = {}

        for name, pricing_config in config.pricing_config.items():
            self.pricing_managers[name] = PricingManager(pricing_config, self.config.train_data_name,
                                                         self.path_manager, self.load_manager)


def prepare_categorical_data(processed_data: pd.DataFrame, train_data: pd.DataFrame) -> pd.DataFrame:
    for col in processed_data.columns:
        if processed_data[col].dtype.name == 'category':
            try:
                processed_data[col] = pd.Categorical(
                    processed_data[col],
                    categories=train_data[col].cat.categories
                )
            except Exception as e:
                logger.error(f"Error processing categorical column {col}: {e}")
    return processed_data


class API:
    def __init__(self, config: ApiConfig):
        self.config = config
        self.path_manager = PathManager(config.service_name)
        self.load_manager = LoadManager(self.path_manager)
        self.model_manager = ModelManager(config)
        self.app = FastAPI(lifespan=self.lifespan)

        InputFeaturesModel = config.feature_validator

        self.setup_app(InputFeaturesModel)

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        # You can run startup/shutdown code here if needed.
        yield

    def setup_app(self, InputFeaturesModel: Any):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

        # Exception handler for request validation errors:
        @self.app.exception_handler(RequestValidationError)
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
                content={"detail": error_details, "message": "Input validation error"}
            )

        @self.app.post("/predict", status_code=200)
        async def predict_competitors(
            data: InputFeaturesModel,
            api_key: str = Depends(self.verify_api_key)
        ):
            processed_data = self.process_input_data(data)
            processed_data_with_competitors = self.predict_competitors(processed_data)
            calculated_prices = self.calculate_prices(processed_data_with_competitors)

            packages = [pricing_manager.pricing_config.package_name
                        for pricing_manager in self.model_manager.pricing_managers.values()]

            prices = calculated_prices[packages].round().to_dict(orient='records')[0]
            pprint.pprint(prices)

            return prices

    async def verify_api_key(self, api_key: str = Security(APIKeyHeader(name="X-API-Key"))):
        if self.config.api_key and (not api_key or api_key != self.config.api_key):
            raise HTTPException(status_code=403, detail="Could not validate API Key")
        return api_key

    def process_input_data(self, data: Any) -> pd.DataFrame:
        data_dict = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        data_dict = {
            key: (value.value if isinstance(value, Enum) else value)
            for key, value in data_dict.items()
        }
        df = pd.DataFrame([data_dict])
        processor = getattr(data_processors, self.config.input_processor)
        processed_data, _, _, _, _ = processor(
            [df],
            self.model_manager.data_name_reference,
            "native",
            self.path_manager,
            self.load_manager
        )
        return prepare_categorical_data(processed_data, self.model_manager.train_data)


    def predict_competitors(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            for name, pricing_manager in self.model_manager.pricing_managers.items():
                data = predict_multiple_models(data, pricing_manager.models, self.model_manager.on_top)
            return data
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            if self.config.debug_mode:
                raise HTTPException(status_code=500, detail=str(e))
            raise HTTPException(status_code=500, detail="An error occurred during prediction")

    def calculate_prices(self, data : pd.DataFrame) -> pd.DataFrame:
        calculated_prices = data.copy()
        for name, pricing_manager in self.model_manager.pricing_managers.items():
            calculated_prices = pricing_logic.calculate_price(calculated_prices, pricing_manager.pricing_config)
        return calculated_prices



def create_app(config_name: str) -> Tuple[FastAPI, ApiConfig]:
    api_config = ApiConfig.load_from_json(config_name)
    api = API(api_config)
    return api.app, api_config

if __name__ == "__main__":
    app, app_config = create_app("mubi")
    uvicorn.run(
        app,
        host=app_config.api_host,
        port=app_config.api_port,
    )