import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8081"))

    MODEL_BASE_PATH: str = os.getenv("MODEL_BASE_PATH", "models")
    DATA_BASE_PATH: str = os.getenv("DATA_BASE_PATH", "data")

    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "False").lower() == "true"

    DEFAULT_MODEL_VERSION: str = os.getenv("DEFAULT_MODEL_VERSION", "mubi_v5")
    CATEGORICAL_FILLNA_VALUE: str = os.getenv("CATEGORICAL_FILLNA_VALUE", "unknown")
    NUMERICAL_FILLNA_VALUE: float = float(os.getenv("NUMERICAL_FILLNA_VALUE", "-999"))

    MODEL_VERSIONS: List[str] = ['mubi_v5', 'mubi_v6']
    TARGET_VARIABLES: List[str] = [
        'BEESAFE-(OC)-price',
        'TUZ-(OC),(NNW),(Assistance=100 km PL)-price',
        'BENEFIA-(OC),(NNW),(Assistance=150 km EU,After breakdown)-price',
        'BALCIA-(OC)-price',
        'WEFOX-(OC),(Assistance=150 km PL)-price',
        'ERGOHESTIA-(OC)-price',
        'UNIQA-(OC),(NNW),(Assistance=75 km PL,After breakdown,Replacement vehicle)-price',
        'TRASTI-(OC)-price',
        'GENERALI-(OC)-price',
        'MTU24-(OC)-price',
        'PROAMA-(OC)-price',
        'LINK4-(OC),(Assistance=100 km PL,Replacement vehicle)-price'
    ]

    # Security Configuration
    API_KEY: str = os.getenv("API_KEY", "")
    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "").split(",")

    @classmethod
    def get_feature_columns(cls) -> List[str]:
        return [
            "vehicle_type",
            "vehicle_maker",
            "vehicle_model",
            "vehicle_make_year",
            "vehicle_engine_size",
            "vehicle_power",
            "vehicle_fuel_type",
            "vehicle_number_of_doors",
            "vehicle_trim",
            "vehicle_ownership_start_year",
            "vehicle_net_weight",
            "vehicle_gross_weight",
            "vehicle_current_mileage",
            "vehicle_planned_annual_mileage",
            "vehicle_is_financed",
            "vehicle_is_leased",
            "vehicle_usage",
            "vehicle_imported",
            "vehicle_steering_wheel_right",
            "vehicle_parking_place",
            "contractor_personal_id",
            "contractor_birth_date",
            "contractor_marital_status",
            "contractor_postal_code",
            "contractor_driver_licence_date",
            "contractor_owner_driver_same",
        ]

config = Config()