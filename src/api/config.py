from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from functools import lru_cache
from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()

import os
import sys
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.path_utils import PathManager


class ApiConfig(BaseModel):
    """Base configuration class for services"""
    # Server settings
    api_host: str = Field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8081")))
    debug_mode: bool = Field(default_factory=lambda: os.getenv("DEBUG_MODE", "false").lower() == "true")
    api_key: str = Field(default_factory=lambda: os.getenv("API_KEY", ""))
    allowed_origins: List[str] = Field(
        default_factory=lambda: os.getenv("ALLOWED_ORIGINS", "").split(",")
    )

    config_name : str
    service_name : str
    model_versions: List[str]
    target_variables: List[str]
    feature_columns: List[str]
    model_kernels: Optional[List[tuple]] = None
    input_processor : str
    additional_settings: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def load_from_json(cls, api_config_name) -> 'ApiConfig':
        config_path = PathManager.get_api_configuration_path(api_config_name)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_data = json.load(f)

        return cls(**config_data)
