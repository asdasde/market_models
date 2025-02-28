from dotenv import load_dotenv

load_dotenv()

import os
import sys
import json
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.path_utils import PathManager

def create_feature_validators(schema: list) -> Type:
    namespace = {"__annotations__": {}, "__validators__": {}}

    for field_info in schema:
        field_name = field_info["name"]
        field_type = field_info["type"]
        validations = field_info.get("validation", [])
        is_nullable = field_info.get("nullable", False)

        # Mapping JSON type to Python type
        if field_type == "str":
            pydantic_type = str
        elif field_type == "int":
            pydantic_type = int
        elif field_type == "bool":
            pydantic_type = bool
        else:
            raise ValueError(f"Unsupported field type: {field_type}")

        field_constraints = {}
        custom_validators = {}

        for validation in validations:
            v_type = validation["type"]
            params = validation.get("params", {})
            error_msg = validation.get("error_message")

            if v_type == "enum":
                allowed_values = params["allowed_values"]
                enum_class = Enum(f"{field_name}_enum", {v: v for v in allowed_values}, type=str)
                pydantic_type = enum_class

            elif v_type == "min_value":
                min_val = params["min_val"]
                field_constraints["ge"] = min_val  # Pydantic's `ge` (greater than or equal)

            elif v_type == "max_value":
                max_val = params["max_val"]
                field_constraints["le"] = max_val  # Pydantic's `le` (less than or equal)

            elif v_type == 'regex_pattern':
                pattern = params['pattern']
                field_constraints['pattern'] = pattern

        if is_nullable:
            pydantic_type = Optional[pydantic_type]

        namespace["__annotations__"][field_name] = pydantic_type

        namespace[field_name] = Field(**field_constraints)


        if custom_validators:
            namespace.update(custom_validators)

    return type("DynamicModel", (BaseModel,), namespace)


class ApiConfig(BaseModel):
    api_host: str = Field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8081")))
    debug_mode: bool = Field(default_factory=lambda: os.getenv("DEBUG_MODE", "false").lower() == "true")
    api_key: str = Field(default_factory=lambda: os.getenv("API_KEY", ""))
    allowed_origins: List[str] = Field(
        default_factory=lambda: os.getenv("ALLOWED_ORIGINS", "").split(",")
    )

    config_name: str
    service_name: str
    train_data_name: str
    target_variables_and_model_config: Dict[str, Optional[str]]
    feature_columns: List[Dict[str, Any]]
    tp_kernel: List[Dict[str, Any]] = None
    tp_take_top_k : int 
    rank1_undercut_factor : float
    input_processor: str
    additional_settings: Dict[str, Any] = Field(default_factory=dict)
    feature_validator: Optional[Type[BaseModel]] = None

    @classmethod
    def load_from_json(cls, api_config_name) -> "ApiConfig":
        config_path = PathManager.get_api_configuration_path(api_config_name)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            config_data = json.load(f)

        instance = cls(**config_data)
        instance.feature_validator = create_feature_validators(instance.feature_columns)
        return instance
