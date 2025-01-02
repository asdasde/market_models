import asyncio
import random
import httpx
import pandas as pd
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities.path_utils import get_raw_data_path

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY", "your_api_key_here")  # Make sure this matches the key in your .env file

df_cols = [
    {
        "vehicle_type": "Sedan",
        "vehicle_maker": "Toyota",
        "vehicle_model": "Camry",
        "vehicle_make_year": 2020,
        "vehicle_engine_size": 2,
        "vehicle_power": 203,
        "vehicle_fuel_type": "Petrol",
        "vehicle_number_of_doors": 4,
        "vehicle_trim": "LE",
        "vehicle_ownership_start_year": 2021,
        "vehicle_net_weight": 1500,
        "vehicle_gross_weight": 2000,
        "vehicle_current_mileage": 30000,
        "vehicle_planned_annual_mileage": 15000,
        "vehicle_is_financed": True,
        "vehicle_is_leased": False,
        "vehicle_usage": "Personal",
        "vehicle_imported": False,
        "vehicle_steering_wheel_right": False,
        "vehicle_parking_place": "Garage",
        "contractor_personal_id": "123456789",
        "contractor_birth_date": "1985_05_15",
        "contractor_marital_status": "Married",
        "contractor_postal_code": "01-001",
        "contractor_driver_licence_date": "2005_07_20",
        "contractor_owner_driver_same": True
    }
]


async def test_prediction():
    # Add headers with API key
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        try:
            rnd = random.randint(0, 100)
            test_data = pd.read_parquet(get_raw_data_path('mubi_all_data', extension='.parquet'))
            test_data = test_data[test_data['contractor_personal_id'] == 84010185779]
            test_row = test_data.iloc[rnd]
            test_features = [test_row[df_cols[0].keys()].to_dict()]
            url = "http://192.168.1.243:8082/predict"

            response = await client.post(url, json=test_features, headers=headers)

            if response.status_code == 403:
                print("Authentication error: Please check your API key")
                return {}, None
            elif response.status_code != 200:
                print(f"Error: Status code {response.status_code}")
                print(f"Response: {response.text}")
                return {}, None

            relative_errors_sum = None
            if response.status_code == 200:
                res = pd.DataFrame(response.json())
                res['real'] = test_row[response.json()['mubi_v5'].keys()]
                res.loc['rank1'] = res.min(axis=0)

                for col in res.columns:
                    if col != 'real':
                        res[f'{col}_relative_error'] = (abs(res['real'] - res[col]) / res['real'] * 100).apply(
                            lambda x: round(x, 2))

                res['best_error'] = res.filter(like='error').min(axis=1)
                res['better_model'] = res.filter(like='error').idxmin(axis=1)

                print(res)
                return res['better_model'].value_counts().to_dict(), res.filter(like='error')

        except httpx.RequestError as e:
            print(f"Request error: {e}")
            return {}, None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {}, None


async def main():
    combined_counts = {}
    relative_errors_sum = None
    relative_errors_counts = None
    n = 40
    for _ in range(n):
        counts, relative_errors = await test_prediction()
        for model, count in counts.items():
            if model in combined_counts:
                combined_counts[model] += count
            else:
                combined_counts[model] = count
        if relative_errors is not None:
            if relative_errors_sum is None:
                relative_errors_sum = relative_errors.fillna(0)
                relative_errors_counts = (~relative_errors.isna()).astype(int)
            else:
                relative_errors_sum += relative_errors.fillna(0)
                relative_errors_counts += (~relative_errors.isna()).astype(int)

    if relative_errors_sum is not None and relative_errors_counts is not None:
        mean_errors = relative_errors_sum / relative_errors_counts
        print("Combined Counts:", combined_counts)
        print("Mean Relative Errors:")
        print(relative_errors_counts)
        print(mean_errors)
    else:
        print("No successful predictions were made")


if __name__ == "__main__":
    asyncio.run(main())