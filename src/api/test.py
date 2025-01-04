import asyncio
import random
import httpx
import pandas as pd
import os
import sys
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.path_utils import get_raw_data_path

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY", "your_api_key_here")

# Constants
BASE_URL = "http://0.0.0.0:8082"
ENDPOINTS = {
    "competitors": "/predict/competitors",
    "technical_price": "/predict/technical_price",
    "optimal_price": "/predict/optimal_price"
}

REQUIRED_COLUMNS = [
    "vehicle_type", "vehicle_maker", "vehicle_model", "vehicle_make_year",
    "vehicle_engine_size", "vehicle_power", "vehicle_fuel_type",
    "vehicle_number_of_doors", "vehicle_trim", "vehicle_ownership_start_year",
    "vehicle_net_weight", "vehicle_gross_weight", "vehicle_current_mileage",
    "vehicle_planned_annual_mileage", "vehicle_is_financed", "vehicle_is_leased",
    "vehicle_usage", "vehicle_imported", "vehicle_steering_wheel_right",
    "vehicle_parking_place", "contractor_personal_id", "contractor_birth_date",
    "contractor_marital_status", "contractor_postal_code",
    "contractor_driver_licence_date", "contractor_owner_driver_same"
]


class APITester:
    def __init__(self, test_data: pd.DataFrame):
        self.test_data = test_data
        self.headers = {
            "X-API-Key": API_KEY,
            "Content-Type": "application/json"
        }
        self.results_dir = "test_results"
        os.makedirs(self.results_dir, exist_ok=True)

    def get_test_case(self) -> Tuple[pd.Series, List[Dict]]:
        """Generate a random test case from the test data."""
        rnd = random.randint(0, len(self.test_data) - 1)
        test_row = self.test_data.iloc[rnd]
        test_features = [test_row[REQUIRED_COLUMNS].to_dict()]
        return test_row, test_features

    async def call_endpoint(self, client: httpx.AsyncClient, endpoint: str,
                            test_features: List[Dict]) -> Tuple[Optional[Dict], int]:
        """Make an API call to the specified endpoint."""
        try:
            url = f"{BASE_URL}{ENDPOINTS[endpoint]}"
            response = await client.post(url, json=test_features, headers=self.headers)

            if response.status_code != 200:
                print(f"Error calling {endpoint}: Status code {response.status_code}")
                return None, response.status_code

            return response.json(), response.status_code

        except Exception as e:
            print(f"Error calling {endpoint}: {str(e)}")
            return None, 500

    def analyze_competitor_results(self, response_data: Dict, test_row: pd.Series) -> pd.DataFrame:
        """Analyze competitor prediction results."""
        df = pd.DataFrame(response_data)
        df['real'] = test_row[response_data['mubi_v5'].keys()]
        df.loc['rank1'] = df.min(axis=0)

        for col in df.columns:
            if col != 'real':
                df[f'{col}_relative_error'] = (abs(df['real'] - df[col]) / df['real'] * 100)

        df['best_error'] = df.filter(like='error').min(axis=1)
        df['better_model'] = df.filter(like='error').idxmin(axis=1)

        return df

    def save_results(self, results: Dict[str, List], iteration: int):
        """Save test results to CSV files."""

        for endpoint, data_list in results.items():
            if data_list:
                df = pd.DataFrame(data_list)
                if endpoint == 'competitors':
                    df = pd.json_normalize(df['mubi_v5'])
                if endpoint == 'optimal_price':
                    rank1 = len(df[df['optimal_price'] < df['cheapest_competitor_price']])
                    total = len(df)
                    print(f'{rank1}/{total}')


                filename = f"{self.results_dir}/{endpoint}_results_iter{iteration}.csv"
                df.to_csv(filename, index=False)
                print(f"Saved {endpoint} results to {filename}")

    async def run_test_iteration(self) -> Dict[str, Dict]:
        test_row, test_features = self.get_test_case()
        results = {}

        async with httpx.AsyncClient() as client:
            for endpoint in ENDPOINTS.keys():
                response_data, status_code = await self.call_endpoint(client, endpoint, test_features)

                if response_data:
                    if endpoint == "competitors":
                        analysis_df = self.analyze_competitor_results(response_data, test_row)
                        results[endpoint] = {
                            "raw_response": response_data,
                            "analysis": analysis_df,
                            "better_model_counts": analysis_df['better_model'].value_counts().to_dict(),
                            "relative_errors": analysis_df.filter(like='error')
                        }
                    else:
                        results[endpoint] = {
                            "raw_response": response_data,
                            "test_case": test_features[0]
                        }

        return results


async def main():
    test_data = pd.read_parquet(get_raw_data_path('mubi_all_data', extension='.parquet'))
    tester = APITester(test_data)

    all_results = {
        "competitors": [],
        "technical_price": [],
        "optimal_price": []
    }

    # Run tests
    n_iterations = 100
    start_time = datetime.now()

    for i in range(n_iterations):
        print(f"\nRunning iteration {i + 1}/{n_iterations}")
        iteration_results = await tester.run_test_iteration()

        # Collect results
        for endpoint in ENDPOINTS.keys():
            if endpoint in iteration_results and iteration_results[endpoint]:
                all_results[endpoint].append(iteration_results[endpoint]["raw_response"])


    tester.save_results(all_results, n_iterations)


    # Print summary statistics
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\nTest completed in {duration:.2f} seconds")
    print(f"Total iterations: {n_iterations}")
    print(f"Results saved in {tester.results_dir}/")


if __name__ == "__main__":
    asyncio.run(main())