import asyncio
import json
import random
import httpx
import os
import sys
from typing import Dict, Tuple, Optional, List


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.load_utils import *

load_dotenv()
API_KEY = os.getenv("API_KEY", "your_api_key_here")


BASE_URL = "https://ml.staging.pl.ominimo.eu"
#BASE_URL = "http://0.0.0.0:8081"
ENDPOINTS = {
    "competitors": "/predict/competitors",
    "technical_price": "/predict/technical_price",
    "optimal_price": "/predict/optimal_price"
}

from config import ApiConfig

conf = ApiConfig.load_from_json('mubi_cheapest_offers')

TRAIN_DATA_NAME = conf.train_data_name
REQUIRED_COLUMNS = [col['name'] for col in conf.feature_columns]

class APITester:
    def __init__(self, test_data: pd.DataFrame):
        self.test_data = test_data
        self.headers = {
            "X-API-Key": API_KEY,
            "Content-Type": "application/json"
        }
        self.results_dir = "test_results"
        os.makedirs(self.results_dir, exist_ok=True)

    def get_test_case(self) -> Tuple[pd.Series, Dict]:
        rnd = random.randint(0, len(self.test_data) - 1)
        test_row = self.test_data.iloc[rnd]
        test_features = test_row[REQUIRED_COLUMNS].to_dict()
        return test_row, test_features

    async def call_endpoint(self, client: httpx.AsyncClient, endpoint: str,
                            test_features: Dict) -> Tuple[Optional[Dict], int]:
        url = f"{BASE_URL}{ENDPOINTS[endpoint]}"
        response = await client.post(url, json=test_features, headers=self.headers)

        if response.status_code != 200:
            print(f"Error calling {endpoint}: Status code {response.status_code}")
            print(response.content)
            return None, response.status_code

        return response.json(), response.status_code
        #
        # except Exception as e:
        #     print(f"Error calling {endpoint}: {str(e)}")
        #     return None, 500

    def analyze_competitor_results(self, response_data: Dict, test_row: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame([response_data])


        df[[f'real_{col}' for col in df.columns]] = test_row[df.columns].values

        for col in df.columns:
            if not col.startswith('real'):
                df[f'{col}_relative_error'] = (abs(df[f'real_{col}'] - df[col]) / df[f'real_{col}'] * 100)

        return df

    def save_results(self, results: Dict[str, List], iteration: int):
        for endpoint, data_list in results.items():
            if data_list:

                if endpoint != 'competitors':
                    df = pd.DataFrame(data_list)
                    if endpoint == 'optimal_price':
                        rank1 = len(df[df['optimal_price'] < df['cheapest_competitor_price']])
                        total = len(df)
                        print(f'{rank1}/{total}')
                else:
                    df = pd.concat(data_list)


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
                            "relative_errors": analysis_df.filter(like='error')
                        }
                    else:
                        results[endpoint] = {
                            "raw_response": response_data,
                            "test_case": test_features
                        }

        return results


async def main():
    path_manager = PathManager('mubi')
    test_data = pd.read_parquet(path_manager.get_raw_data_path('mubi_all_data_new', extension='.parquet'))
    try:
        test_data['policy_start_date'] = test_data['policy_start_date'].dt.strftime('%Y_%m_%d')
    except:
        pass
    tester = APITester(test_data)

    all_results = {
        "competitors": [],
        "technical_price": [],
        "optimal_price": []
    }

    # Run tests
    n_iterations = 1
    start_time = datetime.now()

    for i in range(n_iterations):
        print(f"\nRunning iteration {i + 1}/{n_iterations}")
        iteration_results = await tester.run_test_iteration()

        # Collect results
        for endpoint in ENDPOINTS.keys():
            if endpoint in iteration_results and iteration_results[endpoint]:
                if endpoint == 'competitors':
                    all_results[endpoint].append(iteration_results[endpoint]["analysis"])
                else:
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