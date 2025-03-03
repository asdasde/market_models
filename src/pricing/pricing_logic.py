import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Callable
from config import PricingConfig
import polars as pl
import sys



def calculate_tp_top_k(data : pd.DataFrame,
                       target_variables : List[str],
                       tp_kernel : List[Dict[str, Any]],
                       take_top_k : int = 3,
                       tp_nan_threshold : int = 3) -> np.ndarray:
    """
    :param data: A dataframe containing either one row (used with API) or full datasets (used for simulation)
    :param target_variables: A list of target variables used to calculate TP
    :param tp_kernel: A configuration of market prices which we take into account when calculating TP
    :param take_top_k: For each profile we take top k cheapest prices from tp_kernel, to avoid outpricing from small number of players
    :param tp_nan_threshold: For each profile if more than tp_nan_threshold prices are NaN, we set TP to NaN as well
    :return: The computed target price(s) based on the given market price configurations.
    """


    weights = [x['weight'] for x in tp_kernel]
    cost_estimates = [x['cost_estimate'] for x in tp_kernel]

    prices = data[target_variables].values

    weights = np.array([weights] * len(data))
    cost_estimates = np.array([cost_estimates] * len(data))

    sorted_indices = np.argsort(prices, axis=1)
    top_k_indices = sorted_indices[:, : take_top_k]
    top_k_premiums = np.take_along_axis(prices, top_k_indices, axis=1)
    top_k_weights = np.take_along_axis(weights, top_k_indices, axis=1)
    top_k_cost_estimates = np.take_along_axis(cost_estimates, top_k_indices, axis=1)

    top_k_weights = top_k_weights / top_k_weights.sum(axis=1)[:, None]

    composite = np.sum(top_k_premiums * top_k_weights * top_k_cost_estimates, axis=1)
    nan_mask = (
        np.isnan(top_k_premiums).sum(axis=1) > tp_nan_threshold
    )
    composite[nan_mask] = np.nan
    return composite

def calc_tp_adjustments_poland(data : pd.DataFrame, global_tp_adjustment : float = 1):
    """
    :param data: A dataframe containing either one row (used with API) or full datasets (used for simulation)
    :param global_tp_adjustment: Scales the TP obtained from calculate_tp_top_k with adjustment from the start
    :return: TP with adjustments for variables unused (weakly used by competitors)
    """
    data['TP_adjustment'] = float(global_tp_adjustment)
    data.loc[data['contractor_mtpl_number_of_claims'] == 1, 'TP_adjustment'] *= 1.33
    data.loc[data['contractor_mtpl_number_of_claims'] == 2, 'TP_adjustment'] *= 2
    data.loc[data['contractor_mtpl_number_of_claims'] >= 3, 'TP_adjustment'] *= 3

    data.loc[data['postal_code_population_density'] >= 3000,  'TP_adjustment'] *= 1.1
    data.loc[data['postal_code_population_density'] <= 70,  'TP_adjustment'] *= 0.935

    data.loc[data['vehicle_power'] >= 201, 'TP_adjustment'] *= 1.2

    data.loc[data['driver_experience'].between(0, 1), 'TP_adjustment'] *= 1.2
    data.loc[data['contractor_age'].between(17, 19), 'TP_adjustment'] *= 1.03
    data.loc[data['contractor_age'].between(75, 99), 'TP_adjustment'] *= 1.05

    data['TP'] = data['TP_competitors'] * data['TP_adjustment']
    return data

def calculate_price(data : pd.DataFrame, pricing_config : PricingConfig) -> pd.DataFrame:
    """
    :param data: A dataframe containing either one row (used with API) or full datasets
    :param pricing_config: An object containing all parameters needed for price prediction on specific market subset
    :return: Returns a dataframe containing the final price with all intermediate columns as well
    """

    target_variables = [x + '_model_prediction' for x in pricing_config.target_variables_and_model_config.keys()]
    target_variables_tp = [x['target_variable'] + '_model_prediction' for x in pricing_config.tp_kernel]

    tp = calculate_tp_top_k(data, target_variables_tp, pricing_config.tp_kernel, pricing_config.tp_take_top_k)

    undercut_price_col = f'rank1_undercut_{pricing_config.rank1_undercut_factor}'

    data['rank1_simulated'] = data[target_variables].min(axis = 1)

    data[undercut_price_col] = pricing_config.rank1_undercut_factor * data['rank1_simulated']

    data['TP_competitors'] = tp

    tp_adjustment_func : Callable = getattr(sys.modules[__name__], pricing_config.tp_adjustment_function)

    data = tp_adjustment_func(data, pricing_config.tp_global_adjustment)

    data[pricing_config.package_name] = np.where(
        data['TP'].isna(),
        np.nan,
        np.where(
            data['TP'] > data[undercut_price_col],
            data['TP'],
            data[undercut_price_col]
        )
    )

    return data


def calculate_conversion_factor(data : pl.DataFrame, price_col, conversion_rules : List[Tuple[str, float, float]]):
    expr : pl.Expr = None
    for rank_col, undercut_factor, rank_factor in reversed(conversion_rules):
        condition_expr = pl.col(price_col) <= pl.col(rank_col) * undercut_factor
        if expr is None:
            expr = pl.when(condition_expr).then(rank_factor).otherwise(pl.lit(0))
        else:
            expr = pl.when(condition_expr).then(rank_factor).otherwise(expr)
    print(expr)
    return data.with_columns(
        expr.alias('conversion_factor')
    )


STANDARD_CONVERSION_RULES = [
    ("rank_1-price", 0.80, 0.88),
    ("rank_1-price", 0.85, 0.84),
    ("rank_1-price", 0.90, 0.80),
    ("rank_1-price", 0.925, 0.73),
    ("rank_1-price", 0.95, 0.65),
    ("rank_1-price", 0.96, 0.60),
    ("rank_1-price", 0.97, 0.50),
    ("rank_1-price", 0.98, 0.45),
    ("rank_1-price", 0.99, 0.40),
    ("rank_1-price", 1.0, 0.30),
    ("rank_2-price", 1.0, 0.05),
    ("rank_3-price", 1.0, 0.01)
]




