import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

def aggregate_data(
    retrieved_data: Dict,
    aggregations: List[Dict],
    time_period: str,
    period_type: str = "annual"
) -> Dict:
    """
    Apply aggregations to the retrieved financial data.
    
    Args:
        retrieved_data: Output from data_retriever with 'data', 'error', 'metadata' keys
        aggregations: List of {"metric": str, "aggregation": str} objects
        time_period: Time period for context
        period_type: "annual" or "quarterly"
    
    Returns:
        Dict with aggregated results
    """
    
    if retrieved_data.get("error") or retrieved_data.get("data").empty:
        return retrieved_data
    
    df = retrieved_data["data"].copy()
    
    # If no aggregations specified, return original data
    if not aggregations or all(agg.get("aggregation") is None for agg in aggregations):
        return retrieved_data
    
    results = []
    
    for agg_config in aggregations:
        metric = agg_config.get("metric")
        aggregation = agg_config.get("aggregation")
        
        if not metric or not aggregation:
            continue
            
        # Filter data for this specific metric
        metric_data = df[df["metric"] == metric].copy()
        
        if metric_data.empty:
            continue
            
        # Apply aggregation company-wise
        if aggregation == "growth":
            agg_result = calculate_growth(metric_data, time_period, period_type)
        else:
            agg_result = calculate_statistical_aggregation(metric_data, aggregation)
        
        # Add metric column with aggregation info
        agg_result["metric"] = f"{metric}_{aggregation}"
        
        results.append(agg_result)
    
    # Combine all aggregation results
    combined_df = combine_aggregation_results(results)
    
    return {
        "data": combined_df,
        "error": None,
        "metadata": {
            "aggregations_applied": [f"{agg['metric']}_{agg['aggregation']}" for agg in aggregations if agg.get('aggregation')],
            "time_period": time_period,
            "period_type": period_type,
            "companies_processed": df["company"].unique().tolist() if not df.empty else []
        }
    }

def calculate_statistical_aggregation(df: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    """
    Calculate statistical aggregations (avg, median, sum, min, max) company-wise.
    """
    
    aggregation_functions = {
        "avg": "mean",
        "median": "median", 
        "sum": "sum",
        "min": "min",
        "max": "max"
    }
    
    if aggregation not in aggregation_functions:
        return pd.DataFrame()
    
    # Group by company and sector, then apply aggregation
    grouped = df.groupby(["company", "sector"])["value"]
    
    if aggregation == "avg":
        result = grouped.mean()
    elif aggregation == "median":
        result = grouped.median()
    elif aggregation == "sum":
        result = grouped.sum()
    elif aggregation == "min":
        result = grouped.min()
    elif aggregation == "max":
        result = grouped.max()
    
    # Convert back to DataFrame
    result_df = result.reset_index()
    result_df.columns = ["company", "sector", "value"]
    
    return result_df

def calculate_growth(df: pd.DataFrame, time_period: str, period_type: str) -> pd.DataFrame:
    """
    Calculate growth rate (percentage change from start to end) company-wise.
    """
    
    results = []
    
    # Group by company
    for company, company_data in df.groupby("company"):
        company_data = company_data.sort_values("period")
        
        if len(company_data) < 2:
            continue
            
        start_value = company_data.iloc[0]["value"]
        end_value = company_data.iloc[-1]["value"]
        
        if pd.isna(start_value) or pd.isna(end_value) or start_value == 0:
            growth_rate = np.nan
        else:
            growth_rate = ((end_value - start_value) / start_value) * 100
        
        results.append({
            "company": company,
            "sector": company_data.iloc[0]["sector"],
            "value": growth_rate,
            "start_period": company_data.iloc[0]["period"],
            "end_period": company_data.iloc[-1]["period"],
            "start_value": start_value,
            "end_value": end_value
        })
    
    return pd.DataFrame(results)

def combine_aggregation_results(results: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine multiple aggregation results into a single DataFrame.
    """
    
    if not results:
        return pd.DataFrame()
    
    if len(results) == 1:
        return results[0].copy()
    
    # For multiple aggregations, create a wide format with proper column names
    combined = None
    
    for result_df in results:
        # Get the metric name from the result
        metric_name = result_df['metric'].iloc[0] if 'metric' in result_df.columns else 'unknown'
        
        # Prepare this result for merging
        merge_df = result_df[["company", "sector"]].copy()
        
        # Add the value column with the metric name
        merge_df[metric_name] = result_df["value"]
        
        # For growth, also add additional info columns
        if 'start_value' in result_df.columns:
            merge_df[f"{metric_name}_start"] = result_df["start_value"]
            merge_df[f"{metric_name}_end"] = result_df["end_value"]
            merge_df[f"{metric_name}_period"] = result_df["start_period"] + " to " + result_df["end_period"]
        
        if combined is None:
            combined = merge_df
        else:
            combined = combined.merge(merge_df, on=["company", "sector"], how="outer")
    
    return combined

def aggregator_node(state: dict) -> dict:
    """
    LangGraph node function for data aggregation.
    """
    
    retrieved_data = state.get("retrieved_data", {})
    aggregations = state.get("aggregations", [])
    time_period = state.get("time_period", "2015-2024")
    period_type = state.get("period_type", "annual")
    
    # Apply aggregations
    aggregated_result = aggregate_data(
        retrieved_data=retrieved_data,
        aggregations=aggregations,
        time_period=time_period,
        period_type=period_type
    )
    
    # Update state with aggregated data
    state["aggregated_data"] = aggregated_result
    
    return state 