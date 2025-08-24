import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from .math import MathAgent

def retrieve_data(
    csv_path: str = 'data/data.csv',
    company: Optional[List[str]] = None,
    sector: Optional[List[str]] = None,
    indices: Optional[List[str]] = None,
    fundamental: Optional[str] = None,
    fundamentals: Optional[List[str]] = None,
    time_period: str = "2015-2024",
    period_type: str = "annual"
) -> Dict:
    """
    Retrieve financial data based on parsed query parameters.
    
    Args:
        csv_path: Path to the CSV file
        company: List of company names to filter
        sector: List of sectors to filter  
        indices: List of indices to filter (e.g., ["Nifty 50", "NSE 500"])
        fundamental: Single financial metric to extract (deprecated; will be merged into fundamentals)
        fundamentals: List of financial metrics to extract; preferred over 'fundamental'
        time_period: Time range (e.g., "2020-2024", "2021", "2022-Q2")
        period_type: "annual" or "quarterly"
    
    Returns:
        Dict containing filtered data, metadata, and any errors
    """
    
    try:
        df = pd.read_csv(csv_path)
        
        # Filter by period_type first
        df = df[df['period_type'].str.lower() == period_type.lower()]
        
        # Parse time period
        start_period, end_period = parse_time_period(time_period, period_type)
        
        # Filter by time period
        if period_type == 'annual':
            # For annual, period is an integer year
            df['year'] = pd.to_numeric(df['period'], errors='coerce')
            df = df[(df['year'] >= start_period) & (df['year'] <= end_period)]
        else: # quarterly
            # For quarterly, period is a string 'YYYY-Qn'
            df = df[(df['period'] >= start_period) & (df['period'] <= end_period)]

        # Filter by company if specified
        if company:
            company_mask = df['company'].str.lower().isin([c.lower() for c in company])
            df = df.loc[company_mask].copy()
            if df.empty:
                return {"data": pd.DataFrame(), "error": f"No data found for companies: {company}", "metadata": {}}

        # Filter by sector if specified
        if sector:
            sector_mask = df['sector'].str.lower().isin([s.lower() for s in sector])
            df = df.loc[sector_mask].copy()
            if df.empty:
                return {"data": pd.DataFrame(), "error": f"No data found for sectors: {sector}", "metadata": {}}

        # Filter by indices if specified
        if indices:
            indices_lower = [idx.lower() for idx in indices]
            def check_indices_match(row_indices):
                if pd.isna(row_indices): return False
                row_indices_list = [idx.strip().lower() for idx in str(row_indices).split(',')]
                return any(idx in row_indices_list for idx in indices_lower)
            indices_mask = df['indices'].apply(check_indices_match)
            df = df.loc[indices_mask].copy()
            if df.empty:
                return {"data": pd.DataFrame(), "error": f"No data found for indices: {indices}", "metadata": {}}

        # Handle fundamentals (multiple metrics)
        metrics_to_extract = []
        if fundamentals:
            metrics_to_extract = [m for m in fundamentals if isinstance(m, str) and m]
        # backward compatibility: include single fundamental if provided
        if not metrics_to_extract and fundamental:
            metrics_to_extract = [fundamental]
        
        if metrics_to_extract:
            all_extracted_data = []
            math_agent = MathAgent()
            for metric in metrics_to_extract:
                extracted_data = extract_fundamental_data(df, metric)
                if not extracted_data.empty:
                    all_extracted_data.append(extracted_data)
                else:
                    # Try derived metric via MathAgent formulas
                    try:
                        derived_df = math_agent.compute_derived_metric(df, metric)
                    except Exception:
                        derived_df = None
                    if derived_df is not None and not derived_df.empty:
                        all_extracted_data.append(derived_df)
            
            if not all_extracted_data:
                return {"data": pd.DataFrame(), "error": f"No data found for fundamentals: {metrics_to_extract}", "metadata": {}}
            
            # Combine all fundamental data
            combined_data = pd.concat(all_extracted_data, ignore_index=True)
            return {"data": combined_data, "error": None, "metadata": {"fundamentals": metrics_to_extract}}
            
        # If no fundamental specified, return basic info
        else:
            basic_info = df[['company', 'sector', 'indices', 'period']].copy()
            return {"data": basic_info, "error": None, "metadata": {"fundamentals": []}}
            
    except FileNotFoundError:
        return {"data": pd.DataFrame(), "error": f"CSV file not found: {csv_path}", "metadata": {}}
    except Exception as e:
        return {"data": pd.DataFrame(), "error": f"Error retrieving data: {str(e)}", "metadata": {}}

def parse_time_period(time_period: str, period_type: str) -> tuple:
    """
    Parse time period string into start and end periods.
    """
    if period_type == 'annual':
        try:
            if '-' in time_period:
                start_str, end_str = time_period.split('-')
                start_year = int(start_str.strip())
                end_year = int(end_str.strip())
            else:
                start_year = end_year = int(time_period.strip())
            # Validate years
            start_year = max(2015, min(2024, start_year))
            end_year = max(2015, min(2024, end_year))
            return start_year, end_year
        except (ValueError, AttributeError):
            return 2015, 2024
    
    else: # quarterly
        try:
            # For quarterly, time_period can be YYYY-Qn or YYYY-YYYY or YYYY-Qn to YYYY-Qn
            if ' to ' in time_period: # Quarter range
                start_str, end_str = time_period.split(' to ')
                return start_str.strip(), end_str.strip()
            elif '-Q' in time_period: # Single quarter
                return time_period, time_period
            elif '-' in time_period: # Year range
                start_year, end_year = map(int, time_period.split('-'))
                start_year = max(2015, min(2024, start_year))
                end_year = max(2015, min(2024, end_year))
                return f"{start_year}-Q1", f"{end_year}-Q4"
            else: # Single year
                year = int(time_period)
                year = max(2015, min(2024, year))
                return f"{year}-Q1", f"{year}-Q4"
        except (ValueError, AttributeError):
            return "2015-Q1", "2024-Q4"


def extract_fundamental_data(df: pd.DataFrame, fundamental: str) -> pd.DataFrame:
    """
    Extract specific fundamental data from the normalized dataframe.
    
    Args:
        df: Filtered DataFrame in long format.
        fundamental: Financial metric name (e.g., 'sales', 'net_profit').
    
    Returns:
        DataFrame with company, sector, period, and the fundamental's value.
    """
    # Normalize fundamental name to match column names
    fundamental_col = fundamental.replace(' ', '_')

    if fundamental_col not in df.columns:
        return pd.DataFrame()

    # Columns to keep
    result_cols = ['company', 'sector', 'period', fundamental_col]
    
    # Select and rename the fundamental column to a generic 'value'
    result_df = df[result_cols].copy()
    result_df.rename(columns={fundamental_col: 'value'}, inplace=True)
    
    # Add a 'metric' column
    result_df['metric'] = fundamental
    
    # Reorder columns
    final_df = result_df[['company', 'sector', 'metric', 'period', 'value']].copy()
    
    # Remove rows with null values
    final_df = final_df.dropna(subset=['value']).copy()
    
    # Sort by company and period
    final_df = final_df.sort_values(['company', 'period']).reset_index(drop=True)
    
    return final_df


def retrieve_node(state: dict) -> dict:
    """
    LangGraph node function for data retrieval.
    
    Args:
        state: State dictionary containing parsed query parameters
    
    Returns:
        Updated state with retrieved data
    """
    
    # Extract parameters from state
    company = state.get("company")
    sector = state.get("sector") 
    indices = state.get("indices")
    fundamental = state.get("fundamental")
    fundamentals_list = state.get("fundamentals", [])
    time_period = state.get("time_period", "2015-2024")
    period_type = state.get("period_type", "annual")
    aggregations = state.get("aggregations", [])
    
    # Extract fundamentals from aggregations if available
    fundamentals_from_aggs: List[str] = []
    if aggregations:
        fundamentals_from_aggs = [agg.get("metric") for agg in aggregations if isinstance(agg, dict) and agg.get("metric")]
        fundamentals_from_aggs = list(dict.fromkeys(fundamentals_from_aggs))
    
    # Retrieve data
    result = retrieve_data(
        company=company,
        sector=sector,
        indices=indices,
        fundamental=fundamental,
        fundamentals=fundamentals_list if fundamentals_list else (fundamentals_from_aggs if fundamentals_from_aggs else None),
        time_period=time_period,
        period_type=period_type
    )
    
    # Update state with complete result structure
    state["retrieved_data"] = result  # Store the complete dict with data, error, metadata
    
    return state
