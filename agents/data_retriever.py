import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

def retrieve_data(
    csv_path: str = 'data/data.csv',
    company: Optional[List[str]] = None,
    sector: Optional[List[str]] = None,
    fundamental: Optional[str] = None,
    time_period: str = "2015-2024"
) -> Dict:
    """
    Retrieve financial data based on parsed query parameters.
    
    Args:
        csv_path: Path to the CSV file
        company: List of company names to filter
        sector: List of sectors to filter  
        fundamental: Financial metric to extract
        time_period: Time range (e.g., "2020-2024", "2021-2021")
    
    Returns:
        Dict containing filtered data, metadata, and any errors
    """
    
    try:
        # Load the CSV
        df = pd.read_csv(csv_path)
        
        # Parse time period
        start_year, end_year = parse_time_period(time_period)
        years = list(range(start_year, end_year + 1))
        
        # Filter by company if specified
        if company:
            # Case-insensitive company matching
            company_mask = df['company'].str.lower().isin([c.lower() for c in company])
            df = df.loc[company_mask].copy()
            
            if df.empty:
                return {
                    "data": pd.DataFrame(),
                    "error": f"No data found for companies: {company}",
                    "metadata": {"companies_requested": company, "years": years}
                }
        
        # Filter by sector if specified
        if sector:
            # Case-insensitive sector matching
            sector_mask = df['sector'].str.lower().isin([s.lower() for s in sector])
            df = df.loc[sector_mask].copy()
            
            if df.empty:
                return {
                    "data": pd.DataFrame(),
                    "error": f"No data found for sectors: {sector}",
                    "metadata": {"sectors_requested": sector, "years": years}
                }
        
        # Extract fundamental data if specified
        if fundamental:
            extracted_data = extract_fundamental_data(df, fundamental, years)
            
            if extracted_data.empty:
                return {
                    "data": pd.DataFrame(),
                    "error": f"No data found for fundamental '{fundamental}' in years {years}",
                    "metadata": {
                        "fundamental": fundamental,
                        "years": years,
                        "companies_found": df['company'].tolist()
                    }
                }
            
            return {
                "data": extracted_data,
                "error": None,
                "metadata": {
                    "fundamental": fundamental,
                    "years": years,
                    "companies_found": df['company'].tolist(),
                    "sectors_found": df['sector'].unique().tolist()
                }
            }
        
        # If no fundamental specified, return company/sector info
        else:
            basic_info = df[['company', 'sector']].copy()
            return {
                "data": basic_info,
                "error": None,
                "metadata": {
                    "years": years,
                    "companies_found": df['company'].tolist(),
                    "sectors_found": df['sector'].unique().tolist()
                }
            }
            
    except FileNotFoundError:
        return {
            "data": pd.DataFrame(),
            "error": f"CSV file not found: {csv_path}",
            "metadata": {}
        }
    except Exception as e:
        return {
            "data": pd.DataFrame(),
            "error": f"Error retrieving data: {str(e)}",
            "metadata": {}
        }

def parse_time_period(time_period: str) -> tuple:
    """
    Parse time period string into start and end years.
    
    Args:
        time_period: Time period string (e.g., "2020-2024", "2021-2021")
    
    Returns:
        Tuple of (start_year, end_year)
    """
    try:
        if '-' in time_period:
            start_str, end_str = time_period.split('-')
            start_year = int(start_str.strip())
            end_year = int(end_str.strip())
        else:
            # Single year
            start_year = end_year = int(time_period.strip())
        
        # Validate years are within available range (2015-2024)
        start_year = max(2015, min(2024, start_year))
        end_year = max(2015, min(2024, end_year))
        
        return start_year, end_year
    
    except (ValueError, AttributeError):
        # Default to full range if parsing fails
        return 2015, 2024

def extract_fundamental_data(df: pd.DataFrame, fundamental: str, years: List[int]) -> pd.DataFrame:
    """
    Extract specific fundamental data across specified years.
    
    Args:
        df: Filtered DataFrame
        fundamental: Financial metric name
        years: List of years to extract
    
    Returns:
        DataFrame with company, sector, and yearly data for the fundamental
    """
    
    # Convert spaces to underscores to match CSV column naming convention
    fundamental_normalized = fundamental.replace(' ', '_')
    
    # Create column names for the requested years
    fundamental_cols = [f"{fundamental_normalized}_{year}" for year in years]
    
    # Check which columns actually exist in the data
    available_cols = [col for col in fundamental_cols if col in df.columns]
    
    if not available_cols:
        return pd.DataFrame()
    
    # Extract basic info + fundamental data
    result_cols = ['company', 'sector'] + available_cols
    result_df = df[result_cols].copy()
    
    # Melt the data to long format for easier visualization
    melted_df = pd.melt(
        result_df,
        id_vars=['company', 'sector'],
        value_vars=available_cols,
        var_name='metric_year',
        value_name='value'
    )
    
    # Extract year from column name
    melted_df['year'] = melted_df['metric_year'].str.extract(r'(\d{4})').astype(int)
    melted_df['metric'] = fundamental
    
    # Clean up and reorder columns
    final_df = melted_df[['company', 'sector', 'metric', 'year', 'value']].copy()
    
    # Remove rows with null values
    final_df = final_df.dropna(subset=['value']).copy()
    
    # Sort by company and year
    final_df = final_df.sort_values(['company', 'year']).reset_index(drop=True)
    
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
    fundamental = state.get("fundamental")
    time_period = state.get("time_period", "2015-2024")
    
    # Retrieve data
    result = retrieve_data(
        company=company,
        sector=sector,
        fundamental=fundamental,
        time_period=time_period
    )
    
    # Update state with complete result structure
    state["retrieved_data"] = result  # Store the complete dict with data, error, metadata
    
    return state
