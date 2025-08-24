import os
import pandas as pd
from dotenv import load_dotenv
from google.generativeai import configure, GenerativeModel
from config.gemini_config import model
import re
from fuzzywuzzy import fuzz, process
import json

def get_company_sector_data(csv_path='data/data.csv'):
    """Load and return company, sector, and indices data from CSV"""
    df = pd.read_csv(csv_path)
    companies = df['company'].str.lower().unique().tolist()
    sectors = df['sector'].str.lower().unique().tolist()
    
    # Extract unique indices from the comma-separated indices column
    all_indices = []
    for indices_str in df['indices'].dropna():
        # Split by comma and strip whitespace
        indices_list = [idx.strip().lower() for idx in indices_str.split(',')]
        all_indices.extend(indices_list)
    
    # Get unique indices
    indices = list(set(all_indices))
    
    return companies, sectors, indices, df

def fuzzy_match_entities(query, companies, sectors, indices, threshold=70):
    """
    Use fuzzywuzzy for intelligent fuzzy matching to find potential company/sector/index matches
    instead of sending all names to LLM. Uses advanced string matching algorithms.
    """
    query_lower = query.lower()
    potential_companies = []
    potential_sectors = []
    potential_indices = []
    
    # Company ticker/code mapping
    company_codes = {
        'infy': 'infosys',
        'tcs': 'tcs',  # Already matches
        'wipro': 'wipro',  # Already matches
        'hdfcbank': 'hdfc bank',
        'icicibank': 'icici bank', 
        'axisbank': 'axis bank',
        'reliance': 'reliance industries',
        'ntpc': 'ntpc',  # Already matches
        'sunpharma': 'sun pharma',
        'drreddy': "dr. reddy's"
    }
    
    # Check for company codes first
    for code, company_name in company_codes.items():
        if code in query_lower:
            if company_name in companies:
                potential_companies.append(company_name)
    
    # Enhanced company matching with fuzzywuzzy
    for company in companies:
        company_lower = company.lower()
        
        # Direct substring match (highest priority)
        if company_lower in query_lower:
            potential_companies.append(company)
            continue
            
        # Fuzzy ratio matching
        if fuzz.ratio(company_lower, query_lower) >= threshold:
            potential_companies.append(company)
            continue
            
        # Partial ratio for substring matching (e.g., "HDFC" in "HDFC Bank")
        if fuzz.partial_ratio(company_lower, query_lower) >= threshold + 10:
            potential_companies.append(company)
            continue
            
        # Token set ratio for word order independence
        if fuzz.token_set_ratio(company_lower, query_lower) >= threshold:
            potential_companies.append(company)
            continue

    # Use process.extractOne for best match approach (backup)
    if not potential_companies:
        best_company_match = process.extractOne(query_lower, companies, scorer=fuzz.token_set_ratio)
        if best_company_match and best_company_match[1] >= threshold - 10:  # Slightly lower threshold for backup
            potential_companies.append(best_company_match[0])
    
    # Enhanced sector matching
    # Create expanded sector variations for better matching
    sector_variations = {
        'it': ['it', 'information technology', 'tech', 'technology', 'software', 'computer'],
        'banking': ['banking', 'bank', 'banks', 'finance sector', 'financial sector'],  # More specific
        'energy': ['energy', 'oil', 'gas', 'petroleum', 'power'],
        'pharma': ['pharma', 'pharmaceutical', 'medicine', 'drugs', 'healthcare']
    }
    
    for sector in sectors:
        sector_lower = sector.lower()
        
        # Direct match
        if sector_lower in query_lower:
            potential_sectors.append(sector)
            continue
            
        # Check variations
        if sector_lower in sector_variations:
            for variation in sector_variations[sector_lower]:
                if variation in query_lower:
                    potential_sectors.append(sector)
                    break
            if sector in potential_sectors:
                continue
                
        # Fuzzy matching for sectors
        if fuzz.partial_ratio(sector_lower, query_lower) >= threshold:
            potential_sectors.append(sector)
    
    # Enhanced index matching
    # Create expanded index variations for better matching
    index_variations = {
        'nifty 50': ['nifty', 'nifty 50', 'nifty50', 'nse nifty', 'nifty index'],
        'nse 500': ['nse 500', 'nse500', 'nse index', 'nse broad index'],
        'bse 30': ['bse 30', 'bse30', 'sensex', 'bse sensex', 'bse index']
    }
    
    for index in indices:
        index_lower = index.lower()
        
        # Direct match
        if index_lower in query_lower:
            potential_indices.append(index)
            continue
            
        # Check variations
        if index_lower in index_variations:
            for variation in index_variations[index_lower]:
                if variation in query_lower:
                    potential_indices.append(index)
                    break
            if index in potential_indices:
                continue
                
        # Fuzzy matching for indices
        if fuzz.partial_ratio(index_lower, query_lower) >= threshold:
            potential_indices.append(index)
    
    # Remove duplicates while preserving order
    potential_companies = list(dict.fromkeys(potential_companies))
    potential_sectors = list(dict.fromkeys(potential_sectors))
    potential_indices = list(dict.fromkeys(potential_indices))
    
    return potential_companies, potential_sectors, potential_indices

def classify_and_parse_query(query, csv_path='data/data.csv'):
    """
    Combined function that both classifies and parses the query in one API call.
    Uses local fuzzy matching to decide whether to call LLM at all.
    """
    
    # Load data and do local fuzzy matching
    companies, sectors, indices, df = get_company_sector_data(csv_path)
    potential_companies, potential_sectors, potential_indices = fuzzy_match_entities(query, companies, sectors, indices)
    
    # OPTIMIZATION: If no potential matches found locally, skip LLM entirely
    if not potential_companies and not potential_sectors and not potential_indices:
        # Check if query seems financial in nature (contains financial keywords)
        financial_keywords = ['sales', 'profit', 'revenue', 'earnings', 'eps', 'roe', 'debt', 'dividend', 
                            'stock', 'price', 'market', 'financial', 'performance', 'growth',
                            'analysis', 'trend', 'comparison', 'quarterly', 'annual']
        
        query_lower = query.lower()
        seems_financial = any(keyword in query_lower for keyword in financial_keywords)
        
        if seems_financial:
            # Financial query but no known entities
            return {
                "classification": "financial_unknown",
                "company": None,
                "sector": None,
                "indices": None,
                "fundamental": None,
                "time_period": "2015-2024",
                "period_type": "annual",
                "graph_needed": False,
                "table_needed": False
            }
        else:
            # Non-financial query
            return {
                "classification": "non_financial",
                "company": None,
                "sector": None,
                "indices": None,
                "fundamental": None,
                "time_period": "2015-2024",
                "period_type": "annual",
                "graph_needed": False,
                "table_needed": False
            }
    
    # If we have potential matches, send to LLM with ONLY those matches
    # Convert to proper case for display
    company_mapping = {
        'infosys': 'Infosys', 'tcs': 'TCS', 'wipro': 'Wipro',
        'hdfc bank': 'HDFC Bank', 'icici bank': 'ICICI Bank', 'axis bank': 'Axis Bank',
        'reliance industries': 'Reliance Industries', 'ntpc': 'NTPC', 
        'sun pharma': 'Sun Pharma', "dr. reddy's": "Dr. Reddy's"
    }
    
    sector_mapping = {
        'it': 'IT', 'banking': 'Banking', 'energy': 'Energy', 'pharma': 'Pharma'
    }
    
    indices_mapping = {
        'nifty 50': 'Nifty 50', 'nse 500': 'NSE 500', 'bse 30': 'BSE 30'
    }
    
    # Convert potential matches to proper case
    potential_companies_display = [company_mapping.get(c, c.title()) for c in potential_companies]
    potential_sectors_display = [sector_mapping.get(s, s.title()) for s in potential_sectors]
    potential_indices_display = [indices_mapping.get(i, i.title()) for i in potential_indices]
    
    # include derived metrics from formulas database
    try:
        from agents.math import MathAgent
        derived = MathAgent().available_derived_metrics()
    except Exception:
        derived = []
    base_fundamentals = ['sales', 'net profit', 'roe', 'eps', 'debt equity', 'dividend yield', 'avg price']
    fundamentals = base_fundamentals + [f.replace('_', ' ') for f in derived if f not in base_fundamentals]
    
    prompt = f"""
        You are a financial query classifier and parser. Your task is to analyze the user's query about financial data and extract key information into a structured JSON format. The available data includes both annual and quarterly figures from 2015 to 2024.

        Based on the query and the provided context, return a JSON object with the following fields:

        1.  "classification": Must be "financial_known" because relevant entities were detected in the query.
        2.  "company": A list of company names found in the query. Example: ["Infosys", "TCS"]. If no specific company is mentioned, use null.
        3.  "sector": A list of business sectors found in the query. Example: ["IT", "Banking"]. If no sector is mentioned, use null.
        4.  "indices": A list of stock market indices found in the query. Example: ["Nifty 50", "BSE 30"]. If no index is mentioned, use null.
        5.  "fundamentals": A list of the financial metrics requested (e.g., ["sales", "net profit"]). Use "avg_price" for any stock price related queries. If none are mentioned, use an empty list.
        6.  "time_period": The time frame for the query.
            - For a single year, use "YYYY" (e.g., "2021").
            - For a range of years, use "YYYY-YYYY" (e.g., "2020-2024").
            - For a single quarter, use "YYYY-Qn" (e.g., "2022-Q2").
            - For a range of quarters, use the format "YYYY-Qn to YYYY-Qn" (e.g., "2015-Q3 to 2019-Q2").
            - If a quarter is mentioned without a year (e.g., "Q2," "q3"), default to the latest year, 2024 (e.g., "2024-Q2").
            - If no time period is mentioned, default to "2015-2024".
            - Interpret phrases like "last 5 years" relative to the latest available data (2024), so it would be "2020-2024".
        7.  "period_type": Specify whether the user wants "annual" or "quarterly" data. If the query mentions "quarter," "quarterly," or a specific quarter (like Q1, Q2), set this to "quarterly." Otherwise, default to "annual".
        8.  "aggregations": A list of objects aligned 1:1 with the fundamentals list. Each object should have:
            - "metric": The financial metric (e.g., "sales", "net profit", "eps")
            - "aggregation": The type of aggregation requested ("avg", "median", "sum", "min", "max", "growth")
            Examples:
            - For "avg sales and max eps": fundamentals: ["sales", "eps"], aggregations: [{{"metric": "sales", "aggregation": "avg"}}, {{"metric": "eps", "aggregation": "max"}}]
            - For "growth of pat": fundamentals: ["net profit"], aggregations: [{{"metric": "net profit", "aggregation": "growth"}}]
            - For simple queries without aggregation like "show sales": fundamentals: ["sales"], aggregations: [{{"metric": "sales", "aggregation": null}}]
            Aggregation keyword mapping:
            - "average", "avg", "mean" → "avg"
            - "median" → "median"
            - "total", "sum" → "sum"
            - "highest", "maximum", "max" → "max"
            - "lowest", "minimum", "min" → "min"
            - "growth", "grew", "increase", "change" → "growth"
        IMPORTANT: The metrics in "aggregations" MUST MATCH the order and content of "fundamentals" exactly. If no aggregation is implied for a metric, set its aggregation to null.
        9.  "graph_needed": Set to `true` if the query implies a request for a graph, chart, plot, or any form of visualization. Otherwise, `false`.
        10. "table_needed": Set to `true` if the query implies a request for data in a table or tabular format. Otherwise, `false`.

        Context for parsing:
        - Detected Companies: {potential_companies_display}
        - Detected Sectors: {potential_sectors_display}
        - Detected Indices: {potential_indices_display}
        - Available Fundamentals: {fundamentals}

        Return ONLY a valid JSON object. Do not include any explanatory text.

        Query: "{query}"
    """
    
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.1}
    )
    cleaned = response.text.strip().strip('```json').strip('```').strip()
    
    try:
        result = json.loads(cleaned)
        result["classification"] = "financial_known"  # Force since we have matches

        # Normalize fundamentals to a list and enforce 1:1 with aggregations
        fundamentals_from_result = result.get("fundamentals")
        if isinstance(fundamentals_from_result, list):
            fundamentals_list = [f for f in fundamentals_from_result if isinstance(f, str) and f]
        elif isinstance(result.get("fundamental"), str) and result.get("fundamental"):
            fundamentals_list = [result.get("fundamental")]
        else:
            # derive from aggregations metrics if present
            aggs = result.get("aggregations") or []
            fundamentals_list = []
            for agg in aggs:
                m = (agg or {}).get("metric")
                if isinstance(m, str) and m and m not in fundamentals_list:
                    fundamentals_list.append(m)

        # Build normalized aggregations aligned with fundamentals_list
        aggs_input = result.get("aggregations") or []
        metric_to_agg = {}
        for agg in aggs_input:
            if isinstance(agg, dict):
                m = agg.get("metric")
                if isinstance(m, str) and m and m not in metric_to_agg:
                    metric_to_agg[m] = agg.get("aggregation") if agg.get("aggregation") is not None else None

        normalized_aggs = []
        for m in fundamentals_list:
            normalized_aggs.append({"metric": m, "aggregation": metric_to_agg.get(m)})

        result["fundamentals"] = fundamentals_list
        result["aggregations"] = normalized_aggs
        return result
    except Exception as e:
        # Return default structure on parsing error
        return {
            "classification": "financial_known",
            "company": potential_companies_display if potential_companies_display else None,
            "sector": potential_sectors_display if potential_sectors_display else None,
            "indices": potential_indices_display if potential_indices_display else None,
            "fundamental": None,
            "fundamentals": [],
            "time_period": "2015-2024",
            "graph_needed": False,
            "table_needed": False
        }

def classify_and_parse_node(state: dict) -> dict:
    """
    Combined LangGraph node that handles both classification and parsing
    """
    query = state["query"]
    result = classify_and_parse_query(query)
    
    # Update state with all results
    state["classification"] = result["classification"]
    state["company"] = result["company"]
    state["sector"] = result["sector"]
    state["indices"] = result["indices"]
    #state["fundamental"] = result["fundamental"]
    state["fundamentals"] = result.get("fundamentals", [])
    state["time_period"] = result["time_period"]
    state["period_type"] = result.get("period_type", "annual")
    state["aggregations"] = result.get("aggregations", [])
    state["graph_needed"] = result["graph_needed"]
    state["table_needed"] = result["table_needed"]
    
    return state

def classification_router(state: dict) -> str:
    """Router function for the combined node"""
    return state.get("classification", "uncertain")

# Keep backward compatibility - these functions are now deprecated but kept for reference
def classify_query(query, csv_path='data/data.csv'):
    """Deprecated: Use classify_and_parse_query instead"""
    result = classify_and_parse_query(query, csv_path)
    return result["classification"]

def classify_node(state: dict) -> dict:
    """Deprecated: Use classify_and_parse_node instead"""
    return classify_and_parse_node(state)