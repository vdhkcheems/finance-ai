import os
import pandas as pd
from dotenv import load_dotenv
from google.generativeai import configure, GenerativeModel
from config.gemini_config import model
import re
from fuzzywuzzy import fuzz, process
import json

def get_company_sector_data(csv_path='data/data.csv'):
    """Load and return company and sector data from CSV"""
    df = pd.read_csv(csv_path)
    companies = df['company'].str.lower().unique().tolist()
    sectors = df['sector'].str.lower().unique().tolist()
    return companies, sectors, df

def fuzzy_match_entities(query, companies, sectors, threshold=70):
    """
    Use fuzzywuzzy for intelligent fuzzy matching to find potential company/sector matches
    instead of sending all names to LLM. Uses advanced string matching algorithms.
    """
    query_lower = query.lower()
    potential_companies = []
    potential_sectors = []
    
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
    
    # Remove duplicates while preserving order
    potential_companies = list(dict.fromkeys(potential_companies))
    potential_sectors = list(dict.fromkeys(potential_sectors))
    
    return potential_companies, potential_sectors

def classify_and_parse_query(query, csv_path='data/data.csv'):
    """
    Combined function that both classifies and parses the query in one API call.
    Uses local fuzzy matching to decide whether to call LLM at all.
    """
    
    # Load data and do local fuzzy matching
    companies, sectors, df = get_company_sector_data(csv_path)
    potential_companies, potential_sectors = fuzzy_match_entities(query, companies, sectors)
    
    # OPTIMIZATION: If no potential matches found locally, skip LLM entirely
    if not potential_companies and not potential_sectors:
        # Check if query seems financial in nature (contains financial keywords)
        financial_keywords = ['sales', 'profit', 'revenue', 'earnings', 'roe', 'debt', 'dividend', 
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
                "fundamental": None,
                "time_period": "2015-2024",
                "graph_needed": False,
                "table_needed": False
            }
        else:
            # Non-financial query
            return {
                "classification": "non_financial",
                "company": None,
                "sector": None,
                "fundamental": None,
                "time_period": "2015-2024",
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
    
    # Convert potential matches to proper case
    potential_companies_display = [company_mapping.get(c, c.title()) for c in potential_companies]
    potential_sectors_display = [sector_mapping.get(s, s.title()) for s in potential_sectors]
    
    fundamentals = ['sales', 'net profit', 'roe', 'debt equity', 'dividend yield', 'avg price']
    
    prompt = f"""
        You are a financial query classifier and parser. Analyze the user query and return a JSON object with both classification and parsed information.

        Detected companies: {potential_companies_display}
        Detected sectors: {potential_sectors_display}
        Available fundamentals: {fundamentals}

        Return a JSON object with these fields:
        1. "classification": Must be "financial_known" (since we detected relevant entities)
        2. "company": List of company names from detected companies (e.g., ["Infosys", "TCS"]). Use null if not mentioned.
        3. "sector": List of sectors from detected sectors (e.g., ["IT", "Banking"]). Use null if not mentioned.
        4. "fundamental": Financial metric requested (e.g., "sales", "net profit", "roe", "avg price"). Use null if not mentioned. Use "avg price" for stock price queries.
        5. "time_period": Time period string (e.g., "2019", "2020-2024", "2015-2024"). Use "2015-2024" if not mentioned. For "last 5 years" use "2020-2024". For single year use format "2021-2021".
        6. "graph_needed": true if query implies/requests a graph, chart, plot, or visualization, false otherwise.
        7. "table_needed": true if query implies/requests a table or tabular data, false otherwise.

        Return ONLY valid JSON, no explanations.
        
        Query: "{query}"
    """
    
    response = model.generate_content(prompt)
    cleaned = response.text.strip().strip('```json').strip('```').strip()
    
    try:
        result = json.loads(cleaned)
        result["classification"] = "financial_known"  # Force since we have matches
        return result
    except Exception as e:
        # Return default structure on parsing error
        return {
            "classification": "financial_known",
            "company": potential_companies_display if potential_companies_display else None,
            "sector": potential_sectors_display if potential_sectors_display else None,
            "fundamental": None,
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
    state["fundamental"] = result["fundamental"]
    state["time_period"] = result["time_period"]
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