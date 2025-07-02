# DEPRECATED: This module is no longer used in the main workflow
# The parsing functionality has been merged with classification in query_classifier.py
# to reduce API costs from 2 LLM calls to 1 per query.
# This file is kept for reference or potential future standalone use.

import json
from config.gemini_config import model

companies = ['Infosys', 'Wipro', 'TCS', 'HDFC Bank', 'ICICI Bank', 'Axis Bank', 'Reliance Industries', 'NTPC', 'Sun Pharma', 'Dr. Reddy\'s']
sectors = ['IT', 'Banking', 'Energy', 'Pharma']
fundamentals = ['sales', 'net profit', 'roe', 'debt equity', 'dividend yield', 'avg price']

def parse_query(query: str) -> dict:
    prompt = f"""
        You are a financial query parser. Your job is to extract structured information from user queries related to companies in our dataset.

        Return the following fields as a JSON object:
        1. "company": List of company names (e.g., ["Infosys", "Dr. Reddy's", "NTPC"]). Use null if not mentioned. if one of these {companies}, then write like this only as given in the list.
        2. "sector": List of sectors (e.g., ["IT", "Banking", "Energy", "Pharma"]). Use null if not mentioned. If one of these {sectors}, then write like this only i.e. write Banking and not Banks or something else.
        3. "fundamental": The financial metric requested (e.g., "sales", "net profit", "roe", "stock price", "eps", "debt equity" etc.). Use null if not mentioned. If one of these {fundamentals}, then write like this only, 'avg price' is stock price do if the query asks for stock price return avg price in this.
        4. "time_period": Time period string (e.g., "2019", "2020-2024", "2015-2024"). Put "2015-2024" if not mentioned. The current year is 2024. If the query asks for "last 5 years", use "2020-2024", and so on. If the query has only one year then do like this '2021-2021' repeat the year.
        5. "graph_needed": true if the query implies or requests a graph; false otherwise.
        6. "table_needed": true if the query implies or requests a table; false otherwise.

        Strictly return a **valid JSON object** and nothing else. Do not explain your answer. Use double quotes for keys and string values.
        Query: "{query}"
        """

    response = model.generate_content(prompt)
    cleaned = response.text.strip().strip('```json').strip('```').strip()

    try:
        parsed = json.loads(cleaned)
        return parsed
    except Exception as e:
        return {}

def parse_node(state: dict) -> dict:
    query = state["query"]
    parsed = parse_query(query)
    state["company"] = parsed.get("company")
    state["sector"] = parsed.get("sector")
    state["fundamental"] = parsed.get("fundamental")
    state["time_period"] = parsed.get("time_period")
    state["graph_needed"] = parsed.get("graph_needed")
    state["table_needed"] = parsed.get("table_needed")
    return state