import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import os

def create_visualization(state: dict) -> dict:
    """
    Create visualizations based on the retrieved data and query context.
    
    Args:
        state: Dictionary containing query results and metadata
    
    Returns:
        Updated state with visualization info
    """
    
    # Check if graph is needed
    if not state.get("graph_needed", False):
        state["visualization"] = "No graph requested"
        return state
    
    # Get retrieved data
    data_result = state.get("retrieved_data", {})
    
    # Handle case where data_result might be a DataFrame or dictionary
    if isinstance(data_result, pd.DataFrame):
        # If it's directly a DataFrame, wrap it in the expected structure
        if data_result.empty:
            state["visualization"] = "Cannot create graph: No data available"
            return state
        data_result = {"data": data_result, "metadata": {}, "error": None}
    elif not data_result or (isinstance(data_result, dict) and data_result.get("error")):
        error_msg = data_result.get('error', 'No data available') if isinstance(data_result, dict) else 'No data available'
        state["visualization"] = f"Cannot create graph: {error_msg}"
        return state
    
    df = data_result.get("data")
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        state["visualization"] = "Cannot create graph: No data to visualize"
        return state
    
    metadata = data_result.get("metadata", {})
    
    # Determine visualization type and create graph
    try:
        graph_path = generate_graph(df, metadata, state)
        state["visualization"] = f"Graph saved to: {graph_path}"
        state["graph_path"] = graph_path
        
    except Exception as e:
        state["visualization"] = f"Error creating graph: {str(e)}"
    
    return state

def generate_graph(df: pd.DataFrame, metadata: dict, state: dict) -> str:
    """
    Generate appropriate graph based on data structure and query context.
    
    Args:
        df: DataFrame with financial data
        metadata: Metadata about the query and data
        state: Current state with query context
    
    Returns:
        Path to saved graph file
    """
    
    # Set up the plot style
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Try to construct a 'year' column from 'period' when missing
    if 'year' not in df.columns and 'period' in df.columns:
        df = df.copy()
        # Attempt numeric year first (annual data like '2021')
        year_numeric = pd.to_numeric(df['period'], errors='coerce')
        if year_numeric.notna().any():
            df['year'] = year_numeric.astype('Int64')
        else:
            # Fallback: extract the YYYY from formats like 'YYYY-Qn'
            extracted = df['period'].astype(str).str.extract(r'^(\d{4})')[0]
            df['year'] = pd.to_numeric(extracted, errors='coerce').astype('Int64')

    # Determine graph type based on data structure
    if 'year' in df.columns and 'value' in df.columns and df['year'].nunique() > 1:
        # Multi-year time series
        graph_path = create_time_series_plot(df, metadata, state, ax)
    elif 'company' in df.columns and len(df['company'].unique()) > 1 and df['period'].nunique() > 1:
        # Multiple companies across multiple periods -> grouped bar by year
        graph_path = create_comparison_plot(df, metadata, state, ax)
    elif 'company' in df.columns and len(df['company'].unique()) > 1 and df['period'].nunique() == 1:
        # Multiple companies comparison
        graph_path = create_comparison_plot(df, metadata, state, ax)
    else:
        # Single entity or sector overview
        graph_path = create_overview_plot(df, metadata, state, ax)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return graph_path

def create_time_series_plot(df: pd.DataFrame, metadata: dict, state: dict, ax) -> str:
    """Create time series plot for trends over time"""
    
    fundamental = metadata.get("fundamental", "Financial Metric")
    companies = df['company'].unique()
    indices = state.get("indices", [])
    
    # Create line plot for each company
    for company in companies:
        company_data = df[df['company'] == company]
        if isinstance(company_data, pd.DataFrame):
            company_data = company_data.sort_values('year')
            ax.plot(company_data['year'], company_data['value'], 
                    marker='o', linewidth=2, label=company.title(), markersize=6)
    
    # Customize the plot title based on context
    title_parts = [f'{fundamental.replace("_", " ").title()} Trends Over Time']
    if indices:
        title_parts.append(f'({", ".join(indices)} companies)')
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{fundamental.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    ax.set_title(' '.join(title_parts), fontsize=14, fontweight='bold', pad=20)
    
    # Format x-axis to show years properly
    years = sorted(df['year'].unique())
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Format y-axis based on data magnitude
    format_y_axis(ax, df['value'])
    
    # Generate filename
    companies_str = "_".join([c.replace(" ", "_").lower() for c in companies[:3]])
    indices_str = "_".join([i.replace(" ", "_").lower() for i in indices]) if indices else ""
    filename_parts = [fundamental, companies_str]
    if indices_str:
        filename_parts.append(indices_str)
    filename_parts.append("trend")
    filename = f"graphs/{'_'.join(filename_parts)}.png"
    
    return filename

def create_comparison_plot(df: pd.DataFrame, metadata: dict, state: dict, ax) -> str:
    """Create comparison plot for multiple companies"""
    
    fundamental = metadata.get("fundamental", "Financial Metric")
    indices = state.get("indices", [])
    
    if 'year' in df.columns and df['year'].nunique() > 1:
        # Multi-year comparison - use grouped bar chart
        pivot_df = df.pivot_table(index='year', columns='company', values='value', fill_value=0)
        
        # Create grouped bar chart
        pivot_df.plot(kind='bar', ax=ax, width=0.8)
        
        # Customize title based on context
        title_parts = [f'{fundamental.replace("_", " ").title()} Comparison Across Companies']
        if indices:
            title_parts.append(f'({", ".join(indices)} Index)')
        
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{fundamental.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_title(' '.join(title_parts), fontsize=14, fontweight='bold', pad=20)
        
        # Rotate x-axis labels
        ax.set_xticklabels(pivot_df.index, rotation=45)
        
    else:
        # Single value comparison - use simple bar chart
        companies = df['company'].unique()
        values = []
        for company in companies:
            company_data = df[df['company'] == company]
            if isinstance(company_data, pd.DataFrame) and not company_data.empty:
                # For single period, aggregate deterministically (e.g., mean) in case of duplicates
                values.append(float(company_data['value'].mean()))
            else:
                values.append(0)
        
        # Create colors for bars
        colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(companies)))
        bars = ax.bar(companies, values, color=colors)
        
        # Customize title based on context
        title_parts = [f'{fundamental.replace("_", " ").title()} Comparison']
        if indices:
            title_parts.append(f'({", ".join(indices)} Index)')
        
        ax.set_xlabel('Company', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{fundamental.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_title(' '.join(title_parts), fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{format_number(value)}',
                   ha='center', va='bottom', fontweight='bold')
        
        # Rotate x-axis labels if needed
        ax.set_xticklabels(companies, rotation=45, ha='right')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Format y-axis
    format_y_axis(ax, df['value'])
    
    # Generate filename
    companies = df['company'].unique()
    companies_str = "_".join([c.replace(" ", "_").lower() for c in companies[:3]])
    indices_str = "_".join([i.replace(" ", "_").lower() for i in indices]) if indices else ""
    filename_parts = [fundamental, companies_str]
    if indices_str:
        filename_parts.append(indices_str)
    filename_parts.append("comparison")
    filename = f"graphs/{'_'.join(filename_parts)}.png"
    
    return filename

def create_overview_plot(df: pd.DataFrame, metadata: dict, state: dict, ax) -> str:
    """Create overview plot for single entity or sector"""
    
    fundamental = metadata.get("fundamental", "Financial Metric")
    indices = state.get("indices", [])
    
    if 'year' in df.columns and len(df['year'].unique()) > 1:
        # Time series for single entity
        company_name = df['company'].iloc[0] if 'company' in df.columns else "Entity"
        df_sorted = df.sort_values('year')
        
        ax.plot(df_sorted['year'], df_sorted['value'], 
               marker='o', linewidth=3, markersize=8, color='#2E86AB')
        ax.fill_between(df_sorted['year'], df_sorted['value'], alpha=0.3, color='#2E86AB')
        
        # Customize title based on context
        title_parts = [f'{company_name.title()} - {fundamental.replace("_", " ").title()} Trend']
        if indices:
            title_parts.append(f'({", ".join(indices)} Index)')
        
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{fundamental.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_title(' '.join(title_parts), fontsize=14, fontweight='bold', pad=20)
        
        # Format x-axis
        years = sorted(df['year'].unique())
        ax.set_xticks(years)
        ax.set_xticklabels(years, rotation=45)
        
    else:
        # Single value display
        value = df['value'].iloc[0]
        company_name = df['company'].iloc[0] if 'company' in df.columns else "Entity"
        
        # Create a simple bar chart with single value
        bar = ax.bar([company_name], [value], color='#2E86AB', width=0.6)
        
        # Customize title based on context
        title_parts = [f'{company_name.title()} - {fundamental.replace("_", " ").title()}']
        if indices:
            title_parts.append(f'({", ".join(indices)} Index)')
        
        ax.set_ylabel(f'{fundamental.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_title(' '.join(title_parts), fontsize=14, fontweight='bold', pad=20)
        
        # Add value label on bar
        ax.text(bar[0].get_x() + bar[0].get_width()/2., bar[0].get_height(),
               f'{format_number(value)}',
               ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Format y-axis
    format_y_axis(ax, df['value'])
    
    # Generate filename
    entity_name = df['company'].iloc[0].replace(" ", "_").lower() if 'company' in df.columns else "overview"
    indices_str = "_".join([i.replace(" ", "_").lower() for i in indices]) if indices else ""
    filename_parts = [fundamental, entity_name]
    if indices_str:
        filename_parts.append(indices_str)
    filename_parts.append("overview")
    filename = f"graphs/{'_'.join(filename_parts)}.png"
    
    return filename

def format_y_axis(ax, values):
    """Format y-axis based on data magnitude"""
    max_val = values.max()
    
    if max_val >= 1e9:
        # Billions
        ax.ticklabel_format(style='plain', axis='y')
        ticks = ax.get_yticks()
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{int(val/1e9)}B' if val != 0 else '0' for val in ticks])
    elif max_val >= 1e6:
        # Millions
        ax.ticklabel_format(style='plain', axis='y')
        ticks = ax.get_yticks()
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{int(val/1e6)}M' if val != 0 else '0' for val in ticks])
    elif max_val >= 1e3:
        # Thousands
        ax.ticklabel_format(style='plain', axis='y')
        ticks = ax.get_yticks()
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{int(val/1e3)}K' if val != 0 else '0' for val in ticks])
    else:
        # Regular numbers
        ax.ticklabel_format(style='plain', axis='y')

def format_number(value):
    """Format numbers for display"""
    if value >= 1e9:
        return f'{value/1e9:.1f}B'
    elif value >= 1e6:
        return f'{value/1e6:.1f}M'
    elif value >= 1e3:
        return f'{value/1e3:.1f}K'
    else:
        return f'{value:.1f}'

def visualizer_node(state: dict) -> dict:
    """
    LangGraph node function for visualization.
    
    Args:
        state: State dictionary containing query results
    
    Returns:
        Updated state with visualization information
    """
    
    # Ensure graphs directory exists
    os.makedirs("graphs", exist_ok=True)
    
    # Create visualization
    return create_visualization(state) 