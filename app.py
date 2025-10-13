"""
Private Label Production Summary
================================

A Streamlit application for automating weekly private label inventory reporting.
Processes Headset inventory coverage and Distru inventory assets reports to calculate
Weeks on Hand (WOH) and generate production insights for private label flower products.

Author: DC Retail
Version: 1.8 - Production Ready
Date: 2025

Key Features:
- Combines retail (Headset) and distribution (Distru) inventory data
- Calculates Weeks on Hand (WOH) with outlier control
- Tracks products in stock at distribution for production planning
- Case-insensitive product matching across systems
- Interactive dashboards with drill-down capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import math
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# CONSTANTS
# =============================================================================

# Private Label Brands - hardcoded list for consistency
PRIVATE_LABEL_BRANDS = [
    'Black Label',
    'Black Label Platinum',
    'Block Party',
    'Block Party Exotics',
    'Dope St.',
    'Dope St. Exotics',
    'Dunzo',
    'Fat Stash',
    'High Five',
    "Lil' Buzzies",
    'MikroDose',
    'Nuggies',
    'PTO',
    'Roll & Ready',
    'Side Hustle'
]

# Flower category keywords for filtering
FLOWER_KEYWORDS = ['indica', 'sativa', 'hybrid', 'flower']

# Standard category order for consistent display
CATEGORY_ORDER = ['Indica', 'Hybrid', 'Sativa', 'Unknown']

# WOH calculation parameters
MAX_WOH_WEEKS = 52.0  # Cap at 1 year for outlier control
MIN_DAILY_SALES = 0.1  # Minimum threshold for reliable data

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Private Label Production Summary",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Private Label Production Summary")
st.markdown("**DC Retail** | Weekly inventory analysis and production planning for private label products")

# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================

def initialize_session_state():
    """Initialize session state variables for data persistence"""
    session_vars = ['headset_data', 'distru_data', 'combined_data', 'processed_data', 'app_version']
    
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = None
    
    if st.session_state.app_version is None:
        st.session_state.app_version = "1.8"

initialize_session_state()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_numeric(value, default=0):
    """
    Convert any value to numeric, handling strings, NaN, None, etc.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Numeric value (int or float) or default
    """
    if pd.isna(value) or value is None or value == '':
        return default
    try:
        if isinstance(value, str):
            value = value.strip()
            if value == '':
                return default
        num_val = float(value)
        return int(num_val) if num_val.is_integer() else num_val
    except (ValueError, TypeError, AttributeError):
        return default

def extract_weight_from_product_name(product_name: str) -> Tuple[str, str]:
    """
    Extract weight from product name by removing the last token.
    
    Args:
        product_name: Full product name (e.g., "Dope St. - Gmo Cookie 14g")
        
    Returns:
        Tuple of (product_name_without_weight, weight)
        Example: ("Dope St. - Gmo Cookie", "14g")
    """
    if pd.isna(product_name) or not isinstance(product_name, str):
        return str(product_name), ""
    
    parts = product_name.strip().split()
    if len(parts) >= 2:
        last_part = parts[-1]
        if 'g' in last_part.lower():  # Check if last part looks like a weight
            return ' '.join(parts[:-1]), last_part
    
    return product_name, ""

def extract_brand_from_product_name(product_name: str) -> str:
    """
    Extract brand from product name (text before the first hyphen).
    
    Args:
        product_name: Full product name
        
    Returns:
        Brand name
        
    Examples:
        "Black Label - Cherry Warheads 3.5g" → "Black Label"
        "Block Party - Lemon Cherry Gelato 28g" → "Block Party"
    """
    if pd.isna(product_name) or not isinstance(product_name, str):
        return ""
    
    if ' - ' in product_name:
        return product_name.split(' - ')[0].strip()
    elif '-' in product_name:
        return product_name.split('-')[0].strip()
    else:
        # Fallback - take first two words as brand
        parts = product_name.split()
        if len(parts) >= 2:
            return ' '.join(parts[:2])
        return product_name

def categorize_flower_type(category: str) -> str:
    """
    Standardize flower category names.
    
    Handles patterns like:
    - "Flower (Indica)" → "Indica"
    - "Flower (Sativa)" → "Sativa"
    - "Indica" → "Indica"
    
    Args:
        category: Raw category string
        
    Returns:
        Standardized category: Indica, Sativa, Hybrid, or Unknown
    """
    if pd.isna(category):
        return "Unknown"
    
    category = str(category).lower().strip()
    
    # Handle "Flower (Type)" patterns
    if 'indica' in category:
        return 'Indica'
    elif 'sativa' in category:
        return 'Sativa'
    elif 'hybrid' in category:
        return 'Hybrid'
    elif category == 'flower':
        return 'Unknown'
    
    return category.title()

def sort_by_category_order(df: pd.DataFrame, category_column: str = 'Flower Category') -> pd.DataFrame:
    """
    Sort dataframe by standard category order: Indica, Hybrid, Sativa, Unknown.
    
    Args:
        df: DataFrame to sort
        category_column: Name of the category column
        
    Returns:
        Sorted DataFrame with categorical ordering applied
    """
    if category_column in df.columns:
        df[category_column] = pd.Categorical(
            df[category_column], 
            categories=CATEGORY_ORDER, 
            ordered=True
        )
        df = df.sort_values(category_column)
    
    return df

def calculate_woh(total_inventory: float, daily_sales: float, max_woh: float = MAX_WOH_WEEKS) -> float:
    """
    Calculate Weeks on Hand (WOH) with outlier control.
    
    Formula: WOH = (Total Inventory / Daily Sales) / 7
    - Caps at max_woh to prevent outliers from skewing aggregations
    - Returns max_woh for products with insufficient sales data
    - Always rounds down for conservative estimates
    
    Args:
        total_inventory: Total units in inventory
        daily_sales: Average daily sales rate
        max_woh: Maximum weeks on hand to cap at (default 52 weeks = 1 year)
    
    Returns:
        Weeks on hand, capped at max_woh, rounded down to 1 decimal
    """
    if daily_sales < MIN_DAILY_SALES:
        # Insufficient data - return capped value
        return max_woh
    
    days_supply = total_inventory / daily_sales
    woh = days_supply / 7
    
    # Cap at maximum to prevent outliers
    woh = min(woh, max_woh)
    
    return math.floor(woh * 10) / 10  # Round down to 1 decimal

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_headset_data(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load and validate Headset inventory coverage report.
    
    Args:
        uploaded_file: Uploaded CSV file
        
    Returns:
        DataFrame with validated columns or None if loading fails
    """
    try:
        df = pd.read_csv(uploaded_file, dtype=str)
        
        # Validate required columns
        required_columns = [
            'Store Name', 'Product Name', 'Brand', 'Category', 
            'Total Quantity on Hand', 'In Stock Avg Units per Day'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"❌ Missing required columns in Headset data: {missing_columns}")
            return None
        
        # Convert numeric columns
        df['Total Quantity on Hand'] = df['Total Quantity on Hand'].apply(lambda x: safe_numeric(x, 0))
        df['In Stock Avg Units per Day'] = df['In Stock Avg Units per Day'].apply(lambda x: safe_numeric(x, 0))
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading Headset data: {str(e)}")
        return None

def load_distru_data(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load and validate Distru inventory assets report.
    
    Note: Skips first 2 rows of metadata, uses row 3 as headers.
    
    Args:
        uploaded_file: Uploaded CSV file
        
    Returns:
        DataFrame with validated columns or None if loading fails
    """
    try:
        # Skip first 2 rows of metadata, use row 3 as headers
        df = pd.read_csv(uploaded_file, skiprows=2, dtype=str)
        
        if df.empty:
            st.error("❌ Distru file appears to be empty after skipping metadata rows")
            return None
        
        # Convert numeric columns
        numeric_columns = ['Active Quantity', 'Quantity', 'Total Quantity', 'Available Quantity']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: safe_numeric(x, 0))
        
        # Extract brand from product name if Brand column doesn't exist
        product_name_col = 'Product Name' if 'Product Name' in df.columns else 'Product'
        if 'Brand' not in df.columns and product_name_col in df.columns:
            df['Brand'] = df[product_name_col].apply(extract_brand_from_product_name)
        
        # Standardize column name
        if 'Product' in df.columns and 'Product Name' not in df.columns:
            df['Product Name'] = df['Product']
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading Distru data: {str(e)}")
        return None

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def combine_inventory_data(headset_df: pd.DataFrame, distru_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Combine Headset and Distru data to calculate total inventory and WOH.
    
    Process:
    1. Filter both datasets to private label brands and flower categories
    2. Normalize product names for case-insensitive matching
    3. Aggregate Headset data across all stores
    4. Aggregate Distru data
    5. Add Distru-only products (not in stores)
    6. Merge datasets and calculate metrics
    
    Args:
        headset_df: Headset inventory coverage report
        distru_df: Distru inventory assets report
        
    Returns:
        Combined DataFrame with calculated metrics or None if processing fails
    """
    try:
        # =====================================================================
        # PROCESS HEADSET DATA
        # =====================================================================
        
        # Filter to private label brands
        headset_filtered = headset_df[headset_df['Brand'].isin(PRIVATE_LABEL_BRANDS)].copy()
        
        if headset_filtered.empty:
            st.warning("⚠️ No private label products found in Headset data")
            return None
        
        # Filter to flower categories
        headset_filtered = headset_filtered[
            headset_filtered['Category'].str.lower().str.contains(
                '|'.join(FLOWER_KEYWORDS), na=False
            )
        ].copy()
        
        if headset_filtered.empty:
            st.warning("⚠️ No private label flower products found in Headset data")
            return None
        
        # Extract product weights and standardize categories
        headset_filtered[['Product Base Name', 'Weight']] = headset_filtered['Product Name'].apply(
            lambda x: pd.Series(extract_weight_from_product_name(x))
        )
        headset_filtered['Flower Category'] = headset_filtered['Category'].apply(categorize_flower_type)
        
        # Normalize product names for case-insensitive matching
        headset_filtered['Product Name Normalized'] = headset_filtered['Product Name'].str.lower().str.strip()
        
        # Count stores with inventory for each product
        def count_stores_with_inventory(group):
            return group[group['Total Quantity on Hand'] > 0]['Store Name'].nunique()
        
        # Aggregate across all stores
        headset_summary = headset_filtered.groupby([
            'Product Name', 'Brand', 'Flower Category', 'Product Base Name', 'Weight', 'Product Name Normalized'
        ]).agg({
            'Total Quantity on Hand': 'sum',
            'In Stock Avg Units per Day': 'sum'
        }).reset_index()
        
        # Calculate store counts separately
        store_counts = headset_filtered.groupby('Product Name Normalized').apply(
            count_stores_with_inventory
        ).reset_index()
        store_counts.columns = ['Product Name Normalized', 'Store Count']
        
        headset_summary = headset_summary.merge(store_counts, on='Product Name Normalized', how='left')
        headset_summary['Store Count'] = headset_summary['Store Count'].fillna(0)
        
        # Filter out products with zero inventory
        headset_summary = headset_summary[headset_summary['Total Quantity on Hand'] > 0].copy()
        
        if headset_summary.empty:
            st.warning("⚠️ No private label flower products with inventory found")
            return None
        
        # =====================================================================
        # PROCESS DISTRU DATA
        # =====================================================================
        
        distru_filtered = distru_df.copy()
        
        # Standardize product name column
        if 'Product Name' not in distru_filtered.columns and 'Product' in distru_filtered.columns:
            distru_filtered['Product Name'] = distru_filtered['Product']
        
        # Extract brands if needed
        if 'Brand' not in distru_filtered.columns:
            distru_filtered['Brand'] = distru_filtered['Product Name'].apply(extract_brand_from_product_name)
        
        # Filter to private label brands
        distru_filtered = distru_filtered[distru_filtered['Brand'].isin(PRIVATE_LABEL_BRANDS)].copy()
        
        # Filter to flower categories
        if 'Category' in distru_filtered.columns:
            distru_filtered['Flower Category'] = distru_filtered['Category'].apply(categorize_flower_type)
            distru_filtered = distru_filtered[
                distru_filtered['Category'].str.lower().str.contains(
                    '|'.join(FLOWER_KEYWORDS), na=False
                )
            ].copy()
        else:
            distru_filtered['Flower Category'] = 'Unknown'
        
        # Extract weights and normalize names
        if 'Product Name' in distru_filtered.columns:
            distru_filtered['Product Name Normalized'] = distru_filtered['Product Name'].str.lower().str.strip()
            distru_filtered[['Product Base Name', 'Weight']] = distru_filtered['Product Name'].apply(
                lambda x: pd.Series(extract_weight_from_product_name(x))
            )
        
        # Find quantity column
        quantity_columns = ['Active Quantity', 'Quantity', 'Total Quantity', 'Available Quantity']
        distru_qty_col = None
        for col in quantity_columns:
            if col in distru_filtered.columns:
                distru_qty_col = col
                break
        
        # Aggregate Distru data
        if distru_qty_col is None or distru_filtered.empty:
            distru_summary = pd.DataFrame()
        else:
            distru_summary = distru_filtered.groupby([
                'Product Name', 'Product Name Normalized', 'Brand', 'Product Base Name', 'Weight', 'Flower Category'
            ]).agg({
                distru_qty_col: 'sum'
            }).reset_index()
            distru_summary.rename(columns={distru_qty_col: 'Distru Quantity'}, inplace=True)
        
        # =====================================================================
        # MERGE HEADSET AND DISTRU DATA
        # =====================================================================
        
        if not distru_summary.empty:
            # Find products in Distru but not in stores
            distru_only_products = distru_summary[
                ~distru_summary['Product Name Normalized'].isin(headset_summary['Product Name Normalized'])
            ]
            
            # Add Distru-only products to dataset
            if not distru_only_products.empty:
                for _, row in distru_only_products.iterrows():
                    new_row = {
                        'Product Name': row['Product Name'],
                        'Product Name Normalized': row['Product Name Normalized'],
                        'Brand': row['Brand'],
                        'Flower Category': row.get('Flower Category', 'Unknown'),
                        'Product Base Name': row['Product Base Name'],
                        'Weight': row['Weight'],
                        'Total Quantity on Hand': 0,
                        'In Stock Avg Units per Day': MIN_DAILY_SALES,
                        'Store Count': 0
                    }
                    headset_summary = pd.concat([headset_summary, pd.DataFrame([new_row])], ignore_index=True)
            
            # Merge on normalized names (case-insensitive)
            combined = headset_summary.merge(
                distru_summary[['Product Name Normalized', 'Distru Quantity']], 
                on='Product Name Normalized', 
                how='left'
            )
            combined['Distru Quantity'] = combined['Distru Quantity'].fillna(0)
        else:
            combined = headset_summary.copy()
            combined['Distru Quantity'] = 0
        
        # =====================================================================
        # CALCULATE METRICS
        # =====================================================================
        
        # Total inventory
        combined['Total Inventory'] = combined['Total Quantity on Hand'] + combined['Distru Quantity']
        
        # Filter out products with zero total inventory
        combined = combined[combined['Total Inventory'] > 0].copy()
        
        if combined.empty:
            st.warning("⚠️ No products with inventory found after combining data")
            return None
        
        # Calculate WOH
        combined['WOH'] = combined.apply(
            lambda row: calculate_woh(row['Total Inventory'], row['In Stock Avg Units per Day']), 
            axis=1
        )
        
        # Calculate Distru Days Supply
        combined['Distru Days Supply'] = combined.apply(
            lambda row: math.floor(row['Distru Quantity'] / row['In Stock Avg Units per Day']) 
            if row['In Stock Avg Units per Day'] >= MIN_DAILY_SALES else 0,
            axis=1
        )
        
        # Create Product Group for categorization
        combined['Product Group'] = combined['Brand'] + ' ' + combined['Weight']
        
        # Drop normalized column - no longer needed
        combined = combined.drop(columns=['Product Name Normalized'])
        
        return combined
        
    except Exception as e:
        st.error(f"❌ Error combining inventory data: {str(e)}")
        return None

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_woh_summary_chart(df: pd.DataFrame) -> go.Figure:
    """Create WOH summary chart by brand and category"""
    try:
        summary = df.groupby(['Brand', 'Flower Category']).agg({
            'Total Inventory': 'sum',
            'In Stock Avg Units per Day': 'sum',
            'Product Name': 'nunique'
        }).reset_index()
        
        # Calculate WOH from aggregated totals (don't sum individual WOHs)
        summary['WOH'] = summary.apply(
            lambda row: calculate_woh(row['Total Inventory'], row['In Stock Avg Units per Day']),
            axis=1
        )
        
        fig = px.bar(
            summary, 
            x='Brand', 
            y='WOH',
            color='Flower Category',
            title='Weeks on Hand (WOH) by Brand and Category',
            labels={'WOH': 'Weeks on Hand', 'Brand': 'Private Label Brand'},
            height=500
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating WOH chart: {str(e)}")
        return go.Figure()

def create_daily_sales_chart(df: pd.DataFrame) -> go.Figure:
    """Create daily sales performance chart by brand/weight combination"""
    try:
        sales_summary = df.groupby('Product Group').agg({
            'In Stock Avg Units per Day': 'sum'
        }).reset_index()
        
        sales_summary = sales_summary.sort_values('In Stock Avg Units per Day', ascending=False)
        
        fig = px.bar(
            sales_summary,
            x='Product Group',
            y='In Stock Avg Units per Day',
            title='Daily Sales Performance by Brand/Weight',
            labels={'In Stock Avg Units per Day': 'Daily Sales (units)', 'Product Group': 'Product Type'},
            height=500,
            color='In Stock Avg Units per Day',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating daily sales chart: {str(e)}")
        return go.Figure()

def create_inventory_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create inventory distribution sunburst chart"""
    try:
        weight_summary = df.groupby(['Weight', 'Brand']).agg({
            'Total Inventory': 'sum',
            'Distru Quantity': 'sum'
        }).reset_index()
        
        fig = px.sunburst(
            weight_summary,
            path=['Weight', 'Brand'],
            values='Total Inventory',
            title='Inventory Distribution by Weight and Brand',
            height=500
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating inventory distribution chart: {str(e)}")
        return go.Figure()

def create_distru_stock_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary table for products in stock at Distru"""
    try:
        distru_stock = df[df['Distru Quantity'] > 0].copy()
        
        if distru_stock.empty:
            return pd.DataFrame()
        
        summary = distru_stock.groupby(['Brand', 'Weight', 'Flower Category']).agg({
            'Distru Quantity': 'sum',
            'Total Inventory': 'sum',
            'In Stock Avg Units per Day': 'sum',
            'Product Name': 'nunique'
        }).reset_index()
        
        # Calculate metrics from aggregated values
        summary['Distru Days Supply'] = summary.apply(
            lambda row: math.floor(row['Distru Quantity'] / row['In Stock Avg Units per Day']) 
            if row['In Stock Avg Units per Day'] >= MIN_DAILY_SALES else 0,
            axis=1
        )
        
        summary['WOH'] = summary.apply(
            lambda row: calculate_woh(row['Total Inventory'], row['In Stock Avg Units per Day']),
            axis=1
        )
        
        summary.rename(columns={'Product Name': 'Products at Distro'}, inplace=True)
        summary = summary[[
            'Brand', 'Weight', 'Flower Category', 'Products at Distro', 
            'Distru Quantity', 'Distru Days Supply', 'Total Inventory', 'In Stock Avg Units per Day', 'WOH'
        ]]
        
        summary = summary.sort_values(['Brand', 'Weight'])
        summary = sort_by_category_order(summary, 'Flower Category')
        
        return summary
        
    except Exception as e:
        st.error(f"Error creating Distru stock table: {str(e)}")
        return pd.DataFrame()

def create_expandable_product_summary(df: pd.DataFrame):
    """Create expandable product summary with drill-down by category"""
    
    df_with_inventory = df[df['Total Inventory'] > 0].copy()
    
    if df_with_inventory.empty:
        st.warning("⚠️ No products with inventory to display")
        return
    
    # Group by Product Group (Brand + Weight)
    product_groups = df_with_inventory.groupby('Product Group').agg({
        'Total Inventory': 'sum',
        'Distru Quantity': 'sum',
        'In Stock Avg Units per Day': 'sum',
        'Product Name': 'nunique',
        'Store Count': 'sum'
    }).reset_index()
    
    # Calculate WOH from aggregated values
    product_groups['WOH'] = product_groups.apply(
        lambda row: calculate_woh(row['Total Inventory'], row['In Stock Avg Units per Day']),
        axis=1
    )
    
    # Count products with Distru inventory
    distru_products = df_with_inventory[df_with_inventory['Distru Quantity'] > 0].groupby('Product Group').size().reset_index(name='Distro Products')
    product_groups = product_groups.merge(distru_products, on='Product Group', how='left')
    product_groups['Distro Products'] = product_groups['Distro Products'].fillna(0).astype(int)
    
    product_groups.rename(columns={'Product Name': 'Total Products'}, inplace=True)
    product_groups = product_groups.sort_values('Product Group')
    
    st.subheader("📈 Product Performance Summary")
    
    for _, group_row in product_groups.iterrows():
        product_group = group_row['Product Group']
        total_products = int(group_row['Total Products'])
        distro_products = int(group_row['Distro Products'])
        woh = group_row['WOH']
        daily_sales = group_row['In Stock Avg Units per Day']
        
        with st.expander(f"**{product_group}** - {total_products} products, {distro_products} at Distro, {woh:.1f} WOH, {daily_sales:.1f} daily sales"):
            
            group_products = df_with_inventory[df_with_inventory['Product Group'] == product_group]
            
            # Category summary
            category_agg = group_products.groupby('Flower Category').agg({
                'Total Inventory': 'sum',
                'Distru Quantity': 'sum',
                'In Stock Avg Units per Day': 'sum',
                'Product Name': 'nunique'
            }).reset_index()
            
            category_agg['WOH'] = category_agg.apply(
                lambda row: calculate_woh(row['Total Inventory'], row['In Stock Avg Units per Day']),
                axis=1
            )
            
            # Count products in stock at Distro by category
            distro_by_category = group_products[group_products['Distru Quantity'] > 0].groupby('Flower Category').size().reset_index(name='Distro Products')
            category_summary = category_agg.merge(distro_by_category, on='Flower Category', how='left')
            category_summary['Distro Products'] = category_summary['Distro Products'].fillna(0).astype(int)
            
            category_summary.rename(columns={
                'Product Name': 'Total Products',
                'In Stock Avg Units per Day': 'Daily Sales'
            }, inplace=True)
            
            category_summary = category_summary[[
                'Flower Category', 'Total Products', 'Distro Products', 
                'Total Inventory', 'Distru Quantity', 'Daily Sales', 'WOH'
            ]]
            
            category_summary = sort_by_category_order(category_summary, 'Flower Category')
            category_summary = category_summary.round(1)
            
            st.markdown("**📊 By Category:**")
            st.dataframe(category_summary, use_container_width=True, hide_index=True)
            
            # Individual products by category
            st.markdown("**🌿 Individual Products:**")
            
            for category in [cat for cat in CATEGORY_ORDER if cat in group_products['Flower Category'].unique()]:
                category_products = group_products[
                    (group_products['Flower Category'] == category) & 
                    (group_products['Total Inventory'] > 0)
                ]
                
                if not category_products.empty:
                    distro_count = len(category_products[category_products['Distru Quantity'] > 0])
                    total_count = len(category_products)
                    
                    with st.expander(f"    {category} ({total_count} products, {distro_count} at Distro)"):
                        display_df = category_products[[
                            'Product Name', 'Total Inventory', 'Distru Quantity', 
                            'In Stock Avg Units per Day', 'WOH', 'Store Count'
                        ]].copy()
                        
                        display_df = display_df.rename(columns={'In Stock Avg Units per Day': 'Daily Sales'})
                        
                        for col in ['Total Inventory', 'Distru Quantity', 'Daily Sales', 'WOH']:
                            if col in display_df.columns:
                                display_df[col] = display_df[col].round(1)
                        
                        st.dataframe(display_df, use_container_width=True, hide_index=True)

# =============================================================================
# STREAMLIT UI
# =============================================================================

# Sidebar - File Uploads
st.sidebar.header("📊 Data Sources")

st.sidebar.subheader("📋 Headset Inventory Coverage")
headset_file = st.sidebar.file_uploader(
    "Choose Headset CSV", type=['csv'], key="headset_upload",
    help="Upload the inventory coverage report from Headset"
)

st.sidebar.subheader("📦 Distru Inventory Assets")
distru_file = st.sidebar.file_uploader(
    "Choose Distru CSV", type=['csv'], key="distru_upload",
    help="Upload the inventory assets report from Distru"
)

# Display private label brands
st.sidebar.markdown("---")
st.sidebar.subheader("🏷️ Private Label Brands")
st.sidebar.caption(f"Tracking {len(PRIVATE_LABEL_BRANDS)} brands:")
with st.sidebar.expander("View Brands"):
    for brand in PRIVATE_LABEL_BRANDS:
        st.markdown(f"• {brand}")

# Process Data Button
if st.sidebar.button("🚀 Process Data", type="primary", disabled=not (headset_file and distru_file)):
    with st.spinner("Processing your data..."):
        # Load CSV files
        headset_df = load_headset_data(headset_file)
        distru_df = load_distru_data(distru_file)
        
        if headset_df is None or distru_df is None:
            st.error("❌ Failed to load one or more files")
            st.stop()
        
        # Store raw data in session state
        st.session_state.headset_data = headset_df
        st.session_state.distru_data = distru_df
        
        # Display file info
        st.success(f"✅ Files loaded: Headset ({len(headset_df):,} rows) | Distru ({len(distru_df):,} rows)")
        
        # Combine and process data
        combined_data = combine_inventory_data(headset_df, distru_df)
        
        if combined_data is not None:
            st.session_state.combined_data = combined_data
            st.success(f"✅ Successfully processed {len(combined_data):,} private label flower products")

# Sidebar - Version Info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 App Info")
st.sidebar.markdown(f"**Version:** {st.session_state.app_version}")

# Main Content Area
if st.session_state.combined_data is not None:
    combined_df = st.session_state.combined_data
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎯 Product Analysis", "📦 Distru Stock", "📋 Raw Data"])
    
    with tab1:
        st.header("📊 Private Label Inventory Dashboard")
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Products", f"{len(combined_df):,}")
            
        with col2:
            st.metric("Total Inventory", f"{combined_df['Total Inventory'].sum():,.0f}")
            
        with col3:
            total_inventory = combined_df['Total Inventory'].sum()
            total_daily_sales = combined_df['In Stock Avg Units per Day'].sum()
            total_woh = calculate_woh(total_inventory, total_daily_sales)
            st.metric("Total WOH", f"{total_woh:.1f}", help="Capped at 52 weeks for outlier control")
            
        with col4:
            distru_products = len(combined_df[combined_df['Distru Quantity'] > 0])
            st.metric("Products at Distro", f"{distru_products}")
            
        with col5:
            st.metric("Total Daily Sales", f"{total_daily_sales:.1f}")
        
        # Daily Sales Performance Chart
        st.subheader("📈 Daily Sales Performance by Product Type")
        daily_sales_chart = create_daily_sales_chart(combined_df)
        st.plotly_chart(daily_sales_chart, use_container_width=True)
        
        # Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            woh_chart = create_woh_summary_chart(combined_df)
            st.plotly_chart(woh_chart, use_container_width=True)
        
        with col2:
            inventory_chart = create_inventory_distribution_chart(combined_df)
            st.plotly_chart(inventory_chart, use_container_width=True)
        
        # Brand breakdown table
        st.subheader("📈 Brand Performance Summary")
        brand_summary = combined_df.groupby('Brand').agg({
            'Total Inventory': 'sum',
            'Distru Quantity': 'sum',
            'In Stock Avg Units per Day': 'sum',
            'Product Name': 'nunique',
            'Store Count': 'sum'
        }).reset_index()
        
        brand_summary['WOH'] = brand_summary.apply(
            lambda row: calculate_woh(row['Total Inventory'], row['In Stock Avg Units per Day']),
            axis=1
        )
        
        distru_products_by_brand = combined_df[combined_df['Distru Quantity'] > 0].groupby('Brand')['Product Name'].nunique().reset_index()
        distru_products_by_brand.rename(columns={'Product Name': 'Distro Products'}, inplace=True)
        brand_summary = brand_summary.merge(distru_products_by_brand, on='Brand', how='left')
        brand_summary['Distro Products'] = brand_summary['Distro Products'].fillna(0).astype(int)
        
        brand_summary.rename(columns={
            'Product Name': 'Total Products',
            'Store Count': 'Total Store Presence',
            'In Stock Avg Units per Day': 'Daily Sales'
        }, inplace=True)
        
        brand_summary = brand_summary[[
            'Brand', 'Total Products', 'Distro Products', 'Total Inventory', 
            'Distru Quantity', 'Daily Sales', 'WOH', 'Total Store Presence'
        ]]
        brand_summary = brand_summary.round(1)
        st.dataframe(brand_summary, use_container_width=True)
        
        # Expandable product performance summary
        create_expandable_product_summary(combined_df)
    
    with tab2:
        st.header("🎯 Product Analysis")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_brands = st.multiselect(
                "Select Brands",
                options=sorted(combined_df['Brand'].unique()),
                default=sorted(combined_df['Brand'].unique())
            )
        
        with col2:
            selected_categories = st.multiselect(
                "Select Categories",
                options=sorted(combined_df['Flower Category'].unique()),
                default=sorted(combined_df['Flower Category'].unique())
            )
        
        with col3:
            selected_weights = st.multiselect(
                "Select Weights",
                options=sorted(combined_df['Weight'].unique()),
                default=sorted(combined_df['Weight'].unique())
            )
        
        # Apply filters
        filtered_df = combined_df[
            (combined_df['Brand'].isin(selected_brands)) &
            (combined_df['Flower Category'].isin(selected_categories)) &
            (combined_df['Weight'].isin(selected_weights))
        ]
        
        if filtered_df.empty:
            st.warning("⚠️ No products match the selected filters")
        else:
            # Sort options
            sort_by = st.selectbox(
                "Sort by",
                options=['WOH', 'Total Inventory', 'Distru Quantity', 'Daily Sales', 'Product Name'],
                index=0
            )
            
            sort_ascending = st.checkbox("Sort ascending", value=False)
            
            # Map display name to column name
            sort_column_map = {
                'Daily Sales': 'In Stock Avg Units per Day',
                'WOH': 'WOH',
                'Total Inventory': 'Total Inventory',
                'Distru Quantity': 'Distru Quantity',
                'Product Name': 'Product Name'
            }
            
            filtered_df_sorted = filtered_df.sort_values(sort_column_map[sort_by], ascending=sort_ascending)
            
            st.subheader(f"📋 Product Details ({len(filtered_df_sorted)} products)")
            
            display_df = filtered_df_sorted[[
                'Product Name', 'Brand', 'Flower Category', 'Weight',
                'Total Inventory', 'Distru Quantity', 'In Stock Avg Units per Day',
                'WOH', 'Distru Days Supply', 'Store Count'
            ]].copy()
            
            display_df = display_df.rename(columns={'In Stock Avg Units per Day': 'Daily Sales'})
            
            for col in ['Total Inventory', 'Distru Quantity', 'Daily Sales', 'WOH', 'Distru Days Supply']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(1)
            
            st.dataframe(display_df, use_container_width=True, height=600)
    
    with tab3:
        st.header("📦 Products in Stock at Distru")
        
        distru_stock_df = combined_df[combined_df['Distru Quantity'] > 0].copy()
        
        if distru_stock_df.empty:
            st.warning("⚠️ No products currently in stock at Distru")
        else:
            st.success(f"✅ {len(distru_stock_df)} products in stock at Distru")
            
            # Summary table
            distru_summary = create_distru_stock_table(combined_df)
            
            if not distru_summary.empty:
                st.subheader("📊 Distru Stock Summary")
                distru_summary_display = distru_summary.rename(columns={'In Stock Avg Units per Day': 'Daily Sales'})
                st.dataframe(distru_summary_display, use_container_width=True)
            
            # Detailed product list
            st.subheader("📋 Detailed Product List")
            
            distru_sort_by = st.selectbox(
                "Sort by",
                options=['Distru Quantity', 'Distru Days Supply', 'WOH', 'Product Name'],
                index=0,
                key="distru_sort"
            )
            
            distru_sort_ascending = st.checkbox("Sort ascending", value=False, key="distru_sort_asc")
            
            distru_stock_sorted = distru_stock_df.sort_values(distru_sort_by, ascending=distru_sort_ascending)
            
            distru_display_df = distru_stock_sorted[[
                'Product Name', 'Brand', 'Flower Category', 'Weight',
                'Distru Quantity', 'Distru Days Supply', 'Total Inventory', 'In Stock Avg Units per Day', 'WOH'
            ]].copy()
            
            distru_display_df = distru_display_df.rename(columns={'In Stock Avg Units per Day': 'Daily Sales'})
            
            for col in ['Distru Quantity', 'Distru Days Supply', 'Total Inventory', 'Daily Sales', 'WOH']:
                if col in distru_display_df.columns:
                    distru_display_df[col] = distru_display_df[col].round(1)
            
            st.dataframe(distru_display_df, use_container_width=True, height=600)
    
    with tab4:
        st.header("📋 Raw Data")
        
        # Export functionality
        st.subheader("💾 Export Data")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_buffer = io.StringIO()
            combined_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📄 Download Combined Data CSV",
                data=csv_buffer.getvalue(),
                file_name=f"private_label_combined_{timestamp}.csv",
                mime="text/csv"
            )
        
        with col2:
            distru_stock_df = combined_df[combined_df['Distru Quantity'] > 0]
            if not distru_stock_df.empty:
                distru_csv_buffer = io.StringIO()
                distru_stock_df.to_csv(distru_csv_buffer, index=False)
                
                st.download_button(
                    label="📦 Download Distru Stock CSV",
                    data=distru_csv_buffer.getvalue(),
                    file_name=f"distru_stock_{timestamp}.csv",
                    mime="text/csv"
                )
        
        # Display raw data with search
        st.subheader("🔍 Complete Dataset")
        
        search_term = st.text_input("🔍 Search products (by name, brand, or category)")
        
        if search_term:
            search_mask = (
                combined_df['Product Name'].str.contains(search_term, case=False, na=False) |
                combined_df['Brand'].str.contains(search_term, case=False, na=False) |
                combined_df['Flower Category'].str.contains(search_term, case=False, na=False)
            )
            display_raw_df = combined_df[search_mask]
            st.info(f"Found {len(display_raw_df)} products matching '{search_term}'")
        else:
            display_raw_df = combined_df
        
        st.dataframe(display_raw_df, use_container_width=True, height=600)

else:
    # Welcome screen
    if not headset_file and not distru_file:
        st.info("👈 Upload the required CSV files in the sidebar to get started")
        
        with st.expander("ℹ️ How it Works", expanded=True):
            st.markdown(f"""
            **📊 Upload** → **🔄 Process** → **📈 Analyze** → **📋 Report**
            
            **Key Features:**
            - 🔗 Combines Headset and Distru inventory data
            - 🧮 Calculates Weeks on Hand (WOH) with outlier control (capped at {MAX_WOH_WEEKS:.0f} weeks)
            - 🌿 Focuses on flower products (Indica, Sativa, Hybrid)
            - 📦 Tracks products in stock at distribution for production planning
            - 🔤 Case-insensitive product matching across systems
            - 📊 Interactive dashboards and filtering
            - 💾 CSV export functionality
            
            **Data Requirements:**
            - **Headset CSV:** Store Name, Product Name, Brand, Category, Total Quantity on Hand, In Stock Avg Units per Day
            - **Distru CSV:** Product (or Product Name), Active Quantity (skips first 2 metadata rows automatically)
            
            **Tracked Brands:** {len(PRIVATE_LABEL_BRANDS)} private label brands (see sidebar)
            
            **Version:** {st.session_state.app_version} - Production Ready
            """)
    
    elif headset_file and distru_file:
        st.info("👈 Click the 'Process Data' button in the sidebar to analyze your files")
    
    else:
        missing_files = []
        if not headset_file:
            missing_files.append("Headset Inventory Coverage")
        if not distru_file:
            missing_files.append("Distru Inventory Assets")
        
        st.warning(f"📁 Please upload the {' and '.join(missing_files)} CSV file(s) to continue")

# Footer
st.markdown("---")
st.markdown(f"**Private Label Production Summary v{st.session_state.app_version}** | Built for DC Retail | Focus: Flower Products")