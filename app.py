"""
Private Label Production Summary
================================

A Streamlit application for automating weekly private label inventory reporting.
Processes Headset inventory coverage and Distru inventory assets reports to calculate
Weeks on Hand (WOH) and generate production insights for private label flower products.

Author: DC Retail
Version: 1.5 - Hardcoded private label brands, simplified workflow
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
from plotly.subplots import make_subplots

# =============================================================================
# CONSTANTS
# =============================================================================

# Private Label Brands - hardcoded list
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

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Private Label Production Summary",
    page_icon="📊",
    layout="wide"
)

# App header
st.title("📊 Private Label Production Summary")
st.markdown("**DC Retail** | Weekly inventory analysis and production planning for private label products")

# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================

def initialize_session_state():
    """Initialize session state variables"""
    session_vars = [
        'headset_data', 'distru_data',
        'combined_data', 'processed_data', 'app_version'
    ]
    
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = None
    
    # Set app version
    if st.session_state.app_version is None:
        st.session_state.app_version = "1.5"

initialize_session_state()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_numeric(value, default=0):
    """Convert any value to numeric, handling strings, NaN, None, etc."""
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
    Extract weight from product name by removing the last string.
    Returns: (product_name_without_weight, weight)
    
    Examples:
    - "Dope St. Exotics - Gmo Cookie 14g" → ("Dope St. Exotics - Gmo Cookie", "14g")
    - "Lil' Buzzies - Strawberry Runtz Smalls 7g" → ("Lil' Buzzies - Strawberry Runtz Smalls", "7g")
    """
    if pd.isna(product_name) or not isinstance(product_name, str):
        return str(product_name), ""
    
    parts = product_name.strip().split()
    if len(parts) >= 2:
        # Check if last part looks like a weight (contains 'g')
        last_part = parts[-1]
        if 'g' in last_part.lower():
            return ' '.join(parts[:-1]), last_part
    
    return product_name, ""

def extract_brand_from_product_name(product_name: str) -> str:
    """
    Extract brand from product name (text before the first hyphen).
    
    Examples:
    - "Black Label - Cherry Warheads 3.5g" → "Black Label"
    - "Block Party - Lemon Cherry Gelato 28g" → "Block Party"
    - "Dope St. - Gelonade 14g" → "Dope St."
    """
    if pd.isna(product_name) or not isinstance(product_name, str):
        return ""
    
    if ' - ' in product_name:
        return product_name.split(' - ')[0].strip()
    elif '-' in product_name:
        return product_name.split('-')[0].strip()
    else:
        # Fallback - try to extract first part
        parts = product_name.split()
        if len(parts) >= 2:
            return ' '.join(parts[:2])  # Take first two words as brand
        return product_name

def categorize_flower_type(category: str) -> str:
    """
    Standardize flower category names, handling patterns like:
    - "Flower (Indica)" → "Indica"
    - "Flower (Sativa)" → "Sativa" 
    - "Flower (Hybrid)" → "Hybrid"
    - "Indica" → "Indica"
    """
    if pd.isna(category):
        return "Unknown"
    
    category = str(category).lower().strip()
    
    # Handle "Flower (Type)" patterns
    if 'flower' in category and '(' in category:
        # Extract text within parentheses
        if 'indica' in category:
            return 'Indica'
        elif 'sativa' in category:
            return 'Sativa'
        elif 'hybrid' in category:
            return 'Hybrid'
    
    # Handle direct category names
    if 'indica' in category:
        return 'Indica'
    elif 'sativa' in category:
        return 'Sativa'
    elif 'hybrid' in category:
        return 'Hybrid'
    
    # If it's just "Flower" without subcategory, mark as Unknown for now
    if category == 'flower':
        return 'Unknown'
    
    return category.title()

def sort_by_category_order(df: pd.DataFrame, category_column: str = 'Flower Category') -> pd.DataFrame:
    """
    Sort dataframe by standard category order: Indica, Hybrid, Sativa
    
    Args:
        df: DataFrame to sort
        category_column: Name of the category column
    
    Returns:
        Sorted DataFrame
    """
    # Define category order
    category_order = ['Indica', 'Hybrid', 'Sativa', 'Unknown']
    
    # Create categorical type with fixed order
    if category_column in df.columns:
        df[category_column] = pd.Categorical(
            df[category_column], 
            categories=category_order, 
            ordered=True
        )
        df = df.sort_values(category_column)
    
    return df

def calculate_woh(total_inventory: float, daily_sales: float, max_woh: float = 52.0) -> float:
    """
    Calculate Weeks on Hand (WOH).
    WOH = (Total Inventory / Daily Sales) / 7
    Always round down for conservative estimate.
    Cap at max_woh to handle outliers (default 52 weeks = 1 year).
    
    Args:
        total_inventory: Total units in inventory
        daily_sales: Average daily sales rate
        max_woh: Maximum weeks on hand to cap at (default 52)
    
    Returns:
        Weeks on hand, capped at max_woh
    """
    # Minimum daily sales threshold - products below this have insufficient data
    MIN_DAILY_SALES = 0.1
    
    if daily_sales < MIN_DAILY_SALES:
        # Insufficient data - return capped value
        return max_woh
    
    days_supply = total_inventory / daily_sales
    woh = days_supply / 7
    
    # Cap at maximum to prevent outliers from skewing aggregations
    woh = min(woh, max_woh)
    
    return math.floor(woh * 10) / 10  # Round down to 1 decimal

# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

def load_headset_data(uploaded_file) -> Optional[pd.DataFrame]:
    """Load and validate Headset inventory coverage report"""
    try:
        df = pd.read_csv(uploaded_file, dtype=str)
        
        # Check for required columns
        required_columns = [
            'Store Name', 'Product Name', 'Brand', 'Category', 
            'Total Quantity on Hand', 'In Stock Avg Units per Day'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"❌ Missing required columns in Headset data: {missing_columns}")
            return None
        
        # Convert numeric columns
        numeric_columns = ['Total Quantity on Hand', 'In Stock Avg Units per Day']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: safe_numeric(x, 0))
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading Headset data: {str(e)}")
        return None

def load_distru_data(uploaded_file) -> Optional[pd.DataFrame]:
    """Load and validate Distru inventory assets report - skip metadata rows"""
    try:
        # Skip first 2 rows of metadata, use row 3 as headers
        df = pd.read_csv(uploaded_file, skiprows=2, dtype=str)
        
        # Check if we have data after skipping metadata
        if df.empty:
            st.error("❌ Distru file appears to be empty after skipping metadata rows")
            return None
        
        st.info(f"📋 Distru columns found: {list(df.columns)}")
        
        # Find Active Quantity column
        if 'Active Quantity' not in df.columns:
            st.warning(f"⚠️ 'Active Quantity' column not found.")
            # Try to find similar columns
            qty_columns = [col for col in df.columns if 'quantity' in col.lower()]
            if qty_columns:
                st.info(f"Found potential quantity columns: {qty_columns}")
        
        # Convert numeric columns
        numeric_columns = ['Active Quantity', 'Quantity', 'Total Quantity', 'Available Quantity']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: safe_numeric(x, 0))
        
        # Extract brand from product name if Brand column doesn't exist
        product_name_col = 'Product Name' if 'Product Name' in df.columns else 'Product'
        if 'Brand' not in df.columns and product_name_col in df.columns:
            df['Brand'] = df[product_name_col].apply(extract_brand_from_product_name)
            st.info("✅ Extracted brands from Distru product names")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading Distru data: {str(e)}")
        return None

def load_private_label_brands(uploaded_file) -> Optional[List[str]]:
    """Load private label brands list"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Try to find brand column (could be 'Brand', 'brand', 'Brand Name', etc.)
        brand_columns = [col for col in df.columns if 'brand' in col.lower()]
        
        if brand_columns:
            brands = df[brand_columns[0]].dropna().unique().tolist()
            return [str(brand).strip() for brand in brands if str(brand).strip()]
        else:
            # If no brand column found, use first column
            brands = df.iloc[:, 0].dropna().unique().tolist()
            return [str(brand).strip() for brand in brands if str(brand).strip()]
            
    except Exception as e:
        st.error(f"❌ Error loading private label brands: {str(e)}")
        return None

def combine_inventory_data(headset_df: pd.DataFrame, distru_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Combine Headset and Distru data to calculate total inventory and WOH.
    Focus on private label products and flower categories.
    Fixed to properly count unique products and stores.
    Uses hardcoded PRIVATE_LABEL_BRANDS list.
    """
    try:
        # Use hardcoded private label brands list
        private_label_brands = PRIVATE_LABEL_BRANDS
        
        # Filter Headset data to private label brands only
        headset_filtered = headset_df[headset_df['Brand'].isin(private_label_brands)].copy()
        
        # Debug: Show filtering results
        total_headset_products = len(headset_df)
        filtered_headset_products = len(headset_filtered)
        st.info(f"🎯 Headset filtering: {filtered_headset_products:,} of {total_headset_products:,} products are private label")
        
        if filtered_headset_products == 0:
            unique_brands_in_headset = headset_df['Brand'].dropna().unique()
            st.warning(f"⚠️ No private label brands found in Headset data. Available brands: {list(unique_brands_in_headset)[:10]}...")
            return None
        
        # Filter to flower categories only (exclude pre-rolls, vapes, etc.)
        flower_keywords = ['indica', 'sativa', 'hybrid', 'flower']
        headset_filtered = headset_filtered[
            headset_filtered['Category'].str.lower().str.contains(
                '|'.join(flower_keywords), na=False
            )
        ].copy()
        
        if headset_filtered.empty:
            st.warning("⚠️ No private label flower products found in Headset data")
            return None
        
        # Extract product weights
        headset_filtered[['Product Base Name', 'Weight']] = headset_filtered['Product Name'].apply(
            lambda x: pd.Series(extract_weight_from_product_name(x))
        )
        
        # Standardize categories
        headset_filtered['Flower Category'] = headset_filtered['Category'].apply(categorize_flower_type)
        
        # Create a helper function to count stores with inventory
        def count_stores_with_inventory(group):
            # Count unique stores where Total Quantity on Hand > 0
            stores_with_inventory = group[group['Total Quantity on Hand'] > 0]['Store Name'].nunique()
            return stores_with_inventory
        
        # FIXED: Group by Product Name to get totals across all stores
        # Only count stores that actually have inventory for this product
        headset_summary = headset_filtered.groupby(['Product Name', 'Brand', 'Flower Category', 'Product Base Name', 'Weight']).agg({
            'Total Quantity on Hand': 'sum',
            'In Stock Avg Units per Day': 'sum'
        }).reset_index()
        
        # Calculate store count with inventory separately
        store_counts = headset_filtered.groupby(['Product Name']).apply(count_stores_with_inventory).reset_index()
        store_counts.columns = ['Product Name', 'Store Count']
        
        # Merge store counts back
        headset_summary = headset_summary.merge(store_counts, on='Product Name', how='left')
        headset_summary['Store Count'] = headset_summary['Store Count'].fillna(0)
        
        # Filter out products with zero total inventory
        headset_summary = headset_summary[headset_summary['Total Quantity on Hand'] > 0].copy()
        
        if headset_summary.empty:
            st.warning("⚠️ No private label flower products with inventory found in Headset data")
            return None
        
        # Start with all Distru data
        distru_filtered = distru_df.copy()
        
        # Handle different product name columns (Distru uses "Product", Headset uses "Product Name")
        product_name_col = 'Product Name' if 'Product Name' in distru_filtered.columns else 'Product'
        
        # Standardize column name to match Headset
        if product_name_col == 'Product':
            distru_filtered['Product Name'] = distru_filtered['Product']
        
        # Extract brands from product names if Brand column doesn't exist
        if 'Brand' not in distru_filtered.columns:
            distru_filtered['Brand'] = distru_filtered['Product Name'].apply(extract_brand_from_product_name)
            st.info("✅ Extracted brands from Distru product names")
        
        # Debug: Show detected brands
        detected_brands = distru_filtered['Brand'].dropna().unique()
        st.info(f"🏷️ Brands found in Distru: {list(detected_brands)[:10]}...")  # Show first 10
        
        # Filter Distru data to private label brands
        matching_brands = [b for b in detected_brands if b in private_label_brands]
        st.info(f"🎯 Matching private label brands: {matching_brands}")
        
        if matching_brands:
            distru_filtered = distru_filtered[distru_filtered['Brand'].isin(private_label_brands)].copy()
        else:
            st.warning("⚠️ No matching private label brands found in Distru data")
            distru_filtered = pd.DataFrame()  # Empty dataframe
        
        # Filter Distru to flower categories (only if we have data)
        if not distru_filtered.empty and 'Category' in distru_filtered.columns:
            distru_filtered = distru_filtered[
                distru_filtered['Category'].str.lower().str.contains(
                    '|'.join(flower_keywords), na=False
                )
            ].copy()
        
        # Extract weights from Distru product names
        if not distru_filtered.empty and 'Product Name' in distru_filtered.columns:
            distru_filtered[['Product Base Name', 'Weight']] = distru_filtered['Product Name'].apply(
                lambda x: pd.Series(extract_weight_from_product_name(x))
            )
        
        # Find the quantity column in Distru data
        quantity_columns = ['Active Quantity', 'Quantity', 'Total Quantity', 'Available Quantity']
        distru_qty_col = None
        for col in quantity_columns:
            if col in distru_filtered.columns:
                distru_qty_col = col
                break
        
        if distru_qty_col is None:
            st.warning("⚠️ No quantity column found in Distru data. Proceeding with Headset data only.")
            distru_summary = pd.DataFrame()
        elif distru_filtered.empty:
            st.warning("⚠️ No matching Distru products after filtering. Proceeding with Headset data only.")
            distru_summary = pd.DataFrame()
        else:
            st.success(f"✅ Found Distru quantity column: {distru_qty_col}")
            
            # Ensure all required columns exist before grouping
            required_cols = ['Product Name', 'Brand', 'Product Base Name', 'Weight']
            missing_cols = [col for col in required_cols if col not in distru_filtered.columns]
            
            if missing_cols:
                st.error(f"❌ Missing columns in Distru data for grouping: {missing_cols}")
                distru_summary = pd.DataFrame()
            else:
                # Group Distru data by Product Name
                try:
                    distru_summary = distru_filtered.groupby(['Product Name', 'Brand', 'Product Base Name', 'Weight']).agg({
                        distru_qty_col: 'sum'
                    }).reset_index()
                    distru_summary.rename(columns={distru_qty_col: 'Distru Quantity'}, inplace=True)
                    st.success(f"✅ Processed {len(distru_summary)} Distru products")
                except Exception as e:
                    st.error(f"❌ Error grouping Distru data: {str(e)}")
                    distru_summary = pd.DataFrame()
        
        # Merge Headset and Distru data
        if not distru_summary.empty:
            combined = headset_summary.merge(
                distru_summary[['Product Name', 'Distru Quantity']], 
                on='Product Name', 
                how='left'
            )
            combined['Distru Quantity'] = combined['Distru Quantity'].fillna(0)
        else:
            combined = headset_summary.copy()
            combined['Distru Quantity'] = 0
        
        # Calculate total inventory and WOH
        combined['Total Inventory'] = combined['Total Quantity on Hand'] + combined['Distru Quantity']
        
        # Filter out products with zero total inventory
        combined = combined[combined['Total Inventory'] > 0].copy()
        
        if combined.empty:
            st.warning("⚠️ No products with inventory found after combining data")
            return None
        
        combined['WOH'] = combined.apply(
            lambda row: calculate_woh(row['Total Inventory'], row['In Stock Avg Units per Day']), 
            axis=1
        )
        
        # Calculate Days of Supply for Distru (Distru Quantity / Total Daily Sales)
        combined['Distru Days Supply'] = combined.apply(
            lambda row: math.floor(row['Distru Quantity'] / row['In Stock Avg Units per Day']) 
            if row['In Stock Avg Units per Day'] > 0 else 0,
            axis=1
        )
        
        # Create Product Group (Brand + Weight) for better grouping
        combined['Product Group'] = combined['Brand'] + ' ' + combined['Weight']
        
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
        # Group by Brand and Flower Category
        # CRITICAL: Don't sum WOH! Calculate from aggregated totals
        summary = df.groupby(['Brand', 'Flower Category']).agg({
            'Total Inventory': 'sum',
            'In Stock Avg Units per Day': 'sum',
            'Product Name': 'nunique'  # Count unique products
        }).reset_index()
        
        # Calculate WOH from the aggregated totals
        summary['WOH'] = summary.apply(
            lambda row: calculate_woh(row['Total Inventory'], row['In Stock Avg Units per Day']),
            axis=1
        )
        
        summary.rename(columns={'Product Name': 'Product Count'}, inplace=True)
        
        # Create heatmap
        fig = px.bar(
            summary, 
            x='Brand', 
            y='WOH',
            color='Flower Category',
            title='Weeks on Hand (WOH) by Brand and Category',
            labels={'WOH': 'Total Weeks on Hand', 'Brand': 'Private Label Brand'},
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

def create_inventory_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Create inventory distribution chart"""
    try:
        # Group by Weight and Brand
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
        # Filter to products with Distru stock
        distru_stock = df[df['Distru Quantity'] > 0].copy()
        
        if distru_stock.empty:
            return pd.DataFrame()
        
        # Create summary by Brand, Weight, and Category
        summary = distru_stock.groupby(['Brand', 'Weight', 'Flower Category']).agg({
            'Distru Quantity': 'sum',
            'Total Inventory': 'sum',
            'In Stock Avg Units per Day': 'sum',
            'Product Name': 'nunique'  # Count unique products
        }).reset_index()
        
        # Calculate Distru Days Supply and WOH properly
        summary['Distru Days Supply'] = summary.apply(
            lambda row: math.floor(row['Distru Quantity'] / row['In Stock Avg Units per Day']) 
            if row['In Stock Avg Units per Day'] > 0 else 0,
            axis=1
        )
        
        summary['WOH'] = summary.apply(
            lambda row: calculate_woh(row['Total Inventory'], row['In Stock Avg Units per Day']),
            axis=1
        )
        
        summary.rename(columns={'Product Name': 'Products at Distro'}, inplace=True)
        summary = summary[['Brand', 'Weight', 'Flower Category', 'Products at Distro', 'Distru Quantity', 'Distru Days Supply', 'Total Inventory', 'WOH']]
        
        # Sort by Brand, Weight, then category order (Indica, Hybrid, Sativa)
        summary = summary.sort_values(['Brand', 'Weight'])
        summary = sort_by_category_order(summary, 'Flower Category')
        
        return summary
        
    except Exception as e:
        st.error(f"Error creating Distru stock table: {str(e)}")
        return pd.DataFrame()

def create_expandable_product_summary(df: pd.DataFrame):
    """Create expandable product summary with drill-down by category"""
    
    # Only show products with inventory > 0
    df_with_inventory = df[df['Total Inventory'] > 0].copy()
    
    if df_with_inventory.empty:
        st.warning("⚠️ No products with inventory to display")
        return
    
    # Group by Product Group (Brand + Weight)
    # CRITICAL: Calculate WOH from aggregated totals, don't sum individual WOHs
    product_groups = df_with_inventory.groupby('Product Group').agg({
        'Total Inventory': 'sum',
        'Distru Quantity': 'sum',
        'In Stock Avg Units per Day': 'sum',
        'Product Name': 'nunique',
        'Store Count': 'sum'
    }).reset_index()
    
    # Calculate WOH properly from aggregated values
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
        
        # Create expandable section for each product group
        with st.expander(f"**{product_group}** - {total_products} products, {distro_products} at Distro, {woh:.1f} WOH, {daily_sales:.1f} daily sales"):
            
            # Get products for this group (with inventory only)
            group_products = df_with_inventory[df_with_inventory['Product Group'] == product_group]
            
            # Summary by category - calculate WOH properly
            category_agg = group_products.groupby('Flower Category').agg({
                'Total Inventory': 'sum',
                'Distru Quantity': 'sum',
                'In Stock Avg Units per Day': 'sum',
                'Product Name': 'nunique'
            }).reset_index()
            
            # Calculate WOH from aggregated values
            category_agg['WOH'] = category_agg.apply(
                lambda row: calculate_woh(row['Total Inventory'], row['In Stock Avg Units per Day']),
                axis=1
            )
            
            # Count products in stock at Distro by category
            distro_by_category = group_products[group_products['Distru Quantity'] > 0].groupby('Flower Category').size().reset_index(name='Distro Products')
            category_summary = category_agg.merge(distro_by_category, on='Flower Category', how='left')
            category_summary['Distro Products'] = category_summary['Distro Products'].fillna(0).astype(int)
            
            category_summary.rename(columns={'Product Name': 'Total Products'}, inplace=True)
            category_summary = category_summary[['Flower Category', 'Total Products', 'Distro Products', 'Total Inventory', 'Distru Quantity', 'In Stock Avg Units per Day', 'WOH']]
            
            # Sort by category order: Indica, Hybrid, Sativa
            category_summary = sort_by_category_order(category_summary, 'Flower Category')
            category_summary = category_summary.round(1)
            
            # Display category summary
            st.markdown("**📊 By Category:**")
            st.dataframe(category_summary, use_container_width=True, hide_index=True)
            
            # Individual products by category (only those with inventory)
            st.markdown("**🌿 Individual Products:**")
            
            # Use fixed category order for display
            category_order = ['Indica', 'Hybrid', 'Sativa', 'Unknown']
            available_categories = [cat for cat in category_order if cat in group_products['Flower Category'].unique()]
            
            for category in available_categories:
                category_products = group_products[group_products['Flower Category'] == category]
                # Additional filter for products with inventory (should already be filtered, but double-check)
                category_products = category_products[category_products['Total Inventory'] > 0]
                
                if not category_products.empty:
                    distro_count = len(category_products[category_products['Distru Quantity'] > 0])
                    total_count = len(category_products)
                    
                    with st.expander(f"    {category} ({total_count} products, {distro_count} at Distro)"):
                        display_cols = [
                            'Product Name', 'Total Inventory', 'Distru Quantity', 
                            'In Stock Avg Units per Day', 'WOH', 'Store Count'
                        ]
                        
                        category_display = category_products[display_cols].copy()
                        numeric_cols = ['Total Inventory', 'Distru Quantity', 'In Stock Avg Units per Day', 'WOH']
                        for col in numeric_cols:
                            if col in category_display.columns:
                                category_display[col] = category_display[col].round(1)
                        
                        st.dataframe(category_display, use_container_width=True, hide_index=True)

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
    help="Upload the inventory assets report from Distru (will skip metadata rows)"
)

# Display private label brands being tracked
st.sidebar.markdown("---")
st.sidebar.subheader("🏷️ Private Label Brands")
st.sidebar.caption(f"Tracking {len(PRIVATE_LABEL_BRANDS)} brands:")
with st.sidebar.expander("View Brands"):
    for brand in PRIVATE_LABEL_BRANDS:
        st.markdown(f"• {brand}")

# Process Data Button
if st.sidebar.button("🚀 Process Data", type="primary", 
                    disabled=not (headset_file and distru_file)):
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
        file_info = (f"Headset: {len(headset_df):,} rows | "
                    f"Distru: {len(distru_df):,} rows | "
                    f"Brands: {len(PRIVATE_LABEL_BRANDS)} brands")
        st.success("✅ Files loaded successfully!")
        st.info(file_info)
        
        # Show detected brands in Distru data
        if 'Brand' in distru_df.columns or 'Product' in distru_df.columns or 'Product Name' in distru_df.columns:
            # We'll extract brands during processing
            pass
        else:
            st.warning("⚠️ No product or brand columns found in Distru data")
        
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
            total_products = len(combined_df)
            st.metric("Total Products", f"{total_products:,}")
            
        with col2:
            total_inventory = combined_df['Total Inventory'].sum()
            st.metric("Total Inventory", f"{total_inventory:,.0f}")
            
        with col3:
            # Calculate total WOH properly from aggregated values
            total_inventory = combined_df['Total Inventory'].sum()
            total_daily_sales = combined_df['In Stock Avg Units per Day'].sum()
            total_woh = calculate_woh(total_inventory, total_daily_sales)
            st.metric("Total WOH", f"{total_woh:.1f}", help="Capped at 52 weeks for outlier control")
            
        with col4:
            distru_products = len(combined_df[combined_df['Distru Quantity'] > 0])
            st.metric("Products at Distru", f"{distru_products}")
            
        with col5:
            avg_daily_sales = combined_df['In Stock Avg Units per Day'].sum()
            st.metric("Total Daily Sales", f"{avg_daily_sales:.1f}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            woh_chart = create_woh_summary_chart(combined_df)
            st.plotly_chart(woh_chart, use_container_width=True)
        
        with col2:
            inventory_chart = create_inventory_distribution_chart(combined_df)
            st.plotly_chart(inventory_chart, use_container_width=True)
        
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
                options=['WOH', 'Total Inventory', 'Distru Quantity', 'In Stock Avg Units per Day', 'Product Name'],
                index=0
            )
            
            sort_ascending = st.checkbox("Sort ascending", value=False)
            
            filtered_df_sorted = filtered_df.sort_values(sort_by, ascending=sort_ascending)
            
            # Display filtered data
            st.subheader(f"📋 Product Details ({len(filtered_df_sorted)} products)")
            
            # Format display columns
            display_columns = [
                'Product Name', 'Brand', 'Flower Category', 'Weight',
                'Total Inventory', 'Distru Quantity', 'In Stock Avg Units per Day',
                'WOH', 'Distru Days Supply', 'Store Count'
            ]
            
            display_df = filtered_df_sorted[display_columns].copy()
            
            # Round numeric columns
            numeric_columns = ['Total Inventory', 'Distru Quantity', 'In Stock Avg Units per Day', 'WOH', 'Distru Days Supply']
            for col in numeric_columns:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(1)
            
            st.dataframe(display_df, use_container_width=True, height=600)
    
    with tab3:
        st.header("📦 Products in Stock at Distru")
        
        # Filter to products with Distru stock
        distru_stock_df = combined_df[combined_df['Distru Quantity'] > 0].copy()
        
        if distru_stock_df.empty:
            st.warning("⚠️ No products currently in stock at Distru")
        else:
            st.success(f"✅ {len(distru_stock_df)} products in stock at Distru")
            
            # Create summary table by Brand/Weight/Category
            distru_summary = create_distru_stock_table(combined_df)
            
            if not distru_summary.empty:
                st.subheader("📊 Distru Stock Summary")
                st.dataframe(distru_summary, use_container_width=True)
            
            # Detailed product list
            st.subheader("📋 Detailed Product List")
            
            # Sort options for Distru products
            distru_sort_by = st.selectbox(
                "Sort by",
                options=['Distru Quantity', 'Distru Days Supply', 'WOH', 'Product Name'],
                index=0,
                key="distru_sort"
            )
            
            distru_sort_ascending = st.checkbox("Sort ascending", value=False, key="distru_sort_asc")
            
            distru_stock_sorted = distru_stock_df.sort_values(distru_sort_by, ascending=distru_sort_ascending)
            
            # Display columns for Distru focus
            distru_display_columns = [
                'Product Name', 'Brand', 'Flower Category', 'Weight',
                'Distru Quantity', 'Distru Days Supply', 'Total Inventory', 'WOH'
            ]
            
            distru_display_df = distru_stock_sorted[distru_display_columns].copy()
            
            # Round numeric columns
            numeric_columns = ['Distru Quantity', 'Distru Days Supply', 'Total Inventory', 'WOH']
            for col in numeric_columns:
                if col in distru_display_df.columns:
                    distru_display_df[col] = distru_display_df[col].round(1)
            
            st.dataframe(distru_display_df, use_container_width=True, height=600)
    
    with tab4:
        st.header("📋 Raw Data")
        
        # Export functionality
        st.subheader("💾 Export Data")
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export combined data
            csv_buffer = io.StringIO()
            combined_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📄 Download Combined Data CSV",
                data=csv_buffer.getvalue(),
                file_name=f"private_label_combined_{timestamp}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export Distru stock only
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
        
        # Display raw data
        st.subheader("🔍 Complete Dataset")
        
        # Add search functionality
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
            st.markdown("""
            **📊 Upload** → **🔄 Process** → **📈 Analyze** → **📋 Report**
            
            **Key Features:**
            - 🔗 Combines Headset and Distru inventory data
            - 🧮 Calculates Weeks on Hand (WOH) automatically
            - 🌿 Focuses on flower products (Indica, Sativa, Hybrid)
            - 📦 Identifies products in stock at Distru
            - 📊 Interactive dashboards and filtering
            - 💾 CSV export functionality
            
            **v1.5 Streamlined:**
            - ✅ **Hardcoded Brand List** - No more brand CSV upload needed
            - ✅ **15 Private Label Brands** - Automatically tracks your portfolio
            - ✅ **Faster Workflow** - Only 2 files to upload (Headset + Distru)
            
            **v1.4 Data Science Best Practices:**
            - ✅ **WOH Outlier Control** - Caps at 52 weeks (1 year) to prevent skewed graphs
            - ✅ **Insufficient Data Handling** - Products with < 0.1 daily sales capped at max WOH
            - ✅ **Consistent Category Ordering** - Always displays Indica → Hybrid → Sativa
            - ✅ **Better Data Quality** - Prevents new product rollouts from distorting analytics
            
            **Why 52 weeks?** Beyond 1 year of supply is effectively "infinite" for planning purposes. 
            Capping prevents products with very low sales from dominating aggregate metrics.
            
            **v1.3 Previous Fixes:**
            - ✅ **WOH Calculation Fixed** - Now calculates from aggregated totals, not summing individual WOHs
            - ✅ **Distro Product Counts** - Shows how many products are in-stock at distribution facility
            - ✅ **Production Red Flags** - Track out-of-stocks at Distro for production planning
            - ✅ **Accurate Metrics** - Brand summary now shows "Total Products" and "Distro Products"
            
            **Example:** Black Label 3.5g Hybrid showing "11 products, 4 at Distro, 16.2 WOH"
            
            **v1.2 Previous Improvements:**
            - ✅ **Private Label Filtering Fixed** - Now properly filters to only private label brands
            - ✅ **Category Normalization** - Handles "Flower (Indica)" patterns, shows only Indica/Sativa/Hybrid
            - ✅ **Inventory-Only Display** - Only shows products with inventory > 0 (hides sold-out products)
            - ✅ **Accurate Store Counts** - Only counts stores that actually have the product in stock
            - ✅ **Better Debug Info** - Shows filtering results and brand matching details
            
            **v1.1 Previous Improvements:**
            - ✅ **Fixed Distru Data Loading** - Properly skips metadata rows and finds Active Quantity
            - ✅ **Brand Extraction** - Extracts brands from Distru product names (text before hyphen)
            - ✅ **Fixed Product Counting** - No more double counting of products across stores
            - ✅ **Drill-down Categories** - Expandable product groups with category breakdown
            - ✅ **Product Grouping** - Groups by Brand + Weight (e.g., "Block Party 28g" vs "Block Party 7g")
            - ✅ **Accurate Store Presence** - Counts unique stores per product correctly
            
            **Data Requirements:**
            - **Headset CSV:** Store Name, Product Name, Brand, Category, Total Quantity on Hand, In Stock Avg Units per Day
            - **Distru CSV:** Product (or Product Name), Active Quantity (skips first 2 metadata rows automatically)
            
            **Brand Extraction Examples:**
            - "Black Label - Cherry Warheads 3.5g" → Brand: "Black Label"
            - "Block Party - Lemon Cherry Gelato 28g" → Brand: "Block Party"
            - "Dope St. - Gelonade 14g" → Brand: "Dope St."
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