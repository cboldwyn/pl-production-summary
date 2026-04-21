"""
Haven Current Inventory and Sales
==================================

A Streamlit application for inventory analysis, production planning, and product
performance tracking. Processes Headset inventory coverage and Distru inventory
assets reports to calculate Weeks on Hand (WOH) and generate insights across
private label and all product categories.

Author: DC Retail
Version: 4.0.0 - Dynamic brands list + all-products view
Date: 2026

Key Features:
- Combines retail (Headset) and distribution (Distru) inventory data
- Calculates Weeks on Hand (WOH) with outlier control
- Tracks products in stock at distribution for production planning
- Case-insensitive product and brand matching with normalization
- Vape keyword extraction with breakdown by type
- Interactive dashboards with drill-down capabilities
- Supports Flower (Indica/Sativa/Hybrid), Preroll, and Vape categories
- Persistent data: auto-saves processed data, loads on startup
- Production flags: WOH alerts (warning/urgent/critical) and distro coverage gaps
- Dynamic private label brands list: uploadable via CSV, persisted to disk
- All-products view: see non-private-label product data in Product Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import json
import math
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# CONSTANTS
# =============================================================================

# Default Private Label Brands - fallback when no uploaded list exists
DEFAULT_PRIVATE_LABEL_BRANDS = [
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

# Product category keywords for filtering
# This filters to include: Flower (Indica/Sativa/Hybrid), Preroll, and Vape
# We use keywords because Flower categories appear as "Flower (Indica)" etc.
PRODUCT_KEYWORDS = ['indica', 'sativa', 'hybrid', 'flower', 'preroll', 'vape', 'extract']

# Standard category order for consistent display
CATEGORY_ORDER = ['Indica', 'Hybrid', 'Sativa', 'Preroll', 'Vape', 'Extract', 'Unknown']

# WOH calculation parameters
MAX_WOH_WEEKS = 52.0  # Cap at 1 year for outlier control
MIN_DAILY_SALES = 0.1  # Minimum threshold for reliable data

# Production flag thresholds (weeks on hand)
WOH_WARNING = 8       # Yellow flag
WOH_URGENT = 4        # Red flag
WOH_CRITICAL = 2      # Danger flag

# Persistence paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LATEST_DATA_PATH = os.path.join(DATA_DIR, "latest_combined.parquet")
ALL_PRODUCTS_PATH = os.path.join(DATA_DIR, "latest_all_products.parquet")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")
BRANDS_PATH = os.path.join(DATA_DIR, "private_label_brands.json")
BL_INPUTS_PATH = os.path.join(DATA_DIR, "latest_bl_inputs.parquet")
BL_INPUTS_META_PATH = os.path.join(DATA_DIR, "bl_inputs_metadata.json")

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Haven Current Inventory and Sales",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Haven Current Inventory and Sales")
st.markdown("**DC Retail** | Inventory analysis, production planning, and product performance tracking (Flower, Preroll, Vape)")

# =============================================================================
# SESSION STATE MANAGEMENT
# =============================================================================

def initialize_session_state():
    """Initialize session state variables for data persistence"""
    session_vars = ['headset_data', 'distru_data', 'combined_data', 'all_products_data', 'processed_data', 'app_version', 'saved_metadata', 'private_label_brands', 'bl_inputs_data', 'bl_inputs_metadata']

    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = None

    if st.session_state.app_version is None:
        st.session_state.app_version = "4.0.0"

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

def extract_vape_keywords(product_name: str) -> str:
    """
    Extract vape-specific keywords from product name.
    Based on price-checker v4 vape keyword extraction logic.
    
    Args:
        product_name: Full product name
        
    Returns:
        Comma-separated string of found keywords, or empty string if none found
        
    Examples:
        "Curepen Originals - Blue Dream 1g" → "originals, curepen"
        "DNA Live Resin - Wedding Cake 1g" → "dna, live resin"
    """
    if pd.isna(product_name) or not isinstance(product_name, str):
        return ""
    
    product_lower = str(product_name).lower()
    
    # Vape keywords from price-checker v4
    vape_keywords = [
        'originals', 'ascnd', 'dna', 'exotics', 'disposable', 
        'live resin', 'reload', 'rtu', 'curepen', 'curebar'
    ]
    
    found_keywords = [keyword for keyword in vape_keywords if keyword in product_lower]
    
    return ', '.join(found_keywords) if found_keywords else ""

def normalize_brand_name(brand: str, brands_list: List[str] = None) -> str:
    """
    Normalize brand name to match canonical brand names.

    Performs case-insensitive matching against the provided brands list
    and returns the canonical name.

    Args:
        brand: Brand name from data (e.g., "Pto", "mikrodose")
        brands_list: List of canonical brand names to match against.
                     If None, loads from persistent storage.

    Returns:
        Canonical brand name (e.g., "PTO", "MikroDose") or original if no match
    """
    if pd.isna(brand):
        return brand

    if brands_list is None:
        brands_list = load_brands_list()

    brand_lower = str(brand).lower().strip()

    # Find matching canonical brand name (case-insensitive)
    for canonical_brand in brands_list:
        if canonical_brand.lower() == brand_lower:
            return canonical_brand

    # If no match found, return original
    return str(brand).strip()

def categorize_product_type(category: str) -> str:
    """
    Standardize product category names.
    
    Simple categorization based on keywords:
    - "Flower (Indica)" → "Indica"
    - "Flower (Sativa)" → "Sativa"  
    - "Flower (Hybrid)" → "Hybrid"
    - "Preroll" → "Preroll"
    - "Vape" → "Vape"
    
    Args:
        category: Raw category string from data
        
    Returns:
        Standardized category: Indica, Sativa, Hybrid, Preroll, Vape, or Unknown
    """
    if pd.isna(category):
        return "Unknown"
    
    category = str(category).lower().strip()
    
    # Handle flower subcategories
    if 'indica' in category:
        return 'Indica'
    elif 'sativa' in category:
        return 'Sativa'
    elif 'hybrid' in category:
        return 'Hybrid'
    # Handle prerolls, vapes, and extracts
    elif 'preroll' in category:
        return 'Preroll'
    elif 'vape' in category:
        return 'Vape'
    elif 'extract' in category:
        return 'Extract'
    # Unknown for anything else
    else:
        return 'Unknown'

def sort_by_category_order(df: pd.DataFrame, category_column: str = 'Flower Category') -> pd.DataFrame:
    """
    Sort dataframe by standard category order: Indica, Hybrid, Sativa, Preroll, Vape, Unknown.
    
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
# PERSISTENCE FUNCTIONS
# =============================================================================

def save_processed_data(df: pd.DataFrame, headset_filename: str, distru_filename: str,
                        headset_rows: int, distru_rows: int) -> bool:
    """
    Save processed DataFrame and metadata to disk for persistence.

    Args:
        df: Combined processed DataFrame
        headset_filename: Original Headset CSV filename
        distru_filename: Original Distru CSV filename
        headset_rows: Row count from Headset file
        distru_rows: Row count from Distru file

    Returns:
        True on success, False on failure
    """
    try:
        os.makedirs(DATA_DIR, exist_ok=True)

        # Save DataFrame as Parquet (preserves dtypes)
        save_df = df.copy()
        # Convert Categorical columns to string for clean Parquet storage
        for col in save_df.columns:
            if hasattr(save_df[col], 'cat'):
                save_df[col] = save_df[col].astype(str)
        save_df.to_parquet(LATEST_DATA_PATH, index=False)

        # Save metadata
        metadata = {
            'last_updated': datetime.now().isoformat(),
            'headset_file': headset_filename,
            'distru_file': distru_filename,
            'headset_rows': headset_rows,
            'distru_rows': distru_rows,
            'product_count': len(df),
            'app_version': '3.0.0'
        }
        with open(METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)

        return True

    except Exception as e:
        st.warning(f"Could not save data to disk: {str(e)}")
        return False


def load_saved_data() -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    """
    Load previously saved DataFrame and metadata from disk.

    Returns:
        Tuple of (DataFrame, metadata_dict) or (None, None) if no saved data
    """
    try:
        if not os.path.exists(LATEST_DATA_PATH) or not os.path.exists(METADATA_PATH):
            return None, None

        df = pd.read_parquet(LATEST_DATA_PATH)

        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)

        return df, metadata

    except Exception:
        return None, None


def load_saved_all_products() -> Optional[pd.DataFrame]:
    """Load previously saved all-products DataFrame from disk."""
    try:
        if not os.path.exists(ALL_PRODUCTS_PATH):
            return None
        return pd.read_parquet(ALL_PRODUCTS_PATH)
    except Exception:
        return None


def save_all_products_data(df: pd.DataFrame) -> bool:
    """Save all-products DataFrame to disk for persistence."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        save_df = df.copy()
        for col in save_df.columns:
            if hasattr(save_df[col], 'cat'):
                save_df[col] = save_df[col].astype(str)
        save_df.to_parquet(ALL_PRODUCTS_PATH, index=False)
        return True
    except Exception:
        return False


# =============================================================================
# BRANDS LIST MANAGEMENT
# =============================================================================

def load_brands_list() -> List[str]:
    """
    Load the private label brands list from persistent storage.
    Falls back to DEFAULT_PRIVATE_LABEL_BRANDS if no saved list exists.
    """
    try:
        if os.path.exists(BRANDS_PATH):
            with open(BRANDS_PATH, 'r') as f:
                brands = json.load(f)
            if isinstance(brands, list) and len(brands) > 0:
                return brands
    except Exception:
        pass
    return DEFAULT_PRIVATE_LABEL_BRANDS.copy()


def save_brands_list(brands: List[str]) -> bool:
    """Save brands list to persistent JSON storage."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(BRANDS_PATH, 'w') as f:
            json.dump(sorted(brands), f, indent=2)
        return True
    except Exception as e:
        st.warning(f"Could not save brands list: {str(e)}")
        return False


def parse_brands_csv(uploaded_file) -> Optional[List[str]]:
    """
    Parse a Distru companies CSV export and extract private label brand names.

    Expects columns: 'Name' and 'Private Label' (values: 'Yes'/'No').

    Returns:
        Sorted list of brand names where Private Label == 'Yes', or None on error.
    """
    try:
        df = pd.read_csv(uploaded_file)

        # Find required columns (case-insensitive)
        name_col = None
        pl_col = None
        for col in df.columns:
            if col.strip().lower() == 'name':
                name_col = col
            elif col.strip().lower() == 'private label':
                pl_col = col

        if name_col is None or pl_col is None:
            st.error("CSV must contain 'Name' and 'Private Label' columns")
            return None

        # Filter to Private Label == Yes
        pl_brands = df[df[pl_col].str.strip().str.lower() == 'yes'][name_col].dropna()
        brands = sorted([str(b).strip() for b in pl_brands.tolist() if str(b).strip()])

        if not brands:
            st.warning("No brands with Private Label = 'Yes' found in CSV")
            return None

        return brands
    except Exception as e:
        st.error(f"Error parsing brands CSV: {str(e)}")
        return None


# =============================================================================
# PRODUCTION FLAG FUNCTIONS
# =============================================================================

def calculate_production_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add WOH flag column to each product row based on inventory levels.

    Flags:
        - 'critical': WOH < 2 weeks (danger)
        - 'urgent': WOH < 4 weeks (red)
        - 'warning': WOH < 8 weeks (yellow)
        - 'ok': WOH >= 8 weeks

    Args:
        df: Combined DataFrame with WOH column

    Returns:
        DataFrame with 'WOH Flag' column added
    """
    result = df.copy()

    def assign_flag(woh):
        if woh < WOH_CRITICAL:
            return 'critical'
        elif woh < WOH_URGENT:
            return 'urgent'
        elif woh < WOH_WARNING:
            return 'warning'
        else:
            return 'ok'

    result['WOH Flag'] = result['WOH'].apply(assign_flag)
    return result


def get_woh_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get WOH alerts aggregated at Brand + Weight + Category level.

    Calculates WOH from aggregated inventory and daily sales (not averaged),
    then flags based on thresholds. Returns only flagged items, sorted by severity.

    Args:
        df: Combined DataFrame with inventory data

    Returns:
        DataFrame with columns: Brand, Weight, Flower Category, Total Inventory,
        Distru Quantity, Daily Sales, WOH, Flag
    """
    summary = df.groupby(['Brand', 'Weight', 'Flower Category']).agg({
        'Total Inventory': 'sum',
        'Distru Quantity': 'sum',
        'In Stock Avg Units per Day': 'sum',
        'Product Name': 'nunique'
    }).reset_index()

    summary['WOH'] = summary.apply(
        lambda row: calculate_woh(row['Total Inventory'], row['In Stock Avg Units per Day']),
        axis=1
    )

    def assign_flag(woh):
        if woh < WOH_CRITICAL:
            return 'critical'
        elif woh < WOH_URGENT:
            return 'urgent'
        elif woh < WOH_WARNING:
            return 'warning'
        else:
            return 'ok'

    summary['Flag'] = summary['WOH'].apply(assign_flag)
    summary.rename(columns={
        'Product Name': 'Products',
        'In Stock Avg Units per Day': 'Daily Sales'
    }, inplace=True)

    # Round display values
    for col in ['Daily Sales', 'WOH']:
        if col in summary.columns:
            summary[col] = summary[col].round(1)

    # Filter to flagged items only, sort by severity
    flag_order = {'critical': 0, 'urgent': 1, 'warning': 2}
    flagged = summary[summary['Flag'] != 'ok'].copy()
    flagged['_sort'] = flagged['Flag'].map(flag_order)
    flagged = flagged.sort_values(['_sort', 'WOH']).drop(columns=['_sort'])

    return flagged


def get_distro_coverage_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify Brand + Weight + Category combinations where distro has zero inventory.

    These represent gaps where stores cannot be replenished from distribution.

    Args:
        df: Combined DataFrame with inventory data

    Returns:
        DataFrame with columns: Brand, Weight, Flower Category, Retail Inventory,
        Daily Sales, WOH (retail only), Products
    """
    summary = df.groupby(['Brand', 'Weight', 'Flower Category']).agg({
        'Total Quantity on Hand': 'sum',
        'Distru Quantity': 'sum',
        'In Stock Avg Units per Day': 'sum',
        'Product Name': 'nunique'
    }).reset_index()

    # Filter to combos where distro has zero
    gaps = summary[summary['Distru Quantity'] == 0].copy()

    if gaps.empty:
        return gaps

    # Calculate WOH from retail inventory only
    gaps['WOH'] = gaps.apply(
        lambda row: calculate_woh(row['Total Quantity on Hand'], row['In Stock Avg Units per Day']),
        axis=1
    )

    gaps.rename(columns={
        'Total Quantity on Hand': 'Retail Inventory',
        'In Stock Avg Units per Day': 'Daily Sales',
        'Product Name': 'Products'
    }, inplace=True)

    # Round display values
    for col in ['Daily Sales', 'WOH']:
        if col in gaps.columns:
            gaps[col] = gaps[col].round(1)

    gaps = gaps.drop(columns=['Distru Quantity'])
    gaps = gaps.sort_values('WOH')

    return gaps


def style_flag_dataframe(df: pd.DataFrame, flag_column: str = 'Flag'):
    """
    Apply conditional background colors to rows based on flag values.

    Colors:
        - critical: light red (#ffcccc)
        - urgent: light orange (#ffe0cc)
        - warning: light yellow (#fff3cc)
        - ok: no styling

    Args:
        df: DataFrame to style
        flag_column: Name of the column containing flag values

    Returns:
        Styled DataFrame for use with st.dataframe()
    """
    flag_colors = {
        'critical': 'background-color: #ffcccc',
        'urgent': 'background-color: #ffe0cc',
        'warning': 'background-color: #fff3cc',
        'ok': ''
    }

    def apply_row_style(row):
        flag = row.get(flag_column, 'ok')
        style = flag_colors.get(flag, '')
        return [style] * len(row)

    # Build format dict for float columns that should display as 1 decimal
    format_dict = {}
    for col in df.columns:
        if col in ('Daily Sales', 'WOH', 'In Stock Avg Units per Day'):
            format_dict[col] = '{:.1f}'

    styled = df.style.apply(apply_row_style, axis=1)
    if format_dict:
        styled = styled.format(format_dict)
    return styled


def format_dataframe(df: pd.DataFrame):
    """Apply consistent number formatting (1 decimal) to Daily Sales and WOH columns."""
    format_dict = {}
    for col in df.columns:
        if col in ('Daily Sales', 'WOH', 'In Stock Avg Units per Day'):
            format_dict[col] = '{:.1f}'
    if format_dict:
        return df.style.format(format_dict)
    return df


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


def load_bl_input_materials(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load and filter a Beyond Legends Distru Inventory Assets export down to
    non-cannabis input materials (Packaging + Supplies), then aggregate by SKU
    so each product shows a single total-on-hand row.

    Distru emits one row per License (e.g. Manufacturing vs Distribution).
    Jackie needs total on-hand per SKU for reorder decisions, so we sum
    quantity and cost across licenses and join Location + License into
    comma-delimited strings. Raw per-license rows are not preserved here;
    they remain available by re-exporting from Distru if needed.

    Args:
        uploaded_file: Uploaded CSV from the Distru Inventory Assets report

    Returns:
        Aggregated DataFrame of input materials, or None on failure.
    """
    try:
        df = pd.read_csv(uploaded_file, skiprows=2, dtype=str)

        if df.empty:
            st.error("❌ Beyond Legends file appears to be empty after skipping metadata rows")
            return None

        required = [
            'Product', 'License', 'Location', 'Vendor', 'Unit Type',
            'Active Quantity', 'Category', 'Subcategory',
            'Unit Cost (Actual)', 'Total Cost (Actual)', 'Expiration Date'
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"❌ Beyond Legends file missing required columns: {', '.join(missing)}")
            return None

        if 'SKU' not in df.columns:
            df['SKU'] = ''

        # Numeric + date coercion
        for col in ['Active Quantity', 'Unit Cost (Actual)', 'Total Cost (Actual)']:
            df[col] = df[col].apply(lambda x: safe_numeric(x, 0))
        df['Expiration Date'] = pd.to_datetime(df['Expiration Date'], errors='coerce')

        # Filter to input materials only
        df = df[df['Category'].isin(['Packaging', 'Supplies'])].copy()
        if df.empty:
            st.warning("⚠️ No Packaging or Supplies rows found in this export.")
            return df

        # Clean empty strings for filter / display
        for col in ['Subcategory', 'Vendor', 'Location', 'Unit Type']:
            df[col] = df[col].fillna('').astype(str).str.strip()
            df.loc[df[col] == '', col] = '—'

        # Normalize SKU for grouping: fall back to Product when SKU is blank
        df['SKU'] = df['SKU'].fillna('').astype(str).str.strip()
        df['_group_key'] = df['SKU'].where(df['SKU'] != '', df['Product'])

        def _first_non_empty(series):
            for v in series:
                if pd.notna(v) and str(v).strip() != '':
                    return v
            return ''

        def _join_unique(series):
            vals = sorted({str(v).strip() for v in series if pd.notna(v) and str(v).strip() not in ('', '—')})
            return ', '.join(vals) if vals else '—'

        grouped = df.groupby('_group_key', as_index=False).agg(
            Product=('Product', _first_non_empty),
            SKU=('SKU', _first_non_empty),
            Vendor=('Vendor', _first_non_empty),
            Category=('Category', _first_non_empty),
            Subcategory=('Subcategory', _first_non_empty),
            **{'Unit Type': ('Unit Type', _first_non_empty)},
            **{'Unit Cost (Actual)': ('Unit Cost (Actual)', _first_non_empty)},
            **{'Active Quantity': ('Active Quantity', 'sum')},
            **{'Total Cost (Actual)': ('Total Cost (Actual)', 'sum')},
            Locations=('Location', _join_unique),
            Licenses=('License', _join_unique),
            **{'Earliest Expiration': ('Expiration Date', 'min')},
        )

        grouped = grouped.drop(columns=['_group_key'], errors='ignore')
        return grouped

    except Exception as e:
        st.error(f"❌ Error loading Beyond Legends data: {str(e)}")
        return None


def save_bl_inputs_data(df: pd.DataFrame, source_filename: str) -> bool:
    """Persist BL input materials to Parquet + a small metadata JSON."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        save_df = df.copy()
        for col in save_df.columns:
            if hasattr(save_df[col], 'cat'):
                save_df[col] = save_df[col].astype(str)
        save_df.to_parquet(BL_INPUTS_PATH, index=False)

        meta = {
            'last_updated': datetime.now().isoformat(),
            'source_file': source_filename,
            'sku_count': int(len(df)),
            'total_value': float(df['Total Cost (Actual)'].sum()) if 'Total Cost (Actual)' in df.columns else 0.0,
        }
        with open(BL_INPUTS_META_PATH, 'w') as f:
            json.dump(meta, f, indent=2)
        return True
    except Exception as e:
        st.warning(f"Could not save Beyond Legends input materials to disk: {str(e)}")
        return False


def load_saved_bl_inputs() -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Load persisted BL input materials + metadata, if present."""
    try:
        if not os.path.exists(BL_INPUTS_PATH):
            return None, None
        df = pd.read_parquet(BL_INPUTS_PATH)
        meta = None
        if os.path.exists(BL_INPUTS_META_PATH):
            with open(BL_INPUTS_META_PATH, 'r') as f:
                meta = json.load(f)
        return df, meta
    except Exception:
        return None, None

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def combine_inventory_data(headset_df: pd.DataFrame, distru_df: pd.DataFrame,
                           brands_list: List[str] = None, filter_to_brands: bool = True) -> Optional[pd.DataFrame]:
    """
    Combine Headset and Distru data to calculate total inventory and WOH.

    Args:
        headset_df: Headset inventory coverage report
        distru_df: Distru inventory assets report
        brands_list: Private label brands list. If None, loads from persistent storage.
        filter_to_brands: If True, filter to brands_list only (PL view).
                          If False, include all brands and add 'Private Label' column.

    Returns:
        Combined DataFrame with calculated metrics or None if processing fails
    """
    try:
        # =====================================================================
        # PROCESS HEADSET DATA
        # =====================================================================
        
        # Resolve brands list
        if brands_list is None:
            brands_list = load_brands_list()
        private_label_brands_normalized = [brand.lower().strip() for brand in brands_list]

        # Normalize brand names for case-insensitive matching
        headset_df_normalized = headset_df.copy()
        headset_df_normalized['Brand Normalized'] = headset_df_normalized['Brand'].str.lower().str.strip()

        if filter_to_brands:
            # Filter to private label brands using normalized names
            headset_filtered = headset_df_normalized[
                headset_df_normalized['Brand Normalized'].isin(private_label_brands_normalized)
            ].copy()
        else:
            # Include all brands
            headset_filtered = headset_df_normalized.copy()

        # Normalize brand names to canonical names (fixes "Pto" → "PTO")
        headset_filtered['Brand'] = headset_filtered['Brand'].apply(
            lambda b: normalize_brand_name(b, brands_list)
        )

        if headset_filtered.empty:
            st.warning("⚠️ No products found in Headset data")
            return None
        
        # Filter to tracked product categories
        pre_category_filter = len(headset_filtered)
        headset_filtered = headset_filtered[
            headset_filtered['Category'].str.lower().str.contains(
                '|'.join(PRODUCT_KEYWORDS), na=False
            )
        ].copy()
        post_category_filter = len(headset_filtered)
        
        if headset_filtered.empty:
            st.warning("⚠️ No tracked products found in Headset data (Flower, Preroll, Vape)")
            return None
        
        # Extract product weights and standardize categories
        headset_filtered[['Product Base Name', 'Weight']] = headset_filtered['Product Name'].apply(
            lambda x: pd.Series(extract_weight_from_product_name(x))
        )
        headset_filtered['Flower Category'] = headset_filtered['Category'].apply(categorize_product_type)
        
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
            st.warning("⚠️ No tracked products with inventory found")
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
        
        # Normalize brand names for case-insensitive matching
        distru_filtered['Brand Normalized'] = distru_filtered['Brand'].str.lower().str.strip()

        if filter_to_brands:
            # Filter to private label brands using normalized names
            distru_filtered = distru_filtered[
                distru_filtered['Brand Normalized'].isin(private_label_brands_normalized)
            ].copy()

        # Normalize brand names to canonical names (fixes "Pto" → "PTO")
        distru_filtered['Brand'] = distru_filtered['Brand'].apply(
            lambda b: normalize_brand_name(b, brands_list)
        )
        
        # Filter to tracked product categories
        if 'Category' in distru_filtered.columns:
            distru_filtered['Flower Category'] = distru_filtered['Category'].apply(categorize_product_type)
            distru_filtered = distru_filtered[
                distru_filtered['Category'].str.lower().str.contains(
                    '|'.join(PRODUCT_KEYWORDS), na=False
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
        
        # Extract Vape keywords for Vape products
        combined['Vape Keywords'] = combined.apply(
            lambda row: extract_vape_keywords(row['Product Name']) if row['Flower Category'] == 'Vape' else '',
            axis=1
        )
        
        # Create Product Group for categorization
        combined['Product Group'] = combined['Brand'] + ' ' + combined['Weight']
        
        # Tag private label products when processing all brands
        if not filter_to_brands:
            combined['Private Label'] = combined['Brand'].str.lower().str.strip().isin(
                private_label_brands_normalized
            )

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

        # Round display values
        for col in ['In Stock Avg Units per Day', 'WOH']:
            if col in summary.columns:
                summary[col] = summary[col].round(1)

        return summary
        
    except Exception as e:
        st.error(f"Error creating Distru stock table: {str(e)}")
        return pd.DataFrame()


def render_bl_inputs_tab(df: Optional[pd.DataFrame], meta: Optional[dict] = None):
    """
    Render the Beyond Legends Input Materials tab.

    v1 shows an aggregated SKU-level view of non-cannabis input materials
    (Packaging + Supplies) with filters and CSV download. No weeks-on-hand or
    reorder points in v1 (no consumption data available yet).
    """
    st.header("🧰 Input Materials, Beyond Legends")

    if df is None or df.empty:
        st.info("👈 Upload the Beyond Legends Distru Inventory Assets CSV in the sidebar to populate this tab.")
        return

    # Freshness caption
    if meta:
        try:
            last_updated = datetime.fromisoformat(meta['last_updated'])
            age_days = (datetime.now() - last_updated).days
            if age_days < 1:
                freshness = "Today"
            elif age_days < 2:
                freshness = "Yesterday"
            else:
                freshness = f"{age_days} days old"
            st.caption(
                f"Data from: {last_updated.strftime('%A %m/%d/%Y %I:%M %p')} ({freshness}) | "
                f"Source: {meta.get('source_file', 'N/A')}"
            )
        except (KeyError, ValueError):
            pass

    # Summary KPIs
    total_value = float(df['Total Cost (Actual)'].sum()) if 'Total Cost (Actual)' in df.columns else 0.0
    zero_low = int((df['Active Quantity'] <= 0).sum()) if 'Active Quantity' in df.columns else 0
    top_vendors = (
        df['Vendor'].value_counts().head(3)
        if 'Vendor' in df.columns else pd.Series(dtype=int)
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SKUs", f"{len(df):,}")
    col2.metric("Inventory Value", f"${total_value:,.0f}")
    with col3:
        st.metric("Zero/Low Stock", f"{zero_low:,}")
    with col4:
        st.markdown("**Top Vendors**")
        if not top_vendors.empty:
            lines = [f"• {v} ({int(c)})" for v, c in top_vendors.items()]
            st.markdown("<br/>".join(lines), unsafe_allow_html=True)
        else:
            st.markdown("—")

    st.markdown("---")

    # Filters
    filter_cols = st.columns(4)

    with filter_cols[0]:
        vendor_options = sorted(df['Vendor'].dropna().unique().tolist())
        selected_vendors = st.multiselect(
            "Vendor", vendor_options, default=vendor_options, key="bl_vendor"
        )

    with filter_cols[1]:
        subcat_options = sorted(df['Subcategory'].dropna().unique().tolist())
        selected_subcats = st.multiselect(
            "Subcategory", subcat_options, default=subcat_options, key="bl_subcat"
        )

    # Locations are comma-joined in aggregation; split for the filter list
    all_locs = set()
    for s in df['Locations'].dropna():
        for loc in str(s).split(','):
            loc = loc.strip()
            if loc:
                all_locs.add(loc)
    loc_options = sorted(all_locs)

    with filter_cols[2]:
        selected_locs = st.multiselect(
            "Location", loc_options, default=loc_options, key="bl_loc"
        )

    with filter_cols[3]:
        search_term = st.text_input("Search product name", key="bl_search")

    # Apply filters
    filtered = df.copy()
    if selected_vendors:
        filtered = filtered[filtered['Vendor'].isin(selected_vendors)]
    if selected_subcats:
        filtered = filtered[filtered['Subcategory'].isin(selected_subcats)]
    if selected_locs and selected_locs != loc_options:
        loc_pattern = '|'.join(pd.Series(selected_locs).map(lambda s: s.replace('.', r'\.')))
        filtered = filtered[filtered['Locations'].str.contains(loc_pattern, na=False, regex=True)]
    if search_term:
        filtered = filtered[filtered['Product'].str.contains(search_term, case=False, na=False)]

    # Build display table
    display_cols = [
        'Product', 'Subcategory', 'Vendor', 'Locations',
        'Active Quantity', 'Unit Type', 'Unit Cost (Actual)', 'Total Cost (Actual)'
    ]
    # Conditionally append expiration column
    has_expiry = 'Earliest Expiration' in filtered.columns and filtered['Earliest Expiration'].notna().any()
    if has_expiry:
        display_cols.append('Earliest Expiration')

    display_df = filtered[display_cols].copy()
    display_df = display_df.sort_values('Total Cost (Actual)', ascending=False)

    # Tidy numeric formatting. Show Active Quantity as integer when all values are whole.
    if not display_df.empty:
        qty = display_df['Active Quantity']
        if ((qty.fillna(0) % 1) == 0).all():
            display_df['Active Quantity'] = qty.fillna(0).astype(int)

    zero_cost_ct = int((filtered['Total Cost (Actual)'] <= 0).sum()) if 'Total Cost (Actual)' in filtered.columns else 0
    st.caption(
        f"{len(filtered):,} of {len(df):,} SKUs shown. "
        f"{zero_cost_ct} rows have $0 cost (often service or test items, left in view)."
    )

    st.dataframe(format_dataframe(display_df), use_container_width=True, hide_index=True, height=600)

    # Download button
    timestamp = datetime.now().strftime('%Y%m%d')
    csv_buffer = io.StringIO()
    display_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download filtered CSV",
        data=csv_buffer.getvalue(),
        file_name=f"bl_input_materials_{timestamp}.csv",
        mime="text/csv",
        key="bl_download"
    )


def create_expandable_product_summary(df: pd.DataFrame):
    """Create expandable product summary with drill-down by category, with Vape breakdown by keywords"""
    
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
            for col in ['Total Inventory', 'Distru Quantity']:
                if col in category_summary.columns:
                    category_summary[col] = category_summary[col].astype(int)
            for col in ['Daily Sales', 'WOH']:
                if col in category_summary.columns:
                    category_summary[col] = category_summary[col].round(1)
            
            st.markdown("**📊 By Category:**")
            st.dataframe(format_dataframe(category_summary), use_container_width=True, hide_index=True)
            
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
                    
                    # Special handling for Vape - break down by keywords
                    if category == 'Vape':
                        st.markdown(f"    **💨 {category} ({total_count} products, {distro_count} at Distro)**")
                        
                        # Group vape products by their keywords
                        vape_by_keywords = category_products.groupby('Vape Keywords')
                        
                        # Sort by keywords - put empty keywords last
                        sorted_keyword_groups = sorted(vape_by_keywords.groups.keys(), 
                                                      key=lambda x: (x == '', x))
                        
                        for vape_keywords in sorted_keyword_groups:
                            keyword_products = vape_by_keywords.get_group(vape_keywords)
                            keyword_distro_count = len(keyword_products[keyword_products['Distru Quantity'] > 0])
                            keyword_total_count = len(keyword_products)
                            
                            # Display label
                            if vape_keywords:
                                keyword_label = f"Vape - {vape_keywords}"
                            else:
                                keyword_label = "Vape - No Keywords"
                            
                            with st.expander(f"        {keyword_label} ({keyword_total_count} products, {keyword_distro_count} at Distro)"):
                                display_columns = [
                                    'Product Name', 'Total Inventory', 'Distru Quantity', 
                                    'In Stock Avg Units per Day', 'WOH', 'Store Count'
                                ]
                                
                                display_df = keyword_products[display_columns].copy()
                                display_df = display_df.rename(columns={'In Stock Avg Units per Day': 'Daily Sales'})
                                
                                for col in ['Total Inventory', 'Distru Quantity', 'Store Count']:
                                    if col in display_df.columns:
                                        display_df[col] = display_df[col].astype(int)
                                for col in ['Daily Sales', 'WOH']:
                                    if col in display_df.columns:
                                        display_df[col] = display_df[col].round(1)

                                st.dataframe(format_dataframe(display_df), use_container_width=True, hide_index=True)
                    else:
                        # Non-Vape categories - display normally
                        with st.expander(f"    {category} ({total_count} products, {distro_count} at Distro)"):
                            display_columns = [
                                'Product Name', 'Total Inventory', 'Distru Quantity',
                                'In Stock Avg Units per Day', 'WOH', 'Store Count'
                            ]

                            display_df = category_products[display_columns].copy()
                            display_df = display_df.rename(columns={'In Stock Avg Units per Day': 'Daily Sales'})

                            for col in ['Total Inventory', 'Distru Quantity', 'Daily Sales', 'WOH']:
                                if col in display_df.columns:
                                    display_df[col] = display_df[col].round(1)

                            st.dataframe(format_dataframe(display_df), use_container_width=True, hide_index=True)

# =============================================================================
# AUTO-LOAD SAVED DATA
# =============================================================================

if st.session_state.combined_data is None:
    saved_df, saved_meta = load_saved_data()
    if saved_df is not None:
        # Ensure flag columns exist (handles pre-v3 saved data)
        if 'WOH Flag' not in saved_df.columns:
            saved_df = calculate_production_flags(saved_df)
        st.session_state.combined_data = saved_df
        st.session_state.saved_metadata = saved_meta

if st.session_state.all_products_data is None:
    all_products_df = load_saved_all_products()
    if all_products_df is not None:
        if 'WOH Flag' not in all_products_df.columns:
            all_products_df = calculate_production_flags(all_products_df)
        st.session_state.all_products_data = all_products_df

if st.session_state.bl_inputs_data is None:
    saved_bl, saved_bl_meta = load_saved_bl_inputs()
    if saved_bl is not None:
        st.session_state.bl_inputs_data = saved_bl
        st.session_state.bl_inputs_metadata = saved_bl_meta

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

# Beyond Legends Input Materials (standalone - processes immediately on upload)
st.sidebar.subheader("🧰 Beyond Legends Input Materials")
bl_inputs_file = st.sidebar.file_uploader(
    "Choose Beyond Legends Distru CSV", type=['csv'], key="bl_inputs_upload",
    help="Upload the Distru Inventory Assets export from the Beyond Legends account. "
         "Filters to Packaging + Supplies categories and aggregates by SKU."
)

if bl_inputs_file is not None:
    with st.spinner("Parsing Beyond Legends input materials..."):
        bl_parsed = load_bl_input_materials(bl_inputs_file)
        if bl_parsed is not None and not bl_parsed.empty:
            st.session_state.bl_inputs_data = bl_parsed
            saved_ok = save_bl_inputs_data(bl_parsed, bl_inputs_file.name)
            if saved_ok:
                st.session_state.bl_inputs_metadata = {
                    'last_updated': datetime.now().isoformat(),
                    'source_file': bl_inputs_file.name,
                    'sku_count': int(len(bl_parsed)),
                    'total_value': float(bl_parsed['Total Cost (Actual)'].sum()),
                }
            st.sidebar.success(f"✅ {len(bl_parsed):,} input-material SKUs loaded")

# Private Label Brands Management
st.sidebar.markdown("---")
st.sidebar.subheader("🏷️ Private Label Brands")

# Load current brands list into session state if not already loaded
if st.session_state.private_label_brands is None:
    st.session_state.private_label_brands = load_brands_list()

current_brands = st.session_state.private_label_brands

# Upload brands CSV
brands_csv = st.sidebar.file_uploader(
    "Update Brands List (Distru CSV)", type=['csv'], key="brands_upload",
    help="Upload a Distru companies export CSV to update the private label brands list"
)

if brands_csv is not None:
    parsed_brands = parse_brands_csv(brands_csv)
    if parsed_brands is not None:
        if save_brands_list(parsed_brands):
            st.session_state.private_label_brands = parsed_brands
            current_brands = parsed_brands
            st.sidebar.success(f"Updated to {len(parsed_brands)} brands")

st.sidebar.caption(f"Tracking {len(current_brands)} brands:")
with st.sidebar.expander("View Brands"):
    for brand in current_brands:
        st.markdown(f"• {brand}")

# Auto-process the PL pair when both Headset + Distru are uploaded and the
# file signature has changed since the last processing pass. Streamlit reruns
# the whole script on every widget interaction, so gating on signature change
# avoids re-running the combine on every keystroke / click.
if headset_file is not None and distru_file is not None:
    current_pl_sig = (
        headset_file.name, getattr(headset_file, 'size', None),
        distru_file.name, getattr(distru_file, 'size', None),
    )
    if st.session_state.get('last_pl_sig') != current_pl_sig:
        with st.spinner("Processing Headset + Distru data..."):
            headset_df = load_headset_data(headset_file)
            distru_df = load_distru_data(distru_file)

            if headset_df is None or distru_df is None:
                st.error("❌ Failed to load one or more files")
            else:
                st.session_state.headset_data = headset_df
                st.session_state.distru_data = distru_df

                brands_list = load_brands_list()
                combined_data = combine_inventory_data(
                    headset_df, distru_df, brands_list=brands_list, filter_to_brands=True
                )

                if combined_data is not None:
                    combined_data = calculate_production_flags(combined_data)
                    st.session_state.combined_data = combined_data

                    saved = save_processed_data(
                        combined_data, headset_file.name, distru_file.name,
                        len(headset_df), len(distru_df)
                    )
                    if saved:
                        st.session_state.saved_metadata = {
                            'last_updated': datetime.now().isoformat(),
                            'headset_file': headset_file.name,
                            'distru_file': distru_file.name,
                            'headset_rows': len(headset_df),
                            'distru_rows': len(distru_df),
                            'product_count': len(combined_data),
                        }

                    all_products = combine_inventory_data(
                        headset_df, distru_df, brands_list=brands_list, filter_to_brands=False
                    )
                    if all_products is not None:
                        all_products = calculate_production_flags(all_products)
                        st.session_state.all_products_data = all_products
                        save_all_products_data(all_products)

                    st.sidebar.success(
                        f"✅ {len(combined_data):,} private label products "
                        f"({len(all_products) if all_products is not None else 0:,} total)"
                    )
                    st.session_state['last_pl_sig'] = current_pl_sig
elif headset_file is not None or distru_file is not None:
    missing = []
    if headset_file is None:
        missing.append("Headset")
    if distru_file is None:
        missing.append("Distru")
    st.sidebar.info(f"ℹ️ Upload {' + '.join(missing)} to process PL data.")

# Sidebar - Version Info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 App Info")
st.sidebar.markdown(f"**Version:** {st.session_state.app_version}")
st.sidebar.markdown("**Last Updated:** April 2026")

# Main Content Area
if st.session_state.combined_data is not None or st.session_state.bl_inputs_data is not None:
    combined_df = st.session_state.combined_data  # may be None if only BL data is loaded

    # Display data freshness
    if st.session_state.saved_metadata:
        meta = st.session_state.saved_metadata
        try:
            last_updated = datetime.fromisoformat(meta['last_updated'])
            age = datetime.now() - last_updated
            days_old = age.days
            if days_old < 1:
                freshness = "Today"
            elif days_old < 2:
                freshness = "Yesterday"
            else:
                freshness = f"{days_old} days old"
            st.caption(
                f"Data from: {last_updated.strftime('%A %m/%d/%Y %I:%M %p')} ({freshness}) | "
                f"Sources: {meta.get('headset_file', 'N/A')}, {meta.get('distru_file', 'N/A')}"
            )
        except (KeyError, ValueError):
            pass

    # Create tabs. If the PL combined dataset is loaded, show the full 6-tab
    # layout. Otherwise (BL-only upload) show just the BL Input Materials tab.
    if combined_df is not None:
        tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🚨 Private Label Production Alerts",
            "📊 Private Label Performance & Stock Dashboard",
            "🎯 Product Analysis",
            "📦 Distru Stock",
            "🧰 Input Materials (BL)",
            "📋 Raw Data"
        ])
    else:
        (bl_only_tab,) = st.tabs(["🧰 Input Materials (BL)"])
        with bl_only_tab:
            render_bl_inputs_tab(
                st.session_state.bl_inputs_data,
                st.session_state.bl_inputs_metadata,
            )
        st.info(
            "Showing Beyond Legends input materials only. "
            "Upload the Headset + Distru CSVs in the sidebar to unlock the full dashboard "
            "(production alerts, performance, Distru stock, raw data). Processing is automatic."
        )
        st.stop()

    # =========================================================================
    # ALERTS TAB
    # =========================================================================
    with tab0:
        st.header("🚨 Private Label Production Alerts")

        # Build combined alerts report
        woh_alerts = get_woh_alerts(combined_df)
        distro_gaps = get_distro_coverage_gaps(combined_df)

        # Merge WOH alerts and distro gaps into one table
        # Start with full Brand+Weight+Category summary
        full_summary = combined_df.groupby(['Brand', 'Weight', 'Flower Category']).agg({
            'Total Inventory': 'sum',
            'Distru Quantity': 'sum',
            'In Stock Avg Units per Day': 'sum',
            'Product Name': 'nunique'
        }).reset_index()

        full_summary['WOH'] = full_summary.apply(
            lambda row: calculate_woh(row['Total Inventory'], row['In Stock Avg Units per Day']),
            axis=1
        )

        def assign_flag(woh):
            if woh < WOH_CRITICAL:
                return 'CRITICAL'
            elif woh < WOH_URGENT:
                return 'URGENT'
            elif woh < WOH_WARNING:
                return 'WARNING'
            return ''

        full_summary['WOH Alert'] = full_summary['WOH'].apply(assign_flag)
        full_summary['Distro Gap'] = full_summary['Distru Quantity'].apply(lambda x: 'NO STOCK' if x == 0 else '')

        # Filter to only rows with at least one alert
        alerts_combined = full_summary[
            (full_summary['WOH Alert'] != '') | (full_summary['Distro Gap'] != '')
        ].copy()

        alerts_combined.rename(columns={
            'Product Name': 'Products',
            'In Stock Avg Units per Day': 'Daily Sales'
        }, inplace=True)

        # Sort by Brand, Weight, Category
        alerts_combined = alerts_combined.sort_values(['Brand', 'Weight', 'Flower Category'])

        # Format numbers
        alerts_combined['Total Inventory'] = alerts_combined['Total Inventory'].astype(int)
        alerts_combined['Distru Quantity'] = alerts_combined['Distru Quantity'].astype(int)
        alerts_combined['Daily Sales'] = alerts_combined['Daily Sales'].apply(lambda x: f"{x:.1f}")
        alerts_combined['WOH'] = alerts_combined['WOH'].apply(lambda x: f"{x:.1f}")

        # Metric cards
        critical_count = len(alerts_combined[alerts_combined['WOH Alert'] == 'CRITICAL'])
        urgent_count = len(alerts_combined[alerts_combined['WOH Alert'] == 'URGENT'])
        warning_count = len(alerts_combined[alerts_combined['WOH Alert'] == 'WARNING'])
        gap_count = len(alerts_combined[alerts_combined['Distro Gap'] == 'NO STOCK'])

        acol1, acol2, acol3, acol4 = st.columns(4)
        with acol1:
            st.metric("Critical (< 2 wks)", critical_count)
        with acol2:
            st.metric("Urgent (< 4 wks)", urgent_count)
        with acol3:
            st.metric("Warning (< 8 wks)", warning_count)
        with acol4:
            st.metric("Distro Gaps", gap_count)

        if not alerts_combined.empty:
            display_cols = ['Brand', 'Weight', 'Flower Category', 'Products',
                           'Total Inventory', 'Distru Quantity', 'Daily Sales', 'WOH',
                           'WOH Alert', 'Distro Gap']
            display_alerts = alerts_combined[display_cols].copy()

            # Color styling based on WOH Alert
            flag_colors = {
                'CRITICAL': 'background-color: #ffcccc',
                'URGENT': 'background-color: #ffe0cc',
                'WARNING': 'background-color: #fff3cc',
                '': ''
            }

            def apply_alert_style(row):
                woh_flag = row.get('WOH Alert', '')
                distro_flag = row.get('Distro Gap', '')
                # Prioritize WOH alert color, fall back to distro gap color
                if woh_flag:
                    style = flag_colors.get(woh_flag, '')
                elif distro_flag:
                    style = 'background-color: #ffe0cc'
                else:
                    style = ''
                return [style] * len(row)

            styled = display_alerts.style.apply(apply_alert_style, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # PDF export
            st.subheader("Export")
            report_date = datetime.now().strftime('%m/%d/%Y')
            meta_line = ""
            if st.session_state.saved_metadata:
                try:
                    lu = datetime.fromisoformat(st.session_state.saved_metadata['last_updated'])
                    meta_line = f"Data from: {lu.strftime('%A %m/%d/%Y %I:%M %p')}"
                except (KeyError, ValueError):
                    pass

            # Build HTML for PDF
            rows_html = ""
            for _, row in display_alerts.iterrows():
                woh_flag = row['WOH Alert']
                distro_flag = row['Distro Gap']
                if woh_flag == 'CRITICAL':
                    bg = '#ffcccc'
                elif woh_flag == 'URGENT':
                    bg = '#ffe0cc'
                elif woh_flag == 'WARNING':
                    bg = '#fff3cc'
                elif distro_flag:
                    bg = '#ffe0cc'
                else:
                    bg = '#ffffff'
                rows_html += f"""<tr style="background-color:{bg}">
                    <td>{row['Brand']}</td><td>{row['Weight']}</td><td>{row['Flower Category']}</td>
                    <td style="text-align:right">{row['Products']}</td>
                    <td style="text-align:right">{row['Total Inventory']}</td>
                    <td style="text-align:right">{row['Distru Quantity']}</td>
                    <td style="text-align:right">{row['Daily Sales']}</td>
                    <td style="text-align:right">{row['WOH']}</td>
                    <td>{row['WOH Alert']}</td><td>{row['Distro Gap']}</td>
                </tr>"""

            pdf_html = f"""<!DOCTYPE html>
<html><head><style>
    body {{ font-family: Arial, sans-serif; margin: 20px; font-size: 11px; }}
    h1 {{ font-size: 18px; margin-bottom: 4px; }}
    .meta {{ color: #666; margin-bottom: 12px; font-size: 10px; }}
    .summary {{ margin-bottom: 12px; }}
    .summary span {{ display: inline-block; padding: 4px 10px; margin-right: 8px; border-radius: 3px; font-weight: bold; font-size: 11px; }}
    .critical {{ background: #ffcccc; }}
    .urgent {{ background: #ffe0cc; }}
    .warning {{ background: #fff3cc; }}
    .gap {{ background: #e0e0e0; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th {{ background: #333; color: white; padding: 6px 8px; text-align: left; font-size: 10px; }}
    td {{ padding: 4px 8px; border-bottom: 1px solid #ddd; font-size: 10px; }}
    @media print {{
        body {{ margin: 10px; }}
        * {{ -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; color-adjust: exact !important; }}
    }}
</style></head><body>
<h1>Private Label Production Alerts</h1>
<div class="meta">{report_date} | {meta_line}</div>
<div class="summary">
    <span class="critical">Critical: {critical_count}</span>
    <span class="urgent">Urgent: {urgent_count}</span>
    <span class="warning">Warning: {warning_count}</span>
    <span class="gap">Distro Gaps: {gap_count}</span>
</div>
<table>
<tr><th>Brand</th><th>Weight</th><th>Category</th><th>Products</th><th>Total Inv</th><th>Distru Qty</th><th>Daily Sales</th><th>WOH</th><th>WOH Alert</th><th>Distro Gap</th></tr>
{rows_html}
</table>
</body></html>"""

            st.download_button(
                label="📄 Download Alerts Report (HTML/PDF)",
                data=pdf_html,
                file_name=f"pl_production_alerts_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html",
                help="Open in browser and print to PDF, or save directly as HTML"
            )

        else:
            st.success("No production alerts. All brand/weight/category combinations are healthy.")

    # =========================================================================
    # DASHBOARD TAB
    # =========================================================================
    with tab1:
        st.header("📊 Private Label Performance & Stock Dashboard")

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
        
        # Add WOH flags to brand summary
        def assign_brand_flag(woh):
            if woh < WOH_CRITICAL:
                return 'critical'
            elif woh < WOH_URGENT:
                return 'urgent'
            elif woh < WOH_WARNING:
                return 'warning'
            else:
                return 'ok'

        brand_summary['Flag'] = brand_summary['WOH'].apply(assign_brand_flag)

        brand_summary = brand_summary[[
            'Brand', 'Total Products', 'Distro Products', 'Total Inventory',
            'Distru Quantity', 'Daily Sales', 'WOH', 'Total Store Presence', 'Flag'
        ]]
        for col in ['Total Inventory', 'Distru Quantity', 'Total Store Presence']:
            brand_summary[col] = brand_summary[col].astype(int)
        brand_summary['Daily Sales'] = brand_summary['Daily Sales'].round(1)
        brand_summary['WOH'] = brand_summary['WOH'].round(1)
        st.dataframe(style_flag_dataframe(brand_summary), use_container_width=True, hide_index=True)
        
        # Expandable product performance summary
        create_expandable_product_summary(combined_df)
    
    with tab2:
        st.header("🎯 Product Analysis")

        # View mode selector
        view_mode = st.radio(
            "Product View",
            ["Private Label", "All Products", "Non-Private Label"],
            horizontal=True
        )

        # Select data source based on view mode
        if view_mode == "Private Label":
            analysis_df = combined_df
        else:
            all_products = st.session_state.all_products_data
            if all_products is None:
                st.info("Upload and process new data to view non-private-label products.")
                analysis_df = None
            elif view_mode == "Non-Private Label":
                analysis_df = all_products[all_products['Private Label'] == False].copy()
            else:
                analysis_df = all_products

        if analysis_df is not None and not analysis_df.empty:
            # Filters
            col1, col2, col3 = st.columns(3)

            with col1:
                selected_brands = st.multiselect(
                    "Select Brands",
                    options=sorted(analysis_df['Brand'].unique()),
                    default=sorted(analysis_df['Brand'].unique()),
                    key=f"brands_{view_mode}"
                )

            with col2:
                selected_categories = st.multiselect(
                    "Select Categories",
                    options=sorted(analysis_df['Flower Category'].unique()),
                    default=sorted(analysis_df['Flower Category'].unique()),
                    key=f"categories_{view_mode}"
                )

            with col3:
                selected_weights = st.multiselect(
                    "Select Weights",
                    options=sorted(analysis_df['Weight'].unique()),
                    default=sorted(analysis_df['Weight'].unique()),
                    key=f"weights_{view_mode}"
                )

            # Apply filters
            filtered_df = analysis_df[
                (analysis_df['Brand'].isin(selected_brands)) &
                (analysis_df['Flower Category'].isin(selected_categories)) &
                (analysis_df['Weight'].isin(selected_weights))
            ]

            if filtered_df.empty:
                st.warning("⚠️ No products match the selected filters")
            else:
                # Sort options
                sort_by = st.selectbox(
                    "Sort by",
                    options=['WOH', 'Total Inventory', 'Distru Quantity', 'Daily Sales', 'Product Name'],
                    index=0,
                    key=f"sort_{view_mode}"
                )

                sort_ascending = st.checkbox("Sort ascending", value=False, key=f"asc_{view_mode}")

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

                # Determine columns to display - include Vape Keywords for Vape products and WOH Flag
                base_columns = [
                    'Product Name', 'Brand', 'Flower Category', 'Weight',
                    'Total Inventory', 'Distru Quantity', 'In Stock Avg Units per Day',
                    'WOH', 'Distru Days Supply', 'Store Count'
                ]

                # Add WOH Flag if available
                if 'WOH Flag' in filtered_df_sorted.columns:
                    base_columns.append('WOH Flag')

                # Check if any vape products are in the filtered data
                has_vape = 'Vape' in filtered_df_sorted['Flower Category'].values

                if has_vape and 'Vape Keywords' in filtered_df_sorted.columns:
                    # Insert Vape Keywords column after Product Name
                    display_columns = base_columns[:1] + ['Vape Keywords'] + base_columns[1:]
                else:
                    display_columns = base_columns

                # Only include columns that exist in the dataframe
                display_columns = [c for c in display_columns if c in filtered_df_sorted.columns]

                display_df = filtered_df_sorted[display_columns].copy()
                display_df = display_df.rename(columns={'In Stock Avg Units per Day': 'Daily Sales', 'WOH Flag': 'Flag'})

                for col in ['Total Inventory', 'Distru Quantity', 'Distru Days Supply', 'Store Count']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].astype(int)
                for col in ['Daily Sales', 'WOH']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].round(1)

                if 'Flag' in display_df.columns:
                    st.dataframe(style_flag_dataframe(display_df), use_container_width=True, height=600, hide_index=True)
                else:
                    st.dataframe(format_dataframe(display_df), use_container_width=True, height=600)
        elif analysis_df is not None and analysis_df.empty:
            st.warning("⚠️ No products found for the selected view")
    
    with tab3:
        st.header("📦 Products in Stock at Distru")

        # Coverage summary
        total_combos = combined_df.groupby(['Brand', 'Weight', 'Flower Category']).size().reset_index()
        distro_combos = combined_df[combined_df['Distru Quantity'] > 0].groupby(['Brand', 'Weight', 'Flower Category']).size().reset_index()
        coverage_pct = (len(distro_combos) / len(total_combos) * 100) if len(total_combos) > 0 else 0
        st.info(f"📊 Distro coverage: {len(distro_combos)} of {len(total_combos)} brand/weight/category combinations have stock ({coverage_pct:.0f}%)")

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
                st.dataframe(format_dataframe(distru_summary_display), use_container_width=True)
            
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
            
            # Check if any vape products are in Distru stock
            has_vape_in_distru = 'Vape' in distru_stock_sorted['Flower Category'].values
            
            if has_vape_in_distru:
                display_columns = [
                    'Product Name', 'Vape Keywords', 'Brand', 'Flower Category', 'Weight',
                    'Distru Quantity', 'Distru Days Supply', 'Total Inventory', 'In Stock Avg Units per Day', 'WOH'
                ]
            else:
                display_columns = [
                    'Product Name', 'Brand', 'Flower Category', 'Weight',
                    'Distru Quantity', 'Distru Days Supply', 'Total Inventory', 'In Stock Avg Units per Day', 'WOH'
                ]
            
            distru_display_df = distru_stock_sorted[display_columns].copy()
            distru_display_df = distru_display_df.rename(columns={'In Stock Avg Units per Day': 'Daily Sales'})
            
            for col in ['Distru Quantity', 'Distru Days Supply', 'Total Inventory']:
                if col in distru_display_df.columns:
                    distru_display_df[col] = distru_display_df[col].astype(int)
            for col in ['Daily Sales', 'WOH']:
                if col in distru_display_df.columns:
                    distru_display_df[col] = distru_display_df[col].round(1)
            
            st.dataframe(format_dataframe(distru_display_df), use_container_width=True, height=600)

    with tab4:
        render_bl_inputs_tab(
            st.session_state.bl_inputs_data,
            st.session_state.bl_inputs_metadata,
        )

    with tab5:
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
        
        st.dataframe(format_dataframe(display_raw_df), use_container_width=True, height=600)

else:
    # Welcome screen
    if not headset_file and not distru_file:
        st.info("👈 Upload CSV files in the sidebar to load new data, or check back after someone has uploaded this week's reports.")

        with st.expander("ℹ️ How it Works", expanded=True):
            st.markdown(f"""
            **📊 Upload** -> **💾 Auto-save** -> **📈 Always Available**

            **Key Features:**
            - 💾 **Persistent data:** Upload once, dashboard stays available for everyone until next upload
            - 🚨 **Production alerts:** Automatic flags for low WOH (warning < {WOH_WARNING} wks, urgent < {WOH_URGENT} wks, critical < {WOH_CRITICAL} wks)
            - 📦 **Distro gap detection:** Flags brand/weight/category combos with no distribution inventory
            - 🔗 Combines Headset and Distru inventory data
            - 🧮 Calculates Weeks on Hand (WOH) with outlier control (capped at {MAX_WOH_WEEKS:.0f} weeks)
            - 🌿 Tracks Flower (Indica/Sativa/Hybrid), Preroll, and Vape products
            - 💨 Vape products broken down by keyword type (originals, dna, live resin, etc.)
            - 📊 Interactive dashboards and filtering
            - 💾 CSV export functionality

            **Data Requirements:**
            - **Headset CSV:** Store Name, Product Name, Brand, Category, Total Quantity on Hand, In Stock Avg Units per Day
            - **Distru CSV:** Product (or Product Name), Active Quantity (skips first 2 metadata rows automatically)

            **Tracked Brands:** {len(load_brands_list())} private label brands (see sidebar)

            **Version:** {st.session_state.app_version}
            """)

    elif headset_file and distru_file:
        st.info("Both files are uploaded. If you don't see the dashboard, check the sidebar for an error above and try re-uploading.")

    else:
        missing_files = []
        if not headset_file:
            missing_files.append("Headset Inventory Coverage")
        if not distru_file:
            missing_files.append("Distru Inventory Assets")

        st.warning(f"📁 Please upload the {' and '.join(missing_files)} CSV file(s) to continue")

# Footer
st.markdown("---")
st.markdown(f"**Haven Current Inventory and Sales v{st.session_state.app_version}** | Built for DC Retail | Tracks {len(load_brands_list())} Private Label Brands")