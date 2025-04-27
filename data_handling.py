# -*- coding: utf-8 -*-
# data_handling.py

import os
import pandas as pd
import numpy as np
import streamlit as st # Required for st.error, st.warning etc. within the function
import traceback

# --- Constants needed by load_ongc_data ---
# Determine the application directory
try:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for environments where __file__ is not defined (e.g., interactive)
    APP_DIR = os.getcwd()
CATALOG_FILENAME = "ongc.csv" # Assuming the filename is constant

def load_ongc_data(catalog_path: str) -> pd.DataFrame | None:
    """Loads, filters, and preprocesses data from the OpenNGC CSV file."""
    # Using hardcoded English for error messages within this function for now.
    # Consider passing the translation dictionary 't' if localization is needed here.
    required_cols = ['Name', 'RA', 'Dec', 'Type']
    mag_cols = ['V-Mag', 'B-Mag', 'Mag'] # Prioritize V-Mag, then B-Mag, then generic Mag
    size_col = 'MajAx' # Major Axis for size

    try:
        # Check if the catalog file exists
        if not os.path.exists(catalog_path):
             st.error(f"Error loading catalog: File not found at {catalog_path}")
             st.info(f"Please ensure the file '{CATALOG_FILENAME}' is in the directory: {APP_DIR}") # Uses constants
             return None

        # Read the CSV file
        df = pd.read_csv(catalog_path, sep=';', comment='#', low_memory=False)

        # --- Data Cleaning and Validation ---

        # Check essential columns first
        missing_req_cols = [col for col in required_cols if col not in df.columns]
        if missing_req_cols:
            st.error(f"Missing required columns in catalog '{os.path.basename(catalog_path)}': {', '.join(missing_req_cols)}")
            return None

        # --- Process Coordinates (Strings) ---
        df['RA_str'] = df['RA'].astype(str).str.strip()
        df['Dec_str'] = df['Dec'].astype(str).str.strip()
        df.dropna(subset=['RA_str', 'Dec_str'], inplace=True)
        df = df[df['RA_str'] != '']
        df = df[df['Dec_str'] != '']

        # --- Process Magnitude ---
        mag_col_found = None
        for col in mag_cols:
            if col in df.columns:
                numeric_mags = pd.to_numeric(df[col], errors='coerce')
                if numeric_mags.notna().any():
                    mag_col_found = col
                    print(f"Using magnitude column: {mag_col_found}")
                    break

        if mag_col_found is None:
            st.error(f"No usable magnitude column ({', '.join(mag_cols)}) found with valid numeric data in catalog.")
            return None

        df['Mag'] = pd.to_numeric(df[mag_col_found], errors='coerce')
        df.dropna(subset=['Mag'], inplace=True)

        # --- Process Size Column ---
        if size_col not in df.columns:
            st.warning(f"Size column '{size_col}' not found in catalog. Angular size filtering will be disabled.")
            df[size_col] = np.nan
        else:
            df[size_col] = pd.to_numeric(df[size_col], errors='coerce')
            if not df[size_col].notna().any():
                st.warning(f"No valid numeric data found in size column '{size_col}' after cleaning. Size filter disabled.")
                df[size_col] = np.nan

        # --- Filter by Object Type ---
        dso_types_provided = ['Galaxy', 'Globular Cluster', 'Open Cluster', 'Nebula',
                              'Planetary Nebula', 'Supernova Remnant', 'HII', 'Emission Nebula',
                              'Reflection Nebula', 'Cluster + Nebula', 'Gal', 'GCl', 'Gx', 'OC',
                              'PN', 'SNR', 'Neb', 'EmN', 'RfN', 'C+N', 'Gxy', 'AGN', 'MWSC', 'OCl']
        type_pattern = '|'.join(dso_types_provided)

        if 'Type' in df.columns:
            df_filtered = df[df['Type'].astype(str).str.contains(type_pattern, case=False, na=False)].copy()
        else:
            st.error("Catalog is missing the required 'Type' column.")
            return None

        # --- Select Final Columns ---
        final_cols = ['Name', 'RA_str', 'Dec_str', 'Mag', 'Type', size_col]
        final_cols_exist = [col for col in final_cols if col in df_filtered.columns]
        df_final = df_filtered[final_cols_exist].copy()

        # --- Final Cleanup ---
        df_final.drop_duplicates(subset=['Name'], inplace=True, keep='first')
        df_final.reset_index(drop=True, inplace=True)

        if not df_final.empty:
            print(f"Catalog loaded and processed: {len(df_final)} objects.")
            return df_final
        else:
            st.warning("Catalog file loaded, but no matching objects found after filtering.")
            return None

    except FileNotFoundError:
        st.error(f"Error loading catalog: File not found at {catalog_path}")
        st.info(f"Please ensure the file '{CATALOG_FILENAME}' is in the directory: {APP_DIR}") # Uses constants
        return None
    except pd.errors.ParserError as e:
        st.error(f"Error parsing catalog file '{os.path.basename(catalog_path)}': {e}")
        st.info("Please ensure the file is a valid CSV with ';' separator.")
        return None
    except Exception as e:
        st.error(f"Error loading catalog: An unexpected error occurred: {e}")
        traceback.print_exc()
        return None