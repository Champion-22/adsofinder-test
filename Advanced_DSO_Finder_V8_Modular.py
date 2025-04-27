# -*- coding: utf-8 -*-
# --- Basic Imports ---
from __future__ import annotations
import streamlit as st
import random
from datetime import datetime, date, time, timedelta, timezone
# import io # Removed unused import
import traceback
import os  # Needed for file path joining
import urllib.parse # Needed for robust URL encoding
import pandas as pd
import math # For isnan check

# --- Library Imports (Try after set_page_config) ---
try:
    # Attempt to import necessary libraries
    from astropy.time import Time
    import numpy as np
    import astropy.units as u
    from astropy.coordinates import EarthLocation, SkyCoord, get_sun, AltAz, get_constellation
    from astroplan import Observer
    # from astroplan.constraints import AtNightConstraint # Not strictly needed for current logic
    from astroplan.moon import moon_illumination
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pytz
    from timezonefinder import TimezoneFinder
    from geopy.geocoders import Nominatim, ArcGIS, Photon
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
except ImportError as e:
    # Display error and stop if libraries are missing
    st.error(f"Error: Missing libraries. Please install the required packages. ({e})")
    st.stop()

# --- Import Custom Modules ---
try:
    from localization import translations
    # <<< ADDED IMPORT FOR ASTRO CALCULATIONS >>>
    import astro_calculations
except ModuleNotFoundError as e:
    st.error(f"Error: Could not find a required module file ({e}). Make sure 'localization.py' and 'astro_calculations.py' exist in the same directory.")
    st.stop()
    from astro_calculations import CARDINAL_DIRECTIONS


# --- Page Config (MUST BE FIRST Streamlit command) ---
st.set_page_config(page_title="Advanced DSO Finder", layout="wide")

# --- Global Configuration & Initial Values (Consider moving more to config.py later) ---
INITIAL_LAT = 47.17
INITIAL_LON = 8.01
INITIAL_HEIGHT = 550
INITIAL_TIMEZONE = "Europe/Zurich"

# --- Path to Catalog File ---
try:
    # Determine the application directory
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for environments where __file__ is not defined (e.g., interactive)
    APP_DIR = os.getcwd()
CATALOG_FILENAME = "ongc.csv"
CATALOG_FILEPATH = os.path.join(APP_DIR, CATALOG_FILENAME)

# --- Constants (Consider moving to config.py) ---
# Define cardinal directions (moved here for now, as azimuth_to_direction moved)
# CARDINAL_DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"] # Moved to astro_calculations.py
ALL_DIRECTIONS_KEY = 'All' # Internal key for 'All' option

# --- Initialize TimezoneFinder (cached) ---
@st.cache_resource
def get_timezone_finder():
    """Initializes and returns a TimezoneFinder instance."""
    if TimezoneFinder:
        try:
            # Initialize TimezoneFinder, loading data into memory
            return TimezoneFinder(in_memory=True)
        except Exception as e:
            # Log and warn if initialization fails
            print(f"Error initializing TimezoneFinder: {e}")
            st.warning(f"TimezoneFinder init failed: {e}. Automatic timezone detection disabled.")
            return None
    return None

# Get the TimezoneFinder instance
tf = get_timezone_finder()

# --- Initialize Session State ---
def initialize_session_state():
    """Initializes all required session state keys if they don't exist."""
    defaults = {
        'language': 'de', # Default to German
        'plot_object_name': None,
        'show_plot': False,
        'active_result_plot_data': None,
        'last_results': [],
        'find_button_pressed': False,
        'location_choice_key': 'Search',
        'manual_lat_val': INITIAL_LAT,
        'manual_lon_val': INITIAL_LON,
        'manual_height_val': INITIAL_HEIGHT,
        'location_search_query': "",
        'searched_location_name': None,
        'location_search_status_msg': "",
        'location_search_success': False,
        'selected_timezone': INITIAL_TIMEZONE,
        'manual_min_mag_slider': 0.0,
        'manual_max_mag_slider': 16.0,
        'object_type_filter_exp': [],
        'mag_filter_mode_exp': 'Bortle Scale', # Matches key in translations['de']
        'bortle_slider': 5,
        'min_alt_slider': 20,
        'max_alt_slider': 90, # Added max altitude default
        'moon_phase_slider': 35,
        'size_arcmin_range': [1.0, 120.0],
        'sort_method': 'Duration & Altitude', # Matches key in translations['de']
        'selected_peak_direction': ALL_DIRECTIONS_KEY,
        'plot_type_selection': 'Sky Path', # Matches key in translations['de']
        'custom_target_ra': "",
        'custom_target_dec': "",
        'custom_target_name': "",
        'custom_target_error': "",
        'custom_target_plot_data': None,
        'show_custom_plot': False,
        'expanded_object_name': None,
        'location_is_valid_for_run': False,
        'time_choice_exp': 'Now',
        'window_start_time': None,
        'window_end_time': None,
        'selected_date_widget': date.today()
    }
    # Set default values in session state if keys don't exist
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# --- Helper Functions REMAINING in main script ---
# <<< REMOVED definition of get_magnitude_limit >>>
# <<< REMOVED definition of azimuth_to_direction >>>

# --- Moon Phase SVG (Corrected) ---
def create_moon_phase_svg(illumination: float, size: int = 100) -> str:
    """
    Creates an SVG representation of the moon phase (corrected).

    Args:
        illumination (float): Moon illumination fraction (0=new, 0.5=half, 1=full).
        size (int): Size of the SVG image in pixels.

    Returns:
        str: SVG string representing the moon phase.
    """
    # Validate and clamp illumination value
    if not 0 <= illumination <= 1:
        print(f"Warning: Invalid moon illumination value ({illumination}). Clamping to [0, 1].")
        illumination = max(0.0, min(1.0, illumination))

    radius = size / 2
    cx = cy = radius
    # Use CSS variables for theme compatibility (fallback provided)
    light_color = "var(--text-color, #e0e0e0)" # Use text color for light part
    dark_color = "var(--secondary-background-color, #333333)" # Use secondary bg for dark part

    svg = f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'

    # Draw the background (always the dark side color)
    svg += f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{dark_color}"/>'

    if illumination < 0.01: # New moon - only dark circle needed
        pass
    elif illumination > 0.99: # Full moon - draw full light circle on top
        svg += f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{light_color}"/>'
    else:
        # Calculate the horizontal position of the terminator's center relative to the moon's center
        # x varies from -radius (new moon side) to +radius (full moon side)
        x_terminator_center = radius * (illumination * 2 - 1)

        # Calculate the semi-major axis of the ellipse forming the terminator
        rx_terminator = abs(x_terminator_center)

        # Determine sweep flags for the arcs
        # Sweep flag 1 means "clockwise" or "positive angle" direction
        # Large arc flag 1 means take the longer path between two points

        # We draw the illuminated portion
        if illumination <= 0.5: # Crescent phase (less than half illuminated)
            # Illuminated part is on the right (assuming waxing for simplicity)
            # Path: Move to top, elliptical arc down (terminator), circular arc up (limb)
            large_arc_flag_ellipse = 0
            sweep_flag_ellipse = 1
            large_arc_flag_circle = 0
            sweep_flag_circle = 1
            d = (f"M {cx},{cy - radius} " # Move to top of circle
                 f"A {rx_terminator},{radius} 0 {large_arc_flag_ellipse},{sweep_flag_ellipse} {cx},{cy + radius} " # Elliptical arc for terminator
                 f"A {radius},{radius} 0 {large_arc_flag_circle},{sweep_flag_circle} {cx},{cy - radius} " # Circular arc for limb
                 "Z")
        else: # Gibbous phase (more than half illuminated)
            # Illuminated part includes the left half circle plus an ellipse on the right
            # Path: Move to top, circular arc down (left limb), elliptical arc up (terminator)
            large_arc_flag_circle = 1 # Take the long way around the circle limb
            sweep_flag_circle = 1
            large_arc_flag_ellipse = 1
            sweep_flag_ellipse = 1
            d = (f"M {cx},{cy - radius} " # Move to top of circle
                 f"A {radius},{radius} 0 {large_arc_flag_circle},{sweep_flag_circle} {cx},{cy + radius} " # Circular arc for left limb
                 f"A {rx_terminator},{radius} 0 {large_arc_flag_ellipse},{sweep_flag_ellipse} {cx},{cy - radius} " # Elliptical arc for terminator
                 "Z")

        svg += f'<path d="{d}" fill="{light_color}"/>'

    svg += '</svg>'
    return svg


def load_ongc_data(catalog_path: str) -> pd.DataFrame | None:
    """Loads, filters, and preprocesses data from the OpenNGC CSV file."""
    # Using hardcoded English for error messages within this function.
    required_cols = ['Name', 'RA', 'Dec', 'Type']
    mag_cols = ['V-Mag', 'B-Mag', 'Mag'] # Prioritize V-Mag, then B-Mag, then generic Mag
    size_col = 'MajAx' # Major Axis for size

    try:
        # Check if the catalog file exists
        if not os.path.exists(catalog_path):
             st.error(f"Error loading catalog: File not found at {catalog_path}")
             st.info(f"Please ensure the file '{CATALOG_FILENAME}' is in the directory: {APP_DIR}")
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
        st.info(f"Please ensure the file '{CATALOG_FILENAME}' is in the directory: {APP_DIR}")
        return None
    except pd.errors.ParserError as e:
        st.error(f"Error parsing catalog file '{os.path.basename(catalog_path)}': {e}")
        st.info("Please ensure the file is a valid CSV with ';' separator.")
        return None
    except Exception as e:
        st.error(f"Error loading catalog: An unexpected error occurred: {e}")
        traceback.print_exc()
        return None


# <<< REMOVED definition of _get_fallback_window >>>
# <<< REMOVED definition of get_observable_window >>>
# <<< REMOVED definition of find_observable_objects >>>

# --- Time Formatting ---
def get_local_time_str(utc_time: Time | None, timezone_str: str) -> tuple[str, str]:
    """
    Converts a UTC Time object to a localized time string, or returns "N/A".

    Args:
        utc_time: UTC time as an astropy Time object, or None.
        timezone_str: Timezone string (e.g., 'Europe/Zurich').

    Returns:
        A tuple containing the localized time string (e.g., '2023-12-24 22:15:30')
        and the timezone name, or ("N/A", "N/A") on error or if utc_time is None.
    """
    if utc_time is None:
        return "N/A", "N/A" # Handle None input gracefully

    if not isinstance(utc_time, Time):
        print(f"Error: utc_time must be an astropy Time object. Got {type(utc_time)}")
        return "N/A", "N/A"

    if not isinstance(timezone_str, str) or not timezone_str:
        print(f"Error: timezone_str must be a non-empty string. Got '{timezone_str}' ({type(timezone_str)})")
        return "N/A", "N/A"

    try:
        # Get the timezone object
        local_tz = pytz.timezone(timezone_str)
        # Convert astropy Time to datetime object with UTC timezone
        utc_dt = utc_time.to_datetime(timezone.utc)
        # Convert to the target local timezone
        local_dt = utc_dt.astimezone(local_tz)
        # Format the local time string
        local_time_str = local_dt.strftime('%Y-%m-%d %H:%M:%S')
        # Get the timezone abbreviation (e.g., CET, CEST)
        tz_display_name = local_dt.tzname()
        if not tz_display_name:
            tz_display_name = local_tz.zone # Fallback to full zone name if abbreviation fails
        return local_time_str, tz_display_name
    except pytz.exceptions.UnknownTimeZoneError:
        # Handle case where the timezone string is invalid
        print(f"Error: Unknown timezone '{timezone_str}'.")
        return utc_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), "UTC (TZ Error)"
    except Exception as e:
        # Handle any other conversion errors
        print(f"Error converting time to local timezone '{timezone_str}': {e}")
        traceback.print_exc()
        return utc_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), "UTC (Conv. Error)"


# --- Main App ---
def main():
    """Main function to run the Streamlit application."""

    # --- Initialize Session State ---
    initialize_session_state() # <<< MUST BE CALLED FIRST

    # --- Get Current Language and Translations ---
    lang = st.session_state.language # Now safe to access
    if lang not in translations:
        lang = 'de' # Default to German if invalid language in state
        st.session_state.language = lang
    t = translations.get(lang, translations['en']) # Use .get() for safety

    # --- Load Catalog Data (Cached) ---
    @st.cache_data
    def cached_load_ongc_data(path): # Simplified: removed lang parameter
        """Cached function to load ONGC data."""
        print(f"Cache miss: Loading ONGC data from {path}")
        return load_ongc_data(path) # Call simplified function

    df_catalog_data = cached_load_ongc_data(CATALOG_FILEPATH)

    # --- Title ---
    st.title("Advanced DSO Finder")

    # --- Object Type Glossary ---
    with st.expander(t.get('object_type_glossary_title', "Object Type Glossary")):
        glossary_items = t.get('object_type_glossary', {})
        if glossary_items:
            col1, col2 = st.columns(2)
            col_index = 0
            sorted_items = sorted(glossary_items.items())
            for abbr, full_name in sorted_items:
                if col_index % 2 == 0:
                    col1.markdown(f"**{abbr}:** {full_name}")
                else:
                    col2.markdown(f"**{abbr}:** {full_name}")
                col_index += 1
        else:
            st.info("Glossary not available for the selected language.")

    st.markdown("---")

    # --- Sidebar ---
    with st.sidebar:
        st.header(t.get('settings_header', "Settings"))

        # Show catalog loaded message or error
        if 'catalog_status_msg' not in st.session_state:
            st.session_state.catalog_status_msg = ""
        if df_catalog_data is not None:
            new_msg = t.get('info_catalog_loaded', "Catalog loaded: {} objects.").format(len(df_catalog_data))
            if st.session_state.catalog_status_msg != new_msg:
                st.success(new_msg)
                st.session_state.catalog_status_msg = new_msg
        else:
            new_msg = "Catalog loading failed. Check file or logs."
            if st.session_state.catalog_status_msg != new_msg:
                st.error(new_msg)
                st.session_state.catalog_status_msg = new_msg


        # --- Language Selector ---
        language_options = {'de': 'Deutsch', 'en': 'English', 'fr': 'Français'}
        lang_keys = list(language_options.keys())
        try:
            current_lang_key_for_index = lang
            if current_lang_key_for_index not in lang_keys:
                current_lang_key_for_index = 'de'
            current_lang_index = lang_keys.index(current_lang_key_for_index)
        except ValueError:
            current_lang_index = 0

        selected_lang_key = st.radio(
            t.get('language_select_label', "Language"),
            options=lang_keys,
            format_func=lambda key: language_options[key],
            key='language_radio',
            index=current_lang_index,
            horizontal=True
        )

        if selected_lang_key != st.session_state.language:
            st.session_state.language = selected_lang_key
            st.session_state.location_search_status_msg = ""
            print(f"Language changed to: {selected_lang_key}, Rerun triggered.")
            st.rerun()


        # --- Location Settings ---
        with st.expander(t.get('location_expander', "📍 Location"), expanded=True):
            location_options_map = {
                'Search': t.get('location_option_search', "Search by Name"),
                'Manual': t.get('location_option_manual', "Enter Manually")
            }
            st.radio(
                t.get('location_select_label', "Select Location Method"),
                options=list(location_options_map.keys()),
                format_func=lambda key: location_options_map[key],
                key="location_choice_key",
                horizontal=True
            )

            lat_val, lon_val, height_val = None, None, None
            location_valid_for_tz = False
            current_location_valid = False

            if st.session_state.location_choice_key == "Manual":
                st.number_input(t.get('location_lat_label', "Latitude (°N)"), min_value=-90.0, max_value=90.0, step=0.01, format="%.4f", key="manual_lat_val")
                st.number_input(t.get('location_lon_label', "Longitude (°E)"), min_value=-180.0, max_value=180.0, step=0.01, format="%.4f", key="manual_lon_val")
                st.number_input(t.get('location_elev_label', "Elevation (meters)"), min_value=-500, step=10, format="%d", key="manual_height_val")
                lat_val = st.session_state.manual_lat_val
                lon_val = st.session_state.manual_lon_val
                height_val = st.session_state.manual_height_val
                if isinstance(lat_val, (int, float)) and isinstance(lon_val, (int, float)) and isinstance(height_val, (int, float)):
                    location_valid_for_tz = True
                    current_location_valid = True
                    st.session_state.location_is_valid_for_run = True
                    if st.session_state.location_search_success:
                        st.session_state.location_search_success = False
                        st.session_state.searched_location_name = None
                        st.session_state.location_search_status_msg = ""
                else:
                    st.warning(t.get('location_error_manual_none', "Manual location fields cannot be empty or invalid."))
                    current_location_valid = False
                    st.session_state.location_is_valid_for_run = False

            elif st.session_state.location_choice_key == "Search":
                with st.form("location_search_form"):
                    st.text_input(t.get('location_search_label', "Enter location name:"), key="location_search_query", placeholder=t.get('location_search_placeholder', "e.g., Berlin, Germany"))
                    st.number_input(t.get('location_elev_label', "Elevation (meters)"), min_value=-500, step=10, format="%d", key="manual_height_val")
                    location_search_form_submitted = st.form_submit_button(t.get('location_search_submit_button', "Find Coordinates"))

                status_placeholder = st.empty()
                if st.session_state.location_search_status_msg:
                    if st.session_state.location_search_success: status_placeholder.success(st.session_state.location_search_status_msg)
                    else: status_placeholder.error(st.session_state.location_search_status_msg)

                if location_search_form_submitted and st.session_state.location_search_query:
                    location = None
                    service_used = None
                    final_error = None
                    query = st.session_state.location_search_query
                    user_agent_str = f"AdvancedDSOFinder/{random.randint(1000, 9999)}/streamlit_app_{datetime.now().timestamp()}"

                    with st.spinner(t.get('spinner_geocoding', "Searching for location...")):
                        try:
                            print("Trying Nominatim...")
                            geolocator = Nominatim(user_agent=user_agent_str)
                            location = geolocator.geocode(query, timeout=10)
                            if location: service_used = "Nominatim"; print("Nominatim success.")
                            else: print("Nominatim returned None.")
                        except (GeocoderTimedOut, GeocoderServiceError) as e:
                            print(f"Nominatim failed: {e}. Trying fallback 1 (ArcGIS).")
                            status_placeholder.info(t.get('location_search_info_fallback', "Nominatim failed, trying fallback service (ArcGIS)..."))
                        except Exception as e:
                            print(f"Nominatim failed unexpectedly: {e}. Trying fallback 1 (ArcGIS).")
                            status_placeholder.info(t.get('location_search_info_fallback', "Nominatim failed, trying fallback service (ArcGIS)..."))
                            final_error = e

                        if not location:
                            try:
                                print("Trying ArcGIS...")
                                fallback_geolocator = ArcGIS(timeout=15)
                                location = fallback_geolocator.geocode(query, timeout=15)
                                if location: service_used = "ArcGIS"; print("ArcGIS success.")
                                else: print("ArcGIS returned None.")
                            except (GeocoderTimedOut, GeocoderServiceError) as e2:
                                print(f"ArcGIS failed: {e2}. Trying fallback 2 (Photon).")
                                status_placeholder.info(t.get('location_search_info_fallback2', "ArcGIS failed, trying 2nd fallback service (Photon)..."))
                                if not final_error: final_error = e2
                            except Exception as e2:
                                print(f"ArcGIS failed unexpectedly: {e2}. Trying fallback 2 (Photon).")
                                status_placeholder.info(t.get('location_search_info_fallback2', "ArcGIS failed, trying 2nd fallback service (Photon)..."))
                                if not final_error: final_error = e2

                        if not location:
                            try:
                                print("Trying Photon...")
                                fallback_geolocator2 = Photon(user_agent=user_agent_str, timeout=15)
                                location = fallback_geolocator2.geocode(query, timeout=15)
                                if location: service_used = "Photon"; print("Photon success.")
                                else:
                                    print("Photon returned None.")
                                    if not final_error: final_error = GeocoderServiceError("All services failed or returned None.")
                            except (GeocoderTimedOut, GeocoderServiceError) as e3:
                                print(f"Photon failed: {e3}. All fallbacks exhausted.")
                                if not final_error: final_error = e3
                            except Exception as e3:
                                print(f"Photon failed unexpectedly: {e3}. All fallbacks exhausted.")
                                if not final_error: final_error = e3

                        if location and service_used:
                            found_lat = location.latitude
                            found_lon = location.longitude
                            found_name = location.address
                            st.session_state.searched_location_name = found_name
                            st.session_state.location_search_success = True
                            st.session_state.manual_lat_val = found_lat
                            st.session_state.manual_lon_val = found_lon
                            coord_str = t.get('location_search_coords', "Lat: {:.4f}, Lon: {:.4f}").format(found_lat, found_lon)
                            if service_used == "Nominatim": st.session_state.location_search_status_msg = f"{t.get('location_search_found', 'Found (Nominatim): {}').format(found_name)}\n({coord_str})"
                            elif service_used == "ArcGIS": st.session_state.location_search_status_msg = f"{t.get('location_search_found_fallback', 'Found via Fallback (ArcGIS): {}').format(found_name)}\n({coord_str})"
                            elif service_used == "Photon": st.session_state.location_search_status_msg = f"{t.get('location_search_found_fallback2', 'Found via 2nd Fallback (Photon): {}').format(found_name)}\n({coord_str})"
                            status_placeholder.success(st.session_state.location_search_status_msg)
                            lat_val = found_lat
                            lon_val = found_lon
                            height_val = st.session_state.manual_height_val
                            location_valid_for_tz = True
                            current_location_valid = True
                            st.session_state.location_is_valid_for_run = True
                        else:
                            st.session_state.location_search_success = False
                            st.session_state.searched_location_name = None
                            if final_error:
                                if isinstance(final_error, GeocoderTimedOut): st.session_state.location_search_status_msg = t.get('location_search_error_timeout', "Geocoding service timed out.")
                                elif isinstance(final_error, GeocoderServiceError): st.session_state.location_search_status_msg = t.get('location_search_error_service', "Geocoding service error: {}").format(final_error)
                                else: st.session_state.location_search_status_msg = t.get('location_search_error_fallback2_failed', "All geocoding services (Nominatim, ArcGIS, Photon) failed: {}").format(final_error)
                            else: st.session_state.location_search_status_msg = t.get('location_search_error_not_found', "Location not found.")
                            status_placeholder.error(st.session_state.location_search_status_msg)
                            current_location_valid = False
                            st.session_state.location_is_valid_for_run = False

                elif st.session_state.location_search_success:
                    lat_val = st.session_state.manual_lat_val
                    lon_val = st.session_state.manual_lon_val
                    height_val = st.session_state.manual_height_val
                    location_valid_for_tz = True
                    current_location_valid = True
                    st.session_state.location_is_valid_for_run = True
                    status_placeholder.success(st.session_state.location_search_status_msg)
                else:
                     current_location_valid = False
                     st.session_state.location_is_valid_for_run = False


            # --- Automatic Timezone Detection ---
            st.markdown("---")
            auto_timezone_msg = ""
            if location_valid_for_tz and lat_val is not None and lon_val is not None:
                if tf:
                    try:
                        found_tz = tf.timezone_at(lng=lon_val, lat=lat_val)
                        if found_tz:
                            pytz.timezone(found_tz)
                            st.session_state.selected_timezone = found_tz
                            auto_timezone_msg = f"{t.get('timezone_auto_set_label', 'Detected Timezone:')} **{found_tz}**"
                        else:
                            st.session_state.selected_timezone = 'UTC'
                            auto_timezone_msg = f"{t.get('timezone_auto_fail_label', 'Timezone:')} **UTC** ({t.get('timezone_auto_fail_msg', 'Could not detect timezone, using UTC.')})"
                    except pytz.UnknownTimeZoneError:
                        st.session_state.selected_timezone = 'UTC'
                        auto_timezone_msg = f"{t.get('timezone_auto_fail_label', 'Timezone:')} **UTC** (Invalid TZ: '{found_tz}')"
                    except Exception as tz_find_e:
                        print(f"Error finding timezone for ({lat_val}, {lon_val}): {tz_find_e}")
                        st.session_state.selected_timezone = 'UTC'
                        auto_timezone_msg = f"{t.get('timezone_auto_fail_label', 'Timezone:')} **UTC** (Error)"
                else:
                    auto_timezone_msg = f"{t.get('timezone_auto_fail_label', 'Timezone:')} **{INITIAL_TIMEZONE}** (Auto-detect N/A)"
                    st.session_state.selected_timezone = INITIAL_TIMEZONE
            else:
                auto_timezone_msg = f"{t.get('timezone_auto_fail_label', 'Timezone:')} **{st.session_state.selected_timezone}** (Location Invalid/Not Set)"
            st.markdown(auto_timezone_msg, unsafe_allow_html=True)


        # --- Time Settings ---
        with st.expander(t.get('time_expander', "⏱️ Time & Timezone"), expanded=False):
            time_options_map = {'Now': t.get('time_option_now', "Now (Upcoming Night)"), 'Specific': t.get('time_option_specific', "Specific Night")}
            st.radio(
                t.get('time_select_label', "Select Time"), options=list(time_options_map.keys()),
                format_func=lambda key: time_options_map[key],
                key="time_choice_exp",
                horizontal=True
            )
            is_time_now = (st.session_state.time_choice_exp == "Now")
            if is_time_now: st.caption(f"Current UTC: {Time.now().iso}")
            else:
                st.date_input(
                    t.get('time_date_select_label', "Select Date:"),
                    value=st.session_state.selected_date_widget,
                    min_value=date.today()-timedelta(days=365*10),
                    max_value=date.today()+timedelta(days=365*2),
                    key='selected_date_widget'
                )


        # --- Filter Settings ---
        with st.expander(t.get('filters_expander', "✨ Filters & Conditions"), expanded=False):
            # Magnitude Filter
            st.markdown(t.get('mag_filter_header', "**Magnitude Filter**"))
            mag_filter_options_map = {'Bortle Scale': t.get('mag_filter_option_bortle', "Bortle Scale"), 'Manual': t.get('mag_filter_option_manual', "Manual")}
            if st.session_state.mag_filter_mode_exp not in mag_filter_options_map: st.session_state.mag_filter_mode_exp = 'Bortle Scale'
            st.radio(t.get('mag_filter_method_label', "Filter Method:"), options=list(mag_filter_options_map.keys()), format_func=lambda key: mag_filter_options_map[key], key="mag_filter_mode_exp", horizontal=True)
            st.slider(t.get('mag_filter_bortle_label', "Bortle Scale:"), min_value=1, max_value=9, key='bortle_slider', help=t.get('mag_filter_bortle_help', "Sky darkness: 1=Excellent Dark, 9=Inner-city Sky"))
            if st.session_state.mag_filter_mode_exp == "Manual":
                st.slider(t.get('mag_filter_min_mag_label', "Min. Magnitude:"), min_value=-5.0, max_value=20.0, step=0.5, format="%.1f", help=t.get('mag_filter_min_mag_help', "Brightest object magnitude to include"), key='manual_min_mag_slider')
                st.slider(t.get('mag_filter_max_mag_label', "Max. Magnitude:"), min_value=-5.0, max_value=20.0, step=0.5, format="%.1f", help=t.get('mag_filter_max_mag_help', "Dimest object magnitude to include"), key='manual_max_mag_slider')
                if isinstance(st.session_state.manual_min_mag_slider, (int, float)) and isinstance(st.session_state.manual_max_mag_slider, (int, float)) and st.session_state.manual_min_mag_slider > st.session_state.manual_max_mag_slider: st.warning(t.get('mag_filter_warning_min_max', "Min. Magnitude is greater than Max. Magnitude!"))

            # Altitude Filter
            st.markdown("---")
            st.markdown(t.get('min_alt_header', "**Object Altitude Above Horizon**"))
            min_alt_val = st.session_state.min_alt_slider
            max_alt_val = st.session_state.max_alt_slider
            if min_alt_val > max_alt_val: st.session_state.min_alt_slider = max_alt_val; min_alt_val = max_alt_val
            st.slider(t.get('min_alt_label', "Min. Object Altitude (°):"), min_value=0, max_value=90, key='min_alt_slider', step=1)
            st.slider(t.get('max_alt_label', "Max. Object Altitude (°):"), min_value=0, max_value=90, key='max_alt_slider', step=1)
            if st.session_state.min_alt_slider > st.session_state.max_alt_slider: st.warning("Min. Höhe ist größer als Max. Höhe!")

            # Moon Filter
            st.markdown("---")
            st.markdown(t.get('moon_warning_header', "**Moon Warning**"))
            st.slider(t.get('moon_warning_label', "Warn if Moon > (% Illumination):"), min_value=0, max_value=100, key='moon_phase_slider', step=5)

            # Object Type Filter
            st.markdown("---")
            st.markdown(t.get('object_types_header', "**Object Types**"))
            all_types = []
            if df_catalog_data is not None and not df_catalog_data.empty:
                try:
                    if 'Type' in df_catalog_data.columns: all_types = sorted(list(df_catalog_data['Type'].dropna().astype(str).unique()))
                    else: st.warning("Catalog is missing the 'Type' column.")
                except Exception as e: st.warning(f"{t.get('object_types_error_extract', 'Could not extract object types from catalog')}: {e}")
            if all_types:
                current_selection_in_state = [sel for sel in st.session_state.object_type_filter_exp if sel in all_types]
                if current_selection_in_state != st.session_state.object_type_filter_exp: st.session_state.object_type_filter_exp = current_selection_in_state
                default_for_widget = current_selection_in_state
                st.multiselect(t.get('object_types_label', "Filter Types (leave empty for all):"), options=all_types, default=default_for_widget, key="object_type_filter_exp")
            else:
                st.info("Object types cannot be determined from the catalog. Type filter disabled.")
                st.session_state.object_type_filter_exp = []

            # Angular Size Filter
            st.markdown("---")
            st.markdown(t.get('size_filter_header', "**Angular Size Filter**"))
            size_col_exists = df_catalog_data is not None and 'MajAx' in df_catalog_data.columns and df_catalog_data['MajAx'].notna().any()
            size_slider_disabled = not size_col_exists
            if size_col_exists:
                try:
                    valid_sizes = df_catalog_data['MajAx'].dropna()
                    min_size_possible = max(0.1, float(valid_sizes.min())) if not valid_sizes.empty else 0.1
                    max_size_possible = float(valid_sizes.max()) if not valid_sizes.empty else 120.0
                    current_min_state, current_max_state = st.session_state.size_arcmin_range
                    clamped_min = max(min_size_possible, min(current_min_state, max_size_possible))
                    clamped_max = min(max_size_possible, max(current_max_state, min_size_possible))
                    if clamped_min > clamped_max: clamped_min = clamped_max
                    if (clamped_min, clamped_max) != st.session_state.size_arcmin_range: st.session_state.size_arcmin_range = (clamped_min, clamped_max)
                    slider_step = 0.1 if max_size_possible <= 20 else (0.5 if max_size_possible <= 100 else 1.0)
                    st.slider(t.get('size_filter_label', "Object Size (arcminutes):"), min_value=min_size_possible, max_value=max_size_possible, step=slider_step, format="%.1f arcmin", key='size_arcmin_range', help=t.get('size_filter_help', "Filter objects by their apparent size (major axis). 1 arcminute = 1/60 degree."), disabled=size_slider_disabled)
                except Exception as size_slider_e: st.error(f"Error setting up size slider: {size_slider_e}"); size_slider_disabled = True
            else: st.info("Angular size data ('MajAx') not found or invalid. Size filter disabled."); size_slider_disabled = True
            if size_slider_disabled: st.slider(t.get('size_filter_label', "Object Size (arcminutes):"), min_value=0.0, max_value=1.0, value=(0.0, 1.0), key='size_arcmin_range_disabled', disabled=True)

            # Direction Filter
            st.markdown("---")
            st.markdown(t.get('direction_filter_header', "**Filter by Cardinal Direction**"))
            all_directions_str = t.get('direction_option_all', "All")
            direction_options_display = [all_directions_str] + CARDINAL_DIRECTIONS # CARDINAL_DIRECTIONS needs to be accessible
            direction_options_internal = [ALL_DIRECTIONS_KEY] + CARDINAL_DIRECTIONS
            current_direction_internal = st.session_state.selected_peak_direction
            if current_direction_internal not in direction_options_internal: current_direction_internal = ALL_DIRECTIONS_KEY; st.session_state.selected_peak_direction = current_direction_internal
            try: current_direction_index = direction_options_internal.index(current_direction_internal)
            except ValueError: current_direction_index = 0
            selected_direction_display = st.selectbox(t.get('direction_filter_label', "Show objects culminating towards:"), options=direction_options_display, index=current_direction_index, key='direction_selectbox')
            selected_internal_value = ALL_DIRECTIONS_KEY
            if selected_direction_display != all_directions_str:
                try: selected_internal_index = direction_options_display.index(selected_direction_display); selected_internal_value = direction_options_internal[selected_internal_index]
                except ValueError: selected_internal_value = ALL_DIRECTIONS_KEY
            if selected_internal_value != st.session_state.selected_peak_direction: st.session_state.selected_peak_direction = selected_internal_value


        # --- Result Options ---
        with st.expander(t.get('results_options_expander', "⚙️ Result Options"), expanded=False):
            max_slider_val = len(df_catalog_data) if df_catalog_data is not None and not df_catalog_data.empty else 50
            min_slider_val = 5
            actual_max_slider = max(min_slider_val, max_slider_val)
            slider_disabled = actual_max_slider <= min_slider_val
            default_num_objects = st.session_state.get('num_objects_slider', 20)
            clamped_default = max(min_slider_val, min(default_num_objects, actual_max_slider))
            if clamped_default != default_num_objects: st.session_state.num_objects_slider = clamped_default
            st.slider(t.get('results_options_max_objects_label', "Max. Number of Objects to Display:"), min_value=min_slider_val, max_value=actual_max_slider, step=1, key='num_objects_slider', disabled=slider_disabled)
            sort_options_map = {'Duration & Altitude': t.get('results_options_sort_duration', "Duration & Altitude"), 'Brightness': t.get('results_options_sort_magnitude', "Brightness")}
            if st.session_state.sort_method not in sort_options_map: st.session_state.sort_method = 'Duration & Altitude'
            st.radio(t.get('results_options_sort_method_label', "Sort Results By:"), options=list(sort_options_map.keys()), format_func=lambda key: sort_options_map[key], key='sort_method', horizontal=True)

        # --- Bug Report Button ---
        st.sidebar.markdown("---")
        bug_report_email = "debrun2005@gmail.com"
        bug_report_subject = urllib.parse.quote("Bug Report: Advanced DSO Finder")
        bug_report_body = urllib.parse.quote(t.get('bug_report_body', "\n\n(Please describe the bug and steps to reproduce)"))
        bug_report_link = f"mailto:{bug_report_email}?subject={bug_report_subject}&body={bug_report_body}"
        st.sidebar.markdown(f"<a href='{bug_report_link}' target='_blank'>{t.get('bug_report_button', '🐞 Report Bug')}</a>", unsafe_allow_html=True)

    # --- Main Area ---

    # --- Display Search Parameters ---
    st.subheader(t.get('search_params_header', "Search Parameters"))
    param_col1, param_col2 = st.columns(2)

    # Location Parameter Display
    location_display = t.get('location_error', "Location Error: {}").format("Not Set")
    observer_for_run = None
    if st.session_state.location_is_valid_for_run:
        lat = st.session_state.manual_lat_val; lon = st.session_state.manual_lon_val; height = st.session_state.manual_height_val; tz_str = st.session_state.selected_timezone
        try:
            observer_for_run = Observer(latitude=lat * u.deg, longitude=lon * u.deg, elevation=height * u.m, timezone=tz_str)
            if st.session_state.location_choice_key == "Manual": location_display = t.get('location_manual_display', "Manual ({:.4f}, {:.4f})").format(lat, lon)
            elif st.session_state.searched_location_name: location_display = t.get('location_search_display', "Searched: {} ({:.4f}, {:.4f})").format(st.session_state.searched_location_name, lat, lon)
            else: location_display = f"Lat: {lat:.4f}, Lon: {lon:.4f}"
        except Exception as obs_e: location_display = t.get('location_error', "Location Error: {}").format(f"Observer creation failed: {obs_e}"); st.session_state.location_is_valid_for_run = False; observer_for_run = None
    param_col1.markdown(t.get('search_params_location', "📍 Location: {}").format(location_display))

    # Time Parameter Display
    time_display = ""
    is_time_now_main = (st.session_state.time_choice_exp == "Now")
    if is_time_now_main:
        ref_time_main = Time.now()
        try: local_now_str, local_tz_now = get_local_time_str(ref_time_main, st.session_state.selected_timezone); time_display = t.get('search_params_time_now', "Upcoming Night (from {} UTC)").format(f"{local_now_str} {local_tz_now}")
        except Exception: time_display = t.get('search_params_time_now', "Upcoming Night (from {} UTC)").format(f"{ref_time_main.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    else:
        selected_date_main = st.session_state.selected_date_widget; ref_time_main = Time(datetime.combine(selected_date_main, time(12, 0)), scale='utc'); time_display = t.get('search_params_time_specific', "Night after {}").format(selected_date_main.strftime('%Y-%m-%d'))
    param_col1.markdown(t.get('search_params_time', "⏱️ Time: {}").format(time_display))

    # Magnitude Filter Display
    mag_filter_display = ""
    min_mag_filter, max_mag_filter = -np.inf, np.inf
    if st.session_state.mag_filter_mode_exp == "Bortle Scale":
        # <<< CORRECTED CALL >>>
        max_mag_filter = astro_calculations.get_magnitude_limit(st.session_state.bortle_slider)
        mag_filter_display = t.get('search_params_filter_mag_bortle', "Bortle {} (<= {:.1f} mag)").format(st.session_state.bortle_slider, max_mag_filter)
    else:
        min_mag_filter = st.session_state.manual_min_mag_slider; max_mag_filter = st.session_state.manual_max_mag_slider; mag_filter_display = t.get('search_params_filter_mag_manual', "Manual ({:.1f}-{:.1f} mag)").format(min_mag_filter, max_mag_filter)
    param_col2.markdown(t.get('search_params_filter_mag', "✨ Filter: {}").format(mag_filter_display))

    # Altitude and Type Filter Display
    min_alt_disp = st.session_state.min_alt_slider; max_alt_disp = st.session_state.max_alt_slider; selected_types_disp = st.session_state.object_type_filter_exp
    types_str = ', '.join(selected_types_disp) if selected_types_disp else t.get('search_params_types_all', "All")
    param_col2.markdown(t.get('search_params_filter_alt_types', "🔭 Filter: Alt {}-{}°, Types: {}").format(min_alt_disp, max_alt_disp, types_str))

    # Size Filter Display
    size_min_disp, size_max_disp = st.session_state.size_arcmin_range
    param_col2.markdown(t.get('search_params_filter_size', "📐 Filter: Size {:.1f} - {:.1f} arcmin").format(size_min_disp, size_max_disp))

    # Direction Filter Display
    direction_disp = st.session_state.selected_peak_direction
    if direction_disp == ALL_DIRECTIONS_KEY: direction_disp = t.get('search_params_direction_all', "All")
    param_col2.markdown(t.get('search_params_filter_direction', "🧭 Filter: Direction at Max: {}").format(direction_disp))


    # --- Find Objects Button ---
    st.markdown("---")
    find_button_clicked = st.button(
        t.get('find_button_label', "🔭 Find Observable Objects"),
        key="find_button",
        disabled=(df_catalog_data is None or not st.session_state.location_is_valid_for_run)
    )

    if not st.session_state.location_is_valid_for_run and df_catalog_data is not None:
        st.warning(t.get('info_initial_prompt', "Welcome! Please **Enter Coordinates** or **Search Location** to enable object search."))


    # --- Results Area ---
    results_placeholder = st.container()

    # --- Processing Logic (Triggered by Button Click) ---
    if find_button_clicked:
        st.session_state.find_button_pressed = True; st.session_state.show_plot = False; st.session_state.show_custom_plot = False; st.session_state.active_result_plot_data = None; st.session_state.custom_target_plot_data = None; st.session_state.last_results = []; st.session_state.window_start_time = None; st.session_state.window_end_time = None

        if observer_for_run and df_catalog_data is not None:
            with st.spinner(t.get('spinner_searching', "Calculating window & searching objects...")):
                try:
                    # <<< CORRECTED CALL >>>
                    start_time_calc, end_time_calc, window_status = astro_calculations.get_observable_window(
                        observer_for_run, ref_time_main, is_time_now_main, t
                    )
                    results_placeholder.info(window_status)
                    st.session_state.window_start_time = start_time_calc
                    st.session_state.window_end_time = end_time_calc

                    if start_time_calc and end_time_calc and start_time_calc < end_time_calc:
                        time_resolution = 5 * u.minute
                        observing_times = Time(np.arange(start_time_calc.jd, end_time_calc.jd, time_resolution.to(u.day).value), format='jd', scale='utc')
                        if len(observing_times) < 2: results_placeholder.warning("Warning: Observation window is too short for detailed calculation.")

                        filtered_df = df_catalog_data.copy()
                        filtered_df = filtered_df[(filtered_df['Mag'] >= min_mag_filter) & (filtered_df['Mag'] <= max_mag_filter)]
                        if selected_types_disp: filtered_df = filtered_df[filtered_df['Type'].isin(selected_types_disp)]
                        size_col_exists_main = df_catalog_data is not None and 'MajAx' in df_catalog_data.columns and df_catalog_data['MajAx'].notna().any()
                        if size_col_exists_main:
                            filtered_df = filtered_df.dropna(subset=['MajAx'])
                            filtered_df = filtered_df[(filtered_df['MajAx'] >= size_min_disp) & (filtered_df['MajAx'] <= size_max_disp)]

                        if filtered_df.empty:
                            results_placeholder.warning(t.get('warning_no_objects_found', "No objects found matching all criteria for the calculated observation window.") + " (after initial filtering)")
                            st.session_state.last_results = []
                        else:
                            min_altitude_for_search = st.session_state.min_alt_slider * u.deg
                            # <<< CORRECTED CALL >>>
                            found_objects = astro_calculations.find_observable_objects(
                                observer_for_run.location,
                                observing_times,
                                min_altitude_for_search,
                                filtered_df,
                                t
                            )

                            final_objects = []
                            selected_direction = st.session_state.selected_peak_direction
                            max_alt_filter = st.session_state.max_alt_slider
                            for obj in found_objects:
                                peak_alt = obj.get('Max Altitude (°)', -999)
                                if peak_alt > max_alt_filter: continue
                                if selected_direction != ALL_DIRECTIONS_KEY:
                                    if obj.get('Direction at Max') != selected_direction: continue
                                final_objects.append(obj)

                            sort_key = st.session_state.sort_method
                            if sort_key == 'Brightness': final_objects.sort(key=lambda x: x.get('Magnitude', float('inf')) if x.get('Magnitude') is not None else float('inf'))
                            else: final_objects.sort(key=lambda x: (x.get('Max Cont. Duration (h)', 0), x.get('Max Altitude (°)', 0)), reverse=True)

                            num_to_show = st.session_state.num_objects_slider
                            st.session_state.last_results = final_objects[:num_to_show]

                            if not final_objects: results_placeholder.warning(t.get('warning_no_objects_found', "No objects found matching all criteria for the calculated observation window."))
                            else:
                                results_placeholder.success(t.get('success_objects_found', "{} matching objects found.").format(len(final_objects)))
                                sort_msg_key = 'info_showing_list_duration' if sort_key != 'Brightness' else 'info_showing_list_magnitude'
                                results_placeholder.info(t[sort_msg_key].format(len(st.session_state.last_results)))
                    else:
                        results_placeholder.error(t.get('error_no_window', "No valid astronomical darkness window found for the selected date and location.") + " Cannot proceed with search.")
                        st.session_state.last_results = []

                except Exception as search_e:
                    results_placeholder.error(t.get('error_search_unexpected', "An unexpected error occurred during the search:") + f"\n```\n{search_e}\n```")
                    traceback.print_exc()
                    st.session_state.last_results = []

        else:
            if df_catalog_data is None: results_placeholder.error("Cannot search: Catalog data not loaded.")
            if not observer_for_run: results_placeholder.error("Cannot search: Location is not valid.")
            st.session_state.last_results = []


    # --- Display Results Block ---
    if st.session_state.last_results:
        results_data = st.session_state.last_results
        results_placeholder.subheader(t.get('results_list_header', "Result List"))

        # Moon Phase Display
        window_start = st.session_state.get('window_start_time')
        window_end = st.session_state.get('window_end_time')
        observer_exists = observer_for_run is not None
        if observer_exists and isinstance(window_start, Time) and isinstance(window_end, Time):
            mid_time = window_start + (window_end - window_start) / 2
            try:
                illum = moon_illumination(mid_time)
                moon_phase_percent = illum * 100
                moon_svg = create_moon_phase_svg(illum, size=50)
                moon_col1, moon_col2 = results_placeholder.columns([1, 3])
                with moon_col1: st.markdown(moon_svg, unsafe_allow_html=True)
                with moon_col2:
                    st.metric(label=t.get('moon_metric_label', "Moon Illumination (approx.)"), value=f"{moon_phase_percent:.0f}%")
                    moon_warn_threshold = st.session_state.moon_phase_slider
                    if moon_phase_percent > moon_warn_threshold: st.warning(t.get('moon_warning_message', "Warning: Moon is brighter ({:.0f}%) than threshold ({:.0f}%)!").format(moon_phase_percent, moon_warn_threshold))
            except Exception as moon_e: results_placeholder.warning(t.get('moon_phase_error', "Error calculating moon phase: {}").format(moon_e))
        elif st.session_state.find_button_pressed: results_placeholder.info("Moon phase cannot be calculated (invalid observation window or location).")


        # Plot Type Selection
        plot_options_map = {'Sky Path': t.get('graph_type_sky_path', "Sky Path (Az/Alt)"), 'Altitude Plot': t.get('graph_type_alt_time', "Altitude Plot (Alt/Time)")}
        if st.session_state.plot_type_selection not in plot_options_map: st.session_state.plot_type_selection = 'Sky Path'
        results_placeholder.radio(t.get('graph_type_label', "Graph Type (for all plots):"), options=list(plot_options_map.keys()), format_func=lambda key: plot_options_map[key], key='plot_type_selection', horizontal=True)

        # Display Individual Objects
        for i, obj_data in enumerate(results_data):
            obj_name = obj_data.get('Name', 'N/A'); obj_type = obj_data.get('Type', 'N/A'); obj_mag = obj_data.get('Magnitude'); mag_str = f"{obj_mag:.1f}" if obj_mag is not None else "N/A"; expander_title = t.get('results_expander_title', "{} ({}) - Mag: {:.1f}").format(obj_name, obj_type, obj_mag if obj_mag is not None else 99)
            is_expanded = (st.session_state.expanded_object_name == obj_name)
            object_container = results_placeholder.container()
            with object_container.expander(expander_title, expanded=is_expanded):
                col1, col2, col3 = st.columns([2,2,1])
                col1.markdown(t.get('results_coords_header', "**Details:**"))
                col1.markdown(f"**{t.get('results_export_constellation', 'Constellation')}:** {obj_data.get('Constellation', 'N/A')}")
                size_arcmin = obj_data.get('Size (arcmin)'); col1.markdown(f"**{t.get('results_size_label', 'Size (Major Axis):')}** {t.get('results_size_value', '{:.1f} arcmin').format(size_arcmin) if size_arcmin is not None else 'N/A'}")
                col1.markdown(f"**RA:** {obj_data.get('RA', 'N/A')}"); col1.markdown(f"**Dec:** {obj_data.get('Dec', 'N/A')}")
                col2.markdown(t.get('results_max_alt_header', "**Max. Altitude:**"))
                max_alt = obj_data.get('Max Altitude (°)', 0); az_at_max = obj_data.get('Azimuth at Max (°)', 0); dir_at_max = obj_data.get('Direction at Max', 'N/A'); azimuth_formatted = t.get('results_azimuth_label', "(Azimuth: {:.1f}°{})").format(az_at_max, ""); direction_formatted = t.get('results_direction_label', ", Direction: {}").format(dir_at_max); col2.markdown(f"**{max_alt:.1f}°** {azimuth_formatted}{direction_formatted}")
                col2.markdown(t.get('results_best_time_header', "**Best Time (Local TZ):**"))
                peak_time_utc = obj_data.get('Time at Max (UTC)'); local_time_str, local_tz_name = get_local_time_str(peak_time_utc, st.session_state.selected_timezone); col2.markdown(f"{local_time_str} ({local_tz_name})")
                col2.markdown(t.get('results_cont_duration_header', "**Max. Cont. Duration:**"))
                duration_h = obj_data.get('Max Cont. Duration (h)', 0); col2.markdown(t.get('results_duration_value', "{:.1f} hours").format(duration_h))
                google_query = urllib.parse.quote_plus(f"{obj_name} astronomy"); google_url = f"https://www.google.com/search?q={google_query}"; col3.markdown(f"[{t.get('google_link_text', 'Google')}]({google_url})", unsafe_allow_html=True)
                simbad_query = urllib.parse.quote_plus(obj_name); simbad_url = f"http://simbad.u-strasbg.fr/simbad/sim-basic?Ident={simbad_query}"; col3.markdown(f"[{t.get('simbad_link_text', 'SIMBAD')}]({simbad_url})", unsafe_allow_html=True)
                plot_button_key = f"plot_{obj_name}_{i}"
                if st.button(t.get('results_graph_button', "📈 Show Plot"), key=plot_button_key): st.session_state.plot_object_name = obj_name; st.session_state.active_result_plot_data = obj_data; st.session_state.show_plot = True; st.session_state.show_custom_plot = False; st.session_state.expanded_object_name = obj_name; st.rerun()
                if st.session_state.show_plot and st.session_state.plot_object_name == obj_name:
                    plot_data = st.session_state.active_result_plot_data; min_alt_line = st.session_state.min_alt_slider; max_alt_line = st.session_state.max_alt_slider
                    st.markdown("---")
                    with st.spinner(t.get('results_spinner_plotting', "Creating plot...")):
                        try:
                            fig = create_plot(plot_data, min_alt_line, max_alt_line, st.session_state.plot_type_selection, t) # <<< Pass t here
                            if fig:
                                st.pyplot(fig); close_button_key = f"close_plot_{obj_name}_{i}"
                                if st.button(t.get('results_close_graph_button', "Close Plot"), key=close_button_key): st.session_state.show_plot = False; st.session_state.active_result_plot_data = None; st.session_state.expanded_object_name = None; st.rerun()
                            else: st.error(t.get('results_graph_not_created', "Plot could not be created."))
                        except Exception as plot_err: st.error(t.get('results_graph_error', "Plot Error: {}").format(plot_err)); traceback.print_exc()

        # CSV Export Button
        if results_data:
            csv_export_placeholder = results_placeholder.empty()
            try:
                export_data = []
                for obj in results_data:
                    peak_time_utc = obj.get('Time at Max (UTC)')
                    local_time_str, _ = get_local_time_str(peak_time_utc, st.session_state.selected_timezone)
                    export_data.append({t.get('results_export_name', "Name"): obj.get('Name', 'N/A'), t.get('results_export_type', "Type"): obj.get('Type', 'N/A'), t.get('results_export_constellation', "Constellation"): obj.get('Constellation', 'N/A'), t.get('results_export_mag', "Magnitude"): obj.get('Magnitude'), t.get('results_export_size', "Size (arcmin)"): obj.get('Size (arcmin)'), t.get('results_export_ra', "RA"): obj.get('RA', 'N/A'), t.get('results_export_dec', "Dec"): obj.get('Dec', 'N/A'), t.get('results_export_max_alt', "Max Altitude (°)"): obj.get('Max Altitude (°)', 0), t.get('results_export_az_at_max', "Azimuth at Max (°)"): obj.get('Azimuth at Max (°)', 0), t.get('results_export_direction_at_max', "Direction at Max"): obj.get('Direction at Max', 'N/A'), t.get('results_export_time_max_utc', "Time at Max (UTC)"): peak_time_utc.iso if peak_time_utc else "N/A", t.get('results_export_time_max_local', "Time at Max (Local TZ)"): local_time_str, t.get('results_export_cont_duration', "Max Cont Duration (h)"): obj.get('Max Cont. Duration (h)', 0)})
                df_export = pd.DataFrame(export_data)
                decimal_sep = ',' if lang == 'de' else '.'
                csv_string = df_export.to_csv(index=False, sep=';', encoding='utf-8-sig', decimal=decimal_sep)
                now_str = datetime.now().strftime("%Y%m%d_%H%M")
                csv_filename = t.get('results_csv_filename', "dso_observation_list_{}.csv").format(now_str)
                csv_export_placeholder.download_button(label=t.get('results_save_csv_button', "💾 Save Result List as CSV"), data=csv_string, file_name=csv_filename, mime='text/csv', key='csv_download_button')
            except Exception as csv_e: csv_export_placeholder.error(t.get('results_csv_export_error', "CSV Export Error: {}").format(csv_e))

    elif st.session_state.find_button_pressed: results_placeholder.info(t.get('warning_no_objects_found', "No objects found matching all criteria for the calculated observation window."))


    # --- Custom Target Plotting ---
    st.markdown("---")
    with st.expander(t.get('custom_target_expander', "Plot Custom Target")):
        with st.form("custom_target_form"):
             st.text_input(t.get('custom_target_ra_label', "Right Ascension (RA):"), key="custom_target_ra", placeholder=t.get('custom_target_ra_placeholder', "e.g., 10:45:03.6 or 161.265"))
             st.text_input(t.get('custom_target_dec_label', "Declination (Dec):"), key="custom_target_dec", placeholder=t.get('custom_target_dec_placeholder', "e.g., -16:42:58 or -16.716"))
             st.text_input(t.get('custom_target_name_label', "Target Name (Optional):"), key="custom_target_name", placeholder="My Comet")
             custom_plot_submitted = st.form_submit_button(t.get('custom_target_button', "Create Custom Plot"))

        custom_plot_error_placeholder = st.empty()
        custom_plot_display_area = st.empty()

        if custom_plot_submitted:
             st.session_state.show_plot = False; st.session_state.show_custom_plot = False; st.session_state.custom_target_plot_data = None; st.session_state.custom_target_error = ""
             custom_ra = st.session_state.custom_target_ra; custom_dec = st.session_state.custom_target_dec; custom_name = st.session_state.custom_target_name or t.get('custom_target_name_label', "Target Name (Optional):").replace(":", "")
             window_start_cust = st.session_state.get('window_start_time'); window_end_cust = st.session_state.get('window_end_time'); observer_exists_cust = observer_for_run is not None

             if not custom_ra or not custom_dec: st.session_state.custom_target_error = t.get('custom_target_error_coords', "Invalid RA/Dec format. Use HH:MM:SS.s / DD:MM:SS or decimal degrees."); custom_plot_error_placeholder.error(st.session_state.custom_target_error)
             elif not observer_exists_cust or not isinstance(window_start_cust, Time) or not isinstance(window_end_cust, Time): st.session_state.custom_target_error = t.get('custom_target_error_window', "Cannot create plot. Ensure location and time window are valid (try clicking 'Find Observable Objects' first)."); custom_plot_error_placeholder.error(st.session_state.custom_target_error)
             else:
                 try:
                     custom_coord = SkyCoord(ra=custom_ra, dec=custom_dec, unit=(u.hourangle, u.deg))
                     if window_start_cust < window_end_cust: time_resolution_cust = 5 * u.minute; observing_times_custom = Time(np.arange(window_start_cust.jd, window_end_cust.jd, time_resolution_cust.to(u.day).value), format='jd', scale='utc')
                     else: raise ValueError("Valid time window from main search not available for custom plot.")
                     if len(observing_times_custom) < 2: raise ValueError("Calculated time window for custom plot is too short.")
                     altaz_frame_custom = AltAz(obstime=observing_times_custom, location=observer_for_run.location); custom_altazs = custom_coord.transform_to(altaz_frame_custom); custom_alts = custom_altazs.alt.to(u.deg).value; custom_azs = custom_altazs.az.to(u.deg).value
                     st.session_state.custom_target_plot_data = {'Name': custom_name, 'altitudes': custom_alts, 'azimuths': custom_azs, 'times': observing_times_custom}; st.session_state.show_custom_plot = True; st.session_state.custom_target_error = ""; st.rerun()
                 except ValueError as custom_coord_err: st.session_state.custom_target_error = f"{t.get('custom_target_error_coords', 'Invalid RA/Dec format. Use HH:MM:SS.s / DD:MM:SS or decimal degrees.')} ({custom_coord_err})"; custom_plot_error_placeholder.error(st.session_state.custom_target_error)
                 except Exception as custom_e: st.session_state.custom_target_error = f"Error creating custom plot: {custom_e}"; custom_plot_error_placeholder.error(st.session_state.custom_target_error); traceback.print_exc()

        if st.session_state.show_custom_plot and st.session_state.custom_target_plot_data:
            custom_plot_data = st.session_state.custom_target_plot_data; min_alt_line_cust = st.session_state.min_alt_slider; max_alt_line_cust = st.session_state.max_alt_slider
            with custom_plot_display_area.container():
                 st.markdown("---")
                 with st.spinner(t.get('results_spinner_plotting', "Creating plot...")):
                     try:
                         fig_cust = create_plot(custom_plot_data, min_alt_line_cust, max_alt_line_cust, st.session_state.plot_type_selection, t) # <<< Pass t here
                         if fig_cust:
                             st.pyplot(fig_cust)
                             if st.button(t.get('results_close_graph_button', "Close Plot"), key="close_custom_plot"): st.session_state.show_custom_plot = False; st.session_state.custom_target_plot_data = None; st.rerun()
                         else: st.error(t.get('results_graph_not_created', "Plot could not be created."))
                     except Exception as plot_err_cust: st.error(t.get('results_graph_error', "Plot Error: {}").format(plot_err_cust)); traceback.print_exc()
        elif st.session_state.custom_target_error: custom_plot_error_placeholder.error(st.session_state.custom_target_error)


    # --- Add Donation Link at the bottom ---
    st.markdown("---")
    st.caption(t.get('donation_text', "Like the app? [Support the development on Ko-fi ☕](https://ko-fi.com/advanceddsofinder)"), unsafe_allow_html=True)


# --- Plotting Function (Revised for Robustness and Visibility) ---
def create_plot(plot_data: dict, min_altitude_deg: float, max_altitude_deg: float, plot_type: str, t: dict) -> plt.Figure | None: # <<< Added t parameter
    """Creates either an Altitude vs Time or Sky Path (Alt/Az) plot with improved robustness and theme handling."""
    fig = None
    try:
        # --- Validate Input Data ---
        if not isinstance(plot_data, dict): st.error("Plot Error: Invalid plot_data type (expected dict)."); return None
        times = plot_data.get('times'); altitudes = plot_data.get('altitudes'); azimuths = plot_data.get('azimuths'); obj_name = plot_data.get('Name', 'Object')
        if not isinstance(times, Time) or not isinstance(altitudes, np.ndarray): st.error("Plot Error: Missing or invalid 'times' or 'altitudes' in plot_data."); return None
        if plot_type == 'Sky Path' and not isinstance(azimuths, np.ndarray): st.error("Plot Error: Missing or invalid 'azimuths' for Sky Path plot."); return None
        if len(times) != len(altitudes) or (azimuths is not None and len(times) != len(azimuths)): st.error(f"Plot Error: Mismatched array lengths (times: {len(times)}, alts: {len(altitudes)}, azs: {len(azimuths) if azimuths is not None else 'N/A'})."); return None
        if len(times) < 1: st.error("Plot Error: Not enough data points to create a plot."); return None
        plot_times = times.plot_date

        # --- Theme Detection and Color Setup ---
        try: theme_opts = st.get_option("theme.base"); is_dark_theme = (theme_opts == "dark")
        except Exception: print("Warning: Could not detect Streamlit theme via get_option. Assuming light theme."); is_dark_theme = False
        if is_dark_theme:
            plt.style.use('dark_background'); fig_facecolor = '#0E1117'; ax_facecolor = '#0E1117'; primary_color = 'deepskyblue'; grid_color = '#444444'; label_color = '#FAFAFA'; title_color = '#FFFFFF'; legend_facecolor = '#262730'; min_alt_color = 'tomato'; max_alt_color = 'orange'; spine_color = '#AAAAAA'
        else:
            plt.style.use('default'); fig_facecolor = '#FFFFFF'; ax_facecolor = '#FFFFFF'; primary_color = 'dodgerblue'; grid_color = 'darkgray'; label_color = '#333333'; title_color = '#000000'; legend_facecolor = '#F0F0F0'; min_alt_color = 'red'; max_alt_color = 'darkorange'; spine_color = '#555555'

        # --- Create Figure and Axes ---
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=fig_facecolor, constrained_layout=True); ax.set_facecolor(ax_facecolor)

        # --- Plot Logic ---
        if plot_type == 'Altitude Plot':
            ax.plot(plot_times, altitudes, color=primary_color, alpha=0.9, linewidth=1.5, label=obj_name)
            ax.axhline(min_altitude_deg, color=min_alt_color, linestyle='--', linewidth=1.2, label=t.get('graph_min_altitude_label', "Min Altitude ({:.0f}°)").format(min_altitude_deg), alpha=0.8)
            if max_altitude_deg < 90: ax.axhline(max_altitude_deg, color=max_alt_color, linestyle=':', linewidth=1.2, label=t.get('graph_max_altitude_label', "Max Altitude ({:.0f}°)").format(max_altitude_deg), alpha=0.8)
            ax.set_xlabel("Time (UTC)", color=label_color, fontsize=11); ax.set_ylabel(t.get('graph_ylabel', "Altitude (°)"), color=label_color, fontsize=11); ax.set_title(t.get('graph_title_alt_time', "Altitude Plot for {}").format(obj_name), color=title_color, fontsize=13, weight='bold'); ax.set_ylim(0, 90); ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')); fig.autofmt_xdate(rotation=30)
            ax.grid(True, linestyle='-', alpha=0.5, color=grid_color); ax.tick_params(axis='x', colors=label_color); ax.tick_params(axis='y', colors=label_color)
            for spine in ax.spines.values(): spine.set_color(spine_color); spine.set_linewidth(0.5)
        elif plot_type == 'Sky Path':
            if azimuths is None or not isinstance(azimuths, np.ndarray) or len(azimuths) != len(times): st.error("Plot Error: Invalid or missing azimuth data for Sky Path plot."); plt.close(fig); return None
            ax.remove(); ax = fig.add_subplot(111, projection='polar', facecolor=ax_facecolor)
            az_rad = np.deg2rad(azimuths); radius = 90 - altitudes
            time_delta = times.jd.max() - times.jd.min(); time_norm = (times.jd - times.jd.min()) / (time_delta + 1e-9); colors = plt.cm.plasma(time_norm)
            scatter = ax.scatter(az_rad, radius, c=colors, s=15, alpha=0.8, edgecolors='none', label=obj_name); ax.plot(az_rad, radius, color=primary_color, alpha=0.4, linewidth=0.8)
            ax.plot(np.linspace(0, 2*np.pi, 100), np.full(100, 90 - min_altitude_deg), color=min_alt_color, linestyle='--', linewidth=1.2, label=t.get('graph_min_altitude_label', "Min Altitude ({:.0f}°)").format(min_altitude_deg), alpha=0.8)
            if max_altitude_deg < 90: ax.plot(np.linspace(0, 2*np.pi, 100), np.full(100, 90 - max_altitude_deg), color=max_alt_color, linestyle=':', linewidth=1.2, label=t.get('graph_max_altitude_label', "Max Altitude ({:.0f}°)").format(max_altitude_deg), alpha=0.8)
            ax.set_theta_zero_location('N'); ax.set_theta_direction(-1); ax.set_yticks(np.arange(0, 91, 15)); ax.set_yticklabels([f"{90-alt}°" for alt in np.arange(0, 91, 15)], color=label_color); ax.set_ylim(0, 90); ax.set_title(t.get('graph_title_sky_path', "Sky Path for {}").format(obj_name), va='bottom', color=title_color, fontsize=13, weight='bold', y=1.1)
            ax.grid(True, linestyle=':', alpha=0.5, color=grid_color); ax.spines['polar'].set_color(spine_color); ax.spines['polar'].set_linewidth(0.5)
            try:
                cbar = fig.colorbar(scatter, ax=ax, label="Time Progression (UTC)", pad=0.1, shrink=0.7); cbar.set_ticks([0, 1])
                if len(times) > 0: start_label = times[0].to_datetime(timezone.utc).strftime('%H:%M'); end_label = times[-1].to_datetime(timezone.utc).strftime('%H:%M'); cbar.ax.set_yticklabels([start_label, end_label])
                else: cbar.ax.set_yticklabels(['Start', 'End'])
                cbar.set_label("Time Progression (UTC)", color=label_color, fontsize=10); cbar.ax.yaxis.set_tick_params(color=label_color, labelsize=9); plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=label_color); cbar.outline.set_edgecolor(spine_color); cbar.outline.set_linewidth(0.5)
            except Exception as cbar_err: print(f"Warning: Could not create colorbar for Sky Path plot: {cbar_err}")
        else: st.error(f"Plot Error: Unknown plot type '{plot_type}'"); plt.close(fig); return None

        # --- Common Plot Settings ---
        legend = ax.legend(loc='lower right', fontsize='small', facecolor=legend_facecolor, framealpha=0.8, edgecolor=spine_color)
        for text in legend.get_texts(): text.set_color(label_color)
        return fig
    except Exception as e:
        st.error(f"Plot Error: An unexpected error occurred: {e}"); traceback.print_exc()
        if fig: plt.close(fig)
        return None


# --- Run the App ---
if __name__ == "__main__":
    main()
