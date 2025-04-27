# -*- coding: utf-8 -*-
from __future__ import annotations
import streamlit as st
# Removed random, math imports if only used by moved functions
from datetime import datetime, date, time, timedelta, timezone
import traceback # Keep traceback for error handling in main
import os
# Removed urllib imports if only used by moved functions
import pandas as pd # <<< Re-added pandas import for type hinting

# --- Library Imports ---
try:
    from astropy.time import Time
    import numpy as np
    import astropy.units as u
    # Removed EarthLocation, get_sun, get_constellation if only used by moved functions
    # Keep SkyCoord, AltAz if used for calculation (likely not needed here anymore)
    from astroplan import Observer # Keep Observer for creating the observer object
    # Removed moon_illumination if only used by moved functions
    import pytz # Keep if used for timezone handling here
    from timezonefinder import TimezoneFinder # Keep for initializing tf
    # Removed geopy imports if only used by moved functions
except ImportError as e:
    st.error(f"Import Error: Missing libraries. Please install required packages (check requirements.txt). Details: {e}")
    st.stop()

# --- Import Custom Modules ---
try:
    from localization import translations # Import the main translations dictionary
    import astro_calculations # Import the whole module
    import data_handling # Import the whole module
    # Import UI functions from the new module
    import ui_components # Import the whole module
    # Removed CARDINAL_DIRECTIONS import if only used by UI
except ModuleNotFoundError as e:
    st.error(f"Module Not Found Error: Could not find a required module file ({e}). Ensure 'localization.py', 'astro_calculations.py', 'data_handling.py', and 'ui_components.py' are present.")
    st.stop()


# --- Page Config ---
st.set_page_config(page_title="Advanced DSO Finder", layout="wide")

# --- Global Configuration & Initial Values ---
# Keep initial values needed for session state defaults
INITIAL_LAT = 47.17
INITIAL_LON = 8.01
INITIAL_HEIGHT = 550
INITIAL_TIMEZONE = "Europe/Zurich"
INITIAL_MIN_ALT = 20
INITIAL_MAX_ALT = 90


# --- Path to Catalog File ---
try:
    # Get the directory of the current script
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for environments where __file__ is not defined (e.g., interactive)
    APP_DIR = os.getcwd()
CATALOG_FILENAME = "ongc.csv"
CATALOG_FILEPATH = os.path.join(APP_DIR, CATALOG_FILENAME)

# --- Constants ---
# Keep constants used in this file (if any)
# ALL_DIRECTIONS_KEY might be needed if used in filtering logic here, otherwise can remove
ALL_DIRECTIONS_KEY = 'All' # Used in filtering logic within main()


# --- Initialize TimezoneFinder (cached) ---
# Keep this here as it's used early and passed to UI
@st.cache_resource
def get_timezone_finder():
    """Initializes and returns a TimezoneFinder instance."""
    # Check if the class exists before trying to instantiate
    if 'TimezoneFinder' in globals():
        try:
            return TimezoneFinder(in_memory=True)
        except Exception as e:
            print(f"Error initializing TimezoneFinder: {e}")
            st.warning(f"TimezoneFinder initialization failed: {e}. Timezone detection might be limited.")
            return None
    else:
        print("TimezoneFinder class not imported.")
        st.warning("TimezoneFinder library not found. Timezone detection will default to UTC or initial setting.")
        return None

tf = get_timezone_finder() # Initialize timezone finder

# --- Initialize Session State ---
# Keep this function as it manages the application's state
def initialize_session_state():
    """Initializes all required session state keys if they don't exist."""
    defaults = {
        'language': 'DE', # Default to uppercase 'DE'
        'plot_object_name': None, 'show_plot': False, 'active_result_plot_data': None,
        'last_results': [], 'find_button_pressed': False,
        'location_choice_key': 'Search', 'manual_lat_val': INITIAL_LAT, 'manual_lon_val': INITIAL_LON,
        'manual_height_val': INITIAL_HEIGHT, 'location_search_query': "",
        'searched_location_name': None, 'location_search_status_msg': "",
        'location_search_success': False, 'selected_timezone': INITIAL_TIMEZONE,
        'location_is_valid_for_run': False,
        'manual_min_mag_slider': 0.0, 'manual_max_mag_slider': 16.0,
        'object_type_filter_exp': [], 'mag_filter_mode_exp': 'Bortle Scale',
        'bortle_slider': 5, 'min_alt_slider': INITIAL_MIN_ALT, 'max_alt_slider': INITIAL_MAX_ALT,
        'moon_phase_slider': 35, 'size_arcmin_range': (1.0, 120.0),
        'selected_peak_direction': ALL_DIRECTIONS_KEY,
        'sort_method': 'Duration & Altitude', 'num_objects_slider': 20,
        'plot_type_selection': 'Sky Path',
        'custom_target_ra': "", 'custom_target_dec': "", 'custom_target_name': "",
        'custom_target_error': "", 'custom_target_plot_data': None, 'show_custom_plot': False,
        'expanded_object_name': None, 'time_choice_exp': 'Now',
        'window_start_time': None, 'window_end_time': None,
        'selected_date_widget': date.today(),
    }
    for key, default_value in defaults.items():
        if key not in st.session_state: st.session_state[key] = default_value
    if 'language' in st.session_state: st.session_state.language = st.session_state.language.upper()


# --- Main App Logic ---
def main():
    # 1. Initialize state and load language translations
    initialize_session_state()
    lang = st.session_state.language
    # --- Rename variable 't' to 'trans' ---
    if lang not in translations:
        print(f"Warning: Language '{lang}' not found in translations dictionary. Falling back to 'EN'.")
        lang = 'EN'
        st.session_state.language = lang
    # Assign the selected language dictionary to 'trans'
    trans = translations.get(lang, translations.get('EN', {}))
    # --- End rename ---

    # 2. Load Catalog Data (Cached)
    @st.cache_data
    def cached_load_ongc_data(path: str) -> pd.DataFrame | None:
        """Cached function to load ONGC data."""
        print(f"Cache miss: Loading ONGC data from {path}")
        try: return data_handling.load_ongc_data(path)
        except ModuleNotFoundError: st.error(f"{trans.get('error_module_missing', 'Error: Module missing:')} data_handling.py"); return None
        except FileNotFoundError: st.error(f"{trans.get('error_catalog_not_found', 'Error: Catalog file not found at path:')} {path}"); return None
        except Exception as load_err: st.error(f"{trans.get('error_catalog_load_failed', 'Failed to load catalog')}: {load_err}"); return None

    df_catalog_data = cached_load_ongc_data(CATALOG_FILEPATH)

    # 3. Display Title and Glossary
    st.title(trans.get('app_title', "Advanced DSO Finder"))
    with st.expander(trans.get('object_type_glossary_title', "Object Type Glossary")):
        glossary_items = trans.get('object_type_glossary', {})
        if glossary_items:
            col1, col2 = st.columns(2); col_index = 0
            sorted_items = sorted(glossary_items.items())
            for abbr, full_name in sorted_items:
                target_col = col1 if col_index % 2 == 0 else col2
                target_col.markdown(f"**{abbr}:** {full_name}")
                col_index += 1
        else: st.info(trans.get('glossary_unavailable', "Glossary not available for the selected language."))

    st.markdown("---")

    # 4. Create Sidebar UI (Pass 'trans' instead of 't')
    ui_components.create_sidebar(trans, df_catalog_data, tf)

    # 5. Prepare Observer Object
    observer_run = None
    if st.session_state.location_is_valid_for_run:
        lat = st.session_state.manual_lat_val; lon = st.session_state.manual_lon_val; hgt = st.session_state.manual_height_val; tz_str = st.session_state.selected_timezone
        try: observer_run = Observer(latitude=lat*u.deg, longitude=lon*u.deg, elevation=hgt*u.m, timezone=tz_str)
        except Exception as obs_err: st.error(trans.get('error_observer_creation', "Error creating observer location: {}").format(obs_err)); st.session_state.location_is_valid_for_run = False; observer_run = None

    # 6. Determine Reference Time
    ref_time = None
    is_now_mode_main = (st.session_state.time_choice_exp == "Now")
    if is_now_mode_main: ref_time = Time.now()
    else:
        selected_date_main = st.session_state.selected_date_widget
        try: ref_time = Time(datetime.combine(selected_date_main, time(12, 0)), scale='utc'); print(f"Calculating 'Specific Night' window based on UTC noon: {ref_time.iso}")
        except Exception as time_err: st.error(trans.get('error_ref_time_creation', "Error setting reference time: {}").format(time_err)); ref_time = None

    # 7. Display Search Parameters Summary (Pass 'trans' instead of 't')
    min_mag_filt_calc, max_mag_filt_calc = ui_components.display_search_parameters(
        trans, observer_run, ref_time if ref_time else Time.now()
    )

    st.markdown("---")

    # 8. "Find Objects" Button and Core Logic Trigger
    results_placeholder = st.container()
    find_button_disabled = (df_catalog_data is None or not st.session_state.location_is_valid_for_run or ref_time is None)
    find_button_clicked = st.button(
        trans.get('find_button_label', "🔭 Find Observable Objects"),
        key="find_button", disabled=find_button_disabled
    )

    if not st.session_state.location_is_valid_for_run and df_catalog_data is not None: st.warning(trans.get('info_initial_prompt', "..."))
    elif ref_time is None and not is_now_mode_main: st.warning(trans.get('info_set_ref_time', "..."))
    elif df_catalog_data is None: st.warning(trans.get('info_catalog_missing', "..."))

    if find_button_clicked:
        st.session_state.find_button_pressed = True; st.session_state.show_plot = False; st.session_state.show_custom_plot = False; st.session_state.active_result_plot_data = None; st.session_state.custom_target_plot_data = None; st.session_state.last_results = []; st.session_state.window_start_time = None; st.session_state.window_end_time = None; st.session_state.expanded_object_name = None

        if observer_run and df_catalog_data is not None and ref_time is not None:
            with st.spinner(trans.get('spinner_searching',"Searching for observable objects...")):
                try:
                    start_time_calc, end_time_calc, window_status_msg = astro_calculations.get_observable_window(observer_run, ref_time, is_now_mode_main, trans)
                    results_placeholder.info(window_status_msg); st.session_state.window_start_time = start_time_calc; st.session_state.window_end_time = end_time_calc
                    if start_time_calc and end_time_calc and start_time_calc < end_time_calc:
                        time_resolution_calc = 5 * u.minute; observation_times_calc = Time(np.arange(start_time_calc.jd, end_time_calc.jd, time_resolution_calc.to(u.day).value), format='jd', scale='utc')
                        if len(observation_times_calc) < 2: results_placeholder.warning(trans.get('warning_window_too_short_calc', "..."))
                        filtered_df = df_catalog_data.copy(); filtered_df = filtered_df[(filtered_df['Mag'] >= min_mag_filt_calc) & (filtered_df['Mag'] <= max_mag_filt_calc)]
                        selected_types_calc = st.session_state.object_type_filter_exp
                        if selected_types_calc: filtered_df = filtered_df[filtered_df['Type'].isin(selected_types_calc)]
                        size_data_ok_calc = 'MajAx' in filtered_df.columns and filtered_df['MajAx'].notna().any()
                        if size_data_ok_calc: size_min_calc, size_max_calc = st.session_state.size_arcmin_range; filtered_df = filtered_df.dropna(subset=['MajAx']); filtered_df = filtered_df[(filtered_df['MajAx'] >= size_min_calc) & (filtered_df['MajAx'] <= size_max_calc)]
                        if filtered_df.empty: results_placeholder.warning(trans.get('warning_no_objects_after_filters',"...")); st.session_state.last_results = []
                        else:
                            min_alt_search_calc = st.session_state.min_alt_slider * u.deg
                            found_objects = astro_calculations.find_observable_objects(observer_run.location, observation_times_calc, min_alt_search_calc, filtered_df, trans)
                            final_results = []; selected_direction_calc = st.session_state.selected_peak_direction; max_alt_filter_calc = st.session_state.max_alt_slider
                            for obj_result in found_objects:
                                if obj_result.get('Max Altitude (°)', -999) > max_alt_filter_calc: continue
                                if selected_direction_calc != ALL_DIRECTIONS_KEY and obj_result.get('Direction at Max') != selected_direction_calc: continue
                                final_results.append(obj_result)
                            sort_key = st.session_state.sort_method
                            if sort_key == 'Brightness': final_results.sort(key=lambda x: x.get('Magnitude', float('inf')) if x.get('Magnitude') is not None else float('inf'))
                            else: final_results.sort(key=lambda x: (x.get('Max Cont. Duration (h)', 0), x.get('Max Altitude (°)', 0)), reverse=True)
                            num_to_show = st.session_state.num_objects_slider; st.session_state.last_results = final_results[:num_to_show]
                            if not final_results: results_placeholder.warning(trans.get('warning_no_objects_found_final',"..."))
                            else: results_placeholder.success(trans.get('success_objects_found',"{} objects found matching criteria.").format(len(final_results))); sort_info_key = 'info_showing_list_duration' if sort_key != 'Brightness' else 'info_showing_list_magnitude'; results_placeholder.info(trans.get(sort_info_key, "...").format(len(st.session_state.last_results), sort_key))
                    else: results_placeholder.error(trans.get('error_cannot_search_no_window',"...")); st.session_state.last_results = []
                # --- Correction: Removed extra parenthesis ---
                except Exception as e:
                    error_msg = trans.get('error_search_unexpected',"...") # Corrected: Removed extra ')'
                    results_placeholder.error(f"{error_msg}\n```\n{traceback.format_exc()}\n```")
                    print(f"Search Error: {e}")
                    traceback.print_exc()
                    st.session_state.last_results = []
                # --- End Correction ---
        else:
            if df_catalog_data is None: results_placeholder.error(trans.get('error_prereq_catalog',"..."))
            if not observer_run: results_placeholder.error(trans.get('error_prereq_location',"..."))
            if ref_time is None: results_placeholder.error(trans.get('error_prereq_time',"..."))
            st.session_state.last_results = []

    # 9. Display Results (Pass 'trans' instead of 't')
    if st.session_state.last_results:
        ui_components.display_results(trans, results_placeholder, observer_run)
    elif st.session_state.find_button_pressed: pass

    # 10. Display Custom Target Section (Pass 'trans' instead of 't')
    ui_components.create_custom_target_section(trans, results_placeholder, observer_run)

    # 11. Display Manual Cosmology Calculator (Pass 'trans' instead of 't')
    ui_components.create_manual_cosmology_calculator(trans)

    # 12. Display Donation Link (Pass 'trans' instead of 't')
    ui_components.display_donation_link(trans)

# --- Run the App ---
if __name__ == "__main__":
    main()
