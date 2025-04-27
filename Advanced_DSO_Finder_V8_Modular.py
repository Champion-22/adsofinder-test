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
    from localization import translations
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
        # --- Correction: Use uppercase default language key ---
        'language': 'DE', # Default to uppercase 'DE'
        # --- End Correction ---
        'plot_object_name': None, # Name of the object currently plotted (from results)
        'show_plot': False,       # Flag to show the results plot
        'active_result_plot_data': None, # Data dict for the active results plot
        'last_results': [],       # Stores the list of dicts from the last search
        'find_button_pressed': False, # Flag if the find button was clicked
        # Location state
        'location_choice_key': 'Search', # 'Search' or 'Manual'
        'manual_lat_val': INITIAL_LAT,
        'manual_lon_val': INITIAL_LON,
        'manual_height_val': INITIAL_HEIGHT,
        'location_search_query': "",
        'searched_location_name': None, # Name found via geocoding
        'location_search_status_msg': "", # User-facing message about search status
        'location_search_success': False, # Flag if geocoding was successful
        'selected_timezone': INITIAL_TIMEZONE, # Currently active timezone string
        'location_is_valid_for_run': False, # Flag if current location settings allow search
        # Filter state
        'manual_min_mag_slider': 0.0,
        'manual_max_mag_slider': 16.0,
        'object_type_filter_exp': [], # List of selected object types
        'mag_filter_mode_exp': 'Bortle Scale', # 'Bortle Scale' or 'Manual'
        'bortle_slider': 5,
        'min_alt_slider': INITIAL_MIN_ALT,
        'max_alt_slider': INITIAL_MAX_ALT,
        'moon_phase_slider': 35, # Moon warning threshold
        'size_arcmin_range': (1.0, 120.0), # Tuple (min, max) apparent size
        'selected_peak_direction': ALL_DIRECTIONS_KEY, # Direction filter ('All', 'N', 'NE', ...)
        # Results state
        'sort_method': 'Duration & Altitude', # 'Duration & Altitude' or 'Brightness'
        'num_objects_slider': 20, # Max number of results to show
        'plot_type_selection': 'Sky Path', # 'Sky Path' or 'Altitude Plot'
        # Custom Target state
        'custom_target_ra': "",
        'custom_target_dec': "",
        'custom_target_name': "",
        'custom_target_error': "", # Error message for custom target section
        'custom_target_plot_data': None, # Data dict for the custom plot
        'show_custom_plot': False, # Flag to show the custom plot
        # Other state
        'expanded_object_name': None, # Name of the result expander currently open
        'time_choice_exp': 'Now', # 'Now' or 'Specific'
        'window_start_time': None, # Astropy Time object for observation window start
        'window_end_time': None,   # Astropy Time object for observation window end
        'selected_date_widget': date.today(), # Date selected in the 'Specific Night' picker
    }
    # Initialize session state keys if they don't exist
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    # --- Correction: Ensure language key in session state is uppercase ---
    if 'language' in st.session_state:
        st.session_state.language = st.session_state.language.upper()
    # --- End Correction ---


# --- Helper Functions REMAINING in main script ---
# Most helpers moved to ui_components or astro_calculations

# --- Main App Logic ---
def main():
    # 1. Initialize state and load language translations
    initialize_session_state() # Ensures state exists and language is uppercase
    # --- Correction: Use uppercase language key from state ---
    lang = st.session_state.language # Now guaranteed to be uppercase ('DE', 'EN', 'FR', etc.)
    if lang not in translations:
        print(f"Warning: Language '{lang}' not found in translations dictionary. Falling back to 'EN'.")
        lang = 'EN' # Fallback to uppercase 'EN'
        st.session_state.language = lang # Update state if fallback occurred
    # Load translation dict 't' using the (now guaranteed uppercase) language key
    # Use .get() for safety, although 'EN' should always exist if localization.py is correct
    t = translations.get(lang, translations.get('EN', {})) # Fallback to EN dict or empty dict
    # --- End Correction ---

    # 2. Load Catalog Data (Cached)
    # Define caching function locally or ensure data_handling is imported correctly
    # Corrected type hint using imported pandas as pd
    @st.cache_data
    def cached_load_ongc_data(path: str) -> pd.DataFrame | None:
        """Cached function to load ONGC data."""
        print(f"Cache miss: Loading ONGC data from {path}")
        # Ensure data_handling module is accessible
        try:
            # Assuming data_handling.py is in the same directory or PYTHONPATH
            return data_handling.load_ongc_data(path)
        except ModuleNotFoundError:
             # Use translated error message
             st.error(f"{t.get('error_module_missing', 'Error: Module missing:')} data_handling.py")
             return None
        except FileNotFoundError:
             st.error(f"{t.get('error_catalog_not_found', 'Error: Catalog file not found at path:')} {path}")
             return None
        except Exception as load_err:
            # Use translated error message
            st.error(f"{t.get('error_catalog_load_failed', 'Failed to load catalog')}: {load_err}")
            return None

    df_catalog_data = cached_load_ongc_data(CATALOG_FILEPATH)

    # 3. Display Title and Glossary (Simple version kept here)
    st.title(t.get('app_title', "Advanced DSO Finder"))
    with st.expander(t.get('object_type_glossary_title', "Object Type Glossary")):
        glossary_items = t.get('object_type_glossary', {})
        if glossary_items:
            # Simple two-column layout for glossary terms
            col1, col2 = st.columns(2)
            col_index = 0
            # Sort terms alphabetically for consistent display
            sorted_items = sorted(glossary_items.items())
            for abbr, full_name in sorted_items:
                target_col = col1 if col_index % 2 == 0 else col2
                target_col.markdown(f"**{abbr}:** {full_name}")
                col_index += 1
        else:
            # Message if glossary is not available for the language
            st.info(t.get('glossary_unavailable', "Glossary not available for the selected language."))

    st.markdown("---") # Visual separator

    # 4. Create Sidebar UI (using the imported function)
    # Pass the translation dict 't', catalog data, and timezone finder 'tf'
    ui_components.create_sidebar(t, df_catalog_data, tf)

    # 5. Prepare Observer Object (based on valid location state from sidebar)
    observer_run = None
    if st.session_state.location_is_valid_for_run:
        # Retrieve validated location details from session state
        lat = st.session_state.manual_lat_val
        lon = st.session_state.manual_lon_val
        hgt = st.session_state.manual_height_val
        tz_str = st.session_state.selected_timezone
        try:
            # Attempt to create the Observer object using astroplan
            observer_run = Observer(latitude=lat*u.deg, longitude=lon*u.deg, elevation=hgt*u.m, timezone=tz_str)
        except Exception as obs_err:
            # Handle potential errors during Observer creation (e.g., invalid timezone string)
            st.error(t.get('error_observer_creation', "Error creating observer location: {}").format(obs_err))
            # Invalidate location for run if observer fails
            st.session_state.location_is_valid_for_run = False
            observer_run = None # Ensure observer is None

    # 6. Determine Reference Time for Calculations (based on sidebar time selection)
    ref_time = None
    is_now_mode_main = (st.session_state.time_choice_exp == "Now")
    if is_now_mode_main:
        ref_time = Time.now() # Use current time if 'Now' is selected
    else:
        # Use selected date (noon UTC) for 'Specific Night' mode
        selected_date_main = st.session_state.selected_date_widget
        try:
            # Combine date with noon time, explicitly set scale to UTC
             ref_time = Time(datetime.combine(selected_date_main, time(12, 0)), scale='utc')
             print(f"Calculating 'Specific Night' window based on UTC noon: {ref_time.iso}")
        except Exception as time_err:
             # Handle errors creating the reference time
             st.error(t.get('error_ref_time_creation', "Error setting reference time: {}").format(time_err))
             # Cannot proceed without a valid reference time
             ref_time = None


    # 7. Display Search Parameters Summary (using imported UI function)
    # This function now also returns the actual magnitude filter values used for calculations
    # Provide a default Time.now() if ref_time failed to avoid errors in the UI function
    min_mag_filt_calc, max_mag_filt_calc = ui_components.display_search_parameters(
        t, observer_run, ref_time if ref_time else Time.now()
    )

    st.markdown("---") # Visual separator

    # 8. "Find Objects" Button and Core Logic Trigger
    # Placeholder container for results and status messages below the button
    results_placeholder = st.container()

    # Disable button if prerequisites are not met
    find_button_disabled = (df_catalog_data is None or not st.session_state.location_is_valid_for_run or ref_time is None)
    find_button_clicked = st.button(
        t.get('find_button_label', "🔭 Find Observable Objects"),
        key="find_button", # Consistent key for the button
        disabled=find_button_disabled
    )

    # Show initial prompt or warning if button is disabled
    if not st.session_state.location_is_valid_for_run and df_catalog_data is not None:
        st.warning(t.get('info_initial_prompt', "Please set a valid location in the sidebar to enable search."))
    elif ref_time is None and not is_now_mode_main:
         st.warning(t.get('info_set_ref_time', "Please ensure a valid date is selected."))
    elif df_catalog_data is None:
        st.warning(t.get('info_catalog_missing', "Catalog data could not be loaded. Search disabled."))


    # --- Main Calculation Logic ---
    if find_button_clicked:
        # Reset relevant state variables before starting a new search
        st.session_state.find_button_pressed = True
        st.session_state.show_plot = False
        st.session_state.show_custom_plot = False
        st.session_state.active_result_plot_data = None
        st.session_state.custom_target_plot_data = None
        st.session_state.last_results = []
        st.session_state.window_start_time = None
        st.session_state.window_end_time = None
        st.session_state.expanded_object_name = None # Collapse all result expanders
        # Reset cosmology display states if managed centrally (alternative to per-button toggle)
        # st.session_state.cosmology_display_state = {}

        # Proceed only if observer, catalog data, and ref_time are valid
        if observer_run and df_catalog_data is not None and ref_time is not None:
            with st.spinner(t.get('spinner_searching',"Searching for observable objects...")):
                try:
                    # a. Calculate Observable Window using astro_calculations module
                    start_time_calc, end_time_calc, window_status_msg = astro_calculations.get_observable_window(
                        observer_run, ref_time, is_now_mode_main, t
                    )
                    results_placeholder.info(window_status_msg) # Display window status message
                    # Store calculated window times in session state
                    st.session_state.window_start_time = start_time_calc
                    st.session_state.window_end_time = end_time_calc

                    # b. Proceed only if a valid window was found
                    if start_time_calc and end_time_calc and start_time_calc < end_time_calc:
                        # Generate observation times within the window
                        time_resolution_calc = 5 * u.minute # Time step for calculations
                        observation_times_calc = Time(
                            np.arange(start_time_calc.jd, end_time_calc.jd, time_resolution_calc.to(u.day).value),
                            format='jd', scale='utc'
                        )

                        if len(observation_times_calc) < 2:
                            # Window is too short for meaningful calculation across time steps
                            results_placeholder.warning(t.get('warning_window_too_short_calc', "Observation window is very short. Results based on limited time points."))
                            # Still allow calculation, might catch objects near window edge

                        # c. Filter Catalog Data based on sidebar settings
                        filtered_df = df_catalog_data.copy()
                        # Apply magnitude filter (using values derived earlier)
                        filtered_df = filtered_df[(filtered_df['Mag'] >= min_mag_filt_calc) & (filtered_df['Mag'] <= max_mag_filt_calc)]
                        # Apply type filter
                        selected_types_calc = st.session_state.object_type_filter_exp
                        if selected_types_calc:
                            filtered_df = filtered_df[filtered_df['Type'].isin(selected_types_calc)]
                        # Apply size filter (check availability again, column might be missing)
                        size_data_ok_calc = 'MajAx' in filtered_df.columns and filtered_df['MajAx'].notna().any()
                        if size_data_ok_calc:
                            size_min_calc, size_max_calc = st.session_state.size_arcmin_range
                            filtered_df = filtered_df.dropna(subset=['MajAx']) # Drop rows where size is NA before filtering
                            filtered_df = filtered_df[(filtered_df['MajAx'] >= size_min_calc) & (filtered_df['MajAx'] <= size_max_calc)]

                        # d. Check if any objects remain after initial filtering
                        if filtered_df.empty:
                            results_placeholder.warning(t.get('warning_no_objects_after_filters',"No objects match the selected magnitude, type, or size filters."))
                            st.session_state.last_results = [] # Ensure results list is empty
                        else:
                            # e. Find Observable Objects (Core Calculation using astro_calculations)
                            min_alt_search_calc = st.session_state.min_alt_slider * u.deg
                            # Call the main calculation function
                            found_objects = astro_calculations.find_observable_objects(
                                observer_run.location,
                                observation_times_calc,
                                min_alt_search_calc,
                                filtered_df,
                                t # Pass translation dict for potential internal messages
                            )

                            # f. Apply Post-Calculation Filters (Max Altitude, Direction)
                            final_results = []
                            selected_direction_calc = st.session_state.selected_peak_direction
                            max_alt_filter_calc = st.session_state.max_alt_slider
                            for obj_result in found_objects:
                                # Max altitude filter
                                if obj_result.get('Max Altitude (°)', -999) > max_alt_filter_calc:
                                    continue
                                # Direction filter (skip if 'All' is selected)
                                if selected_direction_calc != ALL_DIRECTIONS_KEY and obj_result.get('Direction at Max') != selected_direction_calc:
                                    continue
                                final_results.append(obj_result)

                            # g. Sort Results based on sidebar selection
                            sort_key = st.session_state.sort_method
                            if sort_key == 'Brightness':
                                # Sort by magnitude (ascending, handle None by putting them last)
                                final_results.sort(key=lambda x: x.get('Magnitude', float('inf')) if x.get('Magnitude') is not None else float('inf'))
                            else: # Default: Duration & Altitude
                                # Sort primarily by duration (desc), secondarily by max altitude (desc)
                                final_results.sort(key=lambda x: (x.get('Max Cont. Duration (h)', 0), x.get('Max Altitude (°)', 0)), reverse=True)

                            # h. Limit number of results and store in session state
                            num_to_show = st.session_state.num_objects_slider
                            st.session_state.last_results = final_results[:num_to_show]

                            # i. Display final status messages based on results
                            if not final_results:
                                results_placeholder.warning(t.get('warning_no_objects_found_final',"No objects found matching all criteria (including altitude and direction)."))
                            else:
                                results_placeholder.success(t.get('success_objects_found',"{} objects found matching criteria.").format(len(final_results)))
                                # Provide info about sorting and number shown
                                sort_info_key = 'info_showing_list_duration' if sort_key != 'Brightness' else 'info_showing_list_magnitude'
                                results_placeholder.info(t.get(sort_info_key, "Showing top {} results sorted by {}.").format(len(st.session_state.last_results), sort_key))

                    else:
                        # Error message if no valid observation window was found
                        # The specific message is already shown by get_observable_window
                        results_placeholder.error(t.get('error_cannot_search_no_window',"Cannot perform search without a valid observation window."))
                        st.session_state.last_results = [] # Ensure results list is empty

                except Exception as e:
                    # Catch-all for unexpected errors during the search process
                    error_msg = t.get('error_search_unexpected',"An unexpected error occurred during the search:")
                    # Use traceback for detailed error logging
                    results_placeholder.error(f"{error_msg}\n```\n{traceback.format_exc()}\n```")
                    print(f"Search Error: {e}")
                    traceback.print_exc()
                    st.session_state.last_results = [] # Clear results on error
        else:
            # Handle cases where button was clicked but prerequisites failed (redundant check, but safe)
            if df_catalog_data is None:
                results_placeholder.error(t.get('error_prereq_catalog',"Error: Catalog data not loaded."))
            if not observer_run:
                 results_placeholder.error(t.get('error_prereq_location',"Error: Location is not valid."))
            if ref_time is None:
                 results_placeholder.error(t.get('error_prereq_time',"Error: Reference time is not valid."))
            st.session_state.last_results = [] # Ensure results are cleared

    # 9. Display Results (if any exist in state) using the UI component
    if st.session_state.last_results:
        # Call the UI function to display the results list, plots, download button, cosmology section
        # Pass the placeholder, observer object (needed for plot calculations within UI), and translations
        ui_components.display_results(t, results_placeholder, observer_run)
    elif st.session_state.find_button_pressed:
        # If button was pressed but no results (e.g., due to filters or errors handled above),
        # an appropriate message should already be in results_placeholder.
        # No additional message needed here unless specific info is desired.
        pass

    # 10. Display Custom Target Section (using imported UI function)
    # Pass the placeholder, observer object (needed for plot calculations), and translations
    ui_components.create_custom_target_section(t, results_placeholder, observer_run)

    # --- NEU: Aufruf des manuellen Kosmologie-Rechners ---
    # Dieser wird jetzt immer angezeigt, unabhängig von den Suchergebnissen
    ui_components.create_manual_cosmology_calculator(t)

    # 11. Display Donation Link (using imported UI function)
    ui_components.display_donation_link(t)

# --- Run the App ---
if __name__ == "__main__":
    # This ensures the main function runs only when the script is executed directly
    main()
