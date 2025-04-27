# -*- coding: utf-8 -*-
from __future__ import annotations
import streamlit as st
# Removed random, math imports if only used by moved functions
from datetime import datetime, date, time, timedelta, timezone
import traceback
import os
# Removed urllib, pandas imports if only used by moved functions
# Removed matplotlib imports

# --- Library Imports ---
try:
    from astropy.time import Time
    import numpy as np
    import astropy.units as u
    # Removed EarthLocation, get_sun, get_constellation if only used by moved functions
    from astropy.coordinates import SkyCoord, AltAz # Keep SkyCoord, AltAz if used for calculation
    from astroplan import Observer
    # Removed moon_illumination if only used by moved functions
    import pytz # Keep if used for timezone handling here
    from timezonefinder import TimezoneFinder
    # Removed geopy imports if only used by moved functions
except ImportError as e:
    st.error(f"Import Error: Missing libraries. Please install required packages (check requirements.txt). Details: {e}")
    st.stop()

# --- Import Custom Modules ---
try:
    from localization import translations
    import astro_calculations
    import data_handling
    # Import UI functions from the new module
    import ui_components
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
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    APP_DIR = os.getcwd()
CATALOG_FILENAME = "ongc.csv"
CATALOG_FILEPATH = os.path.join(APP_DIR, CATALOG_FILENAME)

# --- Constants ---
# Keep constants used in this file (if any)
# ALL_DIRECTIONS_KEY might be needed if used in filtering logic here, otherwise can remove
ALL_DIRECTIONS_KEY = 'All'


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
        'language': 'de',
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
        'selected_date_widget': date.today() # Date selected in the 'Specific Night' picker
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# --- Helper Functions REMAINING in main script ---
# Moved create_moon_phase_svg, get_local_time_str, create_plot to ui_components.py

# --- Main App Logic ---
def main():
    # 1. Initialize state and load language translations
    initialize_session_state()
    lang = st.session_state.language
    if lang not in translations:
        print(f"Warning: Language '{lang}' not found in translations. Falling back to 'en'.")
        lang = 'en'
        st.session_state.language = lang # Correct state if invalid lang was somehow set
    t = translations.get(lang, translations['en']) # Load translation dict

    # 2. Load Catalog Data (Cached)
    # Define caching function locally or ensure data_handling is imported correctly
    @st.cache_data
    def cached_load_ongc_data(path):
        """Cached function to load ONGC data."""
        print(f"Cache miss: Loading ONGC data from {path}")
        # Ensure data_handling module is accessible
        try:
            import data_handling
            return data_handling.load_ongc_data(path)
        except ModuleNotFoundError:
             st.error(f"{t.get('error_module_missing', 'Error: Module missing:')} data_handling.py")
             return None
        except Exception as load_err:
            st.error(f"{t.get('error_catalog_load_failed', 'Failed to load catalog')}: {load_err}")
            return None

    df_catalog_data = cached_load_ongc_data(CATALOG_FILEPATH)

    # 3. Display Title and Glossary
    st.title(t.get('app_title', "Advanced DSO Finder"))
    # Display glossary using UI component function or keep simple version here
    with st.expander(t.get('object_type_glossary_title', "Object Type Glossary")):
        glossary_items = t.get('object_type_glossary', {})
        if glossary_items:
            col1, col2 = st.columns(2); col_index = 0
            sorted_items = sorted(glossary_items.items())
            for abbr, full_name in sorted_items:
                target_col = col1 if col_index % 2 == 0 else col2
                target_col.markdown(f"**{abbr}:** {full_name}")
                col_index += 1
        else: st.info(t.get('glossary_unavailable', "Glossary not available for the selected language."))

    st.markdown("---")

    # 4. Create Sidebar UI (using the imported function)
    # Pass the timezone finder instance (tf)
    ui_components.create_sidebar(t, df_catalog_data, tf)

    # 5. Prepare Observer Object (based on valid location state)
    observer_run = None
    if st.session_state.location_is_valid_for_run:
        lat = st.session_state.manual_lat_val
        lon = st.session_state.manual_lon_val
        hgt = st.session_state.manual_height_val
        tz_str = st.session_state.selected_timezone
        try:
            # Attempt to create the Observer object
            observer_run = Observer(latitude=lat*u.deg, longitude=lon*u.deg, elevation=hgt*u.m, timezone=tz_str)
        except Exception as obs_err:
            # Handle potential errors during Observer creation (e.g., invalid timezone string)
            st.error(t.get('error_observer_creation', "Error creating observer location: {}").format(obs_err))
            # Invalidate location for run if observer fails
            st.session_state.location_is_valid_for_run = False
            observer_run = None # Ensure observer is None

    # 6. Determine Reference Time for Calculations
    ref_time = None
    is_now_mode_main = (st.session_state.time_choice_exp == "Now")
    if is_now_mode_main:
        ref_time = Time.now() # Use current time
    else:
        # Use selected date (noon UTC) for 'Specific Night' mode
        selected_date_main = st.session_state.selected_date_widget
        try:
            # Combine date with noon time, explicitly set scale to UTC
             ref_time = Time(datetime.combine(selected_date_main, time(12, 0)), scale='utc')
             print(f"Calculating 'Specific Night' window based on UTC noon: {ref_time.iso}")
        except Exception as time_err:
             st.error(t.get('error_ref_time_creation', "Error setting reference time: {}").format(time_err))
             # Cannot proceed without a valid reference time
             ref_time = None


    # 7. Display Search Parameters Summary (using imported function)
    # This function now also returns the actual magnitude filter values used
    min_mag_filt_calc, max_mag_filt_calc = ui_components.display_search_parameters(t, observer_run, ref_time if ref_time else Time.now()) # Pass ref_time

    st.markdown("---")

    # 8. "Find Objects" Button and Core Logic Trigger
    results_placeholder = st.container() # Placeholder for messages and results list/plots

    find_button_clicked = st.button(
        t.get('find_button_label', "🔭 Find Observable Objects"),
        key="find_button",
        disabled=(df_catalog_data is None or not st.session_state.location_is_valid_for_run or ref_time is None) # Disable if data/location/time invalid
    )

    # Show initial prompt if location is not yet valid
    if not st.session_state.location_is_valid_for_run and df_catalog_data is not None:
        st.warning(t.get('info_initial_prompt', "Please set a valid location in the sidebar to enable search."))
    elif ref_time is None and not is_now_mode_main:
         st.warning(t.get('info_set_ref_time', "Please ensure a valid date is selected."))


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
        st.session_state.expanded_object_name = None # Collapse all expanders

        # Proceed only if observer and catalog data are valid
        if observer_run and df_catalog_data is not None and ref_time is not None:
            with st.spinner(t.get('spinner_searching',"Searching for observable objects...")):
                try:
                    # a. Calculate Observable Window
                    start_time_calc, end_time_calc, window_status_msg = astro_calculations.get_observable_window(observer_run, ref_time, is_now_mode_main, t)
                    results_placeholder.info(window_status_msg) # Display window status message
                    st.session_state.window_start_time = start_time_calc
                    st.session_state.window_end_time = end_time_calc

                    # b. Proceed only if a valid window was found
                    if start_time_calc and end_time_calc and start_time_calc < end_time_calc:
                        # Generate observation times within the window
                        time_resolution_calc = 5 * u.minute
                        observation_times_calc = Time(np.arange(start_time_calc.jd, end_time_calc.jd, time_resolution_calc.to(u.day).value), format='jd', scale='utc')

                        if len(observation_times_calc) < 2:
                            results_placeholder.warning(t.get('warning_window_too_short', "Observation window is very short, results may be limited."))
                            # Allow proceeding even if short, might find something at edge

                        # c. Filter Catalog Data
                        filtered_df = df_catalog_data.copy()
                        # Apply magnitude filter (using values derived in display_search_parameters)
                        filtered_df = filtered_df[(filtered_df['Mag'] >= min_mag_filt_calc) & (filtered_df['Mag'] <= max_mag_filt_calc)]
                        # Apply type filter
                        selected_types_calc = st.session_state.object_type_filter_exp
                        if selected_types_calc:
                            filtered_df = filtered_df[filtered_df['Type'].isin(selected_types_calc)]
                        # Apply size filter (check availability again)
                        size_data_ok_calc = 'MajAx' in filtered_df.columns and filtered_df['MajAx'].notna().any()
                        if size_data_ok_calc:
                            size_min_calc, size_max_calc = st.session_state.size_arcmin_range
                            filtered_df = filtered_df.dropna(subset=['MajAx']) # Drop rows where size is NA
                            filtered_df = filtered_df[(filtered_df['MajAx'] >= size_min_calc) & (filtered_df['MajAx'] <= size_max_calc)]

                        # d. Check if any objects remain after initial filtering
                        if filtered_df.empty:
                            results_placeholder.warning(t.get('warning_no_objects_after_filters',"No objects match the selected magnitude, type, or size filters."))
                            st.session_state.last_results = []
                        else:
                            # e. Find Observable Objects (Core Calculation)
                            min_alt_search_calc = st.session_state.min_alt_slider * u.deg
                            # Call the calculation function from the astro_calculations module
                            found_objects = astro_calculations.find_observable_objects(
                                observer_run.location,
                                observation_times_calc,
                                min_alt_search_calc,
                                filtered_df,
                                t # Pass translation dict for potential messages inside calculation
                            )

                            # f. Apply Post-Calculation Filters (Max Altitude, Direction)
                            final_results = []
                            selected_direction_calc = st.session_state.selected_peak_direction
                            max_alt_filter_calc = st.session_state.max_alt_slider
                            for obj_result in found_objects:
                                # Max altitude filter
                                if obj_result.get('Max Altitude (°)', -999) > max_alt_filter_calc:
                                    continue
                                # Direction filter
                                if selected_direction_calc != ALL_DIRECTIONS_KEY and obj_result.get('Direction at Max') != selected_direction_calc:
                                    continue
                                final_results.append(obj_result)

                            # g. Sort Results
                            sort_key = st.session_state.sort_method
                            if sort_key == 'Brightness':
                                # Sort by magnitude (ascending, handle None)
                                final_results.sort(key=lambda x: x.get('Magnitude', float('inf')) if x.get('Magnitude') is not None else float('inf'))
                            else: # Default: Duration & Altitude
                                # Sort primarily by duration (desc), secondarily by max altitude (desc)
                                final_results.sort(key=lambda x: (x.get('Max Cont. Duration (h)', 0), x.get('Max Altitude (°)', 0)), reverse=True)

                            # h. Limit number of results and store in session state
                            num_to_show = st.session_state.num_objects_slider
                            st.session_state.last_results = final_results[:num_to_show]

                            # i. Display final status messages
                            if not final_results:
                                results_placeholder.warning(t.get('warning_no_objects_found_final',"No objects found matching all criteria (including altitude and direction)."))
                            else:
                                results_placeholder.success(t.get('success_objects_found',"{} objects found matching criteria.").format(len(final_results)))
                                sort_info_key = 'info_showing_list_duration' if sort_key != 'Brightness' else 'info_showing_list_magnitude'
                                results_placeholder.info(t.get(sort_info_key, "Showing top {} results sorted by {}.").format(len(st.session_state.last_results), sort_key))

                    else:
                        # Error message if no valid observation window was found
                        # The specific message is already shown by get_observable_window
                        results_placeholder.error(t.get('error_cannot_search_no_window',"Cannot perform search without a valid observation window."))
                        st.session_state.last_results = []

                except Exception as e:
                    # Catch-all for unexpected errors during the search process
                    error_msg = t.get('error_search_unexpected',"An unexpected error occurred during the search:")
                    results_placeholder.error(f"{error_msg}\n```\n{traceback.format_exc()}\n```")
                    print(f"Search Error: {e}")
                    traceback.print_exc()
                    st.session_state.last_results = []
        else:
            # Handle cases where button was clicked but prerequisites failed
            if df_catalog_data is None:
                results_placeholder.error(t.get('error_prereq_catalog',"Error: Catalog data not loaded."))
            if not observer_run:
                 results_placeholder.error(t.get('error_prereq_location',"Error: Location is not valid."))
            if ref_time is None:
                 results_placeholder.error(t.get('error_prereq_time',"Error: Reference time is not valid."))
            st.session_state.last_results = [] # Ensure results are cleared

    # 9. Display Results (if any exist in state)
    if st.session_state.last_results:
        # Call the UI function to display the results list, plots, download button
        ui_components.display_results(t, results_placeholder, observer_run)
    elif st.session_state.find_button_pressed:
        # If button was pressed but no results (handled above), show a message
        # (This might be redundant if messages are shown during the process)
        # results_placeholder.info(t.get('info_no_results_to_display',"No results to display."))
        pass # Messages are likely already displayed within the find_button_clicked block

    # 10. Display Custom Target Section (using imported function)
    # Pass observer_run needed for calculations within the section
    ui_components.create_custom_target_section(t, results_placeholder, observer_run)

    # 11. Display Donation Link (using imported function)
    ui_components.display_donation_link(t)

# --- Run the App ---
if __name__ == "__main__":
    main()
