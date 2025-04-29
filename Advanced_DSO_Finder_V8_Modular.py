# -*- coding: utf-8 -*-
# --- Basic Imports ---
from __future__ import annotations
import streamlit as st
import random
from datetime import datetime, date, time, timedelta, timezone
import traceback
import os  # Needed for file path joining
import urllib.parse # Needed for robust URL encoding
import pandas as pd
import math # For isnan check

# --- Library Imports (Try after set_page_config) ---
try:
    from astropy.time import Time
    import numpy as np
    import astropy.units as u
    from astropy.coordinates import EarthLocation, SkyCoord, get_sun, AltAz, get_constellation
    from astroplan import Observer
    from astroplan.moon import moon_illumination
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pytz
    from timezonefinder import TimezoneFinder
    from geopy.geocoders import Nominatim, ArcGIS, Photon
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
except ImportError as e:
    st.error(f"Error: Missing libraries. Please install the required packages. ({e})")
    st.stop()

# --- Localization Import ---
from localization import get_translation # <--- HINZUGEFÜGT

# --- Page Config (MUST BE FIRST Streamlit command) ---
st.set_page_config(page_title="Advanced DSO Finder", layout="wide")

# --- Global Configuration & Initial Values ---
INITIAL_LAT = 47.17
INITIAL_LON = 8.01
INITIAL_HEIGHT = 550
INITIAL_TIMEZONE = "Europe/Zurich"

# --- Path to Catalog File ---
try:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    APP_DIR = os.getcwd()
CATALOG_FILENAME = "ongc.csv"
CATALOG_FILEPATH = os.path.join(APP_DIR, CATALOG_FILENAME)

# Define cardinal directions
CARDINAL_DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
ALL_DIRECTIONS_KEY = 'All' # Internal key for 'All' option

# --- Translations Dictionary Removed (Now in localization.py) ---

# --- Initialize TimezoneFinder (cached) ---
@st.cache_resource
def get_timezone_finder():
    """Initializes and returns a TimezoneFinder instance."""
    if TimezoneFinder:
        try:
            return TimezoneFinder(in_memory=True)
        except Exception as e:
            print(f"Error initializing TimezoneFinder: {e}")
            st.warning(f"TimezoneFinder init failed: {e}. Automatic timezone detection disabled.")
            return None
    return None

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
        'mag_filter_mode_exp': 'Bortle Scale', # Use internal key
        'bortle_slider': 5,
        'min_alt_slider': 20,
        'max_alt_slider': 90,
        'moon_phase_slider': 35,
        'size_arcmin_range': [1.0, 120.0],
        'sort_method': 'Duration & Altitude', # Use internal key
        'selected_peak_direction': ALL_DIRECTIONS_KEY,
        'plot_type_selection': 'Sky Path', # Use internal key
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
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# --- Helper Functions ---
def get_magnitude_limit(bortle_scale: int) -> float:
    """Calculates the approximate limiting magnitude based on Bortle scale."""
    limits = {1: 15.5, 2: 15.5, 3: 14.5, 4: 14.5, 5: 13.5, 6: 12.5, 7: 11.5, 8: 10.5, 9: 9.5}
    return limits.get(bortle_scale, 9.5)

def azimuth_to_direction(azimuth_deg: float) -> str:
    """Converts an azimuth angle (degrees) to a cardinal direction string."""
    if math.isnan(azimuth_deg):
        return "N/A"
    azimuth_deg = azimuth_deg % 360
    index = round((azimuth_deg + 22.5) / 45) % 8
    index = max(0, min(index, len(CARDINAL_DIRECTIONS) - 1))
    return CARDINAL_DIRECTIONS[index]

# --- Moon Phase SVG (Corrected) ---
def create_moon_phase_svg(illumination: float, size: int = 100) -> str:
    """Creates an SVG representation of the moon phase (corrected)."""
    if not 0 <= illumination <= 1:
        print(f"Warning: Invalid moon illumination value ({illumination}). Clamping to [0, 1].")
        illumination = max(0.0, min(1.0, illumination))

    radius = size / 2
    cx = cy = radius
    light_color = "var(--text-color, #e0e0e0)"
    dark_color = "var(--secondary-background-color, #333333)"

    svg = f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
    svg += f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{dark_color}"/>'

    if illumination < 0.01: pass
    elif illumination > 0.99:
        svg += f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{light_color}"/>'
    else:
        x_terminator_center = radius * (illumination * 2 - 1)
        rx_terminator = abs(x_terminator_center)
        if illumination <= 0.5:
            large_arc_flag_ellipse = 0; sweep_flag_ellipse = 1
            large_arc_flag_circle = 0; sweep_flag_circle = 1
            d = (f"M {cx},{cy - radius} "
                 f"A {rx_terminator},{radius} 0 {large_arc_flag_ellipse},{sweep_flag_ellipse} {cx},{cy + radius} "
                 f"A {radius},{radius} 0 {large_arc_flag_circle},{sweep_flag_circle} {cx},{cy - radius} Z")
        else:
            large_arc_flag_circle = 1; sweep_flag_circle = 1
            large_arc_flag_ellipse = 1; sweep_flag_ellipse = 1
            d = (f"M {cx},{cy - radius} "
                 f"A {radius},{radius} 0 {large_arc_flag_circle},{sweep_flag_circle} {cx},{cy + radius} "
                 f"A {rx_terminator},{radius} 0 {large_arc_flag_ellipse},{sweep_flag_ellipse} {cx},{cy - radius} Z")
        svg += f'<path d="{d}" fill="{light_color}"/>'
    svg += '</svg>'
    return svg

def load_ongc_data(catalog_path: str, lang: str) -> pd.DataFrame | None:
    """Loads, filters, and preprocesses data from the OpenNGC CSV file."""
    t_load = get_translation(lang) # <--- GEÄNDERT
    required_cols = ['Name', 'RA', 'Dec', 'Type']
    mag_cols = ['V-Mag', 'B-Mag', 'Mag']
    size_col = 'MajAx'

    try:
        if not os.path.exists(catalog_path):
             st.error(f"{t_load.get('error_loading_catalog', 'Error loading catalog file:').split(':')[0]}: File not found at {catalog_path}")
             st.info(f"Please ensure the file '{CATALOG_FILENAME}' is in the directory: {APP_DIR}")
             return None

        df = pd.read_csv(catalog_path, sep=';', comment='#', low_memory=False)

        missing_req_cols = [col for col in required_cols if col not in df.columns]
        if missing_req_cols:
            st.error(f"Missing required columns in catalog '{os.path.basename(catalog_path)}': {', '.join(missing_req_cols)}")
            return None

        df['RA_str'] = df['RA'].astype(str).str.strip()
        df['Dec_str'] = df['Dec'].astype(str).str.strip()
        df.dropna(subset=['RA_str', 'Dec_str'], inplace=True)
        df = df[(df['RA_str'] != '') & (df['Dec_str'] != '')]

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

        if size_col not in df.columns:
            st.warning(f"Size column '{size_col}' not found in catalog. Angular size filtering will be disabled.")
            df[size_col] = np.nan
        else:
            df[size_col] = pd.to_numeric(df[size_col], errors='coerce')
            if not df[size_col].notna().any():
                st.warning(f"No valid numeric data found in size column '{size_col}' after cleaning. Size filter disabled.")
                df[size_col] = np.nan

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

        final_cols = ['Name', 'RA_str', 'Dec_str', 'Mag', 'Type', size_col]
        final_cols_exist = [col for col in final_cols if col in df_filtered.columns]
        df_final = df_filtered[final_cols_exist].copy()

        df_final.drop_duplicates(subset=['Name'], inplace=True, keep='first')
        df_final.reset_index(drop=True, inplace=True)

        if not df_final.empty:
            print(f"Catalog loaded and processed: {len(df_final)} objects.")
            return df_final
        else:
            st.warning(t_load.get('warning_catalog_empty', 'Catalog file loaded, but no matching objects found after filtering.'))
            return None

    except FileNotFoundError:
        st.error(f"{t_load.get('error_loading_catalog', 'Error loading catalog file:').split(':')[0]}: File not found at {catalog_path}")
        st.info(f"Please ensure the file '{CATALOG_FILENAME}' is in the directory: {APP_DIR}")
        return None
    except pd.errors.ParserError as e:
        st.error(f"Error parsing catalog file '{os.path.basename(catalog_path)}': {e}")
        st.info("Please ensure the file is a valid CSV with ';' separator.")
        return None
    except Exception as e:
        st.error(f"{t_load.get('error_loading_catalog', 'Error loading catalog file:')} An unexpected error occurred: {e}")
        traceback.print_exc()
        return None

# --- Fallback Window ---
def _get_fallback_window(reference_time: Time) -> tuple[Time, Time]:
    """Provides a simple fallback observation window (e.g., 6 PM to 6 AM UTC)."""
    ref_dt_utc = reference_time.to_datetime(timezone.utc)
    ref_date = ref_dt_utc.date()
    fallback_start_dt = datetime.combine(ref_date, time(18, 0), tzinfo=timezone.utc)
    fallback_end_dt = datetime.combine(ref_date + timedelta(days=1), time(6, 0), tzinfo=timezone.utc)
    fallback_start_time = Time(fallback_start_dt, scale='utc')
    fallback_end_time = Time(fallback_end_dt, scale='utc')
    print(f"Using fallback window: {fallback_start_time.iso} to {fallback_end_time.iso}")
    return fallback_start_time, fallback_end_time

# --- Observation Window Calculation ---
def get_observable_window(observer: Observer, reference_time: Time, is_now: bool, lang: str) -> tuple[Time | None, Time | None, str]:
    """Calculates the astronomical darkness window for observation."""
    t = get_translation(lang) # <--- GEÄNDERT
    status_message = ""
    start_time, end_time = None, None
    current_utc_time = Time.now()

    calc_base_time = reference_time
    if is_now:
        current_dt_utc = current_utc_time.to_datetime(timezone.utc)
        noon_today_utc = datetime.combine(current_dt_utc.date(), time(12, 0), tzinfo=timezone.utc)
        if current_dt_utc < noon_today_utc:
            calc_base_time = Time(noon_today_utc - timedelta(days=1))
        else:
            calc_base_time = Time(noon_today_utc)
        print(f"Calculating 'Now' window based on UTC noon: {calc_base_time.iso}")
    else:
        selected_date_noon_utc = datetime.combine(reference_time.to_datetime(timezone.utc).date(), time(12, 0), tzinfo=timezone.utc)
        calc_base_time = Time(selected_date_noon_utc, scale='utc')
        print(f"Calculating specific night window based on UTC noon: {calc_base_time.iso}")

    try:
        if not isinstance(observer, Observer):
            raise TypeError(f"Internal Error: Expected astroplan.Observer, got {type(observer)}")

        astro_set = observer.twilight_evening_astronomical(calc_base_time, which='next')
        astro_rise = observer.twilight_morning_astronomical(astro_set if astro_set else calc_base_time, which='next')

        if astro_set is None or astro_rise is None:
            raise ValueError("Could not determine one or both astronomical twilight times.")
        if astro_rise <= astro_set:
            try:
                sun_alt_ref = observer.sun_altaz(calc_base_time).alt
                sun_alt_12h_later = observer.sun_altaz(calc_base_time + 12*u.hour).alt
                if sun_alt_ref < -18*u.deg and sun_alt_12h_later < -18*u.deg:
                    status_message = t.get('error_polar_night', "Astronomical darkness lasts >24h (Polar night?). Using fallback window.")
                    start_time, end_time = _get_fallback_window(calc_base_time)
                    status_message += t.get('window_fallback_info', "\nUsing fallback window: {} to {} UTC").format(start_time.iso, end_time.iso)
                    return start_time, end_time, status_message
                elif sun_alt_ref > -18*u.deg:
                    times_check = calc_base_time + np.linspace(0, 24, 49)*u.hour
                    sun_alts_check = observer.sun_altaz(times_check).alt
                    if np.min(sun_alts_check) > -18*u.deg:
                        status_message = t.get('error_polar_day', "No astronomical darkness occurs (Polar day?). Using fallback window.")
                        start_time, end_time = _get_fallback_window(calc_base_time)
                        status_message += t.get('window_fallback_info', "\nUsing fallback window: {} to {} UTC").format(start_time.iso, end_time.iso)
                        return start_time, end_time, status_message
            except Exception as check_e: print(f"Error during polar check: {check_e}")
            raise ValueError("Calculated morning twilight is not after evening twilight.")

        start_time = astro_set
        end_time = astro_rise

        if is_now:
            if end_time < current_utc_time:
                status_message = t.get('window_already_passed', "Calculated night window for 'Now' has already passed. Calculating for next night.") + "\n"
                next_noon_utc = datetime.combine(current_utc_time.to_datetime(timezone.utc).date() + timedelta(days=1), time(12, 0), tzinfo=timezone.utc)
                astro_set_next = observer.twilight_evening_astronomical(Time(next_noon_utc), which='next')
                astro_rise_next = observer.twilight_morning_astronomical(astro_set_next if astro_set_next else Time(next_noon_utc), which='next')
                if astro_set_next is None or astro_rise_next is None or astro_rise_next <= astro_set_next:
                    raise ValueError("Could not determine valid twilight times for the *next* night.")
                start_time, end_time = astro_set_next, astro_rise_next
            elif start_time < current_utc_time:
                print(f"Adjusting window start from {start_time.iso} to current time {current_utc_time.iso}")
                start_time = current_utc_time

        start_fmt = start_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M %Z')
        end_fmt = end_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M %Z')
        status_message += t.get('window_info_template', "Observation window: {} to {} UTC (Astronomical Twilight)").format(start_fmt, end_fmt)

    except ValueError as ve:
        error_detail = f"{ve}"
        print(f"Astroplan ValueError calculating window: {error_detail}")
        if 'polar' not in status_message:
            try:
                sun_alt_ref = observer.sun_altaz(calc_base_time).alt
                sun_alt_12h_later = observer.sun_altaz(calc_base_time + 12*u.hour).alt
                if sun_alt_ref < -18*u.deg and sun_alt_12h_later < -18*u.deg:
                    status_message = t.get('error_polar_night', "Astronomical darkness lasts >24h (Polar night?). Using fallback window.")
                elif sun_alt_ref > -18*u.deg:
                    times_check = calc_base_time + np.linspace(0, 24, 49)*u.hour
                    sun_alts_check = observer.sun_altaz(times_check).alt
                    if np.min(sun_alts_check) > -18*u.deg:
                        status_message = t.get('error_polar_day', "No astronomical darkness occurs (Polar day?). Using fallback window.")
                    else: status_message = t.get('window_calc_error', "Error calculating observation window: {}\n{}").format(error_detail, " (Check location/time)")
                else: status_message = t.get('window_calc_error', "Error calculating observation window: {}\n{}").format(error_detail, traceback.format_exc())
            except Exception as check_e:
                print(f"Error checking sun altitude for polar conditions: {check_e}")
                status_message = t.get('window_calc_error', "Error calculating observation window: {}\n{}").format(error_detail, traceback.format_exc())
        start_time, end_time = _get_fallback_window(calc_base_time)
        status_message += t.get('window_fallback_info', "\nUsing fallback window: {} to {} UTC").format(start_time.iso, end_time.iso)

    except Exception as e:
        status_message = t.get('window_calc_error', "Error calculating observation window: {}\n{}").format(e, traceback.format_exc())
        print(f"Unexpected error calculating window: {e}")
        start_time, end_time = _get_fallback_window(calc_base_time)
        status_message += t.get('window_fallback_info', "\nUsing fallback window: {} to {} UTC").format(start_time.iso, end_time.iso)

    if start_time is None or end_time is None or end_time <= start_time:
        if not status_message or "Error" not in status_message and "Fallback" not in status_message:
             status_message += ("\n" if status_message else "") + t.get('error_no_window', "No valid astronomical darkness window found for the selected date and location.")
        start_time_fb, end_time_fb = _get_fallback_window(calc_base_time)
        if t.get('window_fallback_info', "\nUsing fallback window: {} to {} UTC").format(start_time_fb.iso, end_time_fb.iso) not in status_message:
             status_message += t.get('window_fallback_info', "\nUsing fallback window: {} to {} UTC").format(start_time_fb.iso, end_time_fb.iso)
        start_time, end_time = start_time_fb, end_time_fb

    return start_time, end_time, status_message

# --- Object Finding Logic ---
def find_observable_objects(observer_location: EarthLocation,
                            observing_times: Time,
                            min_altitude_limit: u.Quantity,
                            catalog_df: pd.DataFrame,
                            lang: str) -> list[dict]:
    """Finds Deep Sky Objects from the catalog that are observable."""
    t = get_translation(lang) # <--- GEÄNDERT
    observable_objects = []

    if not isinstance(observer_location, EarthLocation): st.error("Internal Error: observer_location type"); return []
    if not isinstance(observing_times, Time) or not observing_times.shape: st.error("Internal Error: observing_times type"); return []
    if not isinstance(min_altitude_limit, u.Quantity): st.error("Internal Error: min_altitude_limit type"); return []
    if not isinstance(catalog_df, pd.DataFrame): st.error("Internal Error: catalog_df type"); return []
    if catalog_df.empty: print("Input catalog_df is empty."); return []
    if len(observing_times) < 2: st.warning("Observing window has < 2 points.")

    altaz_frame = AltAz(obstime=observing_times, location=observer_location)
    min_alt_deg = min_altitude_limit.to(u.deg).value
    time_step_hours = 0
    if len(observing_times) > 1:
        time_step_hours = (observing_times[1] - observing_times[0]).sec / 3600.0

    for index, obj in catalog_df.iterrows():
        try:
            ra_str = obj.get('RA_str')
            dec_str = obj.get('Dec_str')
            dso_name = obj.get('Name', f"Unnamed Object {index}")
            obj_type = obj.get('Type', "Unknown")
            obj_mag = obj.get('Mag', np.nan)
            obj_size = obj.get('MajAx', np.nan)

            if not ra_str or not dec_str: print(f"Skipping '{dso_name}': Missing RA/Dec."); continue

            try: dso_coord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))
            except ValueError as coord_err: print(f"Skipping '{dso_name}': Invalid coords. Error: {coord_err}"); continue

            try:
                dso_altazs = dso_coord.transform_to(altaz_frame)
                dso_alts = dso_altazs.alt.to(u.deg).value
                dso_azs = dso_altazs.az.to(u.deg).value
            except Exception as transform_err: print(f"Skipping '{dso_name}': Transform error. Error: {transform_err}"); continue

            max_alt_this_object = np.max(dso_alts) if len(dso_alts) > 0 else -999
            if max_alt_this_object >= min_alt_deg:
                peak_alt_index = np.argmax(dso_alts)
                peak_alt = dso_alts[peak_alt_index]
                peak_time_utc = observing_times[peak_alt_index]
                peak_az = dso_azs[peak_alt_index]
                peak_direction = azimuth_to_direction(peak_az)

                try: constellation = get_constellation(dso_coord)
                except Exception as const_err: print(f"Warn: Constellation fail for {dso_name}: {const_err}"); constellation = "N/A"

                above_min_alt = dso_alts >= min_alt_deg
                continuous_duration_hours = 0
                if time_step_hours > 0 and np.any(above_min_alt):
                    runs = np.split(np.arange(len(above_min_alt)), np.where(np.diff(above_min_alt))[0]+1)
                    max_duration_indices = 0
                    for run in runs:
                        if run.size > 0 and above_min_alt[run[0]]:
                            max_duration_indices = max(max_duration_indices, len(run))
                    continuous_duration_hours = max_duration_indices * time_step_hours

                result_dict = {
                    'Name': dso_name, 'Type': obj_type, 'Constellation': constellation,
                    'Magnitude': obj_mag if not np.isnan(obj_mag) else None,
                    'Size (arcmin)': obj_size if not np.isnan(obj_size) else None,
                    'RA': ra_str, 'Dec': dec_str,
                    'Max Altitude (°)': peak_alt, 'Azimuth at Max (°)': peak_az,
                    'Direction at Max': peak_direction, 'Time at Max (UTC)': peak_time_utc,
                    'Max Cont. Duration (h)': continuous_duration_hours,
                    'skycoord': dso_coord, 'altitudes': dso_alts,
                    'azimuths': dso_azs, 'times': observing_times
                }
                observable_objects.append(result_dict)

        except Exception as obj_proc_e:
            error_msg = t.get('error_processing_object', "Error processing {}: {}").format(obj.get('Name', f'Object {index}'), obj_proc_e)
            print(error_msg)

    return observable_objects

# --- Time Formatting ---
def get_local_time_str(utc_time: Time | None, timezone_str: str) -> tuple[str, str]:
    """Converts a UTC Time object to a localized time string, or returns "N/A"."""
    if utc_time is None: return "N/A", "N/A"
    if not isinstance(utc_time, Time): print(f"Error: utc_time type. Got {type(utc_time)}"); return "N/A", "N/A"
    if not isinstance(timezone_str, str) or not timezone_str: print(f"Error: timezone_str type. Got '{timezone_str}'"); return "N/A", "N/A"

    try:
        local_tz = pytz.timezone(timezone_str)
        utc_dt = utc_time.to_datetime(timezone.utc)
        local_dt = utc_dt.astimezone(local_tz)
        local_time_str = local_dt.strftime('%Y-%m-%d %H:%M:%S')
        tz_display_name = local_dt.tzname()
        if not tz_display_name: tz_display_name = local_tz.zone
        return local_time_str, tz_display_name
    except pytz.exceptions.UnknownTimeZoneError:
        print(f"Error: Unknown timezone '{timezone_str}'.")
        return utc_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), "UTC (TZ Error)"
    except Exception as e:
        print(f"Error converting time to local timezone '{timezone_str}': {e}")
        traceback.print_exc()
        return utc_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), "UTC (Conv. Error)"

# --- Main App ---
def main():
    """Main function to run the Streamlit application."""
    initialize_session_state()

    # --- Get Current Language and Translations ---
    lang = st.session_state.language
    t = get_translation(lang) # <--- GEÄNDERT
    # Validate language state against available languages in the loaded dictionary
    if lang not in t: # This check might be redundant if get_translation always returns a valid dict
        print(f"Warning: Language '{lang}' not found in translations, falling back.")
        lang = 'de' # Fallback to default language defined in localization.py
        st.session_state.language = lang
        t = get_translation(lang) # Reload translations for the fallback language

    # --- Load Catalog Data (Cached) ---
    @st.cache_data
    def cached_load_ongc_data(path, current_lang):
        """Cached function to load ONGC data."""
        print(f"Cache miss: Loading ONGC data from {path} for lang={current_lang}")
        return load_ongc_data(path, current_lang) # Pass lang here

    df_catalog_data = cached_load_ongc_data(CATALOG_FILEPATH, lang) # Use current lang

    st.title("Advanced DSO Finder")

    # --- Object Type Glossary ---
    with st.expander(t.get('object_type_glossary_title', "Object Type Glossary")):
        glossary_items = t.get('object_type_glossary', {})
        if glossary_items:
            col1, col2 = st.columns(2)
            sorted_items = sorted(glossary_items.items())
            for i, (abbr, full_name) in enumerate(sorted_items):
                target_col = col1 if i % 2 == 0 else col2
                target_col.markdown(f"**{abbr}:** {full_name}")
        else: st.info("Glossary not available for the selected language.")

    st.markdown("---")

    # --- Sidebar ---
    with st.sidebar:
        st.header(t.get('settings_header', "Settings"))

        # Catalog Status
        if 'catalog_status_msg' not in st.session_state: st.session_state.catalog_status_msg = ""
        if df_catalog_data is not None:
            new_msg = t.get('info_catalog_loaded', "Catalog loaded: {} objects.").format(len(df_catalog_data))
            if st.session_state.catalog_status_msg != new_msg: st.success(new_msg); st.session_state.catalog_status_msg = new_msg
        else:
            new_msg = "Catalog loading failed. Check file or logs."
            if st.session_state.catalog_status_msg != new_msg: st.error(new_msg); st.session_state.catalog_status_msg = new_msg

        # --- Language Selector ---
        language_options = {'de': 'Deutsch', 'en': 'English', 'fr': 'Français'}
        lang_keys = list(language_options.keys())
        current_lang_index = lang_keys.index(lang) if lang in lang_keys else 0

        selected_lang_key = st.radio(
            t.get('language_select_label', "Language"),
            options=lang_keys,
            format_func=language_options.get,
            key='language_radio',
            index=current_lang_index,
            horizontal=True
        )

        if selected_lang_key != st.session_state.language:
            st.session_state.language = selected_lang_key
            st.session_state.location_search_status_msg = ""
            st.rerun()

        # --- Location Settings ---
        with st.expander(t.get('location_expander', "📍 Location"), expanded=True):
            # Map internal keys to translated display text
            location_options_map_display = {
                'Search': t.get('location_option_search', "Search by Name"),
                'Manual': t.get('location_option_manual', "Enter Manually")
            }
            location_keys_internal = list(location_options_map_display.keys())

            st.radio(
                t.get('location_select_label', "Select Location Method"),
                options=location_keys_internal,
                format_func=lambda key: location_options_map_display[key],
                key="location_choice_key",
                horizontal=True
            )

            lat_val, lon_val, height_val = None, None, None
            location_valid_for_tz = False
            current_location_valid = False

            if st.session_state.location_choice_key == "Manual":
                st.number_input(t.get('location_lat_label', "Latitude (°N)"), -90.0, 90.0, step=0.01, format="%.4f", key="manual_lat_val")
                st.number_input(t.get('location_lon_label', "Longitude (°E)"), -180.0, 180.0, step=0.01, format="%.4f", key="manual_lon_val")
                st.number_input(t.get('location_elev_label', "Elevation (meters)"), -500, step=10, format="%d", key="manual_height_val")

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
                    st.text_input(t.get('location_search_label', "Enter location name:"), key="location_search_query", placeholder=t.get('location_search_placeholder', "..."))
                    st.number_input(t.get('location_elev_label', "Elevation (meters)"), -500, step=10, format="%d", key="manual_height_val")
                    location_search_form_submitted = st.form_submit_button(t.get('location_search_submit_button', "Find Coordinates"))

                status_placeholder = st.empty()
                if st.session_state.location_search_status_msg:
                    msg_func = status_placeholder.success if st.session_state.location_search_success else status_placeholder.error
                    msg_func(st.session_state.location_search_status_msg)

                if location_search_form_submitted and st.session_state.location_search_query:
                    location, service_used, final_error = None, None, None
                    query = st.session_state.location_search_query
                    user_agent_str = f"AdvancedDSOFinder/{random.randint(1000, 9999)}/streamlit_app_{datetime.now().timestamp()}"

                    with st.spinner(t.get('spinner_geocoding', "Searching for location...")):
                        try: # Nominatim
                            print("Trying Nominatim..."); geolocator = Nominatim(user_agent=user_agent_str); location = geolocator.geocode(query, timeout=10)
                            if location: service_used = "Nominatim"; print("Nominatim success.")
                            else: print("Nominatim returned None.")
                        except (GeocoderTimedOut, GeocoderServiceError, Exception) as e:
                            print(f"Nominatim failed: {e}. Trying fallback 1 (ArcGIS)."); status_placeholder.info(t.get('location_search_info_fallback', "...")); final_error = e if not isinstance(e, (GeocoderTimedOut, GeocoderServiceError)) else e

                        if not location: # ArcGIS
                            try:
                                print("Trying ArcGIS..."); fallback_geolocator = ArcGIS(timeout=15); location = fallback_geolocator.geocode(query, timeout=15)
                                if location: service_used = "ArcGIS"; print("ArcGIS success.")
                                else: print("ArcGIS returned None.")
                            except (GeocoderTimedOut, GeocoderServiceError, Exception) as e2:
                                print(f"ArcGIS failed: {e2}. Trying fallback 2 (Photon)."); status_placeholder.info(t.get('location_search_info_fallback2', "...")); final_error = e2 if not final_error else final_error

                        if not location: # Photon
                            try:
                                print("Trying Photon..."); fallback_geolocator2 = Photon(user_agent=user_agent_str, timeout=15); location = fallback_geolocator2.geocode(query, timeout=15)
                                if location: service_used = "Photon"; print("Photon success.")
                                else: print("Photon returned None."); final_error = GeocoderServiceError("All services failed or returned None.") if not final_error else final_error
                            except (GeocoderTimedOut, GeocoderServiceError, Exception) as e3:
                                print(f"Photon failed: {e3}. All fallbacks exhausted."); final_error = e3 if not final_error else final_error

                        if location and service_used:
                            found_lat, found_lon, found_name = location.latitude, location.longitude, location.address
                            st.session_state.update({
                                'searched_location_name': found_name, 'location_search_success': True,
                                'manual_lat_val': found_lat, 'manual_lon_val': found_lon
                            })
                            coord_str = t.get('location_search_coords', "Lat: {:.4f}, Lon: {:.4f}").format(found_lat, found_lon)
                            found_msg_key = 'location_search_found' if service_used=="Nominatim" else ('location_search_found_fallback' if service_used=="ArcGIS" else 'location_search_found_fallback2')
                            st.session_state.location_search_status_msg = f"{t.get(found_msg_key, 'Found: {}').format(found_name)}\n({coord_str})"
                            status_placeholder.success(st.session_state.location_search_status_msg)
                            lat_val, lon_val, height_val = found_lat, found_lon, st.session_state.manual_height_val
                            location_valid_for_tz = True; current_location_valid = True; st.session_state.location_is_valid_for_run = True
                        else: # Geocoding failed
                            st.session_state.update({'location_search_success': False, 'searched_location_name': None})
                            if final_error:
                                if isinstance(final_error, GeocoderTimedOut): err_key = 'location_search_error_timeout'
                                elif isinstance(final_error, GeocoderServiceError): err_key = 'location_search_error_service'
                                else: err_key = 'location_search_error_fallback2_failed'
                                st.session_state.location_search_status_msg = t.get(err_key, "Geocoding error: {}").format(final_error)
                            else: st.session_state.location_search_status_msg = t.get('location_search_error_not_found', "Location not found.")
                            status_placeholder.error(st.session_state.location_search_status_msg)
                            current_location_valid = False; st.session_state.location_is_valid_for_run = False

                elif st.session_state.location_search_success: # Use stored values if search was successful previously
                    lat_val, lon_val, height_val = st.session_state.manual_lat_val, st.session_state.manual_lon_val, st.session_state.manual_height_val
                    location_valid_for_tz = True; current_location_valid = True; st.session_state.location_is_valid_for_run = True
                    status_placeholder.success(st.session_state.location_search_status_msg)
                else:
                     current_location_valid = False; st.session_state.location_is_valid_for_run = False

            st.markdown("---")
            auto_timezone_msg = ""
            if location_valid_for_tz and lat_val is not None and lon_val is not None:
                if tf:
                    try:
                        found_tz = tf.timezone_at(lng=lon_val, lat=lat_val)
                        if found_tz:
                            pytz.timezone(found_tz) # Validate
                            st.session_state.selected_timezone = found_tz
                            auto_timezone_msg = f"{t.get('timezone_auto_set_label', 'Detected:')} **{found_tz}**"
                        else:
                            st.session_state.selected_timezone = 'UTC'
                            auto_timezone_msg = f"{t.get('timezone_auto_fail_label', 'TZ:')} **UTC** ({t.get('timezone_auto_fail_msg', 'Failed')})"
                    except pytz.UnknownTimeZoneError:
                        st.session_state.selected_timezone = 'UTC'
                        auto_timezone_msg = f"{t.get('timezone_auto_fail_label', 'TZ:')} **UTC** (Invalid TZ: '{found_tz}')"
                    except Exception as tz_find_e:
                        print(f"Error finding TZ: {tz_find_e}")
                        st.session_state.selected_timezone = 'UTC'
                        auto_timezone_msg = f"{t.get('timezone_auto_fail_label', 'TZ:')} **UTC** (Error)"
                else: # TF not available
                    auto_timezone_msg = f"{t.get('timezone_auto_fail_label', 'TZ:')} **{INITIAL_TIMEZONE}** (Auto N/A)"
                    st.session_state.selected_timezone = INITIAL_TIMEZONE
            else: # Location invalid
                auto_timezone_msg = f"{t.get('timezone_auto_fail_label', 'TZ:')} **{st.session_state.selected_timezone}** (Loc Invalid)"

            st.markdown(auto_timezone_msg, unsafe_allow_html=True)

        # --- Time Settings ---
        with st.expander(t.get('time_expander', "⏱️ Time & Timezone"), expanded=False):
            time_options_map_display = {'Now': t.get('time_option_now', "Now"), 'Specific': t.get('time_option_specific', "Specific Night")}
            st.radio(t.get('time_select_label', "Select Time"), options=list(time_options_map_display.keys()),
                     format_func=lambda key: time_options_map_display[key], key="time_choice_exp", horizontal=True)

            if st.session_state.time_choice_exp == "Now": st.caption(f"UTC: {Time.now().iso}")
            else:
                st.date_input(t.get('time_date_select_label', "Select Date:"), value=st.session_state.selected_date_widget,
                              min_value=date.today()-timedelta(days=365*10), max_value=date.today()+timedelta(days=365*2), key='selected_date_widget')

        # --- Filter Settings ---
        with st.expander(t.get('filters_expander', "✨ Filters & Conditions"), expanded=False):
            st.markdown(t.get('mag_filter_header', "**Magnitude Filter**"))
            mag_filter_options_map_display = {'Bortle Scale': t.get('mag_filter_option_bortle', "Bortle"), 'Manual': t.get('mag_filter_option_manual', "Manual")}
            st.radio(t.get('mag_filter_method_label', "Method:"), options=list(mag_filter_options_map_display.keys()),
                     format_func=lambda key: mag_filter_options_map_display[key], key="mag_filter_mode_exp", horizontal=True)

            st.slider(t.get('mag_filter_bortle_label', "Bortle:"), 1, 9, key='bortle_slider', help=t.get('mag_filter_bortle_help', "..."))

            if st.session_state.mag_filter_mode_exp == "Manual":
                st.slider(t.get('mag_filter_min_mag_label', "Min Mag:"), -5.0, 20.0, step=0.5, format="%.1f", help=t.get('mag_filter_min_mag_help', "..."), key='manual_min_mag_slider')
                st.slider(t.get('mag_filter_max_mag_label', "Max Mag:"), -5.0, 20.0, step=0.5, format="%.1f", help=t.get('mag_filter_max_mag_help', "..."), key='manual_max_mag_slider')
                if st.session_state.manual_min_mag_slider > st.session_state.manual_max_mag_slider:
                    st.warning(t.get('mag_filter_warning_min_max', "Min > Max!"))

            st.markdown("---"); st.markdown(t.get('min_alt_header', "**Altitude**"))
            min_alt_val, max_alt_val = st.session_state.min_alt_slider, st.session_state.max_alt_slider
            if min_alt_val > max_alt_val: st.session_state.min_alt_slider = max_alt_val; min_alt_val = max_alt_val
            st.slider(t.get('min_alt_label', "Min Alt (°):"), 0, 90, key='min_alt_slider', step=1)
            st.slider(t.get('max_alt_label', "Max Alt (°):"), 0, 90, key='max_alt_slider', step=1)
            if st.session_state.min_alt_slider > st.session_state.max_alt_slider: st.warning("Min Alt > Max Alt!")

            st.markdown("---"); st.markdown(t.get('moon_warning_header', "**Moon**"))
            st.slider(t.get('moon_warning_label', "Warn > (%):"), 0, 100, key='moon_phase_slider', step=5)

            st.markdown("---"); st.markdown(t.get('object_types_header', "**Types**"))
            all_types = []
            if df_catalog_data is not None and 'Type' in df_catalog_data.columns:
                try: all_types = sorted(list(df_catalog_data['Type'].dropna().astype(str).unique()))
                except Exception as e: st.warning(f"{t.get('object_types_error_extract', 'Type Error')}: {e}")
            if all_types:
                current_selection = [sel for sel in st.session_state.object_type_filter_exp if sel in all_types]
                if current_selection != st.session_state.object_type_filter_exp: st.session_state.object_type_filter_exp = current_selection
                st.multiselect(t.get('object_types_label', "Filter Types:"), options=all_types, default=current_selection, key="object_type_filter_exp")
            else: st.info("No types found/load error."); st.session_state.object_type_filter_exp = []

            st.markdown("---"); st.markdown(t.get('size_filter_header', "**Size**"))
            size_col_exists = df_catalog_data is not None and 'MajAx' in df_catalog_data.columns and df_catalog_data['MajAx'].notna().any()
            size_slider_disabled = not size_col_exists
            if size_col_exists:
                try:
                    valid_sizes = df_catalog_data['MajAx'].dropna(); min_poss = max(0.1, float(valid_sizes.min())) if not valid_sizes.empty else 0.1
                    max_poss = float(valid_sizes.max()) if not valid_sizes.empty else 120.0
                    min_state, max_state = st.session_state.size_arcmin_range
                    clamp_min = max(min_poss, min(min_state, max_poss)); clamp_max = min(max_poss, max(max_state, min_poss))
                    if clamp_min > clamp_max: clamp_min = clamp_max
                    if (clamp_min, clamp_max) != st.session_state.size_arcmin_range: st.session_state.size_arcmin_range = (clamp_min, clamp_max)
                    step = 0.1 if max_poss <= 20 else (0.5 if max_poss <= 100 else 1.0)
                    st.slider(t.get('size_filter_label', "Size (arcmin):"), min_poss, max_poss, step=step, format="%.1f arcmin", key='size_arcmin_range', help=t.get('size_filter_help', "..."), disabled=size_slider_disabled)
                except Exception as size_slider_e: st.error(f"Size slider error: {size_slider_e}"); size_slider_disabled = True
            else: st.info("Size data unavailable."); size_slider_disabled = True
            if size_slider_disabled: st.slider(t.get('size_filter_label', "Size (arcmin):"), 0.0, 1.0, (0.0, 1.0), key='size_arcmin_range_disabled', disabled=True)

            st.markdown("---"); st.markdown(t.get('direction_filter_header', "**Direction**"))
            all_dir_str = t.get('direction_option_all', "All")
            dir_opts_display = [all_dir_str] + CARDINAL_DIRECTIONS
            dir_opts_internal = [ALL_DIRECTIONS_KEY] + CARDINAL_DIRECTIONS
            curr_dir_internal = st.session_state.selected_peak_direction
            if curr_dir_internal not in dir_opts_internal: curr_dir_internal = ALL_DIRECTIONS_KEY; st.session_state.selected_peak_direction = curr_dir_internal
            try: curr_dir_index = dir_opts_internal.index(curr_dir_internal)
            except ValueError: curr_dir_index = 0
            sel_dir_display = st.selectbox(t.get('direction_filter_label', "Direction:"), options=dir_opts_display, index=curr_dir_index, key='direction_selectbox')
            sel_internal = ALL_DIRECTIONS_KEY
            if sel_dir_display != all_dir_str:
                try: sel_internal_idx = dir_opts_display.index(sel_dir_display); sel_internal = dir_opts_internal[sel_internal_idx]
                except ValueError: sel_internal = ALL_DIRECTIONS_KEY
            if sel_internal != st.session_state.selected_peak_direction: st.session_state.selected_peak_direction = sel_internal

        # --- Result Options ---
        with st.expander(t.get('results_options_expander', "⚙️ Results"), expanded=False):
            max_slider = len(df_catalog_data) if df_catalog_data is not None else 50
            min_slider = 5; actual_max = max(min_slider, max_slider); slider_disabled = actual_max <= min_slider
            default_num = st.session_state.get('num_objects_slider', 20)
            clamp_default = max(min_slider, min(default_num, actual_max))
            if clamp_default != default_num: st.session_state.num_objects_slider = clamp_default
            st.slider(t.get('results_options_max_objects_label', "Max Objects:"), min_slider, actual_max, step=1, key='num_objects_slider', disabled=slider_disabled)

            sort_options_map_display = {'Duration & Altitude': t.get('results_options_sort_duration', "Duration"), 'Brightness': t.get('results_options_sort_magnitude', "Brightness")}
            st.radio(t.get('results_options_sort_method_label', "Sort By:"), options=list(sort_options_map_display.keys()),
                     format_func=lambda key: sort_options_map_display[key], key='sort_method', horizontal=True)

        st.sidebar.markdown("---")
        bug_email = "debrun2005@gmail.com"
        bug_subj = urllib.parse.quote("Bug Report: Advanced DSO Finder")
        bug_body = urllib.parse.quote(t.get('bug_report_body', "\n\n(Describe bug)"))
        bug_link = f"mailto:{bug_email}?subject={bug_subj}&body={bug_body}"
        st.sidebar.markdown(f"<a href='{bug_link}' target='_blank'>{t.get('bug_report_button', '🐞 Report Bug')}</a>", unsafe_allow_html=True)

    # --- Main Area ---
    st.subheader(t.get('search_params_header', "Search Parameters"))
    param_col1, param_col2 = st.columns(2)

    location_display = t.get('location_error', "Location Error: {}").format("Not Set")
    observer_for_run = None
    if st.session_state.location_is_valid_for_run:
        lat, lon, h, tz_str = st.session_state.manual_lat_val, st.session_state.manual_lon_val, st.session_state.manual_height_val, st.session_state.selected_timezone
        try:
            observer_for_run = Observer(latitude=lat*u.deg, longitude=lon*u.deg, elevation=h*u.m, timezone=tz_str)
            if st.session_state.location_choice_key == "Manual": loc_disp_key = 'location_manual_display'
            elif st.session_state.searched_location_name: loc_disp_key = 'location_search_display'
            else: loc_disp_key = None # Fallback case
            if loc_disp_key:
                 location_display = t.get(loc_disp_key, "{} ({:.4f}, {:.4f})").format(st.session_state.searched_location_name if loc_disp_key == 'location_search_display' else lat, lon if loc_disp_key == 'location_manual_display' else lon, lat if loc_disp_key == 'location_search_display' else lat) # Simplified format
            else: location_display = f"Lat: {lat:.4f}, Lon: {lon:.4f}"
        except Exception as obs_e:
             location_display = t.get('location_error', "Loc Error: {}").format(f"Observer fail: {obs_e}")
             st.session_state.location_is_valid_for_run = False; observer_for_run = None
    param_col1.markdown(t.get('search_params_location', "📍 Loc: {}").format(location_display))

    time_display = ""
    is_time_now_main = (st.session_state.time_choice_exp == "Now")
    if is_time_now_main:
        ref_time_main = Time.now()
        try: local_now_str, local_tz_now = get_local_time_str(ref_time_main, st.session_state.selected_timezone); time_display = t.get('search_params_time_now', "Now (from {} UTC)").format(f"{local_now_str} {local_tz_now}")
        except Exception: time_display = t.get('search_params_time_now', "Now (from {} UTC)").format(f"{ref_time_main.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    else:
        sel_date_main = st.session_state.selected_date_widget
        ref_time_main = Time(datetime.combine(sel_date_main, time(12, 0)), scale='utc')
        time_display = t.get('search_params_time_specific', "Night after {}").format(sel_date_main.strftime('%Y-%m-%d'))
    param_col1.markdown(t.get('search_params_time', "⏱️ Time: {}").format(time_display))

    mag_filter_display = ""
    min_mag_filter, max_mag_filter = -np.inf, np.inf
    if st.session_state.mag_filter_mode_exp == "Bortle Scale":
        max_mag_filter = get_magnitude_limit(st.session_state.bortle_slider)
        mag_filter_display = t.get('search_params_filter_mag_bortle', "Bortle {} (<= {:.1f} mag)").format(st.session_state.bortle_slider, max_mag_filter)
    else:
        min_mag_filter, max_mag_filter = st.session_state.manual_min_mag_slider, st.session_state.manual_max_mag_slider
        mag_filter_display = t.get('search_params_filter_mag_manual', "Manual ({:.1f}-{:.1f} mag)").format(min_mag_filter, max_mag_filter)
    param_col2.markdown(t.get('search_params_filter_mag', "✨ Mag: {}").format(mag_filter_display))

    min_alt_disp, max_alt_disp = st.session_state.min_alt_slider, st.session_state.max_alt_slider
    sel_types_disp = st.session_state.object_type_filter_exp
    types_str = ', '.join(sel_types_disp) if sel_types_disp else t.get('search_params_types_all', "All")
    param_col2.markdown(t.get('search_params_filter_alt_types', "🔭 Alt {}-{}°, Types: {}").format(min_alt_disp, max_alt_disp, types_str))

    size_min_disp, size_max_disp = st.session_state.size_arcmin_range
    param_col2.markdown(t.get('search_params_filter_size', "📐 Size {:.1f}-{:.1f}'").format(size_min_disp, size_max_disp))

    dir_disp = st.session_state.selected_peak_direction
    if dir_disp == ALL_DIRECTIONS_KEY: dir_disp = t.get('search_params_direction_all', "All")
    param_col2.markdown(t.get('search_params_filter_direction', "🧭 Dir @ Max: {}").format(dir_disp))

    st.markdown("---")
    find_button_clicked = st.button(
        t.get('find_button_label', "🔭 Find Objects"),
        key="find_button",
        disabled=(df_catalog_data is None or not st.session_state.location_is_valid_for_run)
    )

    if not st.session_state.location_is_valid_for_run and df_catalog_data is not None:
        st.warning(t.get('info_initial_prompt', "Enter Coords or Search Loc..."))

    results_placeholder = st.container()

    if find_button_clicked:
        st.session_state.find_button_pressed = True
        st.session_state.update({'show_plot': False, 'show_custom_plot': False, 'active_result_plot_data': None,
                                 'custom_target_plot_data': None, 'last_results': [], 'window_start_time': None, 'window_end_time': None})

        if observer_for_run and df_catalog_data is not None:
            with st.spinner(t.get('spinner_searching', "Calculating...")):
                try:
                    start_time_calc, end_time_calc, window_status = get_observable_window(observer_for_run, ref_time_main, is_time_now_main, lang)
                    results_placeholder.info(window_status)
                    st.session_state.window_start_time = start_time_calc
                    st.session_state.window_end_time = end_time_calc

                    if start_time_calc and end_time_calc and start_time_calc < end_time_calc:
                        time_res = 5 * u.minute
                        obs_times = Time(np.arange(start_time_calc.jd, end_time_calc.jd, time_res.to(u.day).value), format='jd', scale='utc')
                        if len(obs_times) < 2: results_placeholder.warning("Window too short.")

                        filtered_df = df_catalog_data.copy()
                        filtered_df = filtered_df[(filtered_df['Mag'] >= min_mag_filter) & (filtered_df['Mag'] <= max_mag_filter)]
                        if sel_types_disp: filtered_df = filtered_df[filtered_df['Type'].isin(sel_types_disp)]
                        size_col_exists_main = df_catalog_data is not None and 'MajAx' in df_catalog_data.columns and df_catalog_data['MajAx'].notna().any()
                        if size_col_exists_main:
                            filtered_df = filtered_df.dropna(subset=['MajAx'])
                            filtered_df = filtered_df[(filtered_df['MajAx'] >= size_min_disp) & (filtered_df['MajAx'] <= size_max_disp)]

                        if filtered_df.empty:
                            results_placeholder.warning(t.get('warning_no_objects_found', "No objects found...") + " (initial filter)")
                            st.session_state.last_results = []
                        else:
                            min_alt_search = st.session_state.min_alt_slider * u.deg
                            found_objects = find_observable_objects(observer_for_run.location, obs_times, min_alt_search, filtered_df, lang)

                            final_objects = []
                            sel_dir = st.session_state.selected_peak_direction
                            max_alt_filt = st.session_state.max_alt_slider
                            for obj in found_objects:
                                if obj.get('Max Altitude (°)', -999) > max_alt_filt: continue
                                if sel_dir != ALL_DIRECTIONS_KEY and obj.get('Direction at Max') != sel_dir: continue
                                final_objects.append(obj)

                            sort_key = st.session_state.sort_method
                            if sort_key == 'Brightness': final_objects.sort(key=lambda x: x.get('Magnitude', float('inf')) if x.get('Magnitude') is not None else float('inf'))
                            else: final_objects.sort(key=lambda x: (x.get('Max Cont. Duration (h)', 0), x.get('Max Altitude (°)', 0)), reverse=True)

                            num_to_show = st.session_state.num_objects_slider
                            st.session_state.last_results = final_objects[:num_to_show]

                            if not final_objects: results_placeholder.warning(t.get('warning_no_objects_found', "No objects found..."))
                            else:
                                results_placeholder.success(t.get('success_objects_found', "{} objects found.").format(len(final_objects)))
                                sort_msg_key = 'info_showing_list_duration' if sort_key != 'Brightness' else 'info_showing_list_magnitude'
                                results_placeholder.info(t[sort_msg_key].format(len(st.session_state.last_results)))
                    else:
                        results_placeholder.error(t.get('error_no_window', "No valid window...") + " Cannot search.")
                        st.session_state.last_results = []
                except Exception as search_e:
                    results_placeholder.error(t.get('error_search_unexpected', "Search error:") + f"\n```\n{search_e}\n```")
                    traceback.print_exc(); st.session_state.last_results = []
        else:
            if df_catalog_data is None: results_placeholder.error("Cannot search: Catalog not loaded.")
            if not observer_for_run: results_placeholder.error("Cannot search: Location invalid.")
            st.session_state.last_results = []

    # --- Display Results Block ---
    if st.session_state.last_results:
        results_data = st.session_state.last_results
        results_placeholder.subheader(t.get('results_list_header', "Results"))

        window_start = st.session_state.get('window_start_time')
        window_end = st.session_state.get('window_end_time')
        observer_exists = observer_for_run is not None

        if observer_exists and isinstance(window_start, Time) and isinstance(window_end, Time):
            mid_time = window_start + (window_end - window_start) / 2
            try:
                illum = moon_illumination(mid_time); moon_phase_percent = illum * 100
                moon_svg = create_moon_phase_svg(illum, size=50)
                moon_col1, moon_col2 = results_placeholder.columns([1, 3])
                moon_col1.markdown(moon_svg, unsafe_allow_html=True)
                with moon_col2:
                    st.metric(label=t.get('moon_metric_label', "Moon Illum."), value=f"{moon_phase_percent:.0f}%")
                    moon_warn_thresh = st.session_state.moon_phase_slider
                    if moon_phase_percent > moon_warn_thresh:
                        st.warning(t.get('moon_warning_message', "Warn: Moon > ({:.0f}%)!").format(moon_phase_percent, moon_warn_thresh))
            except Exception as moon_e: results_placeholder.warning(t.get('moon_phase_error', "Moon Error: {}").format(moon_e))
        elif st.session_state.find_button_pressed: results_placeholder.info("Moon phase N/A (invalid window/loc).")

        plot_options_map_display = {'Sky Path': t.get('graph_type_sky_path', "Sky Path"), 'Altitude Plot': t.get('graph_type_alt_time', "Alt Plot")}
        results_placeholder.radio(t.get('graph_type_label', "Graph Type:"), options=list(plot_options_map_display.keys()),
                                  format_func=lambda key: plot_options_map_display[key], key='plot_type_selection', horizontal=True)

        for i, obj_data in enumerate(results_data):
            obj_name = obj_data.get('Name', 'N/A'); obj_type = obj_data.get('Type', 'N/A')
            obj_mag = obj_data.get('Magnitude'); mag_str = f"{obj_mag:.1f}" if obj_mag is not None else "N/A"
            exp_title = t.get('results_expander_title', "{} ({}) - Mag: {:.1f}").format(obj_name, obj_type, obj_mag if obj_mag is not None else 99)
            is_expanded = (st.session_state.expanded_object_name == obj_name)
            obj_cont = results_placeholder.container()

            with obj_cont.expander(exp_title, expanded=is_expanded):
                col1, col2, col3 = st.columns([2,2,1])
                col1.markdown(t.get('results_coords_header', "**Details:**"))
                col1.markdown(f"**{t.get('results_export_constellation', 'Const')}:** {obj_data.get('Constellation', 'N/A')}")
                size_am = obj_data.get('Size (arcmin)')
                col1.markdown(f"**{t.get('results_size_label', 'Size:')}** {t.get('results_size_value', '{:.1f}\'').format(size_am) if size_am is not None else 'N/A'}")
                col1.markdown(f"**RA:** {obj_data.get('RA', 'N/A')}"); col1.markdown(f"**Dec:** {obj_data.get('Dec', 'N/A')}")

                col2.markdown(t.get('results_max_alt_header', "**Max Alt:**"))
                max_alt = obj_data.get('Max Altitude (°)', 0); az_max = obj_data.get('Azimuth at Max (°)', 0); dir_max = obj_data.get('Direction at Max', 'N/A')
                az_fmt = t.get('results_azimuth_label', "(Az: {:.1f}°{})").format(az_max, "")
                dir_fmt = t.get('results_direction_label', ", Dir: {}").format(dir_max)
                col2.markdown(f"**{max_alt:.1f}°** {az_fmt}{dir_fmt}")

                col2.markdown(t.get('results_best_time_header', "**Best Time (Local):**"))
                peak_utc = obj_data.get('Time at Max (UTC)')
                local_time, local_tz = get_local_time_str(peak_utc, st.session_state.selected_timezone)
                col2.markdown(f"{local_time} ({local_tz})")

                col2.markdown(t.get('results_cont_duration_header', "**Duration:**"))
                dur_h = obj_data.get('Max Cont. Duration (h)', 0)
                col2.markdown(t.get('results_duration_value', "{:.1f} hrs").format(dur_h))

                g_query = urllib.parse.quote_plus(f"{obj_name} astronomy"); g_url = f"https://www.google.com/search?q={g_query}"
                col3.markdown(f"[{t.get('google_link_text', 'Google')}]({g_url})", unsafe_allow_html=True)
                s_query = urllib.parse.quote_plus(obj_name); s_url = f"http://simbad.u-strasbg.fr/simbad/sim-basic?Ident={s_query}"
                col3.markdown(f"[{t.get('simbad_link_text', 'SIMBAD')}]({s_url})", unsafe_allow_html=True)

                plot_btn_key = f"plot_{obj_name}_{i}"
                if st.button(t.get('results_graph_button', "📈 Plot"), key=plot_btn_key):
                    st.session_state.update({'plot_object_name': obj_name, 'active_result_plot_data': obj_data, 'show_plot': True,
                                             'show_custom_plot': False, 'expanded_object_name': obj_name})
                    st.rerun()

                if st.session_state.show_plot and st.session_state.plot_object_name == obj_name:
                    plot_data = st.session_state.active_result_plot_data
                    min_alt_line, max_alt_line = st.session_state.min_alt_slider, st.session_state.max_alt_slider
                    st.markdown("---")
                    with st.spinner(t.get('results_spinner_plotting', "Plotting...")):
                        try:
                            fig = create_plot(plot_data, min_alt_line, max_alt_line, st.session_state.plot_type_selection, lang)
                            if fig:
                                st.pyplot(fig)
                                close_btn_key = f"close_plot_{obj_name}_{i}"
                                if st.button(t.get('results_close_graph_button', "Close Plot"), key=close_btn_key):
                                    st.session_state.update({'show_plot': False, 'active_result_plot_data': None, 'expanded_object_name': None})
                                    st.rerun()
                            else: st.error(t.get('results_graph_not_created', "Plot failed."))
                        except Exception as plot_err:
                            st.error(t.get('results_graph_error', "Plot Error: {}").format(plot_err)); traceback.print_exc()

        if results_data:
            csv_export_placeholder = results_placeholder.empty()
            try:
                export_data = []
                for obj in results_data:
                    peak_utc = obj.get('Time at Max (UTC)')
                    local_time, _ = get_local_time_str(peak_utc, st.session_state.selected_timezone)
                    export_data.append({
                        t.get('results_export_name', "Name"): obj.get('Name', 'N/A'), t.get('results_export_type', "Type"): obj.get('Type', 'N/A'),
                        t.get('results_export_constellation', "Const"): obj.get('Constellation', 'N/A'), t.get('results_export_mag', "Mag"): obj.get('Magnitude'),
                        t.get('results_export_size', "Size'"): obj.get('Size (arcmin)'), t.get('results_export_ra', "RA"): obj.get('RA', 'N/A'),
                        t.get('results_export_dec', "Dec"): obj.get('Dec', 'N/A'), t.get('results_export_max_alt', "MaxAlt"): obj.get('Max Altitude (°)', 0),
                        t.get('results_export_az_at_max', "Az@Max"): obj.get('Azimuth at Max (°)', 0), t.get('results_export_direction_at_max', "Dir@Max"): obj.get('Direction at Max', 'N/A'),
                        t.get('results_export_time_max_utc', "TimeMaxUTC"): peak_utc.iso if peak_utc else "N/A",
                        t.get('results_export_time_max_local', "TimeMaxLocal"): local_time, t.get('results_export_cont_duration', "Dur(h)"): obj.get('Max Cont. Duration (h)', 0)
                    })
                df_export = pd.DataFrame(export_data)
                dec_sep = ',' if lang == 'de' else '.'
                csv_string = df_export.to_csv(index=False, sep=';', encoding='utf-8-sig', decimal=dec_sep)
                now_str = datetime.now().strftime("%Y%m%d_%H%M")
                csv_fname = t.get('results_csv_filename', "dso_list_{}.csv").format(now_str)
                csv_export_placeholder.download_button(label=t.get('results_save_csv_button', "💾 Save CSV"), data=csv_string, file_name=csv_fname, mime='text/csv', key='csv_download_button')
            except Exception as csv_e: csv_export_placeholder.error(t.get('results_csv_export_error', "CSV Error: {}").format(csv_e))

    elif st.session_state.find_button_pressed:
        results_placeholder.info(t.get('warning_no_objects_found', "No objects found..."))

    # --- Custom Target Plotting ---
    st.markdown("---")
    with st.expander(t.get('custom_target_expander', "Plot Custom Target")):
        with st.form("custom_target_form"):
             st.text_input(t.get('custom_target_ra_label', "RA:"), key="custom_target_ra", placeholder=t.get('custom_target_ra_placeholder', "..."))
             st.text_input(t.get('custom_target_dec_label', "Dec:"), key="custom_target_dec", placeholder=t.get('custom_target_dec_placeholder', "..."))
             st.text_input(t.get('custom_target_name_label', "Name (Opt):"), key="custom_target_name", placeholder="My Comet")
             custom_plot_submitted = st.form_submit_button(t.get('custom_target_button', "Create Plot"))

        custom_plot_error_placeholder = st.empty()
        custom_plot_display_area = st.empty()

        if custom_plot_submitted:
             st.session_state.update({'show_plot': False, 'show_custom_plot': False, 'custom_target_plot_data': None, 'custom_target_error': ""})
             custom_ra, custom_dec = st.session_state.custom_target_ra, st.session_state.custom_target_dec
             custom_name = st.session_state.custom_target_name or t.get('custom_target_name_label', "Target").replace(":", "")
             window_start_cust, window_end_cust = st.session_state.get('window_start_time'), st.session_state.get('window_end_time')
             observer_exists_cust = observer_for_run is not None

             if not custom_ra or not custom_dec:
                 st.session_state.custom_target_error = t.get('custom_target_error_coords', "Invalid RA/Dec.")
                 custom_plot_error_placeholder.error(st.session_state.custom_target_error)
             elif not observer_exists_cust or not isinstance(window_start_cust, Time) or not isinstance(window_end_cust, Time):
                 st.session_state.custom_target_error = t.get('custom_target_error_window', "Invalid window/loc.")
                 custom_plot_error_placeholder.error(st.session_state.custom_target_error)
             else:
                 try:
                     custom_coord = SkyCoord(ra=custom_ra, dec=custom_dec, unit=(u.hourangle, u.deg))
                     if window_start_cust < window_end_cust:
                         time_res_cust = 5 * u.minute
                         obs_times_custom = Time(np.arange(window_start_cust.jd, window_end_cust.jd, time_res_cust.to(u.day).value), format='jd', scale='utc')
                     else: raise ValueError("Invalid time window from main search.")
                     if len(obs_times_custom) < 2: raise ValueError("Time window too short.")

                     altaz_frame_custom = AltAz(obstime=obs_times_custom, location=observer_for_run.location)
                     custom_altazs = custom_coord.transform_to(altaz_frame_custom)
                     st.session_state.custom_target_plot_data = {
                         'Name': custom_name, 'altitudes': custom_altazs.alt.to(u.deg).value,
                         'azimuths': custom_altazs.az.to(u.deg).value, 'times': obs_times_custom }
                     st.session_state.show_custom_plot = True; st.session_state.custom_target_error = ""; st.rerun()
                 except ValueError as custom_coord_err:
                     st.session_state.custom_target_error = f"{t.get('custom_target_error_coords', 'Invalid RA/Dec.')} ({custom_coord_err})"
                     custom_plot_error_placeholder.error(st.session_state.custom_target_error)
                 except Exception as custom_e:
                     st.session_state.custom_target_error = f"Custom plot error: {custom_e}"
                     custom_plot_error_placeholder.error(st.session_state.custom_target_error); traceback.print_exc()

        if st.session_state.show_custom_plot and st.session_state.custom_target_plot_data:
            custom_plot_data = st.session_state.custom_target_plot_data
            min_alt_cust, max_alt_cust = st.session_state.min_alt_slider, st.session_state.max_alt_slider
            with custom_plot_display_area.container():
                 st.markdown("---")
                 with st.spinner(t.get('results_spinner_plotting', "Plotting...")):
                     try:
                         fig_cust = create_plot(custom_plot_data, min_alt_cust, max_alt_cust, st.session_state.plot_type_selection, lang)
                         if fig_cust:
                             st.pyplot(fig_cust)
                             if st.button(t.get('results_close_graph_button', "Close Plot"), key="close_custom_plot"):
                                 st.session_state.update({'show_custom_plot': False, 'custom_target_plot_data': None}); st.rerun()
                         else: st.error(t.get('results_graph_not_created', "Plot failed."))
                     except Exception as plot_err_cust: st.error(t.get('results_graph_error', "Plot Error: {}").format(plot_err_cust)); traceback.print_exc()
        elif st.session_state.custom_target_error: custom_plot_error_placeholder.error(st.session_state.custom_target_error)

    st.markdown("---")
    st.caption(t.get('donation_text', "Like the app? [Donate...](...)"), unsafe_allow_html=True)


# --- Plotting Function ---
#@st.cache_data(show_spinner=False) # Consider caching
def create_plot(plot_data: dict, min_altitude_deg: float, max_altitude_deg: float, plot_type: str, lang: str) -> plt.Figure | None:
    """Creates either an Altitude vs Time or Sky Path plot."""
    t = get_translation(lang) # <--- GEÄNDERT
    fig = None

    try:
        if not isinstance(plot_data, dict): st.error("Plot Error: Invalid data type."); return None
        times, altitudes, azimuths = plot_data.get('times'), plot_data.get('altitudes'), plot_data.get('azimuths')
        obj_name = plot_data.get('Name', 'Object')
        if not isinstance(times, Time) or not isinstance(altitudes, np.ndarray): st.error("Plot Error: Invalid times/alts."); return None
        if plot_type == 'Sky Path' and not isinstance(azimuths, np.ndarray): st.error("Plot Error: Invalid azs for Sky Path."); return None
        if len(times) != len(altitudes) or (azimuths is not None and len(times) != len(azimuths)): st.error("Plot Error: Mismatched arrays."); return None
        if len(times) < 1: st.error("Plot Error: Not enough data."); return None

        plot_times = times.plot_date

        try: theme_opts = st.get_option("theme.base"); is_dark = (theme_opts == "dark")
        except Exception: print("Warn: Theme detect failed."); is_dark = False

        # Simplified theme setup
        plt.style.use('dark_background' if is_dark else 'default')
        label_color = '#FAFAFA' if is_dark else '#333333'
        title_color = '#FFFFFF' if is_dark else '#000000'
        grid_color = '#444444' if is_dark else 'darkgray'
        primary_color = 'deepskyblue' if is_dark else 'dodgerblue'
        min_alt_color = 'tomato' if is_dark else 'red'
        max_alt_color = 'orange' if is_dark else 'darkorange'
        spine_color = '#AAAAAA' if is_dark else '#555555'
        legend_facecolor = '#262730' if is_dark else '#F0F0F0'
        facecolor = '#0E1117' if is_dark else '#FFFFFF'

        fig, ax = plt.subplots(figsize=(10, 6), facecolor=facecolor, constrained_layout=True)
        ax.set_facecolor(facecolor)

        if plot_type == 'Altitude Plot':
            ax.plot(plot_times, altitudes, color=primary_color, alpha=0.9, lw=1.5, label=obj_name)
            ax.axhline(min_altitude_deg, color=min_alt_color, ls='--', lw=1.2, label=t.get('graph_min_altitude_label', "Min Alt ({:.0f}°)").format(min_altitude_deg), alpha=0.8)
            if max_altitude_deg < 90: ax.axhline(max_altitude_deg, color=max_alt_color, ls=':', lw=1.2, label=t.get('graph_max_altitude_label', "Max Alt ({:.0f}°)").format(max_altitude_deg), alpha=0.8)
            ax.set(xlabel="Time (UTC)", ylabel=t.get('graph_ylabel', "Altitude (°)"), title=t.get('graph_title_alt_time', "Alt Plot for {}").format(obj_name), ylim=(0, 90))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')); fig.autofmt_xdate(rotation=30)
        elif plot_type == 'Sky Path':
            if azimuths is None: st.error("Plot Error: Azimuths needed."); plt.close(fig); return None
            ax.remove(); ax = fig.add_subplot(111, projection='polar', facecolor=facecolor)
            az_rad = np.deg2rad(azimuths); radius = 90 - altitudes
            time_delta = times.jd.max() - times.jd.min()
            time_norm = (times.jd - times.jd.min()) / (time_delta + 1e-9)
            colors = plt.cm.plasma(time_norm)
            scatter = ax.scatter(az_rad, radius, c=colors, s=15, alpha=0.8, edgecolors='none', label=obj_name)
            ax.plot(az_rad, radius, color=primary_color, alpha=0.4, lw=0.8)
            ax.plot(np.linspace(0, 2*np.pi, 100), np.full(100, 90 - min_altitude_deg), color=min_alt_color, ls='--', lw=1.2, label=t.get('graph_min_altitude_label', "Min Alt ({:.0f}°)").format(min_altitude_deg), alpha=0.8)
            if max_altitude_deg < 90: ax.plot(np.linspace(0, 2*np.pi, 100), np.full(100, 90 - max_altitude_deg), color=max_alt_color, ls=':', lw=1.2, label=t.get('graph_max_altitude_label', "Max Alt ({:.0f}°)").format(max_altitude_deg), alpha=0.8)
            ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
            ax.set_yticks(np.arange(0, 91, 15)); ax.set_yticklabels([f"{90-alt}°" for alt in np.arange(0, 91, 15)], color=label_color)
            ax.set(ylim=(0, 90), title=t.get('graph_title_sky_path', "Sky Path for {}").format(obj_name))
            ax.title.set(va='bottom', color=title_color, fontsize=13, weight='bold', y=1.1)
            try: # Colorbar
                cbar = fig.colorbar(scatter, ax=ax, label="Time (UTC)", pad=0.1, shrink=0.7)
                if len(times) > 0: start_lbl, end_lbl = times[0].to_datetime(timezone.utc).strftime('%H:%M'), times[-1].to_datetime(timezone.utc).strftime('%H:%M')
                else: start_lbl, end_lbl = 'Start', 'End'
                cbar.set_ticks([0, 1]); cbar.ax.set_yticklabels([start_lbl, end_lbl])
                cbar.set_label("Time (UTC)", color=label_color, fontsize=10); cbar.ax.yaxis.set_tick_params(color=label_color, labelsize=9)
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=label_color); cbar.outline.set_edgecolor(spine_color); cbar.outline.set_linewidth(0.5)
            except Exception as cbar_err: print(f"Warn: Cbar fail: {cbar_err}")
        else: st.error(f"Plot Error: Unknown type '{plot_type}'"); plt.close(fig); return None

        ax.grid(True, linestyle=':', alpha=0.5, color=grid_color); ax.tick_params(axis='x', colors=label_color); ax.tick_params(axis='y', colors=label_color)
        for spine in ax.spines.values(): spine.set_color(spine_color); spine.set_linewidth(0.5)
        legend = ax.legend(loc='lower right', fontsize='small', facecolor=legend_facecolor, framealpha=0.8, edgecolor=spine_color)
        for text in legend.get_texts(): text.set_color(label_color)

        return fig
    except Exception as e:
        st.error(f"Plot Error: Unexpected: {e}"); traceback.print_exc()
        if fig: plt.close(fig)
        return None

# --- Run the App ---
if __name__ == "__main__":
    main()
