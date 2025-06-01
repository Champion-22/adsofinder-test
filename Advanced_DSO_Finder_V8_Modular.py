# -*- coding: utf-8 -*-
# --- Basic Imports ---
from __future__ import annotations
import streamlit as st
import random
from datetime import datetime, date, time, timedelta, timezone
import traceback
import os
import urllib.parse
import pandas as pd
import math
import numpy as np # Needed for Redshift Calc

# --- Library Imports ---
try:
    from astropy.time import Time
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
    from scipy.integrate import quad # Needed for Redshift Calc
except ImportError as e:
    st.error(f"Error: Missing libraries. Please install required packages (check astroplan, astropy, scipy, etc.). ({e})")
    st.stop()

# --- Localization Import ---
# The main() function will handle the actual instantiation of the translator 't'
# by attempting to import from the user-provided 'localization.py' first.

# --- Page Config ---
st.set_page_config(page_title="Advanced DSO Finder", layout="wide")

# --- Global Configuration & Initial Values ---
INITIAL_LAT = 47.17
INITIAL_LON = 8.01
INITIAL_HEIGHT = 550
INITIAL_TIMEZONE = "Europe/Zurich"

# --- Path to Catalog File ---
try: APP_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError: APP_DIR = os.getcwd() # Fallback for environments where __file__ is not defined
CATALOG_FILENAME = "ongc.csv"
CATALOG_FILEPATH = os.path.join(APP_DIR, CATALOG_FILENAME)

# --- Constants ---
CARDINAL_DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
ALL_DIRECTIONS_KEY = 'All'

# --- Constants for Redshift Calculator ---
C_KM_PER_S = 299792.458  # Speed of light in km/s
KM_PER_MPC = 3.085677581491367e+19
KM_PER_AU = 1.495978707e+8
KM_PER_LY = 9.4607304725808e+12
KM_PER_LS = C_KM_PER_S 
GYR_PER_YR = 1e9 
H0_DEFAULT = 67.4
OMEGA_M_DEFAULT = 0.315
OMEGA_LAMBDA_DEFAULT = 0.685

# --- Initialize TimezoneFinder ---
@st.cache_resource
def get_timezone_finder_instance(): # Renamed to avoid conflict if tf is used as a variable elsewhere
    if 'TimezoneFinder' in globals(): 
        try: return TimezoneFinder(in_memory=True)
        except Exception as e: print(f"Error initializing TimezoneFinder: {e}"); st.warning(f"TimezoneFinder initialization failed: {e}. Automatic timezone detection disabled."); return None
    return None
tf_instance = get_timezone_finder_instance()

# --- Initialize Session State ---
def initialize_session_state():
    defaults = {
        # DSO Finder State
        'language': 'de', 'plot_object_name': None, 'show_plot': False, 'active_result_plot_data': None,
        'last_results': [], 'find_button_pressed': False, 'location_choice_key': 'Search',
        'manual_lat_val': INITIAL_LAT, 'manual_lon_val': INITIAL_LON, 'manual_height_val': INITIAL_HEIGHT,
        'location_search_query': "", 'searched_location_name': None, 'location_search_status_msg': "",
        'location_search_success': False, 'selected_timezone': INITIAL_TIMEZONE, 'manual_min_mag_slider': 0.0,
        'manual_max_mag_slider': 16.0, 'object_type_filter_exp': [], 'mag_filter_mode_exp': 'Bortle Scale',
        'bortle_slider': 5, 'min_alt_slider': 20, 'max_alt_slider': 90, 'moon_phase_slider': 35,
        'size_arcmin_range': [1.0, 120.0], 'sort_method': 'Duration & Altitude',
        'selected_peak_direction': ALL_DIRECTIONS_KEY, 'plot_type_selection': 'Sky Path', 'custom_target_ra': "",
        'custom_target_dec': "", 'custom_target_name': "", 'custom_target_error': "", 'custom_target_plot_data': None,
        'show_custom_plot': False, 'expanded_object_name': None, 'location_is_valid_for_run': False,
        'time_choice_exp': 'Now', 'window_start_time': None, 'window_end_time': None, 'selected_date_widget': date.today(),
        # Redshift Calculator State
        'redshift_z_input': 0.1, 'redshift_h0_input': H0_DEFAULT, 'redshift_omega_m_input': OMEGA_M_DEFAULT,
        'redshift_omega_lambda_input': OMEGA_LAMBDA_DEFAULT,
    }
    for key, default_value in defaults.items():
        if key not in st.session_state: st.session_state[key] = default_value

# --- Helper Functions ---
def get_magnitude_limit(bortle_scale: int) -> float:
    limits = {1: 15.5, 2: 15.5, 3: 14.5, 4: 14.5, 5: 13.5, 6: 12.5, 7: 11.5, 8: 10.5, 9: 9.5}
    return limits.get(bortle_scale, 9.5)

def azimuth_to_direction(azimuth_deg: float) -> str:
    if math.isnan(azimuth_deg): return "N/A" 
    azimuth_deg %= 360
    index = round((azimuth_deg + 22.5) / 45) % 8 
    return CARDINAL_DIRECTIONS[max(0, min(index, len(CARDINAL_DIRECTIONS) - 1))]

def create_moon_phase_svg(illumination: float, size: int = 100) -> str:
    if not 0 <= illumination <= 1: 
        print(f"Warning: Invalid moon illumination value ({illumination}) received. Clamping to [0,1].")
        illumination = max(0.0, min(1.0, illumination))
    radius = size / 2; cx = cy = radius
    light_color = "var(--text-color, #E0E0E0)"; dark_color = "var(--secondary-background-color, #333333)"
    svg_parts = [f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">', f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{dark_color}"/>']
    if illumination < 0.01: pass
    elif illumination > 0.99: svg_parts.append(f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{light_color}"/>')
    else:
        x_terminator_offset = radius * (illumination * 2 - 1); rx_ellipse = abs(x_terminator_offset)
        if illumination <= 0.5: laf_e, sf_e, laf_c, sf_c = 0, 1, 0, 1
        else: laf_e, sf_e, laf_c, sf_c = 1, 1, 1, 1
        path_d = (f"M {cx},{cy - radius} A {rx_ellipse},{radius} 0 {laf_e},{sf_e} {cx},{cy + radius} A {radius},{radius} 0 {laf_c},{sf_c} {cx},{cy - radius} Z")
        svg_parts.append(f'<path d="{path_d}" fill="{light_color}"/>')
    svg_parts.append('</svg>')
    return "".join(svg_parts)

def load_ongc_data(catalog_path: str, lang: str, t_loader) -> pd.DataFrame | None:
    required_cols = ['Name', 'RA', 'Dec', 'Type']; mag_cols = ['V-Mag', 'B-Mag', 'Mag']; size_col = 'MajAx'
    try:
        if not os.path.exists(catalog_path):
            st.error(f"{t_loader.get('error_loading_catalog_file_not_found', 'Error: Catalog file not found at path: {}').format(catalog_path)}")
            return None
        df = pd.read_csv(catalog_path, sep=';', comment='#', low_memory=False)
        missing_req_cols = [col for col in required_cols if col not in df.columns]
        if missing_req_cols:
            st.error(f"{t_loader.get('error_missing_catalog_columns', 'Error: Catalog missing required columns: {}').format(', '.join(missing_req_cols))}")
            return None
        df['RA_str'] = df['RA'].astype(str).str.strip(); df['Dec_str'] = df['Dec'].astype(str).str.strip()
        df.dropna(subset=['RA_str', 'Dec_str'], inplace=True); df = df[(df['RA_str'] != '') & (df['Dec_str'] != '')]
        mag_col_found = None
        for col_name in mag_cols:
            if col_name in df.columns and pd.to_numeric(df[col_name], errors='coerce').notna().any():
                mag_col_found = col_name; print(f"Using magnitude column: {mag_col_found}"); break
        if mag_col_found is None:
            st.error(t_loader.get('error_no_usable_magnitude_column', "Error: No usable magnitude column found in catalog from options: {}").format(', '.join(mag_cols)))
            return None
        df['Mag'] = pd.to_numeric(df[mag_col_found], errors='coerce'); df.dropna(subset=['Mag'], inplace=True)
        if size_col not in df.columns:
            st.warning(t_loader.get('warning_size_column_missing', "Warning: Size column '{}' not found in catalog. Size filtering will be disabled.").format(size_col))
            df[size_col] = np.nan
        else:
            df[size_col] = pd.to_numeric(df[size_col], errors='coerce')
            if not df[size_col].notna().any():
                st.warning(t_loader.get('warning_size_column_no_valid_data', "Warning: Size column '{}' contains no valid numeric data. Size filtering may be ineffective.").format(size_col))
                df[size_col] = np.nan
        dso_types_pattern = '|'.join(['Galaxy', 'Globular Cluster', 'Open Cluster', 'Nebula', 'Planetary Nebula', 'Supernova Remnant', 'HII', 'Emission Nebula', 'Reflection Nebula', 'Cluster [+] Nebula', 'Gal', 'GCl', 'Gx', 'OC', 'PN', 'SNR', 'Neb', 'EmN', 'RfN', 'C[+]N', 'Gxy', 'AGN', 'MWSC'])
        if 'Type' in df.columns:
            df_filtered = df[df['Type'].astype(str).str.contains(dso_types_pattern, case=False, na=False)].copy()
        else:
            st.error(t_loader.get('error_type_column_missing_critical', "Critical Error: 'Type' column is missing from catalog for filtering."))
            return None
        final_columns_to_keep = ['Name', 'RA_str', 'Dec_str', 'Mag', 'Type', size_col]
        existing_final_columns = [col for col in final_columns_to_keep if col in df_filtered.columns]
        df_final = df_filtered[existing_final_columns].copy()
        df_final.drop_duplicates(subset=['Name'], inplace=True, keep='first'); df_final.reset_index(drop=True, inplace=True)
        if df_final.empty:
            st.warning(t_loader.get('warning_catalog_empty_after_filters', "Warning: Catalog is empty after initial type filtering or contains no valid DSOs."))
            return None
        print(f"Catalog loaded and pre-processed: {len(df_final)} objects.")
        return df_final
    except FileNotFoundError:
        st.error(f"{t_loader.get('error_loading_catalog_file_not_found', 'Error: Catalog file not found at path: {}').format(catalog_path)}")
        return None
    except pd.errors.EmptyDataError:
        st.error(t_loader.get('error_catalog_empty_data', "Error: Catalog file is empty or not a valid CSV.").format(catalog_path))
        return None
    except Exception as e:
        st.error(f"{t_loader.get('error_loading_catalog_generic', 'An unexpected error occurred while loading/processing the catalog: {}').format(e)}")
        traceback.print_exc(); return None

def _get_fallback_window(reference_time: Time, t_loader) -> tuple[Time, Time]:
    ref_dt = reference_time.to_datetime(timezone.utc); ref_date = ref_dt.date()
    start_dt_utc = datetime.combine(ref_date, time(18, 0, 0), tzinfo=timezone.utc)
    end_dt_utc = datetime.combine(ref_date + timedelta(days=1), time(6, 0, 0), tzinfo=timezone.utc)
    start_time = Time(start_dt_utc, scale='utc'); end_time = Time(end_dt_utc, scale='utc')
    print(f"Using fallback observation window: {start_time.iso} to {end_time.iso} UTC")
    return start_time, end_time

def get_observable_window(observer: Observer, reference_time: Time, is_now_mode: bool, t_loader) -> tuple[Time | None, Time | None, str]:
    status_message = ""; start_time_utc, end_time_utc = None, None; current_utc_time = Time.now()
    if is_now_mode:
        current_dt_utc = current_utc_time.to_datetime(timezone.utc)
        base_calc_time = Time(datetime.combine(current_dt_utc.date(), time(12,0,0), tzinfo=timezone.utc), scale='utc')
    else: base_calc_time = reference_time
    try:
        if not isinstance(observer, Observer): raise TypeError(t_loader.get("error_observer_type_invalid", "Invalid observer object provided."))
        evening_twilight_utc = observer.twilight_evening_astronomical(base_calc_time, which='next')
        morning_twilight_utc = observer.twilight_morning_astronomical(evening_twilight_utc if evening_twilight_utc else base_calc_time, which='next')
        if evening_twilight_utc is None or morning_twilight_utc is None:
            sun_alt_at_base = observer.sun_altaz(base_calc_time).alt; sun_alt_12h_later = observer.sun_altaz(base_calc_time + 12*u.hour).alt
            if sun_alt_at_base < -18*u.deg and sun_alt_12h_later < -18*u.deg:
                status_message = t_loader.get('error_polar_night', "Astronomical darkness persists (Polar Night?). Using fallback window.")
                start_time_utc, end_time_utc = _get_fallback_window(base_calc_time, t_loader)
            elif sun_alt_at_base > -18*u.deg and sun_alt_12h_later > -18*u.deg:
                times_to_check = base_calc_time + np.linspace(0, 24, 49)*u.hour; sun_alts_check = observer.sun_altaz(times_to_check).alt
                if np.min(sun_alts_check.value) > -18:
                    status_message = t_loader.get('error_polar_day', "No astronomical darkness (Polar Day?). Using fallback window.")
                    start_time_utc, end_time_utc = _get_fallback_window(base_calc_time, t_loader)
                else: raise ValueError(t_loader.get("error_twilight_calc_unexpected_polar", "Twilight calculation failed unexpectedly in potential polar conditions."))
            else: raise ValueError(t_loader.get("error_twilight_calc_failed", "Could not calculate astronomical twilight times."))
            if start_time_utc and end_time_utc:
                 status_message += " " + t_loader.get('window_fallback_info_short', "Fallback: {} to {} UTC").format(start_time_utc.iso, end_time_utc.iso)
                 return start_time_utc, end_time_utc, status_message
            else: raise ValueError(status_message or t_loader.get("error_twilight_calc_failed", "Could not calculate astronomical twilight times."))
        if morning_twilight_utc <= evening_twilight_utc:
             status_message = t_loader.get('error_morning_before_evening_twilight', "Morning twilight occurs before or at evening twilight. Check for polar conditions or date selection.")
             if not start_time_utc: 
                 start_time_utc, end_time_utc = _get_fallback_window(base_calc_time, t_loader)
                 status_message += " " + t_loader.get('window_fallback_info_short', "Fallback: {} to {} UTC").format(start_time_utc.iso, end_time_utc.iso)
             return start_time_utc, end_time_utc, status_message
        start_time_utc = evening_twilight_utc; end_time_utc = morning_twilight_utc
        if is_now_mode:
            if end_time_utc < current_utc_time:
                status_message = t_loader.get('window_already_passed', "Calculated observation window for the current night has passed. Advancing to the next night.") + "\n"
                next_night_base_time = Time(datetime.combine(current_utc_time.to_datetime(timezone.utc).date() + timedelta(days=1), time(12,0,0), tzinfo=timezone.utc), scale='utc')
                start_time_utc = observer.twilight_evening_astronomical(next_night_base_time, which='next')
                if start_time_utc: end_time_utc = observer.twilight_morning_astronomical(start_time_utc, which='next')
                else: raise ValueError(t_loader.get("error_twilight_calc_next_night_failed", "Failed to calculate twilight for the next night."))
                if start_time_utc is None or end_time_utc is None or end_time_utc <= start_time_utc:
                    status_message += t_loader.get('error_twilight_recalc_failed_fallback', "Recalculation for next night failed. Using fallback.")
                    start_time_utc, end_time_utc = _get_fallback_window(next_night_base_time, t_loader)
                    status_message += " " + t_loader.get('window_fallback_info_short', "Fallback: {} to {} UTC").format(start_time_utc.iso, end_time_utc.iso)
                    return start_time_utc, end_time_utc, status_message
            elif start_time_utc < current_utc_time:
                print(f"Observation window already started. Adjusting start from {start_time_utc.iso} to current time {current_utc_time.iso}")
                start_time_utc = current_utc_time
        start_display = start_time_utc.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M %Z')
        end_display = end_time_utc.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M %Z')
        status_message += t_loader.get('window_info_template', "Observation window: {} to {} (Astronomical Twilight)").format(start_display, end_display)
    except Exception as e:
        error_info = traceback.format_exc()
        status_message = t_loader.get('window_calc_error', "Error calculating observation window: {}. {}").format(str(e), error_info if len(error_info) < 200 else "See console for full traceback.")
        print(f"Full error during window calculation: {e}\n{error_info}")
        if not start_time_utc or not end_time_utc or (start_time_utc and end_time_utc and end_time_utc <= start_time_utc):
            start_time_utc, end_time_utc = _get_fallback_window(base_calc_time, t_loader)
            status_message += " " + t_loader.get('window_fallback_info_short', "Fallback: {} to {} UTC").format(start_time_utc.iso, end_time_utc.iso)
    if start_time_utc is None or end_time_utc is None or end_time_utc <= start_time_utc:
        if t_loader.get('error_no_window', "No valid window") not in status_message:
            status_message += ("\n" if status_message else "") + t_loader.get('error_no_window_final_fallback', "Could not determine a valid observation window. Using a default fallback.")
        start_fb, end_fb = _get_fallback_window(reference_time, t_loader)
        if (start_time_utc != start_fb or end_time_utc != end_fb) and \
           t_loader.get('window_fallback_info_short', "Fallback: {} to {} UTC").format(start_fb.iso, end_fb.iso) not in status_message:
            status_message += " " + t_loader.get('window_fallback_info_short', "Fallback: {} to {} UTC").format(start_fb.iso, end_fb.iso)
        start_time_utc, end_time_utc = start_fb, end_fb
    return start_time_utc, end_time_utc, status_message

def find_observable_objects(observer_location: EarthLocation, observing_times: Time, min_altitude_limit: u.Quantity, catalog_df: pd.DataFrame, t_loader) -> list[dict]:
    observable_objects = []
    if not isinstance(observer_location, EarthLocation): st.error(t_loader.get("error_internal_observer_location_type", "Internal Error: observer_location is not of type EarthLocation.")); return []
    if not isinstance(observing_times, Time) or not observing_times.shape: st.error(t_loader.get("error_internal_observing_times_type", "Internal Error: observing_times is not a valid Astropy Time array.")); return []
    if not isinstance(min_altitude_limit, u.Quantity): st.error(t_loader.get("error_internal_min_altitude_type", "Internal Error: min_altitude_limit is not an Astropy Quantity.")); return []
    if not isinstance(catalog_df, pd.DataFrame): st.error(t_loader.get("error_internal_catalog_df_type", "Internal Error: catalog_df is not a Pandas DataFrame.")); return []
    if catalog_df.empty: print("Input catalog_df for find_observable_objects is empty."); return []
    if len(observing_times) < 2: st.warning(t_loader.get("warning_obs_window_too_few_points", "Observation window has fewer than 2 time points; duration calculation might be affected."))
    altaz_frame = AltAz(obstime=observing_times, location=observer_location); min_alt_deg_value = min_altitude_limit.to(u.deg).value
    time_step_hours = 0.0
    if len(observing_times) > 1: time_step_hours = (observing_times[1] - observing_times[0]).sec / 3600.0
    for index, obj_row in catalog_df.iterrows():
        try:
            obj_name = obj_row.get('Name', f"Object_{index}"); obj_ra_str = obj_row.get('RA_str'); obj_dec_str = obj_row.get('Dec_str')
            obj_type = obj_row.get('Type', t_loader.get("unknown_type_placeholder", "Unknown")); obj_mag = obj_row.get('Mag', np.nan); obj_size_arcmin = obj_row.get('MajAx', np.nan)
            if not obj_ra_str or not obj_dec_str: print(f"Skipping object '{obj_name}': Missing RA or Dec strings."); continue
            try: sky_coord = SkyCoord(ra=obj_ra_str, dec=obj_dec_str, unit=(u.hourangle, u.deg))
            except ValueError as coord_err: print(f"Skipping object '{obj_name}': Invalid coordinates ('{obj_ra_str}', '{obj_dec_str}'). Error: {coord_err}"); continue
            try: obj_altazs = sky_coord.transform_to(altaz_frame); altitudes_deg = obj_altazs.alt.to(u.deg).value; azimuths_deg = obj_altazs.az.to(u.deg).value
            except Exception as transform_err: print(f"Skipping object '{obj_name}': Error transforming coordinates to AltAz. Error: {transform_err}"); continue
            if not hasattr(altitudes_deg, '__len__') or len(altitudes_deg) == 0: print(f"Skipping object '{obj_name}': No altitude data generated."); continue
            max_altitude_val = np.max(altitudes_deg)
            if max_altitude_val >= min_alt_deg_value:
                peak_altitude_idx = np.argmax(altitudes_deg); peak_altitude = altitudes_deg[peak_altitude_idx]; peak_azimuth = azimuths_deg[peak_altitude_idx]
                peak_time_utc = observing_times[peak_altitude_idx]; peak_direction_cardinal = azimuth_to_direction(peak_azimuth)
                try: constellation_name = get_constellation(sky_coord, short_name=False)
                except Exception as const_err: print(f"Warning: Could not determine constellation for '{obj_name}'. Error: {const_err}"); constellation_name = t_loader.get("constellation_not_available", "N/A")
                is_above_min_alt = altitudes_deg >= min_alt_deg_value; continuous_duration_hours = 0
                if time_step_hours > 0 and np.any(is_above_min_alt):
                    change_indices = np.where(np.diff(is_above_min_alt))[0] + 1; runs = np.split(np.arange(len(is_above_min_alt)), change_indices)
                    max_run_length = 0
                    for run_indices in runs:
                        if run_indices.size > 0 and is_above_min_alt[run_indices[0]]: max_run_length = max(max_run_length, len(run_indices))
                    continuous_duration_hours = max_run_length * time_step_hours
                elif np.all(is_above_min_alt) and len(observing_times) == 1 and time_step_hours == 0: continuous_duration_hours = 0
                result_dict = {'Name': obj_name, 'Type': obj_type, 'Constellation': constellation_name, 'Magnitude': obj_mag if not np.isnan(obj_mag) else None,
                    'Size (arcmin)': obj_size_arcmin if not np.isnan(obj_size_arcmin) else None, 'RA': obj_ra_str, 'Dec': obj_dec_str, 'Max Altitude (°)': peak_altitude,
                    'Azimuth at Max (°)': peak_azimuth, 'Direction at Max': peak_direction_cardinal, 'Time at Max (UTC)': peak_time_utc, 'Max Cont. Duration (h)': continuous_duration_hours,
                    'skycoord': sky_coord, 'altitudes': altitudes_deg, 'azimuths': azimuths_deg, 'times': observing_times}
                observable_objects.append(result_dict)
        except Exception as per_object_err:
            obj_id = obj_row.get('Name', f'Object at index {index}'); print(t_loader.get('error_processing_object', "Error processing object {}: {}").format(obj_id, per_object_err))
    return observable_objects

def get_local_time_str(utc_time: Time | None, timezone_str: str, t_loader) -> tuple[str, str]:
    if utc_time is None: return "N/A", "N/A"
    if not isinstance(utc_time, Time): print(f"Error in get_local_time_str: utc_time is not an Astropy Time object (type: {type(utc_time)})."); return "N/A (Time Error)", "N/A"
    if not isinstance(timezone_str, str) or not timezone_str: print(f"Error in get_local_time_str: timezone_str is invalid (value: '{timezone_str}')."); return utc_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), "UTC (TZ Invalid)"
    try:
        target_tz = pytz.timezone(timezone_str); utc_datetime_obj = utc_time.to_datetime(timezone.utc); local_datetime_obj = utc_datetime_obj.astimezone(target_tz)
        local_time_formatted_str = local_datetime_obj.strftime('%Y-%m-%d %H:%M:%S'); tz_name_display = local_datetime_obj.tzname()
        if not tz_name_display: tz_name_display = target_tz.zone
        return local_time_formatted_str, tz_name_display
    except pytz.exceptions.UnknownTimeZoneError: print(f"Error in get_local_time_str: Unknown timezone '{timezone_str}'. Defaulting to UTC display."); return utc_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), "UTC (TZ Error)"
    except Exception as e: print(f"Error converting UTC time to local time for timezone '{timezone_str}': {e}"); traceback.print_exc(); return utc_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), "UTC (Conversion Error)"

# --- Redshift Calculation Functions ---
def hubble_parameter_inv_integrand(z, omega_m, omega_lambda):
  epsilon = 1e-15; denominator_sq = omega_m * (1 + z)**3 + omega_lambda
  if denominator_sq < 0: denominator_sq = 0
  denominator = np.sqrt(denominator_sq + epsilon)
  return 1.0 / denominator if denominator > epsilon else 0.0

def lookback_time_integrand(z, omega_m, omega_lambda):
  epsilon = 1e-15; e_z_sq = omega_m * (1 + z)**3 + omega_lambda
  if e_z_sq < 0: e_z_sq = 0
  e_z = np.sqrt(e_z_sq + epsilon); denominator = (1 + z) * e_z
  if math.isclose(z, 0):
      e_0_sq = omega_m + omega_lambda;
      if e_0_sq < 0: e_0_sq = 0
      e_0 = np.sqrt(e_0_sq + epsilon)
      return 1.0 / e_0 if e_0 > epsilon else 0.0
  return 1.0 / denominator if abs(denominator) > epsilon else 0.0

@st.cache_data
def calculate_lcdm_distances(redshift: float, h0: float, omega_m: float, omega_lambda: float) -> dict:
    if not all(isinstance(v, (int, float)) for v in [redshift, h0, omega_m, omega_lambda]): return {'error_key': "error_invalid_input"}
    if h0 <= 0: return {'error_key': "error_h0_positive"}
    if omega_m < 0 or omega_lambda < 0: return {'error_key': "error_omega_negative"}
    recessional_velocity_km_s = redshift * C_KM_PER_S
    if redshift < -1 + 1e-9: return {'comoving_mpc': 0.0, 'luminosity_mpc': 0.0, 'ang_diam_mpc': 0.0, 'lookback_gyr': 0.0, 'recessional_velocity_km_s': recessional_velocity_km_s, 'error_key': "error_redshift_too_negative", 'error_args': {'z': redshift}}
    if redshift < 0: return {'comoving_mpc': 0.0, 'luminosity_mpc': 0.0, 'ang_diam_mpc': 0.0, 'lookback_gyr': 0.0, 'recessional_velocity_km_s': recessional_velocity_km_s, 'error_key': "warn_blueshift"}
    if math.isclose(redshift, 0): return {'comoving_mpc': 0.0, 'luminosity_mpc': 0.0, 'ang_diam_mpc': 0.0, 'lookback_gyr': 0.0, 'recessional_velocity_km_s': 0.0, 'error_key': None}
    dh_mpc = C_KM_PER_S / h0; hubble_time_gyr_approx = 977.8 / h0
    try:
        integral_dc_norm, err_dc = quad(hubble_parameter_inv_integrand, 0, redshift, args=(omega_m, omega_lambda), limit=100, epsabs=1.49e-08, epsrel=1.49e-08)
        comoving_distance_mpc = dh_mpc * integral_dc_norm
        integral_lt_norm, err_lt = quad(lookback_time_integrand, 0, redshift, args=(omega_m, omega_lambda), limit=100, epsabs=1.49e-08, epsrel=1.49e-08)
        lookback_time_gyr = hubble_time_gyr_approx * integral_lt_norm
        luminosity_distance_mpc = comoving_distance_mpc * (1 + redshift); angular_diameter_distance_mpc = comoving_distance_mpc / (1 + redshift)
        warning_key, warning_args = None, {}
        if err_dc > 1e-4 or err_lt > 1e-4: warning_key = "warn_integration_accuracy"; warning_args = {'err_dc': f"{err_dc:.2e}", 'err_lt': f"{err_lt:.2e}"}
        return {'comoving_mpc': comoving_distance_mpc, 'luminosity_mpc': luminosity_distance_mpc, 'ang_diam_mpc': angular_diameter_distance_mpc, 'lookback_gyr': lookback_time_gyr, 'recessional_velocity_km_s': recessional_velocity_km_s, 'error_key': None, 'warning_key': warning_key, 'warning_args': warning_args}
    except ImportError: return {'error_key': "error_dep_scipy"}
    except Exception as e: st.exception(e); return {'error_key': "error_calc_failed", 'error_args': {'e': str(e)}}

# --- Redshift Unit Conversion & Formatting ---
def convert_mpc_to_km(d_mpc: float) -> float: return d_mpc * KM_PER_MPC
def convert_km_to_au(d_km: float) -> float: return 0.0 if d_km == 0 else d_km / KM_PER_AU
def convert_km_to_ly(d_km: float) -> float: return 0.0 if d_km == 0 else d_km / KM_PER_LY
def convert_km_to_ls(d_km: float) -> float: return 0.0 if d_km == 0 else d_km / KM_PER_LS
def convert_mpc_to_gly(d_mpc: float) -> float:
    if d_mpc == 0: return 0.0
    km_per_giga_ly = KM_PER_LY * 1e9; distance_km = convert_mpc_to_km(d_mpc)
    if km_per_giga_ly == 0: return np.inf if distance_km != 0 else 0.0
    return distance_km / km_per_giga_ly
def format_large_number(number: float | int, t_loader) -> str:
    if number == 0: return "0"
    if not np.isfinite(number): return str(number)
    try:
        if abs(number) >= 1e6 or (abs(number) < 1e-3 and number !=0 ): return f"{number:,.3e}" 
        elif abs(number) < 1: return f"{number:,.4f}"
        else: return f"{number:,.2f}".replace(",", t_loader.get("thousands_separator", " ")) # Use localized separator
    except (ValueError, TypeError): return str(number)

# --- Redshift Example Helpers ---
def get_lookback_comparison_key(gyr: float) -> str:
    if gyr < 0.001: return "example_lookback_recent"
    if gyr < 0.05: return "example_lookback_humans"
    if gyr < 0.066: return "example_lookback_dinos_extinction"
    if gyr < 0.540: return "example_lookback_multicellular"
    if gyr < 4.5: return "example_lookback_earth_formation"
    if gyr < 13.0: return "example_lookback_early_universe_galaxies"
    return "example_lookback_very_early_universe"
def get_comoving_comparison_key(mpc: float) -> str:
    if mpc < 2: return "example_comoving_local_group"
    if mpc < 20: return "example_comoving_virgo_cluster"
    if mpc < 100: return "example_comoving_coma_cluster"
    if mpc < 300: return "example_comoving_laniakea_supercluster"
    if mpc < 1000: return "example_comoving_large_scale_structure"
    if mpc < 9000: return "example_comoving_distant_quasars"
    return "example_comoving_observable_universe_horizon"

# --- Plotting Function ---
def create_plot(plot_data: dict, min_altitude_deg: float, max_altitude_deg: float, plot_type: str, t_loader) -> plt.Figure | None:
    fig = None
    try:
        if not isinstance(plot_data, dict): st.error(t_loader.get("plot_error_invalid_data_type", "Plot Error: Invalid plot_data type.")); return None
        times_astropy = plot_data.get('times'); altitudes_np = plot_data.get('altitudes'); azimuths_np = plot_data.get('azimuths')
        obj_name = plot_data.get('Name', t_loader.get("plot_default_object_name", "Object"))
        if not isinstance(times_astropy, Time) or not hasattr(times_astropy, 'plot_date'): st.error(t_loader.get("plot_error_invalid_times_data", "Plot Error: 'times' data is not valid.")); return None
        if not isinstance(altitudes_np, np.ndarray): st.error(t_loader.get("plot_error_invalid_altitudes_data", "Plot Error: 'altitudes' data is not valid.")); return None
        if plot_type == 'Sky Path' and not isinstance(azimuths_np, np.ndarray): st.error(t_loader.get("plot_error_invalid_azimuths_data_sky_path", "Plot Error: 'azimuths' data required for Sky Path.")); return None
        if len(times_astropy) != len(altitudes_np) or (azimuths_np is not None and len(times_astropy) != len(azimuths_np)): st.error(t_loader.get("plot_error_data_array_length_mismatch", "Plot Error: Data array lengths mismatch.")); return None
        if len(times_astropy) < 1: st.info(t_loader.get("plot_info_no_data_to_plot", "Plot Info: No data points to plot.")); return None
        plot_times_mpl = times_astropy.plot_date
        try: is_streamlit_dark_theme = (st.get_option("theme.base") == "dark")
        except Exception: is_streamlit_dark_theme = False; print("Warning: Could not determine Streamlit theme. Defaulting to light plot style.")
        plt.style.use('dark_background' if is_streamlit_dark_theme else 'seaborn-v0_8-whitegrid')
        label_color = '#FAFAFA' if is_streamlit_dark_theme else '#333333'; title_color = '#FFFFFF' if is_streamlit_dark_theme else '#000000'; grid_color = '#444444' if is_streamlit_dark_theme else 'darkgray'
        primary_line_color = 'deepskyblue' if is_streamlit_dark_theme else 'dodgerblue'; min_alt_line_color = 'tomato' if is_streamlit_dark_theme else 'red'; max_alt_line_color = 'orange' if is_streamlit_dark_theme else 'darkorange'
        spine_color = '#AAAAAA' if is_streamlit_dark_theme else '#555555'; legend_face_color = '#262730' if is_streamlit_dark_theme else '#F0F0F0'; figure_face_color = '#0E1117' if is_streamlit_dark_theme else '#FFFFFF'
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=figure_face_color, constrained_layout=True); ax.set_facecolor(figure_face_color)
        if plot_type == 'Altitude Plot':
            ax.plot(plot_times_mpl, altitudes_np, color=primary_line_color, alpha=0.9, linewidth=1.5, label=obj_name)
            ax.axhline(min_altitude_deg, color=min_alt_line_color, linestyle='--', linewidth=1.2, label=t_loader.get('graph_min_altitude_label', "Min Alt ({:.0f}°)").format(min_altitude_deg), alpha=0.8)
            if max_altitude_deg < 90: ax.axhline(max_altitude_deg, color=max_alt_line_color, linestyle=':', linewidth=1.2, label=t_loader.get('graph_max_altitude_label', "Max Alt ({:.0f}°)").format(max_altitude_deg), alpha=0.8)
            ax.set_xlabel(t_loader.get("graph_xlabel_time_utc", "Time (UTC)"), color=label_color); ax.set_ylabel(t_loader.get('graph_ylabel_altitude', "Altitude (°)"), color=label_color)
            ax.set_title(t_loader.get('graph_title_alt_time', "Altitude Plot for {}").format(obj_name), color=title_color, fontsize=14, weight='bold'); ax.set_ylim(0, 90)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')); fig.autofmt_xdate(rotation=30)
        elif plot_type == 'Sky Path':
            if azimuths_np is None: st.error(t_loader.get("plot_error_azimuths_missing_sky_path_final", "Plot Error: Azimuth data missing for Sky Path plot.")); plt.close(fig); return None
            ax.remove(); ax = fig.add_subplot(111, projection='polar', facecolor=figure_face_color)
            azimuths_rad = np.deg2rad(azimuths_np); radii_polar = 90 - altitudes_np 
            time_jd_normalized = (times_astropy.jd - times_astropy.jd.min()) / (times_astropy.jd.max() - times_astropy.jd.min() + 1e-9)
            point_colors = plt.cm.viridis(time_jd_normalized)
            scatter = ax.scatter(azimuths_rad, radii_polar, c=point_colors, s=20, alpha=0.8, edgecolors='none', label=obj_name)
            ax.plot(azimuths_rad, radii_polar, color=primary_line_color, alpha=0.5, linewidth=0.8)
            ax.plot(np.linspace(0, 2*np.pi, 100), np.full(100, 90 - min_altitude_deg), color=min_alt_line_color, linestyle='--', linewidth=1.2, label=t_loader.get('graph_min_altitude_label', "Min Alt ({:.0f}°)").format(min_altitude_deg), alpha=0.8)
            if max_altitude_deg < 90: ax.plot(np.linspace(0, 2*np.pi, 100), np.full(100, 90 - max_altitude_deg), color=max_alt_line_color, linestyle=':', linewidth=1.2, label=t_loader.get('graph_max_altitude_label', "Max Alt ({:.0f}°)").format(max_altitude_deg), alpha=0.8)
            ax.set_theta_zero_location('N'); ax.set_theta_direction(-1); ax.set_yticks(np.arange(0, 91, 15)); ax.set_yticklabels([f"{90-alt}°" for alt in np.arange(0, 91, 15)], color=label_color)
            ax.set_ylim(0, 90); ax.set_title(t_loader.get('graph_title_sky_path', "Sky Path for {}").format(obj_name), color=title_color, fontsize=14, weight='bold', va='bottom', y=1.1)
            try:
                cbar = fig.colorbar(scatter, ax=ax, label=t_loader.get("graph_colorbar_label_time_utc", "Time (UTC)"), pad=0.1, shrink=0.7)
                start_time_label = times_astropy[0].to_datetime(timezone.utc).strftime('%H:%M') if len(times_astropy) > 0 else 'Start'
                end_time_label = times_astropy[-1].to_datetime(timezone.utc).strftime('%H:%M') if len(times_astropy) > 0 else 'End'
                cbar.set_ticks([0, 1]); cbar.ax.set_yticklabels([start_time_label, end_time_label])
                cbar.set_label(t_loader.get("graph_colorbar_label_time_utc", "Time (UTC)"), color=label_color, fontsize=10); cbar.ax.yaxis.set_tick_params(color=label_color, labelsize=9)
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=label_color); cbar.outline.set_edgecolor(spine_color); cbar.outline.set_linewidth(0.5)
            except Exception as cbar_err: print(f"Warning: Failed to create colorbar for Sky Path plot. Error: {cbar_err}")
        else: st.error(t_loader.get("plot_error_unknown_plot_type", "Plot Error: Unknown plot type '{}' specified.").format(plot_type)); plt.close(fig); return None
        ax.grid(True, linestyle=':', alpha=0.5, color=grid_color); ax.tick_params(axis='x', colors=label_color); ax.tick_params(axis='y', colors=label_color)
        if hasattr(ax, 'xaxis'): ax.xaxis.label.set_color(label_color); ax.yaxis.label.set_color(label_color)
        if hasattr(ax, 'title'): ax.title.set_color(title_color)
        for spine in ax.spines.values(): spine.set_color(spine_color); spine.set_linewidth(0.5)
        if ax.get_legend_handles_labels()[0]:
            legend = ax.legend(loc='best', fontsize='small', facecolor=legend_face_color, framealpha=0.85, edgecolor=spine_color)
            for text_obj in legend.get_texts(): text_obj.set_color(label_color)
        return fig
    except Exception as e: st.error(t_loader.get("plot_error_unexpected", "Plot Error: An unexpected error occurred: {}").format(e)); traceback.print_exc();
    if fig: plt.close(fig); return None

# --- Main Application ---
def main():
    initialize_session_state()

    # --- Localization Setup ---
    lang = st.session_state.language
    t = None 
    try:
        # Attempt to import from user's localization.py
        from localization import get_translation as get_translation_from_file
        
        # Assuming get_translation_from_file returns a dictionary for the selected language
        # (already handling its own DEFAULT_LANG fallback as per user's localization.py structure)
        t_dict_user = get_translation_from_file(lang)
        
        if not isinstance(t_dict_user, dict):
            print(f"Warning: get_translation from localization.py did not return a dict for lang '{lang}'. Using internal fallback.")
            raise ImportError("localization.py's get_translation did not return a dict.")

        class UserProvidedTranslator:
            def __init__(self, lang_dict_from_user_file):
                self.lang_dict = lang_dict_from_user_file
            
            def get(self, key, default_value=None):
                # The user's get_translation should already handle fallback to their DEFAULT_LANG.
                # This .get() is a simple wrapper. If key is still not in the user's dict,
                # then use the default_value provided in the main script's code.
                return self.lang_dict.get(key, default_value if default_value is not None else key)
        
        t = UserProvidedTranslator(t_dict_user)
        print(f"Successfully loaded translations using localization.py for language: {lang}")

    except ImportError:
        warning_message = "Warning: localization.py not found or error during import. Using internal fallback translations."
        print(warning_message)
        # Fallback to internal translations if localization.py is not available or fails
        _translations_fallback = {
            'de': {
                "app_title": "Advanced DSO Finder (Intern DE)", 'settings_header': "Einstellungen", 'language_select_label': "Sprache", 
                'location_expander': "📍 Standort", 'location_select_label': "Standort-Methode", 'location_option_manual': "Manuell", 
                'location_option_search': "Suche", 'location_search_label': "Ortsname:", 'location_search_submit_button': "Finden",
                'location_search_placeholder': "z.B. Berlin", 'location_search_found': "Gefunden (Nominatim): {}", 
                'location_search_found_fallback': "Gefunden (ArcGIS): {}", 'location_search_found_fallback2': "Gefunden (Photon): {}",
                'location_search_coords': "Lat: {:.4f}, Lon: {:.4f}", 'location_search_error_not_found': "Ort nicht gefunden.",
                'location_search_error_service': "Geocoding Fehler: {}", 'location_search_error_timeout': "Geocoding Zeitüberschreitung.",
                'location_search_error_refused': "Geocoding Verbindung abgelehnt.", 'location_search_info_fallback': "Versuche Fallback Geocoder...",
                'location_search_info_fallback2': "Versuche zweiten Fallback Geocoder...", 
                'location_search_error_fallback_failed': "Alle Geocoder fehlgeschlagen: {}",
                 'location_search_error_fallback2_failed': "Alle Geocoder fehlgeschlagen: {}", 'location_lat_label': "Breite (°N)",
                'location_lon_label': "Länge (°E)", 'location_elev_label': "Höhe (m)", 'location_manual_display': "Manuell ({:.4f}, {:.4f})",
                'location_search_display': "Gesucht: {} ({:.4f}, {:.4f})", 'location_error': "Standortfehler: {}", 
                'location_error_fallback': "FEHLER - Fallback aktiv", 'location_error_manual_none': "Manuelle Felder ungültig.",
                'time_expander': "⏱️ Zeit", 'time_select_label': "Zeit", 'time_option_now': "Jetzt", 'time_option_specific': "Spezifisch",
                'time_date_select_label': "Datum:", 'timezone_auto_set_label': "TZ:", 'timezone_auto_fail_label': "TZ:",
                'timezone_auto_fail_msg': "TZ Auto-Fail, UTC.", 'timezone_error_invalid': "Ungültig", 'timezone_auto_na': "Auto N/A",
                'timezone_loc_invalid': "Standort ungültig", 'filters_expander': "✨ Filter", 'mag_filter_header': "**Magnitude**",
                'mag_filter_method_label': "Methode:", 'mag_filter_option_bortle': "Bortle", 'mag_filter_option_manual': "Manuell",
                'mag_filter_bortle_label': "Bortle:", 'mag_filter_bortle_help': "1=Dunkel, 9=Stadt", 'mag_filter_min_mag_label': "Min:",
                'mag_filter_min_mag_help': "Hellstes Obj.", 'mag_filter_max_mag_label': "Max:", 'mag_filter_max_mag_help': "Schwächstes Obj.",
                'mag_filter_warning_min_max': "Min > Max!", 'min_alt_header': "**Höhe**", 'min_alt_label': "Min (°):", 'max_alt_label': "Max (°):",
                'alt_filter_warning_min_max': "Min Höhe > Max Höhe!", 'moon_warning_header': "**Mond**", 'moon_warning_label': "Warnen > (%):",
                'object_types_header': "**Typen**", 'object_types_error_extract': "Typen Extraktionsfehler.", 'object_types_label': "Typen filtern:",
                'object_types_not_found': "Keine Typen im Katalog.", 'size_filter_header': "**Größe**", 'size_filter_label': "Größe (arcmin):",
                'size_filter_help': "Größe in Bogenminuten", 'size_slider_error': "Größen-Slider Fehler", 'size_data_not_available': "Größendaten N/A.",
                'direction_filter_header': "**Richtung**", 'direction_filter_label': "Richtung:", 'direction_option_all': "Alle",
                'object_type_glossary_title': "Objekttyp Glossar", 'object_type_glossary': {"OCl": "Offener Haufen", "GCl": "Kugelsternhaufen", "Gal": "Galaxie", "PN": "Planetarischer Nebel"},
                'glossary_not_available': "Glossar nicht verfügbar.", 'glossary_format_error': "Glossardaten haben nicht das erwartete Format.",
                'results_options_expander': "⚙️ Ergebnis Opts", 'results_options_max_objects_label': "Max Objs:",
                'results_options_sort_method_label': "Sortieren nach:", 'results_options_sort_duration': "Dauer", 'results_options_sort_magnitude': "Helligkeit",
                'moon_metric_label': "Mond Illum.", 'moon_warning_message': "Warnung: Mond >{:.0f}% (Schwelle: {:.0f}%)!", 'moon_phase_error': "Mond Fehler: {}",
                'moon_phase_not_available': "Mondphase N/A.", 'find_button_label': "🔭 Objekte finden", 'search_params_header': "Suchparameter",
                'search_params_location': "📍 Ort: {}", 'location_not_set': "Nicht gesetzt", 'observer_creation_failed': "Observer Fehler",
                'search_params_time': "⏱️ Zeit: {}", 'search_params_time_now': "Jetzt (ab {} {})", 'search_params_time_now_utc': "Jetzt (ab {} UTC)",
                'search_params_time_specific': "Nacht nach {}", 'search_params_filter_mag': "✨ Mag: {}", 'search_params_filter_mag_bortle': "Bortle {} (<= {:.1f})",
                'search_params_filter_mag_manual': "Manuell ({:.1f}-{:.1f})", 'search_params_filter_alt_types': "🔭 Alt {}-{}°, Typen: {}",
                'search_params_types_all': "Alle", 'search_params_filter_size': "📐 Größe {:.1f}-{:.1f}'", 'search_params_filter_direction': "🧭 Dir @ Max: {}",
                'search_params_direction_all': "Alle", 'spinner_searching': "Berechne...", 'spinner_geocoding': "Suche Ort...",
                'window_info_template': "Fenster: {} bis {}", 'window_already_passed': "Fenster vorbei.", 'warning_window_too_short': "Fenster zu kurz.",
                'error_no_window': "Kein Fenster.", 'error_cannot_search': "Suche nicht möglich.", 'error_polar_night': "Polarnacht?", 'error_polar_day': "Polartag?",
                'window_fallback_info': "\nFallback: {} bis {} UTC", 'window_fallback_info_short': "Fallback: {} bis {} UTC", 'success_objects_found': "{} Objekte gefunden.",
                'info_showing_list_duration': "Zeige {} (Dauer):", 'info_showing_list_magnitude': "Zeige {} (Helligkeit):", 'error_search_unexpected': "Suchfehler:",
                'error_search_no_catalog': "Katalog fehlt.", 'error_search_no_location': "Ort ungültig.", 'results_list_header': "Ergebnisse",
                'results_expander_title': "{} ({}) - Mag: {}", 'results_coords_header': "**Details:**", 'results_export_constellation': "Sternb.",
                'results_size_label': "Größe:", 'results_size_value': "{:.1f}'", 'results_max_alt_header': "**Max Alt:**", 'results_azimuth_label': "(Az: {:.1f}°{})",
                'azimuth_not_available': "(Az: N/A)", 'results_direction_label': ", Dir: {}", 'results_best_time_header': "**Beste Zeit (Lokal):**",
                'results_cont_duration_header': "**Dauer:**", 'results_duration_value': "{:.1f} Std", 'google_link_text': "Google", 'simbad_link_text': "SIMBAD",
                'graph_type_label': "Grafik:", 'graph_type_sky_path': "Himmelsbahn", 'graph_type_alt_time': "Höhenverlauf", 'results_graph_button': "📈 Plot",
                'results_spinner_plotting': "Plotte...", 'results_graph_error': "Plot Fehler: {}", 'results_graph_not_created': "Plot Fail.",
                'results_close_graph_button': "Plot schließen", 'results_save_csv_button': "💾 CSV Speichern", 'results_csv_filename': "dso_liste_{}.csv",
                'results_csv_export_error': "CSV Fehler: {}", 'warning_no_objects_found': "Keine Objekte gefunden.",
                'warning_no_objects_found_filters': "Keine Objekte (Filter).", 'warning_no_objects_found_after_search': "Keine Objekte (Kriterien).",
                'info_initial_prompt': "Koordinaten/Ort eingeben...", 'graph_min_altitude_label': "Min Alt ({:.0f}°)", 'graph_max_altitude_label': "Max Alt ({:.0f}°)",
                'graph_title_alt_time': "Höhenverlauf für {}", 'graph_title_sky_path': "Himmelsbahn für {}", 'graph_ylabel_altitude': "Höhe (°)",
                'graph_xlabel_time_utc': "Zeit (UTC)", 'graph_colorbar_label_time_utc': "Zeit (UTC)", 'custom_target_expander': "Eigenes Ziel plotten",
                'custom_target_ra_label': "RA:", 'custom_target_dec_label': "Dec:", 'custom_target_name_label': "Name (Opt):",
                'custom_target_name_placeholder': "Mein Komet", 'custom_target_default_name': "Eigenes Ziel", 'custom_target_ra_placeholder': "HH:MM:SS.s",
                'custom_target_dec_placeholder': "DD:MM:SS", 'custom_target_button': "Plot erstellen", 'custom_target_error_coords': "Ungültige RA/Dec.",
                'custom_target_error_window': "Fenster/Ort ungültig.", 'custom_target_error_invalid_window_order': "Ungültige Fensterreihenfolge.",
                'custom_target_error_window_short': "Fenster zu kurz.", 'custom_target_error_general': "Plot Fehler", 'error_processing_object': "Fehler bei Obj {}: {}",
                'window_calc_error': "Fenster Fehler: {}. {}", 'error_observer_type_invalid': "Ungültiger Observer.", 'error_twilight_calc_unexpected_polar': "Twilight Fehler Polar.",
                'error_twilight_calc_failed': "Twilight Fehler.", 'error_morning_before_evening_twilight': "Morgen < Abend Twilight.",
                'error_twilight_calc_next_night_failed': "Twilight nächste Nacht Fehler.", 'error_twilight_recalc_failed_fallback': "Twilight Fallback Fehler.",
                'error_no_window_final_fallback': "Kein Fenster, finaler Fallback.", 'error_internal_observer_location_type': "Interner Fehler: Observer Typ.",
                'error_internal_observing_times_type': "Interner Fehler: ObsTimes Typ.", 'error_internal_min_altitude_type': "Interner Fehler: MinAlt Typ.",
                'error_internal_catalog_df_type': "Interner Fehler: CatalogDF Typ.", 'warning_obs_window_too_few_points': "Fenster < 2 Punkte.",
                'unknown_type_placeholder': "Unbekannt", 'constellation_not_available': "N/A", 'plot_error_invalid_data_type': "Plot Fehler: Datentyp.",
                'plot_default_object_name': "Objekt", 'plot_error_invalid_times_data': "Plot Fehler: Zeiten.", 'plot_error_invalid_altitudes_data': "Plot Fehler: Höhen.",
                'plot_error_invalid_azimuths_data_sky_path': "Plot Fehler: Azimute.", 'plot_error_data_array_length_mismatch': "Plot Fehler: Längen.",
                'plot_info_no_data_to_plot': "Plot Info: Keine Daten.", 'plot_error_azimuths_missing_sky_path_final': "Plot Fehler: Azimute fehlen.",
                'plot_error_unknown_plot_type': "Plot Fehler: Unbekannter Typ.", 'plot_error_unexpected': "Plot Fehler: Unerwartet: {}",
                'error_loading_catalog_file_not_found': "Fehler: Katalogdatei nicht gefunden: {}", 'error_missing_catalog_columns': "Fehler: Katalog fehlende Spalten: {}",
                'error_no_usable_magnitude_column': "Fehler: Keine Magnitude Spalte: {}", 'warning_size_column_missing': "Warnung: Größen-Spalte '{}' fehlt.",
                'warning_size_column_no_valid_data': "Warnung: Größen-Spalte '{}' ohne Daten.", 'error_type_column_missing_critical': "Kritischer Fehler: 'Type' Spalte fehlt.",
                'warning_catalog_empty_after_filters': "Warnung: Katalog leer nach Filterung.", 'error_catalog_empty_data': "Fehler: Katalogdatei ist leer.",
                'error_loading_catalog_generic': "Katalog Ladefehler: {}", 'error_catalog_failed': "Katalogfehler.",
                'donation_text': "Gefällt der DSO Finder? [Unterstütze die Entwicklung!](https://ko-fi.com/skyobserver)", 
                'donation_url': "https://ko-fi.com/skyobserver", 'donation_button_text': "Unterstütze auf Ko-fi", 'bug_report_button': "🐞 Fehler melden",
                'bug_report_subject': "Fehlerbericht: Advanced DSO Finder",
                'bug_report_body': "\n\n(Fehlerbeschreibung)", 'recessional_velocity': "Fluchtgeschwindigkeit", 'unit_km_s': "km/s",
                'redshift_calculator_title': "Rotverschiebungsrechner", 'input_params': "Eingabeparameter", 'redshift_z': "Rotverschiebung (z)",
                'redshift_z_tooltip': "Kosmologische Rotverschiebung (negativ für Blauverschiebung)", 'cosmo_params': "Kosmologische Parameter",
                'hubble_h0': "H₀ [km/s/Mpc]", 'omega_m': "Ωm (Materiedichte)", 'omega_lambda': "ΩΛ (Dunkle Energie Dichte)",
                'non_flat_universe_info': "Hinweis: Ωm + ΩΛ = {sum_omega:.3f}. Dies impliziert ein nicht-flaches Universum (Ωk = {omega_k:.3f}).",
                'flat_universe_assumed': "Annahme eines flachen Universums (Ωk ≈ 0).", 'results_for': "Ergebnisse für z = {z:.5f}",
                'lookback_time': "Rückblickzeit", 'unit_Gyr': "Gyr", 'cosmo_distances': "Kosmologische Distanzen",
                'comoving_distance_title': "**Mitbewegte Distanz:**", 'unit_Mpc': "Mpc", 'unit_Gly': "Gly",
                'comoving_other_units_expander': "Weitere Einheiten (Mitbewegt)", 'unit_km_full': "km", 'unit_LJ': "Lj", 'unit_AE': "AE", 'unit_Ls': "Ls",
                'luminosity_distance_title': "**Leuchtkraftdistanz:**", 'explanation_luminosity': "Relevant für Helligkeit.",
                'angular_diameter_distance_title': "**Winkeldurchmesserdistanz:**", 'explanation_angular': "Relevant für Größe.",
                'calculation_note': "Flaches ΛCDM Modell.", 'error_invalid_input': "Ungültige Eingabe.", 'error_h0_positive': "H₀ muss positiv sein.",
                'error_omega_negative': "Ω Parameter negativ.", 'error_redshift_too_negative': "Rotverschiebung z={z} physikalisch nicht sinnvoll (< -1).",
                'warn_blueshift': "Blueshift (z < 0).", 'warn_integration_accuracy': "Integrationsgenauigkeit gering (Fehler dc: {err_dc}, lt: {err_lt}).",
                'error_dep_scipy': "Scipy benötigt.", 'error_calc_failed': "Berechnung fehlgeschlagen: {e}",
                'velocity_positive_caption': "Positiv: Entfernt sich", 'velocity_negative_caption': "Negativ: Nähert sich", 'velocity_zero_caption': "Keine Bewegung",
                'example_lookback_recent': "Gegenwart.", 'example_lookback_humans': "Moderne Menschen.", 'example_lookback_dinos_extinction': "Dino-Aussterben.",
                'example_lookback_multicellular': "Mehrzelliges Leben.", 'example_lookback_earth_formation': "Erdentstehung.",
                'example_lookback_early_universe_galaxies': "Frühe Galaxien.", 'example_lookback_very_early_universe': "Sehr frühes Universum.",
                'example_comoving_local_group': "Lokale Gruppe.", 'example_comoving_virgo_cluster': "Virgo-Haufen.", 'example_comoving_coma_cluster': "Coma-Haufen.",
                'example_comoving_laniakea_supercluster': "Laniakea Superhaufen.", 'example_comoving_large_scale_structure': "Großräumige Strukturen.",
                'example_comoving_distant_quasars': "Ferne Quasare.", 'example_comoving_observable_universe_horizon': "Beobachtbares Universum.",
                'thousands_separator': " " 
            },
            'en': { 
                "app_title": "Advanced DSO Finder (Internal EN)", 'settings_header': "Settings", 'language_select_label': "Language",
                'object_type_glossary_title': "Object Type Glossary", 'donation_text': "Like the DSO Finder? [Support its development!](https://ko-fi.com/skyobserver)",
                'donation_url': "https://ko-fi.com/skyobserver", 'donation_button_text': "Support on Ko-fi", 'bug_report_button': "🐞 Report Bug", 
                'bug_report_subject': "Bug Report: Advanced DSO Finder",
                'bug_report_body': "\n\n(Describe bug)", 'recessional_velocity': "Recessional Velocity",
                'unit_km_s': "km/s", 'redshift_calculator_title': "Redshift Calculator", 'input_params': "Input Parameters", 'redshift_z': "Redshift (z)",
                'redshift_z_tooltip': "Cosmological redshift (negative for blueshift)", 'cosmo_params': "Cosmological Parameters", 'hubble_h0': "H₀ [km/s/Mpc]",
                'omega_m': "Ωm (Matter Density)", 'omega_lambda': "ΩΛ (Dark Energy Density)", 
                'non_flat_universe_info': "Note: Ωm + ΩΛ = {sum_omega:.3f}. This implies a non-flat universe (Ωk = {omega_k:.3f}).",
                'flat_universe_assumed': "Assuming a flat universe (Ωk ≈ 0).", 'results_for': "Results for z = {z:.5f}", 'lookback_time': "Lookback Time",
                'unit_Gyr': "Gyr", 'cosmo_distances': "Cosmological Distances", 'comoving_distance_title': "**Comoving Distance:**", 'unit_Mpc': "Mpc",
                'unit_Gly': "Gly", 'comoving_other_units_expander': "Other Units (Comoving)", 'unit_km_full': "km", 'unit_LJ': "ly", 'unit_AE': "AU", 'unit_Ls': "Ls",
                'luminosity_distance_title': "**Luminosity Distance:**", 'explanation_luminosity': "Relevant for brightness.", 'angular_diameter_distance_title': "**Angular Diameter Distance:**",
                'explanation_angular': "Relevant for size.", 'calculation_note': "Flat ΛCDM model.", 'error_invalid_input': "Invalid input.",
                'error_h0_positive': "H₀ must be positive.", 'error_omega_negative': "Ω params negative.", 'error_redshift_too_negative': "Redshift z={z} physically implausible (< -1).",
                'warn_blueshift': "Blueshift (z < 0).", 'warn_integration_accuracy': "Integration accuracy low (err dc: {err_dc}, lt: {err_lt}).",
                'error_dep_scipy': "Scipy needed.", 'error_calc_failed': "Calc failed: {e}", 'velocity_positive_caption': "Positive: Receding",
                'velocity_negative_caption': "Negative: Approaching", 'velocity_zero_caption': "No motion", 'example_lookback_recent': "Present.",
                'example_lookback_humans': "Modern humans.", 'example_lookback_dinos_extinction': "Dino extinction.", 'example_lookback_multicellular': "Multicellular life.",
                'example_lookback_earth_formation': "Earth formation.", 'example_lookback_early_universe_galaxies': "Early galaxies.",
                'example_lookback_very_early_universe': "Very early universe.", 'example_comoving_local_group': "Local Group.", 'example_comoving_virgo_cluster': "Virgo Cluster.",
                'example_comoving_coma_cluster': "Coma Cluster.", 'example_comoving_laniakea_supercluster': "Laniakea Supercluster.",
                'example_comoving_large_scale_structure': "Large scale structure.", 'example_comoving_distant_quasars': "Distant quasars.",
                'example_comoving_observable_universe_horizon': "Observable universe.", 'thousands_separator': "," 
            },
            'fr': { 
                "app_title": "Advanced DSO Finder (Interne FR)", 'settings_header': "Paramètres", 'language_select_label': "Langue",
                'object_type_glossary_title': "Glossaire des types d'objets", 'donation_text': "Vous aimez le DSO Finder? [Soutenez le développement!](https://ko-fi.com/skyobserver)",
                'donation_url': "https://ko-fi.com/skyobserver", 'donation_button_text': "Soutenir sur Ko-fi", 'bug_report_button': "🐞 Signaler un bug", 
                'bug_report_subject': "Rapport de bug: Advanced DSO Finder",
                'bug_report_body': "\n\n(Décrire le bug)", 'recessional_velocity': "Vitesse de Récession",
                'unit_km_s': "km/s", 'thousands_separator': " " 
            }
        }
        class FallbackTranslationProvider:
            def __init__(self, lang_code, translations_map, default_fallback_lang='de'):
                lang_code_lower = lang_code.lower(); default_fallback_lang_lower = default_fallback_lang.lower()
                self.lang_dict = translations_map.get(lang_code_lower, translations_map.get(default_fallback_lang_lower, {}))
                self.default_lang_dict = translations_map.get(default_fallback_lang_lower, {})
            def get(self, key, default_value=None):
                val = self.lang_dict.get(key)
                if val is None and self.lang_dict is not self.default_lang_dict: val = self.default_lang_dict.get(key)
                if val is None: val = default_value if default_value is not None else key
                return val
        t = FallbackTranslationProvider(lang, _translations_fallback, default_fallback_lang='de')
        if not hasattr(t, 'get'):
            st.error("Critical error: FallbackTranslationProvider 't' not initialized.")
            class EmergencyDummyTranslator:
                def get(self, key, default_value=None): return default_value if default_value is not None else str(key)
            t = EmergencyDummyTranslator()
    if t is None: 
        st.error("CRITICAL: Translator 't' could not be initialized AT ALL.")
        class FinalEmergencyDummyTranslator:
            def get(self, key, default_value=None): return default_value if default_value is not None else str(key)
        t = FinalEmergencyDummyTranslator()

    st.title(t.get("app_title", "Advanced DSO Finder"))
    with st.expander(t.get('object_type_glossary_title', "Object Type Glossary")):
        glossary_items = t.get('object_type_glossary', {})
        if isinstance(glossary_items, dict) and glossary_items:
             col1, col2 = st.columns(2)
             try: sorted_items = sorted(glossary_items.items())
             except AttributeError: sorted_items = []; st.warning(t.get("glossary_format_error", "Glossary data is not in the expected format."))
             for i, (abbr, name) in enumerate(sorted_items): (col1 if i % 2 == 0 else col2).markdown(f"**{abbr}:** {name}")
        elif not glossary_items: st.info(t.get("glossary_not_available", "Glossary is not available or empty for the selected language."))
        elif isinstance(glossary_items, str): st.info(glossary_items)
    st.markdown("---")

    # --- Cached Data Loading ---
    # The translator object 't' is complex and can cause issues with st.cache_data if not handled.
    # We pass 'lang' to the cached function for cache invalidation on language change.
    # The translator object itself is passed with a leading underscore to tell Streamlit not to hash it.
    @st.cache_data
    def cached_load_ongc_data(path, current_lang_for_cache_key, _translator_obj_for_func):
        # The translator is used inside load_ongc_data for messages
        return load_ongc_data(path, current_lang_for_cache_key, _translator_obj_for_func)
    
    df_catalog_data = cached_load_ongc_data(CATALOG_FILEPATH, lang, t)


    with st.sidebar:
        st.header(t.get('settings_header', "Settings"))
        if 'catalog_status_msg' not in st.session_state: st.session_state.catalog_status_msg = ""
        current_catalog_msg = t.get('info_catalog_loaded', "Catalog: {} objects.").format(len(df_catalog_data)) if df_catalog_data is not None else t.get('error_catalog_failed', "Catalog loading failed.")
        msg_func = st.success if df_catalog_data is not None else st.error
        if st.session_state.catalog_status_msg != current_catalog_msg: msg_func(current_catalog_msg); st.session_state.catalog_status_msg = current_catalog_msg
        lang_options_map = {'de': 'Deutsch', 'en': 'English', 'fr': 'Français'}; available_lang_codes = list(lang_options_map.keys())
        try: current_lang_index = available_lang_codes.index(lang)
        except ValueError: current_lang_index = available_lang_codes.index('de')
        selected_lang_key = st.radio(t.get('language_select_label', "Language"), options=available_lang_codes, format_func=lambda lang_code: lang_options_map.get(lang_code, lang_code), key='language_radio_selector', index=current_lang_index, horizontal=True)
        if selected_lang_key != st.session_state.language:
            st.session_state.language = selected_lang_key; st.session_state.location_search_status_msg = ""; st.session_state.catalog_status_msg = ""; st.rerun()
        with st.expander(t.get('location_expander', "📍 Location"), expanded=True):
            loc_opts_map = {'Search': t.get('location_option_search', "Search by Name"), 'Manual': t.get('location_option_manual', "Enter Manually")}
            st.radio(t.get('location_select_label', "Location Method"), options=list(loc_opts_map.keys()), format_func=loc_opts_map.get, key="location_choice_key", horizontal=True)
            lat_val, lon_val, h_val, loc_is_valid_for_timezone_lookup, current_location_is_valid_for_run = None, None, None, False, False
            if st.session_state.location_choice_key == "Manual":
                st.number_input(t.get('location_lat_label', "Latitude (°N)"), -90.0, 90.0, step=0.01, format="%.4f", key="manual_lat_val")
                st.number_input(t.get('location_lon_label', "Longitude (°E)"), -180.0, 180.0, step=0.01, format="%.4f", key="manual_lon_val")
                st.number_input(t.get('location_elev_label', "Elevation (m)"), -500, 9000, step=10, format="%d", key="manual_height_val")
                lat_val, lon_val, h_val = st.session_state.manual_lat_val, st.session_state.manual_lon_val, st.session_state.manual_height_val
                if isinstance(lat_val, (int, float)) and isinstance(lon_val, (int, float)) and isinstance(h_val, (int, float)):
                    loc_is_valid_for_timezone_lookup, current_location_is_valid_for_run = True, True; st.session_state.location_is_valid_for_run = True
                    if st.session_state.location_search_success: st.session_state.location_search_success = False; st.session_state.searched_location_name = None; st.session_state.location_search_status_msg = ""
                else: st.warning(t.get('location_error_manual_none', "Manual location fields are invalid or empty.")); current_location_is_valid_for_run = False; st.session_state.location_is_valid_for_run = False
            elif st.session_state.location_choice_key == "Search":
                with st.form("location_search_form"):
                    st.text_input(t.get('location_search_label', "Location Name:"), key="location_search_query", placeholder=t.get('location_search_placeholder', "e.g., Mount Palomar Observatory"))
                    st.number_input(t.get('location_elev_label', "Elevation (m)"), -500, 9000, step=10, format="%d", key="manual_height_val")
                    search_form_submitted = st.form_submit_button(t.get('location_search_submit_button', "Find Coordinates"))
                status_message_placeholder = st.empty()
                if st.session_state.location_search_status_msg: (status_message_placeholder.success if st.session_state.location_search_success else status_message_placeholder.error)(st.session_state.location_search_status_msg)
                if search_form_submitted and st.session_state.location_search_query:
                    geocoded_location, geocoding_service_name, geocoding_error = None, None, None; search_query = st.session_state.location_search_query; user_agent_string = f"AdvancedDSOFinder/{random.randint(1000,9999)}"
                    with st.spinner(t.get('spinner_geocoding', "Searching for location...")):
                        geocoders_to_try = [("Nominatim", Nominatim(user_agent=user_agent_string, timeout=10)), ("ArcGIS", ArcGIS(timeout=15)), ("Photon", Photon(user_agent=user_agent_string, timeout=15))]
                        for service_name, geolocator in geocoders_to_try:
                            try:
                                print(f"Attempting geocoding with {service_name} for query: '{search_query}'"); geocoded_location = geolocator.geocode(search_query)
                                if geocoded_location: geocoding_service_name = service_name; break
                            except (GeocoderTimedOut, GeocoderServiceError) as e_geo: geocoding_error = e_geo; print(f"{service_name} failed: {e_geo}"); status_message_placeholder.info(t.get(f'location_search_info_fallback{"" if service_name == "Nominatim" else ("2" if service_name == "ArcGIS" else "")}', f"Trying next geocoder..."))
                            except Exception as e_gen: geocoding_error = e_gen; print(f"Generic error with {service_name}: {e_gen}"); break 
                        if geocoded_location and geocoding_service_name:
                            f_lat, f_lon, f_name = geocoded_location.latitude, geocoded_location.longitude, geocoded_location.address
                            st.session_state.update({'searched_location_name': f_name, 'location_search_success': True, 'manual_lat_val': f_lat, 'manual_lon_val': f_lon})
                            coord_str = t.get('location_search_coords', "Lat: {:.4f}, Lon: {:.4f}").format(f_lat, f_lon)
                            f_key = {'Nominatim': 'location_search_found', 'ArcGIS': 'location_search_found_fallback', 'Photon': 'location_search_found_fallback2'}.get(geocoding_service_name, 'location_search_found')
                            st.session_state.location_search_status_msg = f"{t.get(f_key, 'Found: {} ({})').format(f_name, geocoding_service_name)}\n({coord_str})"
                            status_message_placeholder.success(st.session_state.location_search_status_msg)
                            lat_val, lon_val, h_val = f_lat, f_lon, st.session_state.manual_height_val
                            loc_is_valid_for_timezone_lookup, current_location_is_valid_for_run = True, True; st.session_state.location_is_valid_for_run = True
                        else:
                            st.session_state.update({'location_search_success': False, 'searched_location_name': None})
                            if geocoding_error:
                                if isinstance(geocoding_error, GeocoderTimedOut): e_key_loc = 'location_search_error_timeout'; fmt_arg_loc = None
                                elif isinstance(geocoding_error, GeocoderServiceError): e_key_loc = 'location_search_error_service'; fmt_arg_loc = str(geocoding_error)
                                else: e_key_loc = 'location_search_error_fallback2_failed'; fmt_arg_loc = str(geocoding_error) 
                                st.session_state.location_search_status_msg = t.get(e_key_loc, "Geocoding Error: {}").format(fmt_arg_loc) if fmt_arg_loc else t.get(e_key_loc, "Geocoding Error")
                            else: st.session_state.location_search_status_msg = t.get('location_search_error_not_found', "Location not found.")
                            status_message_placeholder.error(st.session_state.location_search_status_msg)
                            current_location_is_valid_for_run = False; st.session_state.location_is_valid_for_run = False
                elif st.session_state.location_search_success:
                    lat_val, lon_val, h_val = st.session_state.manual_lat_val, st.session_state.manual_lon_val, st.session_state.manual_height_val
                    loc_is_valid_for_timezone_lookup, current_location_is_valid_for_run = True, True; st.session_state.location_is_valid_for_run = True
                    status_message_placeholder.success(st.session_state.location_search_status_msg)
                else: current_location_is_valid_for_run = False; st.session_state.location_is_valid_for_run = False
            st.markdown("---")
            tz_display_msg = ""
            if loc_is_valid_for_timezone_lookup and lat_val is not None and lon_val is not None and tf_instance:
                try: found_tz = tf_instance.timezone_at(lng=lon_val, lat=lat_val)
                except Exception as tz_lookup_e: print(f"TimezoneFinder error: {tz_lookup_e}"); found_tz = None
                if found_tz:
                    try: pytz.timezone(found_tz); st.session_state.selected_timezone = found_tz; tz_display_msg = f"{t.get('timezone_auto_set_label', 'Detected Timezone:')} **{found_tz}**"
                    except pytz.UnknownTimeZoneError: st.session_state.selected_timezone = 'UTC'; tz_display_msg = f"{t.get('timezone_auto_fail_label', 'Timezone:')} **UTC** ({t.get('timezone_error_invalid', 'Invalid')}: {found_tz})"
                else: st.session_state.selected_timezone = 'UTC'; tz_display_msg = f"{t.get('timezone_auto_fail_label', 'Timezone:')} **UTC** ({t.get('timezone_auto_fail_msg', 'Could not detect timezone, using UTC.')})"
            elif not tf_instance: tz_display_msg = f"{t.get('timezone_auto_fail_label', 'Timezone:')} **{INITIAL_TIMEZONE}** ({t.get('timezone_auto_na', 'Automatic TZ detection N/A')})"; st.session_state.selected_timezone = INITIAL_TIMEZONE
            else: tz_display_msg = f"{t.get('timezone_auto_fail_label', 'Timezone:')} **{st.session_state.selected_timezone}** ({t.get('timezone_loc_invalid', 'Location Invalid for TZ lookup')})"
            st.markdown(tz_display_msg, unsafe_allow_html=True)
        with st.expander(t.get('time_expander', "⏱️ Time & Timezone"), expanded=False): 
            time_opts_map = {'Now': t.get('time_option_now', "Now (Upcoming Night)"), 'Specific': t.get('time_option_specific', "Specific Night")}
            st.radio(t.get('time_select_label', "Select Time"), options=list(time_opts_map.keys()), format_func=time_opts_map.get, key="time_choice_exp", horizontal=True)
            if st.session_state.time_choice_exp == "Now": st.caption(f"Current UTC: {Time.now().iso}")
            else: st.date_input(t.get('time_date_select_label', "Select Date:"), value=st.session_state.selected_date_widget, key='selected_date_widget')
        with st.expander(t.get('filters_expander', "✨ Filters & Conditions"), expanded=False):
            st.markdown(t.get('mag_filter_header', "**Magnitude Filter**")); mag_opts_map = {'Bortle Scale': t.get('mag_filter_option_bortle', "Bortle Scale"), 'Manual': t.get('mag_filter_option_manual', "Manual Range")}
            st.radio(t.get('mag_filter_method_label', "Filter Method:"), options=list(mag_opts_map.keys()), format_func=mag_opts_map.get, key="mag_filter_mode_exp", horizontal=True)
            st.slider(t.get('mag_filter_bortle_label', "Bortle Scale:"), 1, 9, key='bortle_slider', help=t.get('mag_filter_bortle_help', "Sky darkness: 1=Excellent Dark, 9=Inner-city Sky"))
            if st.session_state.mag_filter_mode_exp == "Manual":
                st.slider(t.get('mag_filter_min_mag_label', "Min. Magnitude:"), -5.0, 20.0, step=0.5, format="%.1f", help=t.get('mag_filter_min_mag_help', "Brightest object magnitude to include"), key='manual_min_mag_slider')
                st.slider(t.get('mag_filter_max_mag_label', "Max. Magnitude:"), -5.0, 20.0, step=0.5, format="%.1f", help=t.get('mag_filter_max_mag_help', "Dimest object magnitude to include"), key='manual_max_mag_slider')
                if st.session_state.manual_min_mag_slider > st.session_state.manual_max_mag_slider: st.warning(t.get('mag_filter_warning_min_max', "Min. Magnitude is greater than Max. Magnitude!"))
            st.markdown("---"); st.markdown(t.get('min_alt_header', "**Object Altitude Above Horizon**"))
            current_min_alt, current_max_alt = st.session_state.min_alt_slider, st.session_state.max_alt_slider
            if current_min_alt > current_max_alt: st.session_state.min_alt_slider = current_max_alt; current_min_alt = current_max_alt
            st.slider(t.get('min_alt_label', "Min. Object Altitude (°):"), 0, 90, key='min_alt_slider', step=1); st.slider(t.get('max_alt_label', "Max. Object Altitude (°):"), 0, 90, key='max_alt_slider', step=1)
            if st.session_state.min_alt_slider > st.session_state.max_alt_slider: st.warning(t.get("alt_filter_warning_min_max", "Minimum altitude is greater than maximum altitude!"))
            st.markdown("---"); st.markdown(t.get('moon_warning_header', "**Moon Warning**")); st.slider(t.get('moon_warning_label', "Warn if Moon > (% Illumination):"), 0, 100, key='moon_phase_slider', step=5)
            st.markdown("---"); st.markdown(t.get('object_types_header', "**Object Types**")); catalog_all_types = []
            if df_catalog_data is not None and 'Type' in df_catalog_data.columns:
                try: catalog_all_types = sorted(list(df_catalog_data['Type'].dropna().astype(str).unique()))
                except Exception as e_types: st.warning(f"{t.get('object_types_error_extract', 'Could not extract object types from catalog.')}: {e_types}")
            if catalog_all_types:
                current_selected_types = [s_type for s_type in st.session_state.object_type_filter_exp if s_type in catalog_all_types]
                if current_selected_types != st.session_state.object_type_filter_exp: st.session_state.object_type_filter_exp = current_selected_types
                st.multiselect(t.get('object_types_label', "Filter Types (leave empty for all):"), options=catalog_all_types, default=current_selected_types, key="object_type_filter_exp")
            else: st.info(t.get("object_types_not_found", "No object types found in catalog.")); st.session_state.object_type_filter_exp = []
            st.markdown("---"); st.markdown(t.get('size_filter_header', "**Angular Size Filter**")); size_data_is_ok = df_catalog_data is not None and 'MajAx' in df_catalog_data.columns and df_catalog_data['MajAx'].notna().any(); size_slider_disabled = not size_data_is_ok
            if size_data_is_ok:
                try:
                    valid_sizes_arcmin = df_catalog_data['MajAx'].dropna(); min_slider_val = max(0.1, float(valid_sizes_arcmin.min())) if not valid_sizes_arcmin.empty else 0.1; max_slider_val = float(valid_sizes_arcmin.max()) if not valid_sizes_arcmin.empty else 120.0
                    current_min_size, current_max_size = st.session_state.size_arcmin_range; clamped_min_size = max(min_slider_val, min(current_min_size, max_slider_val)); clamped_max_size = min(max_slider_val, max(current_max_size, min_slider_val))
                    if clamped_min_size > clamped_max_size: clamped_min_size = clamped_max_size
                    if (clamped_min_size, clamped_max_size) != st.session_state.size_arcmin_range: st.session_state.size_arcmin_range = (clamped_min_size, clamped_max_size)
                    slider_step_val = 0.1 if max_slider_val <= 20 else (0.5 if max_slider_val <= 100 else 1.0)
                    st.slider(t.get('size_filter_label', "Object Size (arcminutes):"), min_slider_val, max_slider_val, step=slider_step_val, format="%.1f'", key='size_arcmin_range', help=t.get('size_filter_help', "Filter objects by their apparent size (major axis). 1 arcminute = 1/60 degree."), disabled=size_slider_disabled)
                except Exception as e_size_slider: st.error(f"{t.get('size_slider_error', 'Error creating size slider.')}: {e_size_slider}"); size_slider_disabled = True
            else: st.info(t.get("size_data_not_available", "Size data not available in catalog.")); size_slider_disabled = True
            if size_slider_disabled: st.slider(t.get('size_filter_label', "Object Size (arcminutes):"), 0.0, 1.0, (0.0, 1.0), key='size_disabled_placeholder', disabled=True)
            st.markdown("---"); st.markdown(t.get('direction_filter_header', "**Filter by Cardinal Direction**")); all_directions_str = t.get('direction_option_all', "All"); display_directions = [all_directions_str] + CARDINAL_DIRECTIONS; internal_directions = [ALL_DIRECTIONS_KEY] + CARDINAL_DIRECTIONS
            current_internal_direction = st.session_state.selected_peak_direction
            if current_internal_direction not in internal_directions: current_internal_direction = ALL_DIRECTIONS_KEY; st.session_state.selected_peak_direction = current_internal_direction
            try: current_direction_idx = internal_directions.index(current_internal_direction)
            except ValueError: current_direction_idx = 0
            selected_display_direction = st.selectbox(t.get('direction_filter_label', "Show objects culminating towards:"), options=display_directions, index=current_direction_idx, key='direction_selectbox')
            selected_internal_direction = ALL_DIRECTIONS_KEY
            if selected_display_direction != all_directions_str:
                try: selected_direction_idx = display_directions.index(selected_display_direction); selected_internal_direction = internal_directions[selected_direction_idx]
                except ValueError: selected_internal_direction = ALL_DIRECTIONS_KEY
            if selected_internal_direction != st.session_state.selected_peak_direction: st.session_state.selected_peak_direction = selected_internal_direction
        with st.expander(t.get('results_options_expander', "⚙️ Result Options"), expanded=False):
            max_slider_limit = len(df_catalog_data) if df_catalog_data is not None else 50; min_slider_limit=5; actual_max_limit=max(min_slider_limit, max_slider_limit); results_slider_disabled=actual_max_limit<=min_slider_limit
            default_num_objects = st.session_state.get('num_objects_slider', 20); clamped_default_num = max(min_slider_limit, min(default_num_objects, actual_max_limit))
            if clamped_default_num != default_num_objects: st.session_state.num_objects_slider = clamped_default_num
            st.slider(t.get('results_options_max_objects_label', "Max. Number of Objects to Display:"), min_slider_limit, actual_max_limit, step=1, key='num_objects_slider', disabled=results_slider_disabled)
            sort_options_map = {'Duration & Altitude': t.get('results_options_sort_duration', "Duration & Altitude"), 'Brightness': t.get('results_options_sort_magnitude', "Brightness")}
            st.radio(t.get('results_options_sort_method_label', "Sort Results By:"), options=list(sort_options_map.keys()), format_func=sort_options_map.get, key='sort_method', horizontal=True)
        
        st.sidebar.markdown("---")
        # Modernized Bug Report Button (Styled Markdown for mailto)
        bug_report_email = "debrun2005@gmail.com"
        bug_report_subject = urllib.parse.quote(t.get("bug_report_subject", "Bug Report: Advanced DSO Finder"))
        bug_report_body_template = urllib.parse.quote(t.get('bug_report_body', "\n\n(Please describe the bug and the steps to reproduce it)\n\nApp Version: X.Y.Z\nOS: ...\nBrowser: ...")) 
        bug_report_mailto_link = f"mailto:{bug_report_email}?subject={bug_report_subject}&body={bug_report_body_template}"
        st.sidebar.markdown(f"""
        <a href="{bug_report_mailto_link}" target="_blank" style="display: inline-block; padding: 0.5em 1em; background-color: #FF4B4B; color: white; text-align: center; border-radius: 0.25rem; text-decoration: none; font-weight: bold;">
        🐞 {t.get('bug_report_button', "Report Bug")}
        </a>""", unsafe_allow_html=True)

        # Modernized Donation Button (using st.link_button)
        donation_url = t.get("donation_url", "https://ko-fi.com/advanceddsofinder") 
        st.link_button(f"☕ {t.get('donation_button_text', 'Support on Ko-fi')}", donation_url, use_container_width=True)


    st.subheader(t.get('search_params_header', "Search Parameters"))
    param_col1, param_col2 = st.columns(2)
    location_display_str = t.get('location_error', "Location Error: {}").format(t.get('location_not_set', "Not Set")); observer_for_run = None
    if st.session_state.location_is_valid_for_run:
        lat, lon, h, tz_str = st.session_state.manual_lat_val, st.session_state.manual_lon_val, st.session_state.manual_height_val, st.session_state.selected_timezone
        try:
            observer_for_run = Observer(latitude=lat*u.deg, longitude=lon*u.deg, elevation=h*u.m, timezone=tz_str)
            if st.session_state.location_choice_key == "Manual": location_display_str = t.get('location_manual_display', "Manual ({:.4f}, {:.4f})").format(lat, lon)
            elif st.session_state.searched_location_name: location_display_str = t.get('location_search_display', "Searched: {} ({:.4f}, {:.4f})").format(st.session_state.searched_location_name, lat, lon)
            else: location_display_str = f"Lat: {lat:.4f}, Lon: {lon:.4f}" 
        except Exception as e_obs: location_display_str = t.get('location_error', "Location Error: {}").format(f"{t.get('observer_creation_failed', 'Observer creation failed')}: {e_obs}"); st.session_state.location_is_valid_for_run = False; observer_for_run = None
    param_col1.markdown(t.get('search_params_location', "📍 Location: {}").format(location_display_str))
    time_display_str = ""; is_now_mode_main = (st.session_state.time_choice_exp == "Now"); ref_time_for_run = Time.now() if is_now_mode_main else Time(datetime.combine(st.session_state.selected_date_widget, time(12,0)), scale='utc')
    if is_now_mode_main:
        try: local_now_str, local_tz_name_str = get_local_time_str(ref_time_for_run, st.session_state.selected_timezone, t); time_display_str = t.get('search_params_time_now', "Upcoming Night (from {} {})").format(local_now_str, local_tz_name_str)
        except Exception: time_display_str = t.get('search_params_time_now_utc', "Upcoming Night (from {} UTC)").format(f"{ref_time_for_run.to_datetime(timezone.utc):%Y-%m-%d %H:%M:%S}")
    else: selected_date_for_run = st.session_state.selected_date_widget; time_display_str = t.get('search_params_time_specific', "Night after {}").format(f"{selected_date_for_run:%Y-%m-%d}")
    param_col1.markdown(t.get('search_params_time', "⏱️ Time: {}").format(time_display_str))
    param_col1.markdown(f"{t.get('search_params_timezone', '🌍 Timezone:')} {st.session_state.selected_timezone}") 
    magnitude_display_str = ""; min_mag_filter, max_mag_filter = -np.inf, np.inf
    if st.session_state.mag_filter_mode_exp == "Bortle Scale": max_mag_filter = get_magnitude_limit(st.session_state.bortle_slider); magnitude_display_str = t.get('search_params_filter_mag_bortle', "Bortle {} (<= {:.1f} mag)").format(st.session_state.bortle_slider, max_mag_filter)
    else: min_mag_filter, max_mag_filter = st.session_state.manual_min_mag_slider, st.session_state.manual_max_mag_slider; magnitude_display_str = t.get('search_params_filter_mag_manual', "Manual ({:.1f}-{:.1f} mag)").format(min_mag_filter, max_mag_filter)
    param_col2.markdown(t.get('search_params_filter_mag', "✨ Filter: {}").format(magnitude_display_str))
    min_alt_display, max_alt_display = st.session_state.min_alt_slider, st.session_state.max_alt_slider; selected_types_display = st.session_state.object_type_filter_exp; types_str_display = ', '.join(selected_types_display) if selected_types_display else t.get('search_params_types_all', "All")
    param_col2.markdown(t.get('search_params_filter_alt_types', "🔭 Filter: Alt {}-{}°, Types: {}").format(min_alt_display, max_alt_display, types_str_display))
    size_min_display, size_max_display = st.session_state.size_arcmin_range; param_col2.markdown(t.get('search_params_filter_size', "📐 Filter: Size {:.1f} - {:.1f} arcmin").format(size_min_display, size_max_display))
    direction_display = st.session_state.selected_peak_direction; direction_display_str = t.get('search_params_direction_all', "All") if direction_display == ALL_DIRECTIONS_KEY else direction_display; param_col2.markdown(t.get('search_params_filter_direction', "🧭 Filter: Direction at Max: {}").format(direction_display_str))
    st.markdown("---")
    find_button_clicked = st.button(t.get('find_button_label', "🔭 Find Observable Objects"), key="find_button_main", use_container_width=True, type="primary", disabled=(df_catalog_data is None or not st.session_state.location_is_valid_for_run))
    if not st.session_state.location_is_valid_for_run and df_catalog_data is not None: st.warning(t.get('info_initial_prompt', "Welcome! Please **Enter Coordinates** or **Search Location** to enable object search."))
    results_display_placeholder = st.container()
    if find_button_clicked:
        st.session_state.find_button_pressed = True; st.session_state.update({'show_plot': False, 'show_custom_plot': False, 'active_result_plot_data': None, 'custom_target_plot_data': None, 'last_results': [], 'window_start_time': None, 'window_end_time': None})
        if observer_for_run and df_catalog_data is not None:
            with st.spinner(t.get('spinner_searching', "Calculating window & searching objects...")):
                try:
                    obs_window_start_time, obs_window_end_time, window_status_msg = get_observable_window(observer_for_run, ref_time_for_run, is_now_mode_main, t)
                    results_display_placeholder.info(window_status_msg)
                    st.session_state.window_start_time = obs_window_start_time; st.session_state.window_end_time = obs_window_end_time
                    if obs_window_start_time and obs_window_end_time and obs_window_start_time < obs_window_end_time:
                        observation_times_array = Time(np.arange(obs_window_start_time.jd, obs_window_end_time.jd, (5*u.min).to(u.day).value), format='jd', scale='utc')
                        if len(observation_times_array) < 2: results_display_placeholder.warning(t.get("warning_window_too_short", "Observation window too short for detailed calculation."))
                        filtered_catalog_df = df_catalog_data.copy(); filtered_catalog_df = filtered_catalog_df[(filtered_catalog_df['Mag'] >= min_mag_filter) & (filtered_catalog_df['Mag'] <= max_mag_filter)]
                        if selected_types_display: filtered_catalog_df = filtered_catalog_df[filtered_catalog_df['Type'].isin(selected_types_display)]
                        if size_data_is_ok: filtered_catalog_df = filtered_catalog_df.dropna(subset=['MajAx']); filtered_catalog_df = filtered_catalog_df[(filtered_catalog_df['MajAx'] >= size_min_display) & (filtered_catalog_df['MajAx'] <= size_max_display)]
                        if filtered_catalog_df.empty: results_display_placeholder.warning(t.get('warning_no_objects_found_filters', "No objects found with current filters (initial pass).")); st.session_state.last_results = []
                        else:
                            min_alt_for_search = st.session_state.min_alt_slider * u.deg
                            found_observable_objects = find_observable_objects(observer_for_run.location, observation_times_array, min_alt_for_search, filtered_catalog_df, t)
                            final_filtered_objects = []
                            selected_peak_dir_filter = st.session_state.selected_peak_direction; max_alt_filter_val = st.session_state.max_alt_slider
                            for obj_item in found_observable_objects:
                                if obj_item.get('Max Altitude (°)', -999) > max_alt_filter_val: continue
                                if selected_peak_dir_filter != ALL_DIRECTIONS_KEY and obj_item.get('Direction at Max') != selected_peak_dir_filter: continue
                                final_filtered_objects.append(obj_item)
                            sort_method_key = st.session_state.sort_method
                            if sort_method_key == 'Brightness': final_filtered_objects.sort(key=lambda x_obj: x_obj.get('Magnitude', float('inf')) if x_obj.get('Magnitude') is not None else float('inf'))
                            else: final_filtered_objects.sort(key=lambda x_obj: (x_obj.get('Max Cont. Duration (h)', 0), x_obj.get('Max Altitude (°)', 0)), reverse=True)
                            num_objects_to_show = st.session_state.num_objects_slider; st.session_state.last_results = final_filtered_objects[:num_objects_to_show]
                            if not final_filtered_objects: results_display_placeholder.warning(t.get('warning_no_objects_found', "No objects found matching all criteria for the calculated observation window."))
                            else: results_display_placeholder.success(t.get('success_objects_found', "{} matching objects found.").format(len(final_filtered_objects))); sort_info_msg_key = 'info_showing_list_duration' if sort_method_key != 'Brightness' else 'info_showing_list_magnitude'; results_display_placeholder.info(t.get(sort_info_msg_key, "Showing {} objects...").format(len(st.session_state.last_results)))
                    else: results_display_placeholder.error(t.get('error_no_window', "No valid astronomical darkness window found for the selected date and location.") + " " + t.get('error_cannot_search', "Cannot perform search.")); st.session_state.last_results = []
                except Exception as e_search: results_display_placeholder.error(t.get('error_search_unexpected', "An unexpected error occurred during the search:") + f"\n```\n{e_search}\n```"); traceback.print_exc(); st.session_state.last_results = []
        else:
             if df_catalog_data is None: results_display_placeholder.error(t.get("error_search_no_catalog", "Cannot search: Catalog missing."))
             if not observer_for_run: results_display_placeholder.error(t.get("error_search_no_location", "Cannot search: Location invalid."))
             st.session_state.last_results = []
    if st.session_state.last_results:
        results_data_list = st.session_state.last_results
        results_display_placeholder.subheader(t.get('results_list_header', "Result List"))
        window_start_time_state, window_end_time_state = st.session_state.get('window_start_time'), st.session_state.get('window_end_time'); observer_exists_for_moon = observer_for_run is not None
        if observer_exists_for_moon and isinstance(window_start_time_state, Time) and isinstance(window_end_time_state, Time):
            mid_observation_time = window_start_time_state + (window_end_time_state - window_start_time_state) / 2
            try: moon_illum_fraction = moon_illumination(mid_observation_time); moon_illum_percent = moon_illum_fraction*100; moon_phase_graphic_svg = create_moon_phase_svg(moon_illum_fraction, 50); moon_col1, moon_col2 = results_display_placeholder.columns([1,3])
            except Exception as e_moon: results_display_placeholder.warning(t.get('moon_phase_error', "Error calculating moon phase: {}").format(e_moon)); moon_illum_percent = -1; moon_phase_graphic_svg = None
            if moon_phase_graphic_svg: moon_col1.markdown(moon_phase_graphic_svg, unsafe_allow_html=True)
            if moon_illum_percent >= 0:
                 with moon_col2:
                    st.metric(label=t.get('moon_metric_label', "Moon Illumination (approx.)"), value=f"{moon_illum_percent:.0f}%")
                    moon_warning_threshold = st.session_state.moon_phase_slider
                    if moon_illum_percent > moon_warning_threshold: st.warning(t.get('moon_warning_message', "Warning: Moon is brighter ({:.0f}%) than threshold ({:.0f}%)!").format(moon_illum_percent, moon_warning_threshold))
        elif st.session_state.find_button_pressed: results_display_placeholder.info(t.get("moon_phase_not_available", "Moon phase information not available."))
        plot_options_map = {'Sky Path': t.get('graph_type_sky_path', "Sky Path (Az/Alt)"), 'Altitude Plot': t.get('graph_type_alt_time', "Altitude Plot (Alt/Time)")}
        results_display_placeholder.radio(t.get('graph_type_label', "Graph Type (for all plots):"), options=list(plot_options_map.keys()), format_func=plot_options_map.get, key='plot_type_selection_main', horizontal=True)
        for i_obj, object_result_data in enumerate(results_data_list):
            obj_name_res, obj_type_res = object_result_data.get('Name','N/A'), object_result_data.get('Type','N/A')
            obj_mag_res = object_result_data.get('Magnitude'); mag_str_res = f"{obj_mag_res:.1f}" if obj_mag_res is not None else "N/A"
            expander_title_str = t.get('results_expander_title', "{} ({}) - Mag: {}").format(obj_name_res, obj_type_res, mag_str_res)
            is_expanded_obj = (st.session_state.expanded_object_name == obj_name_res)
            object_container = results_display_placeholder.container()
            with object_container.expander(expander_title_str, expanded=is_expanded_obj):
                col_details, col_visibility, col_actions = st.columns([2,2,1])
                col_details.markdown(t.get('results_coords_header', "**Details:**")); col_details.markdown(f"**{t.get('results_export_constellation', 'Constellation')}:** {object_result_data.get('Constellation', 'N/A')}")
                obj_size_arcmin_res = object_result_data.get('Size (arcmin)'); col_details.markdown(f"**{t.get('results_size_label', 'Size (Major Axis):')}** {t.get('results_size_value', '{:.1f} arcmin').format(obj_size_arcmin_res) if obj_size_arcmin_res is not None else 'N/A'}")
                col_details.markdown(f"**RA:** {object_result_data.get('RA', 'N/A')}"); col_details.markdown(f"**Dec:** {object_result_data.get('Dec', 'N/A')}")
                col_visibility.markdown(t.get('results_max_alt_header', "**Max. Altitude:**"))
                max_alt_obj = object_result_data.get('Max Altitude (°)', 0); az_at_max_obj = object_result_data.get('Azimuth at Max (°)', 0); dir_at_max_obj = object_result_data.get('Direction at Max', 'N/A')
                az_formatted_str = t.get('results_azimuth_label', "(Azimuth: {:.1f}°{})").format(az_at_max_obj, "") if isinstance(az_at_max_obj, (int, float)) else t.get("azimuth_not_available", "(Azimuth: N/A)")
                dir_formatted_str = t.get('results_direction_label', ", Direction: {}").format(dir_at_max_obj) if dir_at_max_obj != "N/A" else ""
                col_visibility.markdown(f"**{max_alt_obj:.1f}°** {az_formatted_str}{dir_formatted_str}")
                col_visibility.markdown(t.get('results_best_time_header', "**Best Time (Local TZ):**"))
                peak_time_obj_utc = object_result_data.get('Time at Max (UTC)'); local_peak_time_str, local_peak_tz_name = get_local_time_str(peak_time_obj_utc, st.session_state.selected_timezone, t); col_visibility.markdown(f"{local_peak_time_str} ({local_peak_tz_name})")
                col_visibility.markdown(t.get('results_cont_duration_header', "**Max. Cont. Duration:**")); duration_obj_hours = object_result_data.get('Max Cont. Duration (h)', 0); col_visibility.markdown(t.get('results_duration_value', "{:.1f} hours").format(duration_obj_hours))
                google_query_str = urllib.parse.quote_plus(f"{obj_name_res} astronomy"); google_url_str = f"https://www.google.com/search?q={google_query_str}"; col_actions.markdown(f"[{t.get('google_link_text', 'Google')}]({google_url_str})", unsafe_allow_html=True)
                simbad_query_str = urllib.parse.quote_plus(obj_name_res); simbad_url_str = f"http://simbad.u-strasbg.fr/simbad/sim-basic?Ident={simbad_query_str}"; col_actions.markdown(f"[{t.get('simbad_link_text', 'SIMBAD')}]({simbad_url_str})", unsafe_allow_html=True)
                plot_button_key = f"plot_button_{obj_name_res}_{i_obj}"
                if st.button(t.get('results_graph_button', "📈 Show Plot"), key=plot_button_key):
                    st.session_state.update({'plot_object_name': obj_name_res, 'active_result_plot_data': object_result_data, 'show_plot': True, 'show_custom_plot': False, 'expanded_object_name': obj_name_res}); st.rerun()
                if st.session_state.show_plot and st.session_state.plot_object_name == obj_name_res:
                    plot_data_for_func = st.session_state.active_result_plot_data; min_alt_plot, max_alt_plot = st.session_state.min_alt_slider, st.session_state.max_alt_slider; st.markdown("---")
                    with st.spinner(t.get('results_spinner_plotting', "Creating plot...")):
                        try:
                            figure_object_plot = create_plot(plot_data_for_func, min_alt_plot, max_alt_plot, st.session_state.plot_type_selection_main, t)
                            if figure_object_plot:
                                st.pyplot(figure_object_plot); close_plot_button_key = f"close_plot_button_{obj_name_res}_{i_obj}"
                                if st.button(t.get('results_close_graph_button', "Close Plot"), key=close_plot_button_key): st.session_state.update({'show_plot': False, 'active_result_plot_data': None, 'expanded_object_name': None}); st.rerun()
                            else: st.error(t.get('results_graph_not_created', "Plot could not be created."))
                        except Exception as e_plot_render: st.error(t.get('results_graph_error', "Plot Error: {}").format(e_plot_render)); traceback.print_exc()
        if results_data_list:
            csv_download_placeholder = results_display_placeholder.empty()
            try:
                export_data_list = []; tz_for_csv = st.session_state.selected_timezone
                for obj_to_export in results_data_list:
                    peak_time_utc_export = obj_to_export.get('Time at Max (UTC)'); local_time_export_str, _ = get_local_time_str(peak_time_utc_export, tz_for_csv, t)
                    export_data_list.append({ t.get('results_export_name',"Name"): obj_to_export.get('Name'), t.get('results_export_type',"Type"): obj_to_export.get('Type'), t.get('results_export_constellation',"Constellation"): obj_to_export.get('Constellation'),
                        t.get('results_export_mag',"Magnitude"): obj_to_export.get('Magnitude'), t.get('results_export_size',"Size (arcmin)"): obj_to_export.get('Size (arcmin)'), t.get('results_export_ra',"RA"): obj_to_export.get('RA'),
                        t.get('results_export_dec',"Dec"): obj_to_export.get('Dec'), t.get('results_export_max_alt',"Max Altitude (°)") : obj_to_export.get('Max Altitude (°)', np.nan),
                        t.get('results_export_az_at_max',"Azimuth at Max (°)") : obj_to_export.get('Azimuth at Max (°)', np.nan), t.get('results_export_direction_at_max',"Direction at Max"): obj_to_export.get('Direction at Max'),
                        t.get('results_export_time_max_utc',"Time at Max (UTC)"): peak_time_utc_export.iso if peak_time_utc_export else 'N/A', t.get('results_export_time_max_local',"Time at Max (Local TZ)"): local_time_export_str,
                        t.get('results_export_cont_duration',"Max Cont Duration (h)") : obj_to_export.get('Max Cont. Duration (h)', np.nan) })
                df_export_csv = pd.DataFrame(export_data_list); decimal_char_csv = ',' if lang == 'de' else '.'; csv_string_data = df_export_csv.to_csv(index=False, sep=';', encoding='utf-8-sig', decimal=decimal_char_csv)
                current_datetime_str = datetime.now().strftime("%Y%m%d_%H%M"); csv_file_name_str = t.get('results_csv_filename', "dso_observation_list_{}.csv").format(current_datetime_str)
                csv_download_placeholder.download_button(label=t.get('results_save_csv_button', "💾 Save Result List as CSV"), data=csv_string_data, file_name=csv_file_name_str, mime='text/csv', key='csv_download_button')
            except Exception as e_csv_export: csv_download_placeholder.error(t.get('results_csv_export_error', "CSV Export Error: {}").format(e_csv_export))
    elif st.session_state.find_button_pressed: results_display_placeholder.info(t.get('warning_no_objects_found_after_search', "No objects found matching your criteria."))
    st.markdown("---")
    with st.expander(t.get('custom_target_expander', "Plot Custom Target")):
        with st.form("custom_target_form"):
             st.text_input(t.get('custom_target_ra_label', "Right Ascension (RA):"), key="custom_target_ra_input", placeholder=t.get('custom_target_ra_placeholder', "e.g., 10:45:03.6 or 161.265"))
             st.text_input(t.get('custom_target_dec_label', "Declination (Dec):"), key="custom_target_dec_input", placeholder=t.get('custom_target_dec_placeholder', "e.g., -16:42:58 or -16.716"))
             st.text_input(t.get('custom_target_name_label', "Target Name (Optional):"), key="custom_target_name_input", placeholder=t.get("custom_target_name_placeholder", "My Comet"))
             custom_target_submit_button = st.form_submit_button(t.get('custom_target_button', "Create Custom Plot"))
        custom_target_error_placeholder = st.empty(); custom_target_plot_placeholder = st.empty()
        if custom_target_submit_button:
             st.session_state.update({'show_plot': False, 'show_custom_plot': False, 'custom_target_plot_data': None, 'custom_target_error': ""})
             custom_ra_str, custom_dec_str = st.session_state.custom_target_ra_input, st.session_state.custom_target_dec_input; custom_target_display_name = st.session_state.custom_target_name_input or t.get('custom_target_default_name', "Custom Target")
             window_start_custom, window_end_custom = st.session_state.get('window_start_time'), st.session_state.get('window_end_time'); observer_exists_custom = observer_for_run is not None
             if not custom_ra_str or not custom_dec_str: st.session_state.custom_target_error = t.get('custom_target_error_coords', "Invalid RA/Dec format. Use HH:MM:SS.s / DD:MM:SS or decimal degrees."); custom_target_error_placeholder.error(st.session_state.custom_target_error)
             elif not observer_exists_custom or not isinstance(window_start_custom, Time) or not isinstance(window_end_custom, Time): st.session_state.custom_target_error = t.get('custom_target_error_window', "Cannot create plot. Ensure location and time window are valid (try clicking 'Find Observable Objects' first)."); custom_target_error_placeholder.error(st.session_state.custom_target_error)
             else:
                 try:
                     custom_skycoord = SkyCoord(ra=custom_ra_str, dec=custom_dec_str, unit=(u.hourangle, u.deg))
                     if window_start_custom < window_end_custom: custom_obs_times_array = Time(np.arange(window_start_custom.jd, window_end_custom.jd, (5*u.min).to(u.day).value), format='jd', scale='utc')
                     else: raise ValueError(t.get("custom_target_error_invalid_window_order", "Invalid window time order for custom target."))
                     if len(custom_obs_times_array) < 2: raise ValueError(t.get("custom_target_error_window_short", "Time window too short for custom plot."))
                     altaz_frame_custom = AltAz(obstime=custom_obs_times_array, location=observer_for_run.location); custom_target_altazs = custom_skycoord.transform_to(altaz_frame_custom)
                     st.session_state.custom_target_plot_data = {'Name': custom_target_display_name, 'altitudes': custom_target_altazs.alt.to(u.deg).value, 'azimuths': custom_target_altazs.az.to(u.deg).value, 'times': custom_obs_times_array}
                     st.session_state.show_custom_plot = True; st.session_state.custom_target_error = ""; st.rerun()
                 except ValueError as e_custom_coord: st.session_state.custom_target_error = f"{t.get('custom_target_error_coords', 'Invalid RA/Dec format. Use HH:MM:SS.s / DD:MM:SS or decimal degrees.')} ({e_custom_coord})"; custom_target_error_placeholder.error(st.session_state.custom_target_error)
                 except Exception as e_custom_plot: st.session_state.custom_target_error = f"{t.get('custom_target_error_general', 'General error plotting custom target')}: {e_custom_plot}"; custom_target_error_placeholder.error(st.session_state.custom_target_error); traceback.print_exc()
        if st.session_state.show_custom_plot and st.session_state.custom_target_plot_data:
            custom_plot_render_data = st.session_state.custom_target_plot_data; min_alt_custom_plot, max_alt_custom_plot = st.session_state.min_alt_slider, st.session_state.max_alt_slider
            with custom_target_plot_placeholder.container():
                 st.markdown("---");
                 with st.spinner(t.get('results_spinner_plotting', "Creating plot...")):
                     try:
                         figure_custom_plot = create_plot(custom_plot_render_data, min_alt_custom_plot, max_alt_custom_plot, st.session_state.plot_type_selection_main, t)
                         if figure_custom_plot:
                             st.pyplot(figure_custom_plot);
                             if st.button(t.get('results_close_graph_button', "Close Plot"), key="close_custom_plot_button"): st.session_state.update({'show_custom_plot': False, 'custom_target_plot_data': None}); st.rerun()
                         else: st.error(t.get('results_graph_not_created', "Plot could not be created."))
                     except Exception as e_custom_plot_render: st.error(t.get('results_graph_error', "Plot Error: {}").format(e_custom_plot_render)); traceback.print_exc()
        elif st.session_state.custom_target_error: custom_target_error_placeholder.error(st.session_state.custom_target_error)
    st.markdown("---")
    with st.expander(t.get("redshift_calculator_title", "Redshift Calculator"), expanded=False):
        st.subheader(t.get("input_params", "Input Parameters"))
        rc_z_input = st.number_input(label=t.get("redshift_z", "Redshift (z)"), min_value=-0.999, value=st.session_state.redshift_z_input, step=0.01, format="%.5f", key="redshift_z_input_widget", help=t.get("redshift_z_tooltip", "Cosmological redshift (negative for blueshift)."))
        st.subheader(t.get("cosmo_params", "Cosmological Parameters"))
        col_cosmo_params1, col_cosmo_params2, col_cosmo_params3 = st.columns(3)
        with col_cosmo_params1: rc_h0_input = st.number_input(label=t.get("hubble_h0", "Hubble Constant (H₀) [km/s/Mpc]"), min_value=1.0, value=st.session_state.redshift_h0_input, step=0.1, format="%.1f", key="redshift_h0_input_widget")
        with col_cosmo_params2: rc_om_input = st.number_input(label=t.get("omega_m", "Matter Density (Ωm)"), min_value=0.0, max_value=2.0, value=st.session_state.redshift_omega_m_input, step=0.001, format="%.3f", key="redshift_omega_m_input_widget")
        with col_cosmo_params3: rc_ol_input = st.number_input(label=t.get("omega_lambda", "Dark Energy Density (ΩΛ)"), min_value=0.0, max_value=2.0, value=st.session_state.redshift_omega_lambda_input, step=0.001, format="%.3f", key="redshift_omega_lambda_input_widget")
        omega_k_calc = 1.0 - rc_om_input - rc_ol_input
        if not math.isclose(omega_k_calc, 0.0, abs_tol=1e-3): st.info(t.get("non_flat_universe_info", "Note: Ωm + ΩΛ = {sum_omega:.3f}. This implies a non-flat universe (Ωk = {omega_k:.3f}). Calculations will use this geometry.").format(sum_omega=(rc_om_input + rc_ol_input), omega_k=omega_k_calc))
        else: st.caption(t.get("flat_universe_assumed", "Assuming a flat universe (Ωk ≈ 0)."))
        st.markdown("---"); st.subheader(t.get("results_for", "Results for z = {z:.5f}").format(z=rc_z_input))
        redshift_calc_results = calculate_lcdm_distances(rc_z_input, rc_h0_input, rc_om_input, rc_ol_input)
        error_key_redshift = redshift_calc_results.get('error_key')
        if error_key_redshift and error_key_redshift != "warn_blueshift":
            error_args_redshift = redshift_calc_results.get('error_args', {}); error_text_redshift = t.get(error_key_redshift, "Calculation Error: {}").format(**error_args_redshift); st.error(error_text_redshift)
        else:
            if error_key_redshift == "warn_blueshift": st.warning(t.get("warn_blueshift", "Warning: Redshift is negative (Blueshift). Cosmological distances are 0 or not directly applicable here."))
            warning_key_redshift = redshift_calc_results.get('warning_key')
            if warning_key_redshift: warning_args_redshift = redshift_calc_results.get('warning_args', {}); st.info(t.get(warning_key_redshift, "Calculation Warning: {}").format(**warning_args_redshift))
            velocity_km_s_res = redshift_calc_results.get('recessional_velocity_km_s', 0.0); lookback_gyr_res = redshift_calc_results.get('lookback_gyr', 0.0)
            comov_mpc_res = redshift_calc_results.get('comoving_mpc', 0.0); lum_mpc_res = redshift_calc_results.get('luminosity_mpc', 0.0); angd_mpc_res = redshift_calc_results.get('ang_diam_mpc', 0.0)
            comov_gly_res = convert_mpc_to_gly(comov_mpc_res); lum_gly_res = convert_mpc_to_gly(lum_mpc_res); angd_gly_res = convert_mpc_to_gly(angd_mpc_res)
            comov_km_res = convert_mpc_to_km(comov_mpc_res); comov_ly_res = convert_km_to_ly(comov_km_res); comov_au_res = convert_km_to_au(comov_km_res); comov_ls_res = convert_km_to_ls(comov_km_res)
            comov_km_formatted_res = format_large_number(comov_km_res, t)
            res_col_vel, res_col_lookback = st.columns(2)
            with res_col_vel:
                st.metric(label=t.get("recessional_velocity", "Recessional Velocity"), value=f"{velocity_km_s_res:,.1f}", delta=t.get("unit_km_s", "km/s"))
                if rc_z_input > 0: st.caption(t.get("velocity_positive_caption", "Positive: Object is receding (redshift)"))
                elif rc_z_input < 0: st.caption(t.get("velocity_negative_caption", "Negative: Object is approaching (blueshift)"))
                else: st.caption(t.get("velocity_zero_caption", "Zero: No significant cosmological relative motion"))
            with res_col_lookback:
                st.metric(label=t.get("lookback_time", "Lookback Time"), value=f"{lookback_gyr_res:.4f}", delta=t.get("unit_Gyr", "Gyr (Billion Years)"))
                lookback_example_key = get_lookback_comparison_key(lookback_gyr_res); st.caption(f"*{t.get(lookback_example_key, 'Contextual example for lookback time...')}*")
            st.markdown("---"); st.subheader(t.get("cosmo_distances", "Cosmological Distances"))
            dist_col_comov, dist_col_lum, dist_col_angd = st.columns(3)
            with dist_col_comov:
                st.markdown(t.get("comoving_distance_title", "**Comoving Distance:**"))
                st.text(f"  {comov_mpc_res:,.4f} {t.get('unit_Mpc', 'Mpc')}")
                st.text(f"  {comov_gly_res:,.4f} {t.get('unit_Gly', 'Gly (Billion Lightyears)')}")
                comoving_example_key = get_comoving_comparison_key(comov_mpc_res)
                st.caption(f"*{t.get(comoving_example_key, 'Contextual example for comoving distance...')}*")
                # Removed nested expander here
                st.markdown(f"_{t.get('comoving_other_units_expander', 'Other Units (Comoving)')}_:") # Label for the section
                st.text(f"  {comov_km_formatted_res} {t.get('unit_km_full', 'km')}")
                st.text(f"  {comov_ly_res:,.2e} {t.get('unit_LJ', 'ly')}")
                st.text(f"  {comov_au_res:,.2e} {t.get('unit_AE', 'AU')}")
                st.text(f"  {comov_ls_res:,.2e} {t.get('unit_Ls', 'Ls')}")
            with dist_col_lum:
                st.markdown(t.get("luminosity_distance_title", "**Luminosity Distance:**"))
                st.text(f"  {lum_mpc_res:,.4f} {t.get('unit_Mpc', 'Mpc')}")
                st.text(f"  {lum_gly_res:,.4f} {t.get('unit_Gly', 'Gly (Billion Lightyears)')}")
                st.caption(f"*{t.get('explanation_luminosity', 'Relevant for brightness: Objects appear as bright as expected at this distance (important for standard candles).')}*")
            with dist_col_angd:
                st.markdown(t.get("angular_diameter_distance_title", "**Angular Diameter Distance:**"))
                st.text(f"  {angd_mpc_res:,.4f} {t.get('unit_Mpc', 'Mpc')}")
                st.text(f"  {angd_gly_res:,.4f} {t.get('unit_Gly', 'Gly (Billion Lightyears)')}")
                st.caption(f"*{t.get('explanation_angular', 'Relevant for size: Objects have the expected apparent size at this distance (important for standard rulers).')}*")
            st.caption(t.get("calculation_note", "Calculation based on the ΛCDM model. For non-flat models, Ωk is derived from Ωm and ΩΛ."))
    st.markdown("---")
    st.caption(t.get('donation_text', "Like the DSO Finder? [Support the development on Ko-fi ☕](https://ko-fi.com/advanceddsofinder)"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
