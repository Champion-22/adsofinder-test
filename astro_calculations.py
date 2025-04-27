# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import math
from datetime import timedelta, time, datetime, timezone
import pytz # Keep for timezone validation if needed elsewhere

# Added imports needed for cosmology calculations
from scipy.integrate import quad
import warnings # To handle integration warnings

# Imports needed for functions previously in main
import traceback # For printing error details
import pandas as pd # For DataFrame type hint
from astroplan import Observer # For Observer type hint and calculations within get_observable_window


# --- Constants ---
# Cardinal directions
CARDINAL_DIRECTIONS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

# Constants for Cosmology Calculations (from Redshift_Calculator.py)
C_KM_PER_S = 299792.458  # Speed of light in km/s
KM_PER_MPC = 3.085677581491367e+19 # Kilometers per Megaparsec
KM_PER_LY = 9.4607304725808e+12   # Kilometers per Lightyear
GYR_PER_YR = 1e9 # Gigayears per Year

# Standard Cosmological Parameters (from Redshift_Calculator.py)
# These can be used as defaults if specific parameters aren't provided
H0_DEFAULT = 67.4  # Hubble constant in km/s/Mpc (Planck 2018)
OMEGA_M_DEFAULT = 0.315 # Matter density parameter (Planck 2018)
OMEGA_LAMBDA_DEFAULT = 0.685 # Dark energy density parameter (Planck 2018)
# Assuming flat universe (Omega_k = 0), so Omega_Lambda = 1 - Omega_M


# --- Observation Planning Functions ---

def get_magnitude_limit(bortle_scale: int) -> float:
    """Estimates the naked-eye limiting magnitude based on Bortle scale."""
    # Simple estimation, can be refined
    limits = {1: 7.8, 2: 7.3, 3: 6.8, 4: 6.3, 5: 6.0, 6: 5.7, 7: 5.3, 8: 4.8, 9: 4.3}
    return limits.get(bortle_scale, 5.0) # Default to 5.0 if scale is invalid

def get_observable_window(observer: Observer, ref_time: Time, is_now: bool, t: dict) -> tuple[Time | None, Time | None, str]:
    """
    Calculates the observable astronomical night window around a reference time.

    Args:
        observer: The astroplan Observer object.
        ref_time: The reference Astropy Time object (either Time.now() or noon UTC of selected date).
        is_now: Boolean indicating if the 'Now' option was selected.
        t: Translation dictionary.

    Returns:
        A tuple containing:
        - start_time: Astropy Time object for the start of the window (evening astronomical twilight end). None if no window.
        - end_time: Astropy Time object for the end of the window (morning astronomical twilight start). None if no window.
        - status_message: A string describing the calculated window.
    """
    try:
        # Define astronomical twilight limit (-18 degrees)
        sun_alt_limit = -18 * u.deg

        # Determine the time range to search for sunset/sunrise
        if is_now:
            search_start = ref_time - timedelta(hours=12) # Approximate previous noon
            search_end = ref_time + timedelta(hours=12)   # Approximate next noon
            time_range = Time([search_start, search_end])
            print(f"Calculating 'Now' window based on range: {search_start.iso} to {search_end.iso}")
        else:
            search_start = ref_time # ref_time is already noon UTC of the selected date
            search_end = ref_time + timedelta(days=1)
            time_range = Time([search_start, search_end])
            print(f"Calculating 'Specific Night' window based on range: {search_start.iso} to {search_end.iso}")

        # Find evening twilight end (sun setting below -18 deg)
        try:
            if not isinstance(observer, Observer): raise TypeError("Invalid Observer object provided.")
            evening_twilight_end = observer.sun_set_time(time_range[0], which='next', horizon=sun_alt_limit)
        except Exception as set_err: print(f"Error finding sunset: {set_err}"); evening_twilight_end = None

        # Find morning twilight start (sun rising above -18 deg)
        search_for_rise_start = evening_twilight_end if evening_twilight_end else time_range[0]
        try:
            if not isinstance(observer, Observer): raise TypeError("Invalid Observer object provided.")
            morning_twilight_start = observer.sun_rise_time(search_for_rise_start, which='next', horizon=sun_alt_limit)
        except Exception as rise_err: print(f"Error finding sunrise: {rise_err}"); morning_twilight_start = None

        # --- Validate and determine the final window ---
        start_time_final = None; end_time_final = None
        status_msg_key = 'error_no_window' # Default to error

        if not isinstance(observer, Observer): raise TypeError("Invalid Observer object for altaz calculation.")

        if evening_twilight_end and morning_twilight_start:
            if evening_twilight_end < morning_twilight_start and morning_twilight_start <= time_range[1]:
                 start_time_final = evening_twilight_end; end_time_final = morning_twilight_start
                 status_msg_key = 'info_window_calculated'
            elif evening_twilight_end >= time_range[1]: status_msg_key = 'info_window_polar_day'
            elif morning_twilight_start <= time_range[0]: status_msg_key = 'info_window_polar_day'
            elif evening_twilight_end >= morning_twilight_start:
                 status_msg_key = 'info_window_polar_night'
                 if observer.altaz(time_range[0], observer.sun).alt < sun_alt_limit:
                      start_time_final = time_range[0]; end_time_final = time_range[1]
                      status_msg_key = 'info_window_polar_night_full'
        elif evening_twilight_end and not morning_twilight_start:
            if evening_twilight_end < time_range[1]:
                 start_time_final = evening_twilight_end; end_time_final = time_range[1]
                 status_msg_key = 'info_window_polar_night_start'
            else: status_msg_key = 'info_window_polar_day'
        elif not evening_twilight_end and morning_twilight_start:
             if morning_twilight_start > time_range[0]:
                  start_time_final = time_range[0]; end_time_final = morning_twilight_start
                  status_msg_key = 'info_window_polar_day_end'
             else:
                  status_msg_key = 'info_window_polar_night'
                  if observer.altaz(time_range[0], observer.sun).alt < sun_alt_limit:
                       start_time_final = time_range[0]; end_time_final = time_range[1]
                       status_msg_key = 'info_window_polar_night_full'
        else:
             sun_alt_start = observer.altaz(time_range[0], observer.sun).alt
             if sun_alt_start < sun_alt_limit:
                  start_time_final = time_range[0]; end_time_final = time_range[1]
                  status_msg_key = 'info_window_polar_night_full'
             else: status_msg_key = 'info_window_polar_day'

        # Format the status message using the translation dict
        start_str = start_time_final.iso if start_time_final else "N/A"
        end_str = end_time_final.iso if end_time_final else "N/A"
        status_message = t.get(status_msg_key, "Window status unknown.").format(start_utc=start_str, end_utc=end_str)

        return start_time_final, end_time_final, status_message

    except Exception as e:
        print(f"Error in get_observable_window: {e}")
        traceback.print_exc()
        return None, None, f"{t.get('error_window_unexpected', 'Unexpected error calculating observation window:')} {e}"


def find_observable_objects(location: EarthLocation, times: Time, min_altitude: u.Quantity, catalog_df: pd.DataFrame, t: dict) -> list[dict]:
    """
    Finds objects from the catalog that are observable above a minimum altitude during the specified times.

    Args:
        location: EarthLocation of the observer.
        times: Astropy Time array of observation times.
        min_altitude: Minimum altitude threshold (Astropy Quantity, e.g., 20*u.deg).
        catalog_df: Pandas DataFrame containing the object catalog (must include 'RA_deg', 'Dec_deg', 'Name', 'Type', 'Mag', 'MajAx', 'Constellation', 'z').
        t: Translation dictionary.

    Returns:
        A list of dictionaries, where each dictionary represents an observable object
        and contains calculated information like max altitude, time at max, etc.
        Returns an empty list if no objects are observable or on error.
    """
    results = []
    if not isinstance(times, Time) or len(times) < 2:
        print("Error: Invalid times array provided to find_observable_objects.")
        return results # Need at least two time points

    if catalog_df is None or catalog_df.empty:
        print("Error: Empty or invalid catalog DataFrame provided.")
        return results

    # Ensure required columns exist
    required_cols = ['RA_deg', 'Dec_deg', 'Name', 'Type', 'Mag', 'MajAx', 'Constellation']
    # 'z' is optional, handled by .get() later
    if not all(col in catalog_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in catalog_df.columns]
        print(f"Error: Catalog DataFrame missing required columns: {missing}")
        return results # Cannot proceed without essential columns

    # Prepare AltAz frame for coordinate transformation
    altaz_frame = AltAz(obstime=times, location=location)

    # Iterate through each object in the catalog
    for index, obj in catalog_df.iterrows():
        try:
            # Get object coordinates (ensure RA/Dec are present and valid)
            ra = obj.get('RA_deg')
            dec = obj.get('Dec_deg')
            if ra is None or dec is None or not np.isfinite(ra) or not np.isfinite(dec):
                 print(f"Skipping object {obj.get('Name', index)} due to invalid coordinates.")
                 continue

            obj_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

            # Transform coordinates to AltAz for all times at once
            obj_altazs = obj_coord.transform_to(altaz_frame)
            altitudes = obj_altazs.alt # Altitude array
            azimuths = obj_altazs.az   # Azimuth array

            # Find times when the object is above the minimum altitude
            observable_mask = altitudes >= min_altitude
            observable_indices = np.where(observable_mask)[0]

            if not np.any(observable_mask):
                continue # Object never reaches minimum altitude

            # --- Calculate Max Altitude and related info ---
            max_alt_index = np.argmax(altitudes)
            max_altitude = altitudes[max_alt_index].to(u.deg).value
            time_at_max_alt = times[max_alt_index]
            azimuth_at_max_alt = azimuths[max_alt_index].to(u.deg).value
            direction_at_max = get_cardinal_direction(azimuth_at_max_alt)

            # --- Calculate Max Continuous Duration ---
            max_cont_duration_hours = 0.0
            if len(observable_indices) > 0:
                blocks = np.split(observable_indices, np.where(np.diff(observable_indices) != 1)[0] + 1)
                if len(times) > 1: time_step_hours = (times[1] - times[0]).to(u.hour).value
                else: time_step_hours = 0
                for block in blocks:
                    if len(block) > 1:
                        duration = (len(block) - 1) * time_step_hours
                        max_cont_duration_hours = max(max_cont_duration_hours, duration)

            # --- Store Results ---
            results.append({
                'Name': obj.get('Name', f'Obj_{index}'),
                'Type': obj.get('Type', 'Unknown'),
                'Constellation': obj.get('Constellation', '?'),
                'Magnitude': obj.get('Mag'), # Can be None
                'Size (arcmin)': obj.get('MajAx'), # Can be None
                'RA': obj_coord.ra.to_string(unit=u.hourangle, sep='hms', precision=1),
                'Dec': obj_coord.dec.to_string(unit=u.deg, sep='dms', precision=0),
                'RA_deg': ra, # Keep numeric values if needed
                'Dec_deg': dec,
                'Max Altitude (°)': round(max_altitude, 2),
                'Time at Max (UTC)': time_at_max_alt,
                'Azimuth at Max (°)': round(azimuth_at_max_alt, 2),
                'Direction at Max': direction_at_max,
                'Max Cont. Duration (h)': round(max_cont_duration_hours, 2),
                # Include raw data for plotting if needed by UI component
                'altitudes': altitudes.to(u.deg).value,
                'azimuths': azimuths.to(u.deg).value,
                'times': times,
                'z': obj.get('z') # Include redshift if present in catalog
            })

        except Exception as e:
            print(f"Error processing object {obj.get('Name', index)}: {e}")
            traceback.print_exc()
            continue # Skip to the next object

    return results


def get_cardinal_direction(azimuth_deg: float) -> str:
    """Converts azimuth angle (degrees) to the nearest cardinal direction."""
    az = azimuth_deg % 360 # Normalize azimuth to 0-360
    index = round(az / 45.0) % 8 # Use float division
    return CARDINAL_DIRECTIONS[index]


# --- Cosmology Calculation Functions (from Redshift_Calculator.py) ---

def hubble_parameter_inv_integrand(z_prime, omega_m, omega_lambda):
    """Integrand 1 / E(z') for calculating comoving distance."""
    epsilon = 1e-15
    denominator_sq = omega_m * (1 + z_prime)**3 + omega_lambda
    denominator_sq = max(denominator_sq, 0)
    denominator = np.sqrt(denominator_sq + epsilon)
    if denominator < epsilon:
        warnings.warn(f"Hubble parameter integrand denominator near zero at z'={z_prime}. Returning 0.")
        return 0.0
    return 1.0 / denominator

def lookback_time_integrand(z_prime, omega_m, omega_lambda):
    """Integrand 1 / [(1 + z') * E(z')] for calculating lookback time."""
    epsilon = 1e-15
    term_in_sqrt = omega_m * (1 + z_prime)**3 + omega_lambda
    term_in_sqrt = max(term_in_sqrt, 0)
    denominator = (1 + z_prime) * np.sqrt(term_in_sqrt + epsilon)
    if abs(denominator) < epsilon:
        warnings.warn(f"Lookback time integrand denominator near zero at z'={z_prime}. Returning 0.")
        return 0.0
    return 1.0 / denominator

# Consider adding @st.cache_data if this calculation is intensive and called repeatedly with same inputs
# However, it's likely called per object, so caching might not be effective unless parameters change rarely.
def calculate_lcdm_distances(redshift: float, h0: float = H0_DEFAULT, omega_m: float = OMEGA_M_DEFAULT, omega_lambda: float = OMEGA_LAMBDA_DEFAULT) -> dict:
    """
    Calculates cosmological distances and lookback time for a given redshift
    assuming a flat Lambda-CDM model.

    Args:
        redshift (float): The cosmological redshift (z).
        h0 (float): Hubble constant in km/s/Mpc. Defaults to H0_DEFAULT.
        omega_m (float): Matter density parameter. Defaults to OMEGA_M_DEFAULT.
        omega_lambda (float): Dark energy density parameter. Defaults to OMEGA_LAMBDA_DEFAULT.

    Returns:
        A dictionary containing calculated values and status keys.
    """
    # --- Input Validation ---
    if not isinstance(redshift, (int, float)) or \
       not isinstance(h0, (int, float)) or \
       not isinstance(omega_m, (int, float)) or \
       not isinstance(omega_lambda, (int, float)):
       return {'error_msg': "error_invalid_input"}
    if redshift < 0:
       return {'comoving_mpc': 0.0, 'luminosity_mpc': 0.0, 'ang_diam_mpc': 0.0, 'lookback_gyr': 0.0, 'error_msg': "warn_blueshift", 'integration_warning_key': None, 'integration_warning_args': {}}
    if math.isclose(redshift, 0):
        return {'comoving_mpc': 0.0, 'luminosity_mpc': 0.0, 'ang_diam_mpc': 0.0, 'lookback_gyr': 0.0, 'error_msg': None, 'integration_warning_key': None, 'integration_warning_args': {}}
    if h0 <= 0: return {'error_msg': "error_h0_positive"}
    if omega_m < 0 or omega_lambda < 0: return {'error_msg': "error_omega_negative"}

    # --- Calculation ---
    dh = C_KM_PER_S / h0 # Hubble distance in Mpc
    try:
        integral_dc, err_dc = quad(hubble_parameter_inv_integrand, 0, redshift, args=(omega_m, omega_lambda), limit=100)
        comoving_distance_mpc = dh * integral_dc

        # Hubble time in Gyr
        seconds_per_gyr = 365.25 * 24 * 3600 * GYR_PER_YR
        hubble_time_gyr = (KM_PER_MPC / h0) / seconds_per_gyr
        integral_lt, err_lt = quad(lookback_time_integrand, 0, redshift, args=(omega_m, omega_lambda), limit=100)
        lookback_time_gyr = hubble_time_gyr * integral_lt

        luminosity_distance_mpc = comoving_distance_mpc * (1 + redshift)
        angular_diameter_distance_mpc = comoving_distance_mpc / (1 + redshift)

        # Check integration accuracy
        warning_msg_key = None; warning_msg_args = {}
        integration_warning_threshold = 1e-5
        rel_err_dc = abs(err_dc / integral_dc) if not math.isclose(integral_dc, 0) else 0
        rel_err_lt = abs(err_lt / integral_lt) if not math.isclose(integral_lt, 0) else 0
        if rel_err_dc > integration_warning_threshold or rel_err_lt > integration_warning_threshold:
           warning_msg_key = "warn_integration_accuracy"
           warning_msg_args = {'err_dc': err_dc, 'err_lt': err_lt}

        return {
            'comoving_mpc': comoving_distance_mpc, 'luminosity_mpc': luminosity_distance_mpc,
            'ang_diam_mpc': angular_diameter_distance_mpc, 'lookback_gyr': lookback_time_gyr,
            'error_msg': None, 'integration_warning_key': warning_msg_key,
            'integration_warning_args': warning_msg_args
        }
    except ImportError:
        print("Error: SciPy (required for integration) not found.")
        return {'error_msg': "error_dep_scipy"}
    except Exception as e:
        print(f"Error during cosmological calculation: {e}")
        traceback.print_exc()
        return {'error_msg': "error_calc_failed", 'error_args': {'e': str(e)}}


# --- Einheitenumrechnungsfunktionen (from Redshift_Calculator.py) ---
# Diese sind jetzt hier, da sie rein mathematisch sind und evtl. auch von der UI benötigt werden

def convert_mpc_to_km(d_mpc: float) -> float:
    """Converts Megaparsecs to kilometers."""
    return d_mpc * KM_PER_MPC

def convert_km_to_ly(d_km: float) -> float:
    """Converts kilometers to lightyears."""
    return 0.0 if d_km == 0 or KM_PER_LY == 0 else d_km / KM_PER_LY

def convert_mpc_to_gly(d_mpc: float) -> float:
    """Converts Megaparsecs to Gigalightyears."""
    if d_mpc == 0: return 0.0
    km_per_gly = KM_PER_LY * GYR_PER_YR
    if km_per_gly == 0: return 0.0
    distance_km = convert_mpc_to_km(d_mpc)
    return distance_km / km_per_gly

# Add other conversions if needed (AU, Ls, etc.)
KM_PER_AU = 1.495978707e+8 # Kilometers per Astronomical Unit
KM_PER_LS = C_KM_PER_S # Kilometers per Lightsecond

def convert_km_to_au(d_km: float) -> float:
   """Converts kilometers to Astronomical Units."""
   return 0.0 if d_km == 0 or KM_PER_AU == 0 else d_km / KM_PER_AU

def convert_km_to_ls(d_km: float) -> float:
   """Converts kilometers to Lightseconds."""
   return 0.0 if d_km == 0 or KM_PER_LS == 0 else d_km / KM_PER_LS

# Entferne UI-Hilfsfunktionen wie format_large_number, get_lookback_comparison etc.
