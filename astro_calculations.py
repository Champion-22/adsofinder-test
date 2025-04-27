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

# --- Added missing imports ---
import traceback # For printing error details
import pandas as pd # For DataFrame type hint
from astroplan import Observer # For Observer type hint and calculations within get_observable_window
# --- End of added imports ---


# --- Constants ---
# Cardinal directions (keep if used elsewhere, e.g., main script logic)
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

# Corrected type hint for observer
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
        # If 'Now', check from previous noon to next noon.
        # If 'Specific Night', check from noon of selected date to noon of the next day.
        if is_now:
            search_start = ref_time - timedelta(hours=12) # Approximate previous noon
            search_end = ref_time + timedelta(hours=12)   # Approximate next noon
            time_range = Time([search_start, search_end])
            print(f"Calculating 'Now' window based on range: {search_start.iso} to {search_end.iso}")
        else:
            # ref_time is already noon UTC of the selected date
            search_start = ref_time
            search_end = ref_time + timedelta(days=1)
            time_range = Time([search_start, search_end])
            print(f"Calculating 'Specific Night' window based on range: {search_start.iso} to {search_end.iso}")

        # Find evening twilight end (sun setting below -18 deg)
        # Use 'next' to find the first time after search_start
        try:
            # Ensure observer object is valid before calling methods
            if not isinstance(observer, Observer):
                 raise TypeError("Invalid Observer object provided to get_observable_window.")
            evening_twilight_end = observer.sun_set_time(time_range[0], which='next', horizon=sun_alt_limit)
        except Exception as set_err: # Catch potential errors like sun never setting/rising
             print(f"Error finding sunset: {set_err}")
             evening_twilight_end = None

        # Find morning twilight start (sun rising above -18 deg)
        # Use 'next' starting from *after* the potential sunset time (or search_start if sunset failed)
        search_for_rise_start = evening_twilight_end if evening_twilight_end else time_range[0]
        try:
            if not isinstance(observer, Observer):
                 raise TypeError("Invalid Observer object provided to get_observable_window.")
            morning_twilight_start = observer.sun_rise_time(search_for_rise_start, which='next', horizon=sun_alt_limit)
        except Exception as rise_err:
            print(f"Error finding sunrise: {rise_err}")
            morning_twilight_start = None

        # --- Validate and determine the final window ---
        start_time_final = None
        end_time_final = None
        status_msg_key = 'error_no_window' # Default to error

        # Check observer validity again before calculating altaz
        if not isinstance(observer, Observer):
             raise TypeError("Invalid Observer object for altaz calculation.")

        if evening_twilight_end and morning_twilight_start:
            # Check if the found times fall within the expected range and order
            if evening_twilight_end < morning_twilight_start and morning_twilight_start <= time_range[1]:
                 start_time_final = evening_twilight_end
                 end_time_final = morning_twilight_start
                 status_msg_key = 'info_window_calculated'
            elif evening_twilight_end >= time_range[1]: # Sunset happens after end of search range (e.g., polar day)
                 status_msg_key = 'info_window_polar_day'
            elif morning_twilight_start <= time_range[0]: # Sunrise happens before start of search range (e.g., polar day recovery)
                 status_msg_key = 'info_window_polar_day' # Or a specific message?
            elif evening_twilight_end >= morning_twilight_start: # Found times are out of order (e.g., spanning across noon incorrectly)
                 status_msg_key = 'info_window_polar_night' # Likely polar night or issue near poles
                 # Try to find a valid window within the 24h period anyway for polar night
                 # This logic might need refinement for extreme latitudes
                 if observer.altaz(time_range[0], observer.sun).alt < sun_alt_limit:
                      start_time_final = time_range[0] # Start at beginning of range
                      end_time_final = time_range[1]   # End at end of range
                      status_msg_key = 'info_window_polar_night_full' # Indicate it's likely dark the whole time

        elif evening_twilight_end and not morning_twilight_start: # Found sunset but no sunrise in range (likely polar night starting/ongoing)
            if evening_twilight_end < time_range[1]: # Sunset is within the range
                 start_time_final = evening_twilight_end
                 end_time_final = time_range[1] # Window extends to end of search period
                 status_msg_key = 'info_window_polar_night_start'
            else: # Sunset is after the search range (polar day)
                 status_msg_key = 'info_window_polar_day'

        elif not evening_twilight_end and morning_twilight_start: # Found sunrise but no sunset (likely polar day ending/ongoing)
             if morning_twilight_start > time_range[0]: # Sunrise is within the range
                  start_time_final = time_range[0] # Window starts at beginning of search period
                  end_time_final = morning_twilight_start
                  status_msg_key = 'info_window_polar_day_end'
             else: # Sunrise is before the search range (polar night)
                  status_msg_key = 'info_window_polar_night' # Or full polar night? Check sun alt at start
                  if observer.altaz(time_range[0], observer.sun).alt < sun_alt_limit:
                       start_time_final = time_range[0]
                       end_time_final = time_range[1]
                       status_msg_key = 'info_window_polar_night_full'

        else: # Neither sunset nor sunrise found (likely deep polar day/night or error)
             sun_alt_start = observer.altaz(time_range[0], observer.sun).alt
             if sun_alt_start < sun_alt_limit: # Sun is down at the start -> Polar night
                  start_time_final = time_range[0]
                  end_time_final = time_range[1]
                  status_msg_key = 'info_window_polar_night_full'
             else: # Sun is up at the start -> Polar day
                  status_msg_key = 'info_window_polar_day'


        # Format the status message using the translation dict
        start_str = start_time_final.iso if start_time_final else "N/A"
        end_str = end_time_final.iso if end_time_final else "N/A"
        status_message = t.get(status_msg_key, "Window status unknown.").format(start_utc=start_str, end_utc=end_str)

        return start_time_final, end_time_final, status_message

    except Exception as e:
        print(f"Error in get_observable_window: {e}")
        # Use imported traceback
        traceback.print_exc()
        return None, None, f"{t.get('error_window_unexpected', 'Unexpected error calculating observation window:')} {e}"

# Corrected type hint for catalog_df
def find_observable_objects(location: EarthLocation, times: Time, min_altitude: u.Quantity, catalog_df: pd.DataFrame, t: dict) -> list[dict]:
    """
    Finds objects from the catalog that are observable above a minimum altitude during the specified times.

    Args:
        location: EarthLocation of the observer.
        times: Astropy Time array of observation times.
        min_altitude: Minimum altitude threshold (Astropy Quantity, e.g., 20*u.deg).
        catalog_df: Pandas DataFrame containing the object catalog (must include 'RA_deg', 'Dec_deg', 'Name', 'Type', 'Mag', 'MajAx', 'Constellation').
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
    # Add 'z' to required if cosmology calculation relies on it being present,
    # otherwise keep it optional with .get() later.
    # required_cols.append('z')
    if not all(col in catalog_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in catalog_df.columns]
        print(f"Error: Catalog DataFrame missing required columns: {missing}")
        # Optionally provide a default value or raise an error
        # For now, we'll try to proceed but might fail later if data is used
        # return results # Or return here if columns are essential

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
                # Find contiguous blocks of observable indices
                blocks = np.split(observable_indices, np.where(np.diff(observable_indices) != 1)[0] + 1)
                # Calculate time step accurately
                if len(times) > 1:
                    time_step_hours = (times[1] - times[0]).to(u.hour).value # Assuming constant time steps
                else:
                    time_step_hours = 0 # Cannot calculate duration with one time point

                for block in blocks:
                    if len(block) > 1:
                        # Duration is (number of steps) * time_step
                        # A block of length N has N-1 steps between points
                        duration = (len(block) - 1) * time_step_hours
                        max_cont_duration_hours = max(max_cont_duration_hours, duration)
                    elif len(block) == 1 and time_step_hours > 0:
                         # If only one point is observable, maybe assign half a time step?
                         # Or keep duration 0 as it's just one snapshot? Let's keep it 0 for now.
                         pass


            # --- Store Results ---
            # Use .get() with defaults for potentially missing columns
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
            # Use imported traceback
            traceback.print_exc()
            continue # Skip to the next object

    return results


def get_cardinal_direction(azimuth_deg: float) -> str:
    """Converts azimuth angle (degrees) to the nearest cardinal direction."""
    az = azimuth_deg % 360 # Normalize azimuth to 0-360
    # Ensure index calculation is robust for edge cases like 359 degrees rounding to 8
    index = round(az / 45.0) % 8 # Use float division
    return CARDINAL_DIRECTIONS[index]


# --- Cosmology Calculation Functions (from Redshift_Calculator.py) ---

def hubble_parameter_inv_integrand(z_prime, omega_m, omega_lambda):
  """
  Integrand 1 / E(z') for calculating comoving distance.
  E(z') = sqrt(omega_m * (1 + z')**3 + omega_k * (1 + z')**2 + omega_lambda)
  Assumes flat universe (omega_k = 0).
  """
  # Small epsilon to avoid division by zero or sqrt of negative if parameters are slightly off
  epsilon = 1e-15
  # Calculate the denominator E(z')
  denominator_sq = omega_m * (1 + z_prime)**3 + omega_lambda
  # Ensure denominator_sq is not negative due to potential floating point issues
  denominator_sq = max(denominator_sq, 0)
  denominator = np.sqrt(denominator_sq + epsilon)
  # Avoid division by zero
  if denominator < epsilon:
      # This should ideally not happen for standard cosmology if z_prime >= 0
      warnings.warn(f"Hubble parameter integrand denominator near zero at z'={z_prime}. Returning 0.")
      return 0.0
  return 1.0 / denominator

def lookback_time_integrand(z_prime, omega_m, omega_lambda):
  """
  Integrand 1 / [(1 + z') * E(z')] for calculating lookback time.
  E(z') is the same as in hubble_parameter_inv_integrand.
  """
  epsilon = 1e-15
  # Calculate E(z')^2
  term_in_sqrt = omega_m * (1 + z_prime)**3 + omega_lambda
  term_in_sqrt = max(term_in_sqrt, 0) # Ensure non-negative

  # Calculate the full denominator (1 + z') * E(z')
  denominator = (1 + z_prime) * np.sqrt(term_in_sqrt + epsilon)

  # Handle edge case z_prime = 0 separately to avoid potential issues if omega_m + omega_lambda is exactly 0 (though unlikely)
  # if math.isclose(z_prime, 0):
  #     denom_at_zero = np.sqrt(omega_m + omega_lambda + epsilon)
  #     if denom_at_zero < epsilon: return 0.0 # Avoid division by zero
  #     return 1.0 / denom_at_zero

  # Avoid division by zero for the general case
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
                           Omega_k (curvature) is assumed to be 0 (flat universe).

  Returns:
      A dictionary containing calculated values:
      - 'comoving_mpc': Comoving distance in Mpc.
      - 'luminosity_mpc': Luminosity distance in Mpc.
      - 'ang_diam_mpc': Angular diameter distance in Mpc.
      - 'lookback_gyr': Lookback time in Gyr.
      - 'error_msg': String key for error/warning message (or None if OK).
      - 'integration_warning_key': String key for integration warning (or None).
      - 'integration_warning_args': Dict with error values for warning message.
      Returns error message key if input is invalid or calculation fails.
  """
  # --- Input Validation ---
  if not isinstance(redshift, (int, float)) or \
     not isinstance(h0, (int, float)) or \
     not isinstance(omega_m, (int, float)) or \
     not isinstance(omega_lambda, (int, float)):
       # Return a key that can be translated by the UI
       return {'error_msg': "error_invalid_input"}

  if redshift < 0:
     # Handle blueshift: distances are 0, lookback time is 0.
     return {'comoving_mpc': 0.0, 'luminosity_mpc': 0.0, 'ang_diam_mpc': 0.0, 'lookback_gyr': 0.0, 'error_msg': "warn_blueshift", 'integration_warning_key': None, 'integration_warning_args': {}}
  if math.isclose(redshift, 0):
      # At z=0, all distances and lookback time are zero.
      return {'comoving_mpc': 0.0, 'luminosity_mpc': 0.0, 'ang_diam_mpc': 0.0, 'lookback_gyr': 0.0, 'error_msg': None, 'integration_warning_key': None, 'integration_warning_args': {}}
  if h0 <= 0:
      return {'error_msg': "error_h0_positive"}
  if omega_m < 0 or omega_lambda < 0:
      # While physically unlikely, prevent negative inputs.
      return {'error_msg': "error_omega_negative"}

  # Optional: Check for flatness if needed, though calculations here assume it.
  # if not math.isclose(omega_m + omega_lambda, 1.0, abs_tol=1e-3):
  #     # UI should handle displaying this warning based on inputs
  #     pass

  # --- Calculation ---
  dh = C_KM_PER_S / h0 # Hubble distance in Mpc

  try:
    # Calculate Comoving Distance (Radial)
    # Integrate 1 / E(z') from 0 to z
    # quad returns (result, absolute_error)
    integral_dc, err_dc = quad(hubble_parameter_inv_integrand, 0, redshift, args=(omega_m, omega_lambda), limit=100) # Increase limit for potentially difficult integrals
    comoving_distance_mpc = dh * integral_dc

    # Calculate Lookback Time
    # Integrate 1 / [(1 + z') * E(z')] from 0 to z
    # Hubble time in Gyr (approx 978 Gyr / (h0/100))
    # Use the more precise calculation: 1/H0 in seconds, convert Mpc to km, convert s to Gyr
    seconds_per_gyr = 365.25 * 24 * 3600 * GYR_PER_YR
    hubble_time_gyr = (KM_PER_MPC / h0) / seconds_per_gyr

    # Alternative Hubble time calculation: 977.8 / h0 is a common approximation in Gyr
    # hubble_time_gyr_approx = 977.8 / h0
    integral_lt, err_lt = quad(lookback_time_integrand, 0, redshift, args=(omega_m, omega_lambda), limit=100)
    lookback_time_gyr = hubble_time_gyr * integral_lt # Use the calculated Hubble time

    # Calculate Luminosity Distance
    luminosity_distance_mpc = comoving_distance_mpc * (1 + redshift)

    # Calculate Angular Diameter Distance
    angular_diameter_distance_mpc = comoving_distance_mpc / (1 + redshift)

    # Check integration accuracy
    warning_msg_key = None
    warning_msg_args = {}
    integration_warning_threshold = 1e-5 # Relative error threshold
    # Check relative error: error / result (avoid division by zero)
    rel_err_dc = abs(err_dc / integral_dc) if not math.isclose(integral_dc, 0) else 0
    rel_err_lt = abs(err_lt / integral_lt) if not math.isclose(integral_lt, 0) else 0

    if rel_err_dc > integration_warning_threshold or rel_err_lt > integration_warning_threshold:
       warning_msg_key = "warn_integration_accuracy"
       # Pass the absolute errors for display if needed
       warning_msg_args = {'err_dc': err_dc, 'err_lt': err_lt}

    # Return results dictionary
    return {
        'comoving_mpc': comoving_distance_mpc,
        'luminosity_mpc': luminosity_distance_mpc,
        'ang_diam_mpc': angular_diameter_distance_mpc,
        'lookback_gyr': lookback_time_gyr,
        'error_msg': None, # No calculation error
        'integration_warning_key': warning_msg_key,
        'integration_warning_args': warning_msg_args
    }

  except ImportError:
        # This shouldn't happen if scipy is installed, but good practice
        print("Error: SciPy (required for integration) not found.")
        return {'error_msg': "error_dep_scipy"}
  except Exception as e:
        # Catch any other unexpected errors during integration/calculation
        print(f"Error during cosmological calculation: {e}")
        # Use imported traceback
        traceback.print_exc()
        # Return a generic calculation error key and the exception details
        return {'error_msg': "error_calc_failed", 'error_args': {'e': str(e)}}


# --- Einheitenumrechnungsfunktionen (from Redshift_Calculator.py) ---
# These might be useful for displaying results in the UI later

def convert_mpc_to_km(d_mpc: float) -> float:
    """Converts Megaparsecs to kilometers."""
    return d_mpc * KM_PER_MPC

def convert_km_to_ly(d_km: float) -> float:
    """Converts kilometers to lightyears."""
    # Avoid division by zero if KM_PER_LY is somehow 0
    return 0.0 if d_km == 0 or KM_PER_LY == 0 else d_km / KM_PER_LY

def convert_mpc_to_gly(d_mpc: float) -> float:
    """Converts Megaparsecs to Gigalightyears."""
    if d_mpc == 0: return 0.0
    km_per_gly = KM_PER_LY * GYR_PER_YR # Kilometers per Gigalightyear
    # Avoid division by zero
    if km_per_gly == 0: return 0.0
    distance_km = convert_mpc_to_km(d_mpc)
    return distance_km / km_per_gly

# Add other conversions if needed by UI (AU, Ls, etc.)
# Example:
# KM_PER_AU = 1.495978707e+8 # Kilometers per Astronomical Unit
# def convert_km_to_au(d_km: float) -> float:
#    """Converts kilometers to Astronomical Units."""
#    return 0.0 if d_km == 0 or KM_PER_AU == 0 else d_km / KM_PER_AU

