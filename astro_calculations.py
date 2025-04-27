# -*- coding: utf-8 -*-
# astro_calculations.py

# --- Basic Imports ---
from __future__ import annotations
import math
import traceback
from datetime import datetime, time, timedelta, timezone

# --- Library Imports ---
# Import only what's needed by the functions in *this* module
import pandas as pd # <--- HINZUGEFÜGT: Fehlender Pandas-Import
try:
    from astropy.time import Time
    import numpy as np
    import astropy.units as u
    from astropy.coordinates import EarthLocation, SkyCoord, AltAz, get_constellation
    from astroplan import Observer
except ImportError as e:
    # Handle missing libraries specific to this module if necessary
    # Or rely on the main script to catch this early
    print(f"Error: Missing astro libraries in astro_calculations.py. ({e})")
    raise # Re-raise the error to stop execution if critical

# --- Constants used in this module ---
# TODO: Consider moving these to a central config.py module later
CARDINAL_DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# --- Helper Functions ---
def get_magnitude_limit(bortle_scale: int) -> float:
    """Calculates the approximate limiting magnitude based on Bortle scale."""
    # These limits are approximate and can vary.
    limits = {1: 15.5, 2: 15.5, 3: 14.5, 4: 14.5, 5: 13.5, 6: 12.5, 7: 11.5, 8: 10.5, 9: 9.5}
    return limits.get(bortle_scale, 9.5) # Default to Bortle 9 if scale is invalid

def azimuth_to_direction(azimuth_deg: float) -> str:
    """Converts an azimuth angle (degrees) to a cardinal direction string."""
    if math.isnan(azimuth_deg):
        return "N/A" # Handle potential NaN input
    azimuth_deg = azimuth_deg % 360 # Normalize to 0-360
    # Calculate the index in the CARDINAL_DIRECTIONS list
    # Each direction covers 45 degrees (360 / 8 = 45)
    # We add 22.5 degrees (half of 45) to center the bins, then divide by 45
    index = round((azimuth_deg + 22.5) / 45) % 8
    # Ensure index stays within bounds (although % 8 should handle it)
    index = max(0, min(index, len(CARDINAL_DIRECTIONS) - 1))
    return CARDINAL_DIRECTIONS[index]

# --- Fallback Window ---
def _get_fallback_window(reference_time: Time) -> tuple[Time, Time]:
    """
    Provides a simple fallback observation window (e.g., 6 PM to 6 AM UTC).
    """
    # Get the date part of the reference time
    ref_dt_utc = reference_time.to_datetime(timezone.utc)
    ref_date = ref_dt_utc.date()

    # Define fallback start and end times (e.g., 18:00 UTC to 06:00 UTC next day)
    fallback_start_dt = datetime.combine(ref_date, time(18, 0), tzinfo=timezone.utc)
    fallback_end_dt = datetime.combine(ref_date + timedelta(days=1), time(6, 0), tzinfo=timezone.utc)

    # Convert back to Astropy Time objects
    fallback_start_time = Time(fallback_start_dt, scale='utc')
    fallback_end_time = Time(fallback_end_dt, scale='utc')

    print(f"Using fallback window: {fallback_start_time.iso} to {fallback_end_time.iso}")
    return fallback_start_time, fallback_end_time

# --- Observation Window Calculation ---
def get_observable_window(observer: Observer, reference_time: Time, is_now: bool, t: dict) -> tuple[Time | None, Time | None, str]:
    """
    Calculates the astronomical darkness window for observation.

    Args:
        observer: The astroplan Observer object.
        reference_time: The reference time for calculation (Time object).
        is_now: Boolean indicating if "Now" was selected (affects window start).
        t: The translation dictionary for the current language.

    Returns:
        A tuple containing:
            - start_time: Astropy Time object for window start (or None).
            - end_time: Astropy Time object for window end (or None).
            - status_message: String describing the window or errors.
    """
    status_message = ""
    start_time, end_time = None, None # Initialize here
    current_utc_time = Time.now() # Get current UTC time

    # Determine the calculation base time (noon UTC of the target night)
    calc_base_time = reference_time
    if is_now:
        current_dt_utc = current_utc_time.to_datetime(timezone.utc)
        noon_today_utc = datetime.combine(current_dt_utc.date(), time(12, 0), tzinfo=timezone.utc)
        # Base calculation on previous noon if it's before noon today
        if current_dt_utc < noon_today_utc:
            calc_base_time = Time(noon_today_utc - timedelta(days=1))
        else:
            calc_base_time = Time(noon_today_utc)
        print(f"Calculating 'Now' window based on UTC noon: {calc_base_time.iso}")
    else:
        # For specific date, use noon UTC of that date
        selected_date_noon_utc = datetime.combine(reference_time.to_datetime(timezone.utc).date(), time(12, 0), tzinfo=timezone.utc)
        calc_base_time = Time(selected_date_noon_utc, scale='utc')
        print(f"Calculating specific night window based on UTC noon: {calc_base_time.iso}")


    try:
        # Validate observer input type
        if not isinstance(observer, Observer):
            raise TypeError(f"Internal Error: Expected astroplan.Observer, got {type(observer)}")

        # Calculate astronomical twilight times
        astro_set = observer.twilight_evening_astronomical(calc_base_time, which='next')
        astro_rise = observer.twilight_morning_astronomical(astro_set if astro_set else calc_base_time, which='next')

        # Check if twilight times are valid
        if astro_set is None or astro_rise is None:
            raise ValueError("Could not determine one or both astronomical twilight times.")
        if astro_rise <= astro_set:
            # This case can happen near poles or if twilight persists > 24h
            # Check if it's likely polar night/day before raising generic error
            try:
                # Check sun altitude at reference time and 12 hours later
                sun_alt_ref = observer.sun_altaz(calc_base_time).alt
                sun_alt_12h_later = observer.sun_altaz(calc_base_time + 12*u.hour).alt
                if sun_alt_ref < -18*u.deg and sun_alt_12h_later < -18*u.deg:
                    # Likely polar night
                    status_message = t.get('error_polar_night', "Astronomical darkness lasts >24h (Polar night?). Using fallback window.")
                    start_time, end_time = _get_fallback_window(calc_base_time)
                    status_message += t.get('window_fallback_info', "\nUsing fallback window: {} to {} UTC").format(start_time.iso, end_time.iso)
                    return start_time, end_time, status_message
                elif sun_alt_ref > -18*u.deg: # Check polar day more thoroughly
                    # Check sun altitude over a 24-hour period
                    times_check = calc_base_time + np.linspace(0, 24, 49)*u.hour
                    sun_alts_check = observer.sun_altaz(times_check).alt
                    if np.min(sun_alts_check) > -18*u.deg:
                        # Likely polar day
                        status_message = t.get('error_polar_day', "No astronomical darkness occurs (Polar day?). Using fallback window.")
                        start_time, end_time = _get_fallback_window(calc_base_time)
                        status_message += t.get('window_fallback_info', "\nUsing fallback window: {} to {} UTC").format(start_time.iso, end_time.iso)
                        return start_time, end_time, status_message
            except Exception as check_e:
                print(f"Error during polar check: {check_e}")
            # If not clearly polar night/day, raise the original error
            raise ValueError("Calculated morning twilight is not after evening twilight.")


        start_time = astro_set
        end_time = astro_rise

        # Adjust window if 'Now' is selected and current time is within or after the calculated window
        if is_now:
            if end_time < current_utc_time:
                # Window has already passed, calculate for the next night
                status_message = t.get('window_already_passed', "Calculated night window for 'Now' has already passed. Calculating for next night.") + "\n"
                next_noon_utc = datetime.combine(current_utc_time.to_datetime(timezone.utc).date() + timedelta(days=1), time(12, 0), tzinfo=timezone.utc)
                astro_set_next = observer.twilight_evening_astronomical(Time(next_noon_utc), which='next')
                astro_rise_next = observer.twilight_morning_astronomical(astro_set_next if astro_set_next else Time(next_noon_utc), which='next')

                if astro_set_next is None or astro_rise_next is None or astro_rise_next <= astro_set_next:
                    raise ValueError("Could not determine valid twilight times for the *next* night.")

                start_time = astro_set_next
                end_time = astro_rise_next

            elif start_time < current_utc_time:
                # Window is ongoing, adjust start time to now
                print(f"Adjusting window start from {start_time.iso} to current time {current_utc_time.iso}")
                start_time = current_utc_time

        # Format window times for display
        start_fmt = start_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M %Z')
        end_fmt = end_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M %Z')
        status_message += t.get('window_info_template', "Observation window: {} to {} UTC (Astronomical Twilight)").format(start_fmt, end_fmt)

    except ValueError as ve:
        # Handle specific ValueErrors from astroplan (e.g., twilight calculation issues)
        error_detail = f"{ve}"
        print(f"Astroplan ValueError calculating window: {error_detail}")
        # Check for polar conditions if not already handled
        if 'polar' not in status_message: # Avoid double messages if already caught above
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
                    else: # Normal error if twilight calculation failed but it's not polar day/night
                        status_message = t.get('window_calc_error', "Error calculating observation window: {}\n{}").format(error_detail, " (Check location/time)")
                else: # Normal error if twilight calculation failed
                    status_message = t.get('window_calc_error', "Error calculating observation window: {}\n{}").format(error_detail, traceback.format_exc())
            except Exception as check_e:
                print(f"Error checking sun altitude for polar conditions: {check_e}")
                status_message = t.get('window_calc_error', "Error calculating observation window: {}\n{}").format(error_detail, traceback.format_exc())
        # Apply fallback window
        start_time, end_time = _get_fallback_window(calc_base_time)
        status_message += t.get('window_fallback_info', "\nUsing fallback window: {} to {} UTC").format(start_time.iso, end_time.iso)


    except Exception as e:
        # Handle unexpected errors during window calculation
        status_message = t.get('window_calc_error', "Error calculating observation window: {}\n{}").format(e, traceback.format_exc())
        print(f"Unexpected error calculating window: {e}")
        start_time, end_time = _get_fallback_window(calc_base_time)
        status_message += t.get('window_fallback_info', "\nUsing fallback window: {} to {} UTC").format(start_time.iso, end_time.iso)

    # Final check and fallback if times are still invalid
    if start_time is None or end_time is None or end_time <= start_time:
        if not status_message or "Error" not in status_message and "Fallback" not in status_message:
             status_message += ("\n" if status_message else "") + t.get('error_no_window', "No valid astronomical darkness window found for the selected date and location.")

        start_time_fb, end_time_fb = _get_fallback_window(calc_base_time)
        # Only add fallback info if it wasn't already added
        if t.get('window_fallback_info', "\nUsing fallback window: {} to {} UTC").format(start_time_fb.iso, end_time_fb.iso) not in status_message:
             status_message += t.get('window_fallback_info', "\nUsing fallback window: {} to {} UTC").format(start_time_fb.iso, end_time_fb.iso)
        start_time, end_time = start_time_fb, end_time_fb


    return start_time, end_time, status_message


# --- Object Finding Logic ---
def find_observable_objects(observer_location: EarthLocation,
                            observing_times: Time,
                            min_altitude_limit: u.Quantity,
                            catalog_df: pd.DataFrame,
                            t: dict) -> list[dict]:
    """
    Finds Deep Sky Objects from the catalog that are observable
    above a minimum altitude for the given observer and times.
    Note: Max altitude filtering is done *after* this function in main().

    Args:
        observer_location: The observer's location (EarthLocation).
        observing_times: Times at which to check object visibility (Time array).
        min_altitude_limit: Minimum altitude for an object to be considered observable.
        catalog_df: DataFrame containing the DSO catalog data.
        t: The translation dictionary for the current language.

    Returns:
        A list of dictionaries, where each dictionary represents an observable DSO.
        Returns empty list if no objects are found or errors occur.
    """
    observable_objects = []

    # --- Input Validation ---
    # (Removed Streamlit UI calls like st.error from here, rely on main script for user feedback)
    if not isinstance(observer_location, EarthLocation):
        print(f"Internal Error: observer_location must be an astropy EarthLocation. Got {type(observer_location)}")
        return []
    if not isinstance(observing_times, Time) or not observing_times.shape: # Check if it's a valid Time array
        print(f"Internal Error: observing_times must be a non-empty astropy Time array. Got {type(observing_times)}")
        return []
    if not isinstance(min_altitude_limit, u.Quantity) or not min_altitude_limit.unit.is_equivalent(u.deg):
        print(f"Internal Error: min_altitude_limit must be an astropy Quantity in angular units. Got {type(min_altitude_limit)}")
        return []
    # Use 'pd' which is now defined due to the import at the top
    if not isinstance(catalog_df, pd.DataFrame): # <-- USES pd
        print(f"Internal Error: catalog_df must be a pandas DataFrame. Got {type(catalog_df)}")
        return []
    if catalog_df.empty:
        print("Input catalog_df is empty. No objects to process.")
        return []
    if len(observing_times) < 2:
        print("Warning: Observing window has less than 2 time points. Duration calculation might be inaccurate.")


    # Pre-calculate AltAz frame for efficiency
    altaz_frame = AltAz(obstime=observing_times, location=observer_location)
    min_alt_deg = min_altitude_limit.to(u.deg).value
    time_step_hours = 0 # Initialize outside loop
    if len(observing_times) > 1:
        time_diff_seconds = (observing_times[1] - observing_times[0]).sec
        time_step_hours = time_diff_seconds / 3600.0


    # --- Iterate through Catalog Objects ---
    for index, obj in catalog_df.iterrows(): # <-- USES pd implicitly via DataFrame type
        try:
            # --- Get Object Data ---
            ra_str = obj.get('RA_str', None)
            dec_str = obj.get('Dec_str', None)
            dso_name = obj.get('Name', f"Unnamed Object {index}")
            obj_type = obj.get('Type', "Unknown")
            obj_mag = obj.get('Mag', np.nan)
            obj_size = obj.get('MajAx', np.nan)

            if not ra_str or not dec_str:
                print(f"Skipping object '{dso_name}': Missing RA or Dec string.")
                continue

            # --- Handle Coordinates ---
            try:
                dso_coord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))
            except ValueError as coord_err:
                print(f"Skipping object '{dso_name}': Invalid coordinate format ('{ra_str}', '{dec_str}'). Error: {coord_err}")
                continue

            # --- Calculate Alt/Az ---
            # Add extra check for potential coordinate transformation errors
            try:
                dso_altazs = dso_coord.transform_to(altaz_frame)
                dso_alts = dso_altazs.alt.to(u.deg).value
                dso_azs = dso_altazs.az.to(u.deg).value
            except Exception as transform_err:
                print(f"Skipping object '{dso_name}': Error transforming coordinates. Error: {transform_err}")
                continue

            # --- Check if Observable (Reaches Minimum Altitude) ---
            max_alt_this_object = np.max(dso_alts) if len(dso_alts) > 0 else -999 # Handle empty array case
            if max_alt_this_object >= min_alt_deg:
                # Object is potentially observable (reaches min alt)

                # --- Find Peak Altitude Details ---
                peak_alt_index = np.argmax(dso_alts)
                peak_alt = dso_alts[peak_alt_index]
                peak_time_utc = observing_times[peak_alt_index]
                peak_az = dso_azs[peak_alt_index]
                peak_direction = azimuth_to_direction(peak_az)

                # --- Get Constellation ---
                try:
                    constellation = get_constellation(dso_coord)
                except Exception as const_err:
                    print(f"Warning: Could not determine constellation for {dso_name}: {const_err}")
                    constellation = "N/A"

                # --- Calculate Continuous Duration Above Minimum Altitude ---
                above_min_alt = dso_alts >= min_alt_deg
                continuous_duration_hours = 0
                if time_step_hours > 0 and np.any(above_min_alt):
                    # Find contiguous blocks where above_min_alt is True
                    # Based on https://stackoverflow.com/a/4495197/1169513
                    runs = np.split(np.arange(len(above_min_alt)), np.where(np.diff(above_min_alt))[0]+1)
                    max_duration_indices = 0
                    for run in runs:
                        if run.size > 0 and above_min_alt[run[0]]: # Check if this run is 'True' and not empty
                            max_duration_indices = max(max_duration_indices, len(run))

                    # Duration is (number of steps - 1) * time_step, but since we count indices,
                    # len(run) directly gives the number of time points.
                    # For N points, there are N-1 intervals.
                    # However, counting points * step size gives a good approximation.
                    continuous_duration_hours = max_duration_indices * time_step_hours


                # --- Store Result (Max Altitude filter applied later) ---
                result_dict = {
                    'Name': dso_name,
                    'Type': obj_type,
                    'Constellation': constellation,
                    'Magnitude': obj_mag if not np.isnan(obj_mag) else None,
                    'Size (arcmin)': obj_size if not np.isnan(obj_size) else None,
                    'RA': ra_str,
                    'Dec': dec_str,
                    'Max Altitude (°)': peak_alt, # Store calculated peak altitude
                    'Azimuth at Max (°)': peak_az,
                    'Direction at Max': peak_direction,
                    'Time at Max (UTC)': peak_time_utc,
                    'Max Cont. Duration (h)': continuous_duration_hours,
                    'skycoord': dso_coord,
                    'altitudes': dso_alts,
                    'azimuths': dso_azs,
                    'times': observing_times
                }
                observable_objects.append(result_dict)

        except Exception as obj_proc_e:
            # Catch any other unexpected error during object processing
            # Use 't' dictionary passed as parameter for error message
            error_msg = t.get('error_processing_object', "Error processing {}: {}").format(obj.get('Name', f'Object at index {index}'), obj_proc_e)
            print(error_msg)
            # traceback.print_exc() # Uncomment for detailed traceback during debugging

    return observable_objects