# -*- coding: utf-8 -*-
from __future__ import annotations
import streamlit as st
import random
from datetime import datetime, date, time, timedelta, timezone
import traceback
import os
import urllib.parse
import pandas as pd
import math

# --- Library Imports ---
try:
    from astropy.time import Time
    import numpy as np
    import astropy.units as u
    from astropy.coordinates import SkyCoord, AltAz
    # Removed moon_illumination import as it's only used in create_moon_phase_svg which is now local
    from astroplan import Observer # <<< Added Observer import
    from astroplan.moon import moon_illumination # <<< Re-added moon_illumination
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pytz
    from timezonefinder import TimezoneFinder # <<< Added TimezoneFinder import
    from geopy.geocoders import Nominatim, ArcGIS, Photon
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
except ImportError as e:
    # This error should ideally be caught in the main script,
    # but added here for robustness if used independently.
    st.error(f"Import Error in UI module: {e}")
    st.stop()

# --- Import Custom Modules ---
# Assuming these are in the same directory or accessible via PYTHONPATH
try:
    from astro_calculations import CARDINAL_DIRECTIONS # Needed for direction filter
    # Note: translations 't' will be passed as an argument
except ModuleNotFoundError as e:
    st.error(f"Module Not Found Error in UI module: {e}")
    st.stop()

# --- Constants ---
ALL_DIRECTIONS_KEY = 'All' # Define locally or import if defined elsewhere centrally

# --- UI Helper Functions (Moved from main script) ---

def create_moon_phase_svg(illumination: float, size: int = 100) -> str:
    """Creates an SVG representation of the moon phase."""
    if not 0 <= illumination <= 1: print(f"Warn: Invalid moon illum ({illumination}). Clamping."); illumination = max(0.0, min(1.0, illumination))
    radius = size / 2; cx = cy = radius
    # Use Streamlit theme variables for colors
    light_color = "var(--text-color, #e0e0e0)"; dark_color = "var(--secondary-background-color, #333333)"
    svg = f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
    svg += f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{dark_color}"/>'
    if illumination < 0.01: pass # New moon
    elif illumination > 0.99: svg += f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{light_color}"/>' # Full moon
    else:
        # Calculate the horizontal position for the terminator
        x = radius * (illumination * 2 - 1)
        rx = abs(x) # Radius of the ellipse for the terminator arc

        if illumination <= 0.5: # Waxing or Waning Crescent/Quarter
            # Draw the dark part (left) and the illuminated part (right half-circle)
            # Path: Move to top-center, arc to bottom-center (ellipse), arc back to top-center (circle)
            large_arc_ellipse = 0; sweep_ellipse = 1 # Ellipse from top to bottom on the right
            large_arc_circle = 0; sweep_circle = 1  # Circle arc for the right edge
            d=f"M {cx},{cy-radius} A {rx},{radius} 0 {large_arc_ellipse},{sweep_ellipse} {cx},{cy+radius} A {radius},{radius} 0 {large_arc_circle},{sweep_circle} {cx},{cy-radius} Z"
        else: # Waxing or Waning Gibbous
            # Draw the illuminated part (left half-circle) and the illuminated ellipse part (right)
            # Path: Move to top-center, arc to bottom-center (circle), arc back to top-center (ellipse)
            large_arc_circle = 1; sweep_circle = 1  # Circle arc for the left edge
            large_arc_ellipse = 0; sweep_ellipse = 1 # Ellipse from bottom to top on the right
            d=f"M {cx},{cy-radius} A {radius},{radius} 0 {large_arc_circle},{sweep_circle} {cx},{cy+radius} A {rx},{radius} 0 {large_arc_ellipse},{sweep_ellipse} {cx},{cy-radius} Z"
        svg += f'<path d="{d}" fill="{light_color}"/>'
    svg += '</svg>'; return svg

def get_local_time_str(utc_time: Time | None, timezone_str: str) -> tuple[str, str]:
    """Converts a UTC Time object to a localized time string, or returns "N/A"."""
    if utc_time is None: return "N/A", "N/A"
    if not isinstance(utc_time, Time): print(f"Err: utc_time type {type(utc_time)}"); return "N/A", "N/A"
    if not isinstance(timezone_str, str) or not timezone_str: print(f"Err: tz type '{timezone_str}'"); return "N/A", "N/A"
    try:
        local_tz = pytz.timezone(timezone_str); utc_dt = utc_time.to_datetime(timezone.utc); local_dt = utc_dt.astimezone(local_tz)
        local_time_str = local_dt.strftime('%Y-%m-%d %H:%M:%S'); tz_display_name = local_dt.tzname()
        if not tz_display_name: tz_display_name = local_tz.zone # Fallback if tzname() is None
        return local_time_str, tz_display_name
    except pytz.exceptions.UnknownTimeZoneError: print(f"Err: Unknown TZ '{timezone_str}'."); return utc_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), "UTC (TZ Err)"
    except Exception as e: print(f"Err converting time: {e}"); traceback.print_exc(); return utc_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), "UTC (Conv Err)"

def create_plot(plot_data: dict, min_altitude_deg: float, max_altitude_deg: float, plot_type: str, t: dict) -> plt.Figure | None:
    """Creates either an Altitude vs Time or Sky Path (Alt/Az) plot."""
    fig = None
    try:
        # --- Data Validation ---
        if not isinstance(plot_data, dict):
            st.error(t.get('plot_error_invalid_data_type', "Plot Error: Invalid plot data type provided."))
            return None
        times = plot_data.get('times')
        altitudes = plot_data.get('altitudes')
        azimuths = plot_data.get('azimuths') # May be None for Altitude Plot
        obj_name = plot_data.get('Name', t.get('plot_object_default_name', 'Object'))

        # Check essential data presence and types
        if not isinstance(times, Time) or not isinstance(altitudes, np.ndarray):
            st.error(t.get('plot_error_missing_time_alt', "Plot Error: Missing or invalid time/altitude data."))
            return None
        if plot_type == 'Sky Path' and not isinstance(azimuths, np.ndarray):
            st.error(t.get('plot_error_missing_azimuth', "Plot Error: Missing or invalid azimuth data required for Sky Path plot."))
            return None

        # Check array lengths
        if len(times) != len(altitudes) or (azimuths is not None and len(times) != len(azimuths)):
            st.error(t.get('plot_error_mismatched_lengths', "Plot Error: Time, altitude, and azimuth arrays have mismatched lengths."))
            return None
        if len(times) < 1:
            st.error(t.get('plot_error_no_data_points', "Plot Error: Not enough data points to create a plot."))
            return None

        plot_times = times.plot_date # Convert Astropy Time to matplotlib-compatible format

        # --- Theming ---
        try:
            # Check Streamlit's theme setting
            theme_opts = st.get_option("theme.base")
            is_dark_theme = (theme_opts == "dark")
        except Exception:
            # Fallback if theme option is not available (older Streamlit versions?)
            print("Warning: Could not detect Streamlit theme. Assuming light theme.")
            is_dark_theme = False

        # Define color palettes for dark and light themes
        if is_dark_theme:
            plt.style.use('dark_background')
            fc = '#0E1117' # Figure facecolor (match Streamlit dark bg)
            pc = 'deepskyblue' # Primary plot color
            gc = '#444' # Grid color
            lc = '#CCC' # Label color (lighter gray)
            tc = '#FFF' # Title color
            lfc = '#262730' # Legend facecolor
            min_c = 'tomato' # Min altitude line color
            max_c = 'orange' # Max altitude line color
            sc = '#555' # Spine color
        else:
            plt.style.use('default') # Use Matplotlib's default light style
            fc = '#FFFFFF' # Figure facecolor
            pc = 'dodgerblue' # Primary plot color
            gc = 'darkgray' # Grid color
            lc = '#333' # Label color (dark gray)
            tc = '#000' # Title color
            lfc = '#F0F0F0' # Legend facecolor
            min_c = 'red' # Min altitude line color
            max_c = 'darkorange' # Max altitude line color
            sc = '#888' # Spine color (slightly darker gray)


        # --- Plot Creation ---
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=fc, constrained_layout=True)
        ax.set_facecolor(fc) # Set axes background color

        if plot_type == 'Altitude Plot':
            # Plot altitude vs. time
            ax.plot(plot_times, altitudes, color=pc, alpha=0.9, lw=1.5, label=obj_name)

            # Add horizontal lines for min/max altitude filters
            ax.axhline(min_altitude_deg, color=min_c, linestyle='--', linewidth=1.2, label=t.get('graph_min_altitude_label', "Min Alt ({:.0f}°)").format(min_altitude_deg), alpha=0.8)
            if max_altitude_deg < 90: # Only show max alt line if it's not zenith
                 ax.axhline(max_altitude_deg, color=max_c, linestyle=':', linewidth=1.2, label=t.get('graph_max_altitude_label', "Max Alt ({:.0f}°)").format(max_altitude_deg), alpha=0.8)

            # Formatting
            ax.set_xlabel(t.get('graph_xlabel_time', "Time (UTC)"), color=lc, fontsize=11)
            ax.set_ylabel(t.get('graph_ylabel_alt', "Altitude (°)"), color=lc, fontsize=11)
            ax.set_title(t.get('graph_title_alt_time', "Altitude vs. Time: {}").format(obj_name), color=tc, fontsize=13, weight='bold')
            ax.set_ylim(0, 90) # Altitude limits
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) # Format time axis
            fig.autofmt_xdate(rotation=30) # Rotate date labels for better readability
            ax.grid(True, linestyle='-', alpha=0.5, color=gc)
            ax.tick_params(axis='x', colors=lc)
            ax.tick_params(axis='y', colors=lc)
            for spine in ax.spines.values(): spine.set_color(sc); spine.set_linewidth(0.5)

        elif plot_type == 'Sky Path':
            # Requires azimuths data, checked earlier
            if azimuths is None: # Double check just in case
                 st.error(t.get('plot_error_missing_azimuth', "Plot Error: Missing azimuth data required for Sky Path plot."))
                 plt.close(fig)
                 return None

            # Create a polar plot
            ax.remove() # Remove the default Cartesian axes
            ax = fig.add_subplot(111, projection='polar', facecolor=fc)

            # Convert Alt/Az to polar coordinates (theta=azimuth, r=zenith angle)
            az_rad = np.deg2rad(azimuths)
            radius = 90 - altitudes # Zenith angle

            # Color points by time
            time_delta = times.jd.max() - times.jd.min()
            time_normalized = (times.jd - times.jd.min()) / (time_delta + 1e-9) if time_delta > 0 else np.zeros_like(times.jd) # Avoid division by zero
            colors = plt.cm.viridis(time_normalized) # Use viridis colormap

            # Plot the path
            scatter = ax.scatter(az_rad, radius, c=colors, s=15, alpha=0.8, edgecolors='none', label=obj_name)
            ax.plot(az_rad, radius, color=pc, alpha=0.4, lw=0.8) # Connect points with a faint line

            # Add circles for min/max altitude filters
            ax.plot(np.linspace(0, 2 * np.pi, 100), np.full(100, 90 - min_altitude_deg), color=min_c, linestyle='--', linewidth=1.2, label=t.get('graph_min_altitude_label',"Min Alt ({:.0f}°)").format(min_altitude_deg), alpha=0.8)
            if max_altitude_deg < 90:
                ax.plot(np.linspace(0, 2 * np.pi, 100), np.full(100, 90 - max_altitude_deg), color=max_c, linestyle=':', linewidth=1.2, label=t.get('graph_max_altitude_label',"Max Alt ({:.0f}°)").format(max_altitude_deg), alpha=0.8)

            # Formatting the polar plot
            ax.set_theta_zero_location('N') # North at top
            ax.set_theta_direction(-1) # Clockwise azimuth (East is right)
            ax.set_yticks(np.arange(0, 91, 15)) # Altitude grid lines every 15 degrees (radial)
            ax.set_yticklabels([f"{90-a}°" for a in np.arange(0, 91, 15)], color=lc) # Label with altitude
            ax.set_ylim(0, 90) # Radius limit (0=zenith, 90=horizon)
            ax.set_title(t.get('graph_title_sky_path',"Sky Path: {}").format(obj_name), va='bottom', color=tc, fontsize=13, weight='bold', y=1.1) # Adjust title position
            ax.grid(True, linestyle=':', alpha=0.5, color=gc)
            ax.spines['polar'].set_color(sc) # Color the outer circle
            ax.spines['polar'].set_linewidth(0.5)

            # Add a colorbar indicating time
            try:
                cbar = fig.colorbar(scatter, ax=ax, label=t.get('graph_colorbar_label', "Time (UTC)"), pad=0.1, shrink=0.7)
                cbar.set_ticks([0, 1]) # Ticks at the start and end
                if len(times) > 0:
                    start_label = times[0].to_datetime(timezone.utc).strftime('%H:%M')
                    end_label = times[-1].to_datetime(timezone.utc).strftime('%H:%M')
                    cbar.ax.set_yticklabels([start_label, end_label])
                else:
                    cbar.ax.set_yticklabels(['Start', 'End']) # Fallback labels
                cbar.set_label(t.get('graph_colorbar_label', "Time (UTC)"), color=lc, fontsize=10)
                cbar.ax.yaxis.set_tick_params(color=lc, labelsize=9)
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=lc)
                cbar.outline.set_edgecolor(sc)
                cbar.outline.set_linewidth(0.5)
            except Exception as e:
                print(f"Warning: Colorbar creation failed: {e}")

        else:
            # Should not happen if plot_type is validated, but good practice
            st.error(t.get('plot_error_unknown_type', "Plot Error: Unknown plot type requested: '{}'").format(plot_type))
            plt.close(fig)
            return None

        # --- Legend ---
        leg = ax.legend(loc='lower right', fontsize='small', facecolor=lfc, framealpha=0.8, edgecolor=sc)
        for text in leg.get_texts():
            text.set_color(lc) # Set legend text color

        return fig

    except Exception as e:
        st.error(t.get('plot_error_unexpected', "Plot Error: An unexpected error occurred during plot creation: {}").format(e))
        traceback.print_exc()
        if fig: plt.close(fig) # Ensure figure is closed if an error occurs
        return None


# --- Main UI Component Functions ---

# Pass TimezoneFinder class as type hint
def create_sidebar(t: dict, df_catalog_data: pd.DataFrame | None, tf: TimezoneFinder | None) -> None:
    """Creates the sidebar UI elements."""
    with st.sidebar:
        st.header(t.get('settings_header', "Settings"))

        # Catalog Status
        if 'catalog_status_msg' not in st.session_state: st.session_state.catalog_status_msg = ""
        if df_catalog_data is not None:
            new_msg = t.get('info_catalog_loaded', "Catalog loaded: {} objects.").format(len(df_catalog_data))
            # Only update message if it changes to avoid unnecessary reruns if status is stable
            if st.session_state.catalog_status_msg != new_msg:
                st.success(new_msg)
                st.session_state.catalog_status_msg = new_msg
        else:
            new_msg = t.get('error_catalog_load_failed', "Catalog loading failed. Check file or logs.")
            if st.session_state.catalog_status_msg != new_msg:
                st.error(new_msg)
                st.session_state.catalog_status_msg = new_msg

        # Language Selection
        language_options = {'de': 'Deutsch', 'en': 'English', 'fr': 'Français'}
        lang_keys = list(language_options.keys())
        try:
            current_lang_key = st.session_state.language
            current_lang_idx = lang_keys.index(current_lang_key)
        except ValueError:
            current_lang_idx = 0 # Default to first language if current is invalid

        selected_lang_key = st.radio(
            t.get('language_select_label', "Language"),
            lang_keys,
            format_func=lambda k: language_options[k],
            key='language_radio', # Use a distinct key
            index=current_lang_idx,
            horizontal=True
        )
        # Update language state and rerun if changed
        if selected_lang_key != st.session_state.language:
            st.session_state.language = selected_lang_key
            st.session_state.location_search_status_msg = "" # Reset search message on language change
            print(f"Language changed to: {selected_lang_key}. Rerunning.")
            st.rerun()

        # --- Location Expander ---
        with st.expander(t.get('location_expander', "📍 Location"), expanded=True):
            loc_opts = {'Search': t.get('location_option_search', "Search"), 'Manual': t.get('location_option_manual', "Manual")}
            st.radio(t.get('location_select_label', "Location Method"), list(loc_opts.keys()), format_func=lambda k: loc_opts[k], key="location_choice_key", horizontal=True)

            lat, lon, hgt = None, None, None
            loc_valid_for_tz_lookup = False
            current_location_is_valid = False

            if st.session_state.location_choice_key == "Manual":
                st.number_input(t.get('location_lat_label', "Latitude (°N)"), -90.0, 90.0, step=0.01, format="%.4f", key="manual_lat_val")
                st.number_input(t.get('location_lon_label', "Longitude (°E)"), -180.0, 180.0, step=0.01, format="%.4f", key="manual_lon_val")
                st.number_input(t.get('location_elev_label', "Elevation (m)"), -500, step=10, format="%d", key="manual_height_val")

                lat = st.session_state.manual_lat_val
                lon = st.session_state.manual_lon_val
                hgt = st.session_state.manual_height_val

                # Validate manual input
                if isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and isinstance(hgt, (int, float)):
                    loc_valid_for_tz_lookup = True
                    current_location_is_valid = True
                    st.session_state.location_is_valid_for_run = True
                    # If switching from successful search to manual, clear search results
                    if st.session_state.location_search_success:
                        st.session_state.location_search_success = False
                        st.session_state.searched_location_name = None
                        st.session_state.location_search_status_msg = ""
                else:
                    st.warning(t.get('location_error_manual_invalid', "Manual coordinate or elevation fields are invalid."))
                    current_location_is_valid = False
                    st.session_state.location_is_valid_for_run = False

            elif st.session_state.location_choice_key == "Search":
                with st.form("loc_search_form"):
                    st.text_input(t.get('location_search_label', "Location Name/Address:"), key="location_search_query", placeholder=t.get('location_search_placeholder', "e.g., Paris, France or Zip Code"))
                    # Keep elevation manual even in search mode
                    st.number_input(t.get('location_elev_label', "Elevation (m)"), -500, step=10, format="%d", key="manual_height_val")
                    search_submitted = st.form_submit_button(t.get('location_search_submit_button', "Find Coordinates"))

                status_placeholder = st.empty() # Placeholder for success/error messages

                # Display previous search status
                if st.session_state.location_search_status_msg:
                    if st.session_state.location_search_success:
                        status_placeholder.success(st.session_state.location_search_status_msg)
                    else:
                        status_placeholder.error(st.session_state.location_search_status_msg)

                if search_submitted and st.session_state.location_search_query:
                    location_result = None
                    service_used = None
                    error_occurred = None
                    query = st.session_state.location_search_query
                    user_agent = f"AdvancedDSOFinder/{random.randint(1000,9999)}" # Unique user agent

                    with st.spinner(t.get('spinner_geocoding', "Searching for location...")):
                        # Try Nominatim first
                        try:
                            geolocator = Nominatim(user_agent=user_agent)
                            location_result = geolocator.geocode(query, timeout=10)
                            service_used = "Nominatim"
                            print("Nominatim geocoding successful.")
                        except (GeocoderTimedOut, GeocoderServiceError) as e:
                            print(f"Nominatim failed: {e}")
                            status_placeholder.info(t.get('location_search_info_fallback', "Nominatim failed, trying ArcGIS..."))
                        except Exception as e:
                            print(f"Nominatim error: {e}")
                            status_placeholder.info(t.get('location_search_info_fallback', "Nominatim failed, trying ArcGIS..."))
                            error_occurred = e

                        # Fallback to ArcGIS
                        if not location_result:
                            try:
                                fallback_geolocator = ArcGIS(timeout=15)
                                location_result = fallback_geolocator.geocode(query, timeout=15)
                                service_used = "ArcGIS"
                                print("ArcGIS geocoding successful.")
                            except (GeocoderTimedOut, GeocoderServiceError) as e2:
                                print(f"ArcGIS failed: {e2}")
                                status_placeholder.info(t.get('location_search_info_fallback2', "ArcGIS failed, trying Photon..."))
                                if not error_occurred: error_occurred = e2
                            except Exception as e2:
                                print(f"ArcGIS error: {e2}")
                                status_placeholder.info(t.get('location_search_info_fallback2', "ArcGIS failed, trying Photon..."))
                                if not error_occurred: error_occurred = e2

                        # Fallback to Photon
                        if not location_result:
                             try:
                                 fallback_geolocator2 = Photon(user_agent=user_agent, timeout=15)
                                 location_result = fallback_geolocator2.geocode(query, timeout=15)
                                 service_used = "Photon"
                                 print("Photon geocoding successful.")
                             except (GeocoderTimedOut, GeocoderServiceError) as e3:
                                 print(f"Photon failed: {e3}")
                                 if not error_occurred: error_occurred = e3
                             except Exception as e3:
                                 print(f"Photon error: {e3}")
                                 if not error_occurred: error_occurred = e3

                        # Process results
                        if location_result and service_used:
                            found_lat = location_result.latitude
                            found_lon = location_result.longitude
                            found_name = location_result.address
                            st.session_state.searched_location_name = found_name
                            st.session_state.location_search_success = True
                            st.session_state.manual_lat_val = found_lat # Update manual fields with found coords
                            st.session_state.manual_lon_val = found_lon
                            # Keep manually entered height: hgt = st.session_state.manual_height_val

                            coords_str = t.get('location_search_coords',"Lat:{:.4f}, Lon:{:.4f}").format(found_lat, found_lon)
                            if service_used == "Nominatim":
                                st.session_state.location_search_status_msg = f"{t.get('location_search_found','Found (Nominatim): {}').format(found_name)}\n({coords_str})"
                            elif service_used == "ArcGIS":
                                st.session_state.location_search_status_msg = f"{t.get('location_search_found_fallback','Found (ArcGIS): {}').format(found_name)}\n({coords_str})"
                            elif service_used == "Photon":
                                st.session_state.location_search_status_msg = f"{t.get('location_search_found_fallback2','Found (Photon): {}').format(found_name)}\n({coords_str})"

                            status_placeholder.success(st.session_state.location_search_status_msg)
                            lat = found_lat; lon = found_lon; hgt = st.session_state.manual_height_val
                            loc_valid_for_tz_lookup = True
                            current_location_is_valid = True
                            st.session_state.location_is_valid_for_run = True
                        else:
                            # Geocoding failed
                            st.session_state.location_search_success = False
                            st.session_state.searched_location_name = None
                            if error_occurred:
                                if isinstance(error_occurred, GeocoderTimedOut):
                                    st.session_state.location_search_status_msg = t.get('location_search_error_timeout',"Geocoding timed out.")
                                elif isinstance(error_occurred, GeocoderServiceError):
                                     st.session_state.location_search_status_msg = t.get('location_search_error_service',"Geocoding service error: {}").format(error_occurred)
                                else:
                                     st.session_state.location_search_status_msg = t.get('location_search_error_fallback2_failed',"All geocoding services failed: {}").format(error_occurred)
                            else:
                                st.session_state.location_search_status_msg = t.get('location_search_error_not_found',"Location not found.")
                            status_placeholder.error(st.session_state.location_search_status_msg)
                            current_location_is_valid = False
                            st.session_state.location_is_valid_for_run = False

                # If not searching now, but a previous search was successful, use those coords
                elif st.session_state.location_search_success:
                    lat = st.session_state.manual_lat_val
                    lon = st.session_state.manual_lon_val
                    hgt = st.session_state.manual_height_val
                    loc_valid_for_tz_lookup = True
                    current_location_is_valid = True
                    st.session_state.location_is_valid_for_run = True
                    status_placeholder.success(st.session_state.location_search_status_msg) # Keep showing success message
                else:
                    # No valid search result and not searching now
                    current_location_is_valid = False
                    st.session_state.location_is_valid_for_run = False

            # --- Timezone Detection ---
            st.markdown("---")
            timezone_message = ""
            if loc_valid_for_tz_lookup and lat is not None and lon is not None:
                if tf: # Check if TimezoneFinder instance is available and valid
                    try:
                        # Attempt to find timezone
                        found_timezone_val = tf.timezone_at(lng=lon, lat=lat)
                        if found_timezone_val:
                            # Validate the timezone with pytz
                            pytz.timezone(found_timezone_val)
                            st.session_state.selected_timezone = found_timezone_val
                            timezone_message = f"{t.get('timezone_auto_set_label','Detected Timezone:')} **{found_timezone_val}**"
                        else:
                            # TimezoneFinder returned None
                            st.session_state.selected_timezone = 'UTC' # Default to UTC
                            timezone_message = f"{t.get('timezone_auto_fail_label','Timezone:')} **UTC** ({t.get('timezone_auto_fail_msg','Detection failed')})"
                    except pytz.UnknownTimeZoneError:
                         # Found timezone name is invalid according to pytz
                         st.session_state.selected_timezone = 'UTC'
                         invalid_tz_name = locals().get('found_timezone_val', 'Unknown') # Get the invalid name if possible
                         timezone_message = t.get('timezone_auto_fail_invalid_label','Timezone:') + f" **UTC** ({t.get('timezone_auto_fail_invalid_msg','Invalid Name:')} '{invalid_tz_name}')"
                    except Exception as e:
                        # Other errors during timezone lookup
                        print(f"Timezone lookup error: {e}")
                        st.session_state.selected_timezone = 'UTC'
                        timezone_message = f"{t.get('timezone_auto_fail_error_label','Timezone:')} **UTC** ({t.get('timezone_auto_fail_error_msg','Error')})"
                else:
                    # TimezoneFinder instance (tf) is not available
                    timezone_message = f"{t.get('timezone_finder_unavailable_label','Timezone:')} **{st.session_state.selected_timezone}** ({t.get('timezone_finder_unavailable_msg','Finder N/A')})"
                    # Keep the previously set timezone or initial default
            else:
                # Location is not valid for timezone lookup
                timezone_message = f"{t.get('timezone_invalid_location_label','Timezone:')} **{st.session_state.selected_timezone}** ({t.get('timezone_invalid_location_msg','Location Invalid')})"

            st.markdown(timezone_message, unsafe_allow_html=True)

        # --- Time Expander ---
        with st.expander(t.get('time_expander', "⏱️ Time"), expanded=False):
            time_opts = {'Now': t.get('time_option_now',"Now"), 'Specific': t.get('time_option_specific',"Specific Night")}
            st.radio(t.get('time_select_label',"Select Time Frame"), list(time_opts.keys()), format_func=lambda k:time_opts[k], key="time_choice_exp", horizontal=True)

            is_now_time_choice = (st.session_state.time_choice_exp == "Now")
            if is_now_time_choice:
                # Display current UTC time when 'Now' is selected
                st.caption(f"UTC: {Time.now().iso}")
            else:
                # Show date picker for 'Specific Night'
                st.date_input(
                    t.get('time_date_select_label',"Select Date for Night:"),
                    value=st.session_state.selected_date_widget,
                    min_value=date.today() - timedelta(days=365*10), # Limit past dates
                    max_value=date.today() + timedelta(days=365*2),  # Limit future dates
                    key='selected_date_widget'
                )

        # --- Filters Expander ---
        with st.expander(t.get('filters_expander', "✨ Filters"), expanded=False):
            # Magnitude Filter
            st.markdown(t.get('mag_filter_header', "**Magnitude**"))
            mag_opts = {'Bortle Scale': t.get('mag_filter_option_bortle',"Bortle Scale"), 'Manual': t.get('mag_filter_option_manual',"Manual")}
            # Ensure the selected mode is valid, default to Bortle if not
            if st.session_state.mag_filter_mode_exp not in mag_opts:
                st.session_state.mag_filter_mode_exp = 'Bortle Scale'
            st.radio(t.get('mag_filter_method_label',"Filter Method:"), list(mag_opts.keys()), format_func=lambda k:mag_opts[k], key="mag_filter_mode_exp", horizontal=True)

            # Bortle Scale Slider (always visible, but only used if mode is Bortle)
            st.slider(t.get('mag_filter_bortle_label',"Bortle Scale:"), 1, 9, key='bortle_slider', help=t.get('mag_filter_bortle_help',"Select Bortle class (1=darkest, 9=brightest city)"))

            # Manual Magnitude Sliders (only visible if mode is Manual)
            if st.session_state.mag_filter_mode_exp == "Manual":
                st.slider(t.get('mag_filter_min_mag_label',"Min Magnitude:"), -5.0, 20.0, step=0.5, format="%.1f", help=t.get('mag_filter_min_mag_help',"Brightest magnitude limit (lower numbers are brighter)"), key='manual_min_mag_slider')
                st.slider(t.get('mag_filter_max_mag_label',"Max Magnitude:"), -5.0, 20.0, step=0.5, format="%.1f", help=t.get('mag_filter_max_mag_help',"Faintest magnitude limit (higher numbers are fainter)"), key='manual_max_mag_slider')
                # Add warning if min > max
                if isinstance(st.session_state.manual_min_mag_slider,(int,float)) and isinstance(st.session_state.manual_max_mag_slider,(int,float)) and st.session_state.manual_min_mag_slider > st.session_state.manual_max_mag_slider:
                    st.warning(t.get('mag_filter_warning_min_max',"Minimum magnitude cannot be greater than maximum magnitude!"))

            # Altitude Filter
            st.markdown("---")
            st.markdown(t.get('min_alt_header', "**Altitude**"))
            # Ensure min_alt <= max_alt before displaying sliders
            min_alt_filter = st.session_state.min_alt_slider
            max_alt_filter = st.session_state.max_alt_slider
            if min_alt_filter > max_alt_filter:
                 # If invalid state somehow occurs, reset min to max to avoid error
                 st.session_state.min_alt_slider = max_alt_filter
                 min_alt_filter = max_alt_filter

            st.slider(t.get('min_alt_label',"Minimum Altitude (°):"), 0, 90, key='min_alt_slider', step=1, help=t.get('min_alt_help', "Objects must be above this altitude."))
            st.slider(t.get('max_alt_label',"Maximum Altitude (°):"), 0, 90, key='max_alt_slider', step=1, help=t.get('max_alt_help', "Objects must be below this altitude (useful for obstructions)."))
            # Add warning if min > max after user interaction
            if st.session_state.min_alt_slider > st.session_state.max_alt_slider:
                st.warning(t.get('alt_filter_warning_min_max', "Minimum altitude cannot be greater than maximum altitude!"))

            # Moon Filter
            st.markdown("---")
            st.markdown(t.get('moon_warning_header',"**Moon**"))
            st.slider(t.get('moon_warning_label',"Warn if Moon Illumination > (%):"), 0, 100, key='moon_phase_slider', step=5, help=t.get('moon_warning_help', "Show a warning in results if moon is brighter than this percentage."))

            # Object Type Filter
            st.markdown("---")
            st.markdown(t.get('object_types_header',"**Object Types**"))
            all_types_list = []
            if df_catalog_data is not None and not df_catalog_data.empty:
                try:
                    if 'Type' in df_catalog_data.columns:
                        # Get unique, non-null, string types and sort them
                        all_types_list = sorted(list(df_catalog_data['Type'].dropna().astype(str).unique()))
                    else:
                        st.warning(t.get('object_types_warning_missing_column', "Catalog is missing the 'Type' column."))
                except Exception as e:
                    st.warning(f"{t.get('object_types_error_extract','Error extracting object types')}: {e}")

            if all_types_list:
                # Ensure currently selected types are valid options
                current_selected_types = [s_type for s_type in st.session_state.object_type_filter_exp if s_type in all_types_list]
                # Update session state if invalid types were present
                if current_selected_types != st.session_state.object_type_filter_exp:
                    st.session_state.object_type_filter_exp = current_selected_types

                st.multiselect(
                    t.get('object_types_label',"Filter by Object Type(s):"),
                    all_types_list,
                    default=current_selected_types, # Use the validated list
                    key="object_type_filter_exp",
                    help=t.get('object_types_help',"Select specific object types (leave empty for all).")
                )
            else:
                # Handle case where no types could be loaded
                st.info(t.get('object_types_info_no_types', "No object types available in catalog. Type filter disabled."))
                st.session_state.object_type_filter_exp = [] # Ensure filter is empty

            # Size Filter
            st.markdown("---")
            st.markdown(t.get('size_filter_header',"**Apparent Size**"))
            # Check if size data ('MajAx') is available and usable
            size_data_available = (
                df_catalog_data is not None and
                'MajAx' in df_catalog_data.columns and
                df_catalog_data['MajAx'].notna().any() # Check if there's at least one non-NA value
            )
            size_filter_disabled = not size_data_available

            if size_data_available:
                try:
                    valid_sizes = df_catalog_data['MajAx'].dropna()
                    # Determine min/max from actual data, with safe fallbacks
                    min_size_limit = max(0.1, float(valid_sizes.min())) if not valid_sizes.empty else 0.1
                    max_size_limit = float(valid_sizes.max()) if not valid_sizes.empty else 120.0

                    # Get current slider values from session state
                    current_min_size, current_max_size = st.session_state.size_arcmin_range

                    # Clamp current values to be within data limits
                    clamped_min = max(min_size_limit, min(current_min_size, max_size_limit))
                    clamped_max = min(max_size_limit, max(current_max_size, min_size_limit))

                    # Ensure min <= max after clamping
                    if clamped_min > clamped_max: clamped_min = clamped_max

                    # Update session state only if clamping changed the values
                    if (clamped_min, clamped_max) != st.session_state.size_arcmin_range:
                        st.session_state.size_arcmin_range = (clamped_min, clamped_max)

                    # Determine appropriate step size based on max limit
                    step_size = 0.1 if max_size_limit <= 20 else (0.5 if max_size_limit <= 100 else 1.0)

                    # Display the slider
                    st.slider(
                        label=t.get('size_filter_label',"Size Range (arcminutes):"),
                        min_value=min_size_limit,
                        max_value=max_size_limit,
                        value=st.session_state.size_arcmin_range, # Use potentially updated state
                        step=step_size,
                        key='size_arcmin_range', # Re-assign key to ensure value updates correctly
                        help=t.get('size_filter_help',"Filter objects by their apparent major axis size."),
                        disabled=size_filter_disabled # Should be False here
                    )
                except Exception as e:
                    st.error(f"{t.get('size_filter_error', 'Error setting up size slider')}: {e}")
                    size_filter_disabled = True # Disable if error occurs
            else:
                # Inform user if size data is missing
                st.info(t.get('size_filter_info_missing', "Size data ('MajAx') missing or invalid in catalog. Size filter disabled."))
                size_filter_disabled = True

            # Show a disabled slider if the filter is inactive
            if size_filter_disabled:
                st.slider(
                    t.get('size_filter_label',"Size Range (arcminutes):"),
                    0.0, 1.0, value=(0.0, 1.0), # Dummy values
                    key='size_arcmin_range_disabled', # Use different key for disabled state
                    disabled=True
                )

            # Direction Filter
            st.markdown("---")
            st.markdown(t.get('direction_filter_header',"**Culmination Direction**"))
            all_directions_text = t.get('direction_option_all',"All Directions")
            # Combine translated "All" with cardinal directions for display
            direction_display_options = [all_directions_text] + CARDINAL_DIRECTIONS
            # Internal keys: Use a specific key for "All" and the directions themselves
            direction_internal_keys = [ALL_DIRECTIONS_KEY] + CARDINAL_DIRECTIONS

            # Get current selection from session state
            current_direction_key = st.session_state.selected_peak_direction
            # Ensure current selection is valid, default to "All" if not
            if current_direction_key not in direction_internal_keys:
                current_direction_key = ALL_DIRECTIONS_KEY
                st.session_state.selected_peak_direction = current_direction_key

            # Find the index corresponding to the current key
            try:
                current_index = direction_internal_keys.index(current_direction_key)
            except ValueError:
                current_index = 0 # Default to "All" index

            # Display the selectbox
            selected_direction_display = st.selectbox(
                t.get('direction_filter_label',"Filter by Direction at Max Altitude:"),
                direction_display_options,
                index=current_index,
                key='direction_selectbox', # Use a distinct key
                help=t.get('direction_filter_help',"Show only objects culminating (reaching max altitude) in the selected direction.")
            )

            # Map the selected display text back to the internal key
            selected_internal_key = ALL_DIRECTIONS_KEY # Default to "All"
            if selected_direction_display != all_directions_text:
                try:
                    selected_index = direction_display_options.index(selected_direction_display)
                    selected_internal_key = direction_internal_keys[selected_index]
                except ValueError:
                    # Should not happen if lists are aligned, but fallback just in case
                    selected_internal_key = ALL_DIRECTIONS_KEY

            # Update session state if the selection changed
            if selected_internal_key != st.session_state.selected_peak_direction:
                st.session_state.selected_peak_direction = selected_internal_key

        # --- Results Options Expander ---
        with st.expander(t.get('results_options_expander',"⚙️ Results Options"), expanded=False):
            # Max Objects Slider
            max_possible_objects = len(df_catalog_data) if df_catalog_data is not None and not df_catalog_data.empty else 50
            min_objects_limit = 5
            actual_max_limit = max(min_objects_limit, max_possible_objects)
            slider_disabled = (actual_max_limit <= min_objects_limit)

            # Get current value from session state, default to 20
            default_num_objects = st.session_state.get('num_objects_slider', 20)
            # Clamp the default value to be within the allowed range
            clamped_default = max(min_objects_limit, min(default_num_objects, actual_max_limit))
            # Update session state if clamping changed the value
            if clamped_default != default_num_objects:
                st.session_state.num_objects_slider = clamped_default

            st.slider(
                t.get('results_options_max_objects_label',"Max Objects to Display:"),
                min_value=min_objects_limit,
                max_value=actual_max_limit,
                value=st.session_state.num_objects_slider, # Use potentially updated state
                step=1,
                key='num_objects_slider', # Re-assign key
                disabled=slider_disabled,
                help=t.get('results_options_max_objects_help',"Limit the number of objects shown in the results list.")
            )

            # Sort Method Radio
            sort_method_map = {
                'Duration & Altitude': t.get('results_options_sort_duration',"Duration & Altitude"),
                'Brightness': t.get('results_options_sort_magnitude',"Brightness (Magnitude)")
            }
            # Ensure current sort method is valid, default if not
            if st.session_state.sort_method not in sort_method_map:
                st.session_state.sort_method = 'Duration & Altitude'

            st.radio(
                t.get('results_options_sort_method_label',"Sort Results By:"),
                list(sort_method_map.keys()),
                format_func=lambda k: sort_method_map[k],
                key='sort_method',
                horizontal=True,
                help=t.get('results_options_sort_method_help',"Choose how to order the found objects.")
            )

        # --- Bug Report Link ---
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**{t.get('bug_report', 'Found a bug or have feedback?')}**")
        bug_email_address = "debrun2005@gmail.com"
        bug_email_subject = urllib.parse.quote(t.get('bug_report_subject', "Bug Report/Feedback: Advanced DSO Finder"))
        bug_email_body = urllib.parse.quote(t.get('bug_report_body', "Please describe the issue or suggestion:\n\n(Include steps to reproduce if reporting a bug)\n\nApp Version/Date: [If known]\nBrowser/OS: [If relevant]"))
        mailto_link = f"mailto:{bug_email_address}?subject={bug_email_subject}&body={bug_email_body}"
        st.sidebar.link_button(t.get('bug_report_button', '🐞 Report Issue / Suggestion'), mailto_link)

# Pass Observer class as type hint
def display_search_parameters(t: dict, observer_run: Observer | None, ref_time: Time) -> tuple[float, float]:
    """Displays the summary of the current search parameters in the main area."""
    st.subheader(t.get('search_params_header', "Search Parameters Summary"))
    p1, p2 = st.columns(2)

    # Location Display
    location_display_text = t.get('location_not_set', "Location not set or invalid.")
    if st.session_state.location_is_valid_for_run and observer_run:
        lat = observer_run.location.lat.deg
        lon = observer_run.location.lon.deg
        # Check if location came from search or manual input
        if st.session_state.location_choice_key == "Manual":
            location_display_text = t.get('location_manual_display',"Manual ({:.4f}°N, {:.4f}°E)").format(lat, lon)
        elif st.session_state.searched_location_name:
            # Use the stored name from the successful search
            location_display_text = t.get('location_search_display',"Searched: {} ({:.4f}°N, {:.4f}°E)").format(st.session_state.searched_location_name, lat, lon)
        else:
            # Fallback if search was successful but name wasn't stored (shouldn't happen)
            location_display_text = f"Lat:{lat:.4f}, Lon:{lon:.4f}"
    elif not st.session_state.location_is_valid_for_run:
         # Explicitly state if location is invalid based on session state
         location_display_text = t.get('location_invalid_for_run', "Location Invalid")

    p1.markdown(f"📍 **{t.get('search_params_location_label','Location:')}** {location_display_text}")

    # Time Display
    time_display_text = ""
    is_now_mode = (st.session_state.time_choice_exp == "Now")
    if is_now_mode:
        try:
            # Try to get local time string
            local_now_str, tz_now_name = get_local_time_str(ref_time, st.session_state.selected_timezone)
            time_display_text = t.get('search_params_time_now',"Now (Local: {} {})").format(local_now_str, tz_now_name)
        except Exception:
            # Fallback to UTC if local time conversion fails
            time_display_text = t.get('search_params_time_now_utc',"Now (UTC: {})").format(ref_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))
    else:
        # Display selected date for 'Specific Night' mode
        selected_date = st.session_state.selected_date_widget
        time_display_text = t.get('search_params_time_specific',"Night of {}").format(selected_date.strftime('%Y-%m-%d'))

    p1.markdown(f"⏱️ **{t.get('search_params_time_label','Time:')}** {time_display_text}")

    # Magnitude Filter Display
    magnitude_display_text = ""
    # Also store filter values used for the actual search calculation
    min_mag_filter, max_mag_filter = -np.inf, np.inf # Initialize with wide range
    if st.session_state.mag_filter_mode_exp == "Bortle Scale":
        # Import needs to be inside or passed if astro_calculations is not globally available
        from astro_calculations import get_magnitude_limit # Assuming it's importable
        try:
            max_mag_filter = get_magnitude_limit(st.session_state.bortle_slider)
            magnitude_display_text = t.get('search_params_filter_mag_bortle',"Bortle {} (≤ Mag {:.1f})").format(st.session_state.bortle_slider, max_mag_filter)
        except Exception as e:
             magnitude_display_text = t.get('search_params_filter_mag_bortle_error',"Bortle {} (Error calculating limit)").format(st.session_state.bortle_slider)
             print(f"Error getting magnitude limit: {e}")
    else: # Manual mode
        min_mag_filter = st.session_state.manual_min_mag_slider
        max_mag_filter = st.session_state.manual_max_mag_slider
        magnitude_display_text = t.get('search_params_filter_mag_manual',"Manual (Mag {:.1f} to {:.1f})").format(min_mag_filter, max_mag_filter)

    p2.markdown(f"✨ **{t.get('search_params_filter_mag_label','Magnitude:')}** {magnitude_display_text}")

    # Altitude Filter Display
    min_alt_display = st.session_state.min_alt_slider
    max_alt_display = st.session_state.max_alt_slider
    p2.markdown(f"🔭 **{t.get('search_params_filter_alt_label','Altitude:')}** {min_alt_display}° – {max_alt_display}°")

    # Object Type Filter Display
    selected_types = st.session_state.object_type_filter_exp
    types_display_string = ', '.join(selected_types) if selected_types else t.get('search_params_types_all',"All")
    p2.markdown(f"**{t.get('search_params_filter_type_label','Types:')}** {types_display_string}")

    # Size Filter Display
    size_min, size_max = st.session_state.size_arcmin_range
    p2.markdown(f"📐 **{t.get('search_params_filter_size_label','Size:')}** {size_min:.1f}′ – {size_max:.1f}′")

    # Direction Filter Display
    direction_key = st.session_state.selected_peak_direction
    direction_display_text = t.get('search_params_direction_all',"All") if direction_key == ALL_DIRECTIONS_KEY else direction_key
    p2.markdown(f"🧭 **{t.get('search_params_filter_direction_label','Culm. Dir.:')}** {direction_display_text}")

    # Return the actual filter values used for calculation if needed by main logic
    return min_mag_filter, max_mag_filter

# Pass Observer class as type hint
def display_results(t: dict, results_ph: st.container, observer_run: Observer | None) -> None:
    """Displays the results list, plots, and download button."""
    results_data = st.session_state.last_results
    results_ph.subheader(t.get('results_list_header',"Results"))

    window_start = st.session_state.get('window_start_time')
    window_end = st.session_state.get('window_end_time')
    observer_exists = observer_run is not None

    # Display Moon Phase Metric and Warning
    if observer_exists and isinstance(window_start, Time) and isinstance(window_end, Time):
        mid_time = window_start + (window_end - window_start) / 2 # Calculate midpoint of observation window
        try:
            illumination = moon_illumination(mid_time)
            moon_percentage = illumination * 100
            moon_svg_icon = create_moon_phase_svg(illumination, size=50) # Generate SVG icon

            # Use columns for better layout: icon on left, metric/warning on right
            mc1, mc2 = results_ph.columns([1, 3])
            with mc1:
                st.markdown(moon_svg_icon, unsafe_allow_html=True)
            with mc2:
                st.metric(label=t.get('moon_metric_label',"Moon Illumination"), value=f"{moon_percentage:.0f}%")
                # Check against user's warning threshold
                moon_warning_threshold = st.session_state.moon_phase_slider
                if moon_percentage > moon_warning_threshold:
                    st.warning(t.get('moon_warning_message',"Warning: Moon illumination ({:.0f}%) exceeds threshold ({}%)!").format(moon_percentage, moon_warning_threshold))
        except Exception as e:
            results_ph.warning(t.get('moon_phase_error',"Could not calculate moon phase: {}").format(e))
            print(f"Error calculating moon phase: {e}")
    elif st.session_state.find_button_pressed: # Only show message if search was attempted
        results_ph.info(t.get('moon_phase_info_cannot_calc',"Moon phase calculation requires a valid location and time window."))

    # Plot Type Selection
    plot_type_map = {
        'Sky Path': t.get('graph_type_sky_path',"Sky Path (Polar)"),
        'Altitude Plot': t.get('graph_type_alt_time',"Altitude vs. Time")
    }
    # Ensure selected plot type is valid
    if st.session_state.plot_type_selection not in plot_type_map:
        st.session_state.plot_type_selection = 'Sky Path' # Default to Sky Path

    results_ph.radio(
        t.get('graph_type_label',"Select Plot Type:"),
        list(plot_type_map.keys()),
        format_func=lambda k: plot_type_map[k],
        key='plot_type_selection',
        horizontal=True
    )

    # Display each result object in an expander
    for i, obj_data in enumerate(results_data):
        # --- Prepare Expander Title ---
        obj_name = obj_data.get('Name', '?')
        obj_type = obj_data.get('Type', '?')
        obj_mag = obj_data.get('Magnitude')
        # Format magnitude safely as string
        mag_str = f"{obj_mag:.1f}" if obj_mag is not None else t.get('magnitude_unknown', 'N/A')
        # Use a guaranteed safe template for the title
        expander_title_template = "{} ({}) - Mag: {}"
        expander_title = expander_title_template.format(obj_name, obj_type, mag_str)

        # Check if this expander should be open (based on last plot clicked)
        is_expanded = (st.session_state.expanded_object_name == obj_name)
        # Create a container for each object to manage layout
        obj_container = results_ph.container()

        with obj_container.expander(expander_title, expanded=is_expanded):
            # --- Display Object Details ---
            col1, col2, col3 = st.columns([2, 2, 1]) # Adjust column ratios as needed

            # Column 1: Basic Info & Coordinates
            col1.markdown(f"**{t.get('results_details_header','Details:')}**")
            constellation = obj_data.get('Constellation', '?')
            col1.markdown(f"**{t.get('results_export_constellation','Constellation')}:** {constellation}")
            size_arcmin = obj_data.get('Size (arcmin)')
            size_display = t.get('results_size_value','{:.1f}′').format(size_arcmin) if size_arcmin is not None else '?'
            col1.markdown(f"**{t.get('results_size_label','Size:')}** {size_display}")
            ra_str = obj_data.get('RA', '?')
            dec_str = obj_data.get('Dec', '?')
            col1.markdown(f"**RA:** {ra_str}")
            col1.markdown(f"**Dec:** {dec_str}")

            # Column 2: Observation Specifics
            col2.markdown(f"**{t.get('results_max_alt_header','Max Altitude:')}**")
            max_alt_value = obj_data.get('Max Altitude (°)', 0)
            az_at_max = obj_data.get('Azimuth at Max (°)', 0)
            direction_at_max = obj_data.get('Direction at Max', '?')
            # Use safe format strings directly
            safe_az_fmt = "(Az:{:.1f}°)"
            safe_dir_fmt = ", Dir:{}"
            az_display = safe_az_fmt.format(az_at_max)
            dir_display = safe_dir_fmt.format(direction_at_max)
            col2.markdown(f"**{max_alt_value:.1f}°** {az_display}{dir_display}")

            col2.markdown(f"**{t.get('results_best_time_header','Best Time (Local):')}**")
            peak_time_utc = obj_data.get('Time at Max (UTC)')
            local_time_str, local_tz_name = get_local_time_str(peak_time_utc, st.session_state.selected_timezone)
            col2.markdown(f"{local_time_str} ({local_tz_name})")

            col2.markdown(f"**{t.get('results_cont_duration_header','Max Cont. Duration:')}**")
            duration_hours = obj_data.get('Max Cont. Duration (h)', 0)
            col2.markdown(t.get('results_duration_value',"{:.1f} hours").format(duration_hours))

            # Column 3: External Links & Plot Button
            # Google Search Link
            google_query = urllib.parse.quote_plus(f"{obj_name} astronomy")
            google_url = f"https://google.com/search?q={google_query}"
            col3.markdown(f"[{t.get('google_link_text','Google')}]({google_url})", unsafe_allow_html=True)

            # SIMBAD Link
            simbad_query = urllib.parse.quote_plus(obj_name)
            simbad_url = f"http://simbad.cds.unistra.fr/simbad/sim-basic?Ident={simbad_query}"
            col3.markdown(f"[{t.get('simbad_link_text','SIMBAD')}]({simbad_url})", unsafe_allow_html=True)

            # Plot Button
            plot_button_key = f"plot_{obj_name}_{i}" # Unique key for each plot button
            if st.button(t.get('results_graph_button',"📈 Plot"), key=plot_button_key):
                st.session_state.plot_object_name = obj_name
                st.session_state.active_result_plot_data = obj_data # Store data needed for plotting
                st.session_state.show_plot = True
                st.session_state.show_custom_plot = False # Ensure custom plot is hidden
                st.session_state.expanded_object_name = obj_name # Keep this expander open
                st.rerun() # Rerun to display the plot

            # --- Display Plot if requested ---
            if st.session_state.show_plot and st.session_state.plot_object_name == obj_name:
                plot_data_to_use = st.session_state.active_result_plot_data
                min_alt_for_plot = st.session_state.min_alt_slider
                max_alt_for_plot = st.session_state.max_alt_slider
                st.markdown("---") # Separator before plot

                with st.spinner(t.get('results_spinner_plotting',"Generating plot...")):
                    try:
                        # Call the plotting function (now part of this module)
                        figure = create_plot(plot_data_to_use, min_alt_for_plot, max_alt_for_plot, st.session_state.plot_type_selection, t)
                    except Exception as e:
                        st.error(f"{t.get('plot_error_unexpected', 'Unexpected plot error')}: {e}")
                        traceback.print_exc()
                        figure = None

                    if figure:
                        st.pyplot(figure)
                        # Add a button to close the plot
                        close_button_key = f"close_{obj_name}_{i}"
                        if st.button(t.get('results_close_graph_button',"Close Plot"), key=close_button_key):
                            st.session_state.show_plot = False
                            st.session_state.active_result_plot_data = None
                            st.session_state.expanded_object_name = None # Collapse expander on close
                            st.rerun()
                    else:
                        # Error message if plot creation failed but was attempted
                        st.error(t.get('results_graph_not_created',"Failed to create the plot."))

    # --- CSV Download Button ---
    if results_data: # Only show download if there are results
        csv_placeholder = results_ph.container() # Container for the download button
        try:
            export_rows = []
            # Prepare data for CSV export
            for obj_csv_data in results_data:
                peak_time_utc_csv = obj_csv_data.get('Time at Max (UTC)')
                local_time_csv, _ = get_local_time_str(peak_time_utc_csv, st.session_state.selected_timezone)
                export_rows.append({
                    t.get('results_export_name', "Name"): obj_csv_data.get('Name', 'N/A'),
                    t.get('results_export_type', "Type"): obj_csv_data.get('Type', 'N/A'),
                    t.get('results_export_constellation', "Constellation"): obj_csv_data.get('Constellation', 'N/A'),
                    t.get('results_export_mag', "Magnitude"): obj_csv_data.get('Magnitude'),
                    t.get('results_export_size', "Size (arcmin)"): obj_csv_data.get('Size (arcmin)'),
                    t.get('results_export_ra', "RA"): obj_csv_data.get('RA', 'N/A'),
                    t.get('results_export_dec', "Dec"): obj_csv_data.get('Dec', 'N/A'),
                    t.get('results_export_max_alt', "Max Altitude (°)"): obj_csv_data.get('Max Altitude (°)', 0),
                    t.get('results_export_az_at_max', "Azimuth at Max (°)"): obj_csv_data.get('Azimuth at Max (°)', 0),
                    t.get('results_export_direction_at_max', "Direction at Max"): obj_csv_data.get('Direction at Max', 'N/A'),
                    t.get('results_export_time_max_utc', "Time at Max (UTC)"): peak_time_utc_csv.iso if peak_time_utc_csv else "N/A",
                    t.get('results_export_time_max_local', "Time at Max (Local TZ)"): local_time_csv,
                    t.get('results_export_cont_duration', "Max Cont Duration (h)"): obj_csv_data.get('Max Cont. Duration (h)', 0)
                })

            # Create DataFrame and CSV string
            df_export = pd.DataFrame(export_rows)
            # Use appropriate decimal separator based on language (e.g., comma for German)
            decimal_separator = ',' if st.session_state.language == 'de' else '.'
            csv_string = df_export.to_csv(index=False, sep=';', encoding='utf-8-sig', decimal=decimal_separator)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            csv_filename = t.get('results_csv_filename',"dso_results_{}.csv").format(timestamp)

            # Display download button
            csv_placeholder.download_button(
                label=t.get('results_save_csv_button',"💾 Download Results as CSV"),
                data=csv_string,
                file_name=csv_filename,
                mime='text/csv',
                key='csv_download_button' # Unique key
            )
        except Exception as e:
            csv_placeholder.error(t.get('results_csv_export_error',"Error generating CSV file: {}").format(e))
            print(f"CSV Export Error: {e}")

# Pass Observer class as type hint
def create_custom_target_section(t: dict, results_ph: st.container, observer_run: Observer | None) -> None:
    """Creates the UI section for plotting a custom RA/Dec target."""
    st.markdown("---") # Separator
    with st.expander(t.get('custom_target_expander',"Plot Custom RA/Dec Target")):
        with st.form("custom_target_form"): # Use a unique key for the form
             st.text_input(t.get('custom_target_ra_label',"Right Ascension (RA):"), key="custom_target_ra", placeholder=t.get('custom_target_ra_placeholder',"e.g., 10h 08m 22.3s or 10.1395h"))
             st.text_input(t.get('custom_target_dec_label',"Declination (Dec):"), key="custom_target_dec", placeholder=t.get('custom_target_dec_placeholder',"e.g., +11d 58m 02s or 11.9672d"))
             st.text_input(t.get('custom_target_name_label',"Target Name (Optional):"), key="custom_target_name", placeholder=t.get('custom_target_name_default_placeholder',"Custom Target"))
             custom_plot_submitted = st.form_submit_button(t.get('custom_target_button',"Plot Custom Target"))

        error_placeholder = st.empty() # For displaying errors related to custom target
        plot_display_area = st.container() # Container for the custom plot

        if custom_plot_submitted:
             # Reset state before attempting to plot
             st.session_state.show_plot = False # Hide results plot
             st.session_state.show_custom_plot = False
             st.session_state.custom_target_plot_data = None
             st.session_state.custom_target_error = ""

             # Get input values
             custom_ra_input = st.session_state.custom_target_ra
             custom_dec_input = st.session_state.custom_target_dec
             custom_name_input = st.session_state.custom_target_name or t.get('custom_target_name_default',"Custom Target")

             # Validate inputs and conditions for plotting
             window_start_time = st.session_state.get('window_start_time')
             window_end_time = st.session_state.get('window_end_time')
             observer_exists = observer_run is not None

             if not custom_ra_input or not custom_dec_input:
                 st.session_state.custom_target_error = t.get('custom_target_error_coords_missing',"RA and Dec coordinates are required.")
                 error_placeholder.error(st.session_state.custom_target_error)
             elif not observer_exists or not isinstance(window_start_time, Time) or not isinstance(window_end_time, Time):
                 st.session_state.custom_target_error = t.get('custom_target_error_window_invalid',"Valid location and observation window are required (run a search first).")
                 error_placeholder.error(st.session_state.custom_target_error)
             else:
                 # Attempt to calculate and plot
                 try:
                     # Parse coordinates using SkyCoord (handles various formats)
                     custom_skycoord = SkyCoord(ra=custom_ra_input, dec=custom_dec_input, unit=(u.hourangle, u.deg), frame='icrs') # Assume ICRS frame

                     # Ensure window is valid
                     if window_start_time >= window_end_time:
                         raise ValueError(t.get('custom_target_error_window_order', "Observation window start time must be before end time."))

                     # Generate time steps within the window
                     time_resolution = 5 * u.minute
                     observation_times_custom = Time(np.arange(window_start_time.jd, window_end_time.jd, time_resolution.to(u.day).value), format='jd', scale='utc')

                     if len(observation_times_custom) < 2:
                         raise ValueError(t.get('custom_target_error_window_short', "Observation window is too short for plotting."))

                     # Calculate Alt/Az coordinates for the custom target
                     altaz_frame = AltAz(obstime=observation_times_custom, location=observer_run.location)
                     custom_altazs = custom_skycoord.transform_to(altaz_frame)
                     custom_altitudes = custom_altazs.alt.to(u.deg).value
                     custom_azimuths = custom_altazs.az.to(u.deg).value

                     # Store plot data in session state
                     st.session_state.custom_target_plot_data = {
                         'Name': custom_name_input,
                         'altitudes': custom_altitudes,
                         'azimuths': custom_azimuths,
                         'times': observation_times_custom
                     }
                     st.session_state.show_custom_plot = True
                     st.session_state.custom_target_error = "" # Clear any previous error
                     st.rerun() # Rerun to display the plot

                 except ValueError as e:
                     # Handle coordinate parsing errors or other ValueErrors
                     st.session_state.custom_target_error = f"{t.get('custom_target_error_parsing','Error parsing coordinates or invalid input:')} {e}"
                     error_placeholder.error(st.session_state.custom_target_error)
                 except Exception as e:
                     # Catch any other unexpected errors during calculation/plotting
                     st.session_state.custom_target_error = f"{t.get('custom_target_error_unexpected','Unexpected error plotting custom target:')} {e}"
                     error_placeholder.error(st.session_state.custom_target_error)
                     traceback.print_exc()

        # Display the custom plot if data is available and requested
        if st.session_state.show_custom_plot and st.session_state.custom_target_plot_data:
            custom_plot_data = st.session_state.custom_target_plot_data
            min_alt_cust_plot = st.session_state.min_alt_slider
            max_alt_cust_plot = st.session_state.max_alt_slider

            with plot_display_area:
                 st.markdown("---") # Separator
                 with st.spinner(t.get('results_spinner_plotting',"Generating plot...")):
                     try:
                         # Create the plot using the helper function
                         custom_figure = create_plot(custom_plot_data, min_alt_cust_plot, max_alt_cust_plot, st.session_state.plot_type_selection, t)
                     except Exception as e:
                         st.error(f"{t.get('plot_error_unexpected','Plot Error')}: {e}")
                         traceback.print_exc()
                         custom_figure = None

                     if custom_figure:
                         st.pyplot(custom_figure)
                         # Add button to close the custom plot
                         if st.button(t.get('results_close_graph_button',"Close Plot"), key="close_custom_plot_button"): # Unique key
                             st.session_state.show_custom_plot = False
                             st.session_state.custom_target_plot_data = None
                             st.rerun()
                     elif custom_plot_submitted: # Show error only if plot failed after submission
                          st.error(t.get('results_graph_not_created',"Failed to create the custom plot."))

        # Display persistent error message if one occurred
        elif st.session_state.custom_target_error:
            error_placeholder.error(st.session_state.custom_target_error)


def display_donation_link(t: dict) -> None:
    """Displays the Ko-fi donation link button."""
    st.markdown("---")
    kofi_url = "https://ko-fi.com/advanceddsofinder"
    # Get translated text, provide a sensible default
    kofi_text = t.get('donation_button_text', "Support Development on Ko-fi ☕")
    st.link_button(kofi_text, kofi_url)
