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
# NOTE: Unresolved import errors below usually mean the package is not installed
# in the selected Python environment or the editor is not using the right environment.
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
    st.error(f"Import Error: Missing libraries. Please install required packages (check requirements.txt). Details: {e}")
    st.stop()

# --- Import Custom Modules ---
try:
    from localization import translations
    import astro_calculations
    import data_handling
    from astro_calculations import CARDINAL_DIRECTIONS
except ModuleNotFoundError as e:
    st.error(f"Module Not Found Error: Could not find a required module file ({e}). Ensure 'localization.py', 'astro_calculations.py', and 'data_handling.py' are present.")
    st.stop()


# --- Page Config ---
st.set_page_config(page_title="Advanced DSO Finder", layout="wide")

# --- Global Configuration & Initial Values ---
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
ALL_DIRECTIONS_KEY = 'All'

# --- Initialize TimezoneFinder (cached) ---
@st.cache_resource
def get_timezone_finder():
    """Initializes and returns a TimezoneFinder instance."""
    if TimezoneFinder:
        try: return TimezoneFinder(in_memory=True)
        except Exception as e: print(f"Error initializing TimezoneFinder: {e}"); st.warning(f"TimezoneFinder init failed: {e}."); return None
    return None

tf = get_timezone_finder()

# --- Initialize Session State ---
def initialize_session_state():
    """Initializes all required session state keys if they don't exist."""
    defaults = {
        'language': 'de', 'plot_object_name': None, 'show_plot': False,
        'active_result_plot_data': None, 'last_results': [], 'find_button_pressed': False,
        'location_choice_key': 'Search', 'manual_lat_val': INITIAL_LAT, 'manual_lon_val': INITIAL_LON,
        'manual_height_val': INITIAL_HEIGHT, 'location_search_query': "",
        'searched_location_name': None, 'location_search_status_msg': "",
        'location_search_success': False, 'selected_timezone': INITIAL_TIMEZONE,
        'manual_min_mag_slider': 0.0, 'manual_max_mag_slider': 16.0,
        'object_type_filter_exp': [], 'mag_filter_mode_exp': 'Bortle Scale',
        'bortle_slider': 5, 'min_alt_slider': INITIAL_MIN_ALT, 'max_alt_slider': INITIAL_MAX_ALT,
        'moon_phase_slider': 35, 'size_arcmin_range': [1.0, 120.0],
        'sort_method': 'Duration & Altitude', 'selected_peak_direction': ALL_DIRECTIONS_KEY,
        'plot_type_selection': 'Sky Path', 'custom_target_ra': "", 'custom_target_dec': "",
        'custom_target_name': "", 'custom_target_error': "", 'custom_target_plot_data': None,
        'show_custom_plot': False, 'expanded_object_name': None, 'location_is_valid_for_run': False,
        'time_choice_exp': 'Now', 'window_start_time': None, 'window_end_time': None,
        'selected_date_widget': date.today()
    }
    for key, default_value in defaults.items():
        if key not in st.session_state: st.session_state[key] = default_value

# --- Helper Functions REMAINING in main script ---

def create_moon_phase_svg(illumination: float, size: int = 100) -> str:
    """Creates an SVG representation of the moon phase."""
    if not 0 <= illumination <= 1: print(f"Warn: Invalid moon illum ({illumination}). Clamping."); illumination = max(0.0, min(1.0, illumination))
    radius = size / 2; cx = cy = radius
    light_color = "var(--text-color, #e0e0e0)"; dark_color = "var(--secondary-background-color, #333333)"
    svg = f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
    svg += f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{dark_color}"/>'
    if illumination < 0.01: pass
    elif illumination > 0.99: svg += f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{light_color}"/>'
    else:
        x = radius * (illumination * 2 - 1); rx = abs(x)
        if illumination <= 0.5:
            lae=0; se=1; lac=0; sc=1
            d=f"M {cx},{cy-radius} A {rx},{radius} 0 {lae},{se} {cx},{cy+radius} A {radius},{radius} 0 {lac},{sc} {cx},{cy-radius} Z"
        else:
            lac=1; sc=1; lae=1; se=1
            d=f"M {cx},{cy-radius} A {radius},{radius} 0 {lac},{sc} {cx},{cy+radius} A {rx},{radius} 0 {lae},{se} {cx},{cy-radius} Z"
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
        if not tz_display_name: tz_display_name = local_tz.zone
        return local_time_str, tz_display_name
    except pytz.exceptions.UnknownTimeZoneError: print(f"Err: Unknown TZ '{timezone_str}'."); return utc_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), "UTC (TZ Err)"
    except Exception as e: print(f"Err converting time: {e}"); traceback.print_exc(); return utc_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), "UTC (Conv Err)"

def create_plot(plot_data: dict, min_altitude_deg: float, max_altitude_deg: float, plot_type: str, t: dict) -> plt.Figure | None:
    """Creates either an Altitude vs Time or Sky Path (Alt/Az) plot."""
    fig = None
    try:
        if not isinstance(plot_data, dict): st.error("Plot Error: Invalid plot_data type."); return None
        times = plot_data.get('times'); altitudes = plot_data.get('altitudes'); azimuths = plot_data.get('azimuths'); obj_name = plot_data.get('Name', 'Object')
        if not isinstance(times, Time) or not isinstance(altitudes, np.ndarray): st.error("Plot Error: Missing time/altitude data."); return None
        if plot_type == 'Sky Path' and not isinstance(azimuths, np.ndarray): st.error("Plot Error: Missing azimuth data for Sky Path."); return None
        if len(times) != len(altitudes) or (azimuths is not None and len(times) != len(azimuths)): st.error("Plot Error: Mismatched array lengths."); return None
        if len(times) < 1: st.error("Plot Error: Not enough data points."); return None
        plot_times = times.plot_date

        try: theme_opts = st.get_option("theme.base"); is_dark_theme = (theme_opts == "dark")
        except Exception: print("Warn: Assuming light theme."); is_dark_theme = False
        if is_dark_theme: plt.style.use('dark_background'); fc = '#0E1117'; pc = 'deepskyblue'; gc = '#444'; lc = '#FAF'; tc = '#FFF'; lfc = '#262730'; min_c = 'tomato'; max_c = 'orange'; sc = '#AAA'
        else: plt.style.use('default'); fc = '#FFF'; pc = 'dodgerblue'; gc = 'darkgray'; lc = '#333'; tc = '#000'; lfc = '#F0F0F0'; min_c = 'red'; max_c = 'darkorange'; sc = '#555'

        fig, ax = plt.subplots(figsize=(10, 6), facecolor=fc, constrained_layout=True); ax.set_facecolor(fc)

        if plot_type == 'Altitude Plot':
            ax.plot(plot_times, altitudes, color=pc, alpha=0.9, lw=1.5, label=obj_name)
            ax.axhline(min_altitude_deg, color=min_c, ls='--', lw=1.2, label=t.get('graph_min_altitude_label', "Min Alt ({:.0f}°)").format(min_altitude_deg), alpha=0.8)
            if max_altitude_deg < 90: ax.axhline(max_altitude_deg, color=max_c, ls=':', lw=1.2, label=t.get('graph_max_altitude_label', "Max Alt ({:.0f}°)").format(max_altitude_deg), alpha=0.8)
            ax.set_xlabel("Time (UTC)", color=lc, fontsize=11); ax.set_ylabel(t.get('graph_ylabel', "Alt (°)"), color=lc, fontsize=11); ax.set_title(t.get('graph_title_alt_time', "Altitude: {}").format(obj_name), color=tc, fontsize=13, weight='bold'); ax.set_ylim(0,90); ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')); fig.autofmt_xdate(rotation=30)
            ax.grid(True, linestyle='-', alpha=0.5, color=gc); ax.tick_params(axis='x', colors=lc); ax.tick_params(axis='y', colors=lc)
            for spine in ax.spines.values(): spine.set_color(sc); spine.set_linewidth(0.5)
        elif plot_type == 'Sky Path':
            if azimuths is None: st.error("Plot Err: No azimuths for Sky Path."); plt.close(fig); return None
            ax.remove(); ax = fig.add_subplot(111, projection='polar', facecolor=fc)
            az_rad = np.deg2rad(azimuths); radius = 90 - altitudes
            t_del=times.jd.max()-times.jd.min(); t_norm=(times.jd-times.jd.min())/(t_del+1e-9) if t_del > 0 else np.zeros_like(times.jd); colors=plt.cm.plasma(t_norm)
            scatter=ax.scatter(az_rad, radius, c=colors, s=15, alpha=0.8, ec='none', label=obj_name); ax.plot(az_rad, radius, color=pc, alpha=0.4, lw=0.8)
            ax.plot(np.linspace(0,2*np.pi,100), np.full(100, 90-min_altitude_deg), color=min_c, ls='--', lw=1.2, label=t.get('graph_min_altitude_label',"Min Alt ({:.0f}°)").format(min_altitude_deg), alpha=0.8)
            if max_altitude_deg < 90: ax.plot(np.linspace(0,2*np.pi,100), np.full(100, 90-max_altitude_deg), color=max_c, ls=':', lw=1.2, label=t.get('graph_max_altitude_label',"Max Alt ({:.0f}°)").format(max_altitude_deg), alpha=0.8)
            ax.set_theta_zero_location('N'); ax.set_theta_direction(-1); ax.set_yticks(np.arange(0,91,15)); ax.set_yticklabels([f"{90-a}°" for a in np.arange(0,91,15)], color=lc); ax.set_ylim(0,90); ax.set_title(t.get('graph_title_sky_path',"Sky Path: {}").format(obj_name), va='bottom', color=tc, fontsize=13, weight='bold', y=1.1)
            ax.grid(True, linestyle=':', alpha=0.5, color=gc); ax.spines['polar'].set_color(sc); ax.spines['polar'].set_linewidth(0.5)
            try:
                cbar = fig.colorbar(scatter, ax=ax, label="Time (UTC)", pad=0.1, shrink=0.7); cbar.set_ticks([0,1])
                if len(times)>0: s_lbl=times[0].to_datetime(timezone.utc).strftime('%H:%M'); e_lbl=times[-1].to_datetime(timezone.utc).strftime('%H:%M'); cbar.ax.set_yticklabels([s_lbl, e_lbl])
                else: cbar.ax.set_yticklabels(['S','E'])
                cbar.set_label("Time (UTC)", color=lc, fontsize=10); cbar.ax.yaxis.set_tick_params(color=lc, labelsize=9); plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=lc); cbar.outline.set_edgecolor(sc); cbar.outline.set_linewidth(0.5)
            except Exception as e: print(f"Warn: Cbar err:{e}")
        else: st.error(f"Plot Err: Unknown type '{plot_type}'?"); plt.close(fig); return None
        leg = ax.legend(loc='lower right', fontsize='small', facecolor=lfc, framealpha=0.8, edgecolor=sc)
        for txt in leg.get_texts(): txt.set_color(lc)
        return fig
    except Exception as e:
        st.error(f"Plot Err: Unexpected: {e}")
        traceback.print_exc()
        if fig: plt.close(fig)
        return None

# --- Main App ---
def main():
    initialize_session_state()

    lang = st.session_state.language
    if lang not in translations:
        lang = 'de'; st.session_state.language = lang
    t = translations.get(lang, translations['en'])

    @st.cache_data
    def cached_load_ongc_data(path):
        print(f"Cache miss: Loading ONGC data from {path}")
        return data_handling.load_ongc_data(path)

    df_catalog_data = cached_load_ongc_data(CATALOG_FILEPATH)

    st.title("Advanced DSO Finder")

    with st.expander(t.get('object_type_glossary_title', "Object Type Glossary")):
        glossary_items = t.get('object_type_glossary', {})
        if glossary_items:
            col1, col2 = st.columns(2); col_index = 0
            sorted_items = sorted(glossary_items.items())
            for abbr, full_name in sorted_items:
                target_col = col1 if col_index % 2 == 0 else col2
                target_col.markdown(f"**{abbr}:** {full_name}")
                col_index += 1
        else: st.info("Glossary not available for the selected language.")

    st.markdown("---")

    with st.sidebar:
        st.header(t.get('settings_header', "Settings"))

        if 'catalog_status_msg' not in st.session_state: st.session_state.catalog_status_msg = ""
        if df_catalog_data is not None:
            new_msg = t.get('info_catalog_loaded', "Catalog loaded: {} objects.").format(len(df_catalog_data))
            if st.session_state.catalog_status_msg != new_msg: st.success(new_msg); st.session_state.catalog_status_msg = new_msg
        else:
            new_msg = "Catalog loading failed. Check file or logs."
            if st.session_state.catalog_status_msg != new_msg: st.error(new_msg); st.session_state.catalog_status_msg = new_msg

        language_options = {'de': 'Deutsch', 'en': 'English', 'fr': 'Français'}; lang_keys = list(language_options.keys())
        try: current_lang_key = lang; current_lang_idx = lang_keys.index(current_lang_key)
        except ValueError: current_lang_idx = 0
        sel_lang_key = st.radio(t.get('language_select_label', "Lang"), lang_keys, format_func=lambda k: language_options[k], key='language_radio', index=current_lang_idx, horizontal=True)
        if sel_lang_key != st.session_state.language: st.session_state.language = sel_lang_key; st.session_state.location_search_status_msg = ""; print(f"Lang changed: {sel_lang_key}. Rerun."); st.rerun()

        with st.expander(t.get('location_expander', "📍 Location"), expanded=True):
            loc_opts = {'Search': t.get('location_option_search', "Search"), 'Manual': t.get('location_option_manual', "Manual")}
            st.radio(t.get('location_select_label', "Loc Method"), list(loc_opts.keys()), format_func=lambda k: loc_opts[k], key="location_choice_key", horizontal=True)
            lat, lon, hgt = None, None, None; loc_valid_tz = False; curr_loc_valid = False
            if st.session_state.location_choice_key == "Manual":
                st.number_input(t.get('location_lat_label', "Lat (°N)"), -90.0, 90.0, step=0.01, format="%.4f", key="manual_lat_val")
                st.number_input(t.get('location_lon_label', "Lon (°E)"), -180.0, 180.0, step=0.01, format="%.4f", key="manual_lon_val")
                st.number_input(t.get('location_elev_label', "Elev (m)"), -500, step=10, format="%d", key="manual_height_val")
                lat = st.session_state.manual_lat_val; lon = st.session_state.manual_lon_val; hgt = st.session_state.manual_height_val
                if isinstance(lat, (int,float)) and isinstance(lon, (int,float)) and isinstance(hgt, (int,float)):
                    loc_valid_tz = True; curr_loc_valid = True; st.session_state.location_is_valid_for_run = True
                    if st.session_state.location_search_success: st.session_state.location_search_success = False; st.session_state.searched_location_name = None; st.session_state.location_search_status_msg = ""
                else: st.warning(t.get('location_error_manual_none', "Manual fields invalid.")); curr_loc_valid = False; st.session_state.location_is_valid_for_run = False
            elif st.session_state.location_choice_key == "Search":
                with st.form("loc_search_form"):
                    st.text_input(t.get('location_search_label', "Loc Name:"), key="location_search_query", placeholder=t.get('location_search_placeholder', "..."))
                    st.number_input(t.get('location_elev_label', "Elev (m)"), -500, step=10, format="%d", key="manual_height_val")
                    search_submitted = st.form_submit_button(t.get('location_search_submit_button', "Find Coords"))
                status_ph = st.empty()
                if st.session_state.location_search_status_msg:
                    if st.session_state.location_search_success: status_ph.success(st.session_state.location_search_status_msg)
                    else: status_ph.error(st.session_state.location_search_status_msg)
                if search_submitted and st.session_state.location_search_query:
                    loc=None; service=None; err=None; query=st.session_state.location_search_query; agent=f"AdvDSOFinder/{random.randint(1000,9999)}"
                    with st.spinner(t.get('spinner_geocoding', "Searching...")):
                        try: geo=Nominatim(user_agent=agent); loc=geo.geocode(query, timeout=10); service="Nominatim"; print("Nominatim success.")
                        except (GeocoderTimedOut, GeocoderServiceError) as e: print(f"Nominatim fail: {e}"); status_ph.info(t.get('location_search_info_fallback', "Fallback 1..."))
                        except Exception as e: print(f"Nominatim error: {e}"); status_ph.info(t.get('location_search_info_fallback', "Fallback 1...")); err=e
                        if not loc:
                           try: fgeo=ArcGIS(timeout=15); loc=fgeo.geocode(query,timeout=15); service="ArcGIS"; print("ArcGIS success.")
                           except (GeocoderTimedOut, GeocoderServiceError) as e2: print(f"ArcGIS fail: {e2}"); status_ph.info(t.get('location_search_info_fallback2', "Fallback 2...")); err=e2 if not err else err
                           except Exception as e2: print(f"ArcGIS error: {e2}"); status_ph.info(t.get('location_search_info_fallback2', "Fallback 2...")); err=e2 if not err else err
                        if not loc:
                           try: fgeo2=Photon(user_agent=agent, timeout=15); loc=fgeo2.geocode(query,timeout=15); service="Photon"; print("Photon success.")
                           except (GeocoderTimedOut, GeocoderServiceError) as e3: print(f"Photon fail: {e3}"); err=e3 if not err else err
                           except Exception as e3: print(f"Photon error: {e3}"); err=e3 if not err else err
                        if loc and service:
                            found_lat=loc.latitude; found_lon=loc.longitude; found_name=loc.address; st.session_state.searched_location_name=found_name; st.session_state.location_search_success=True; st.session_state.manual_lat_val=found_lat; st.session_state.manual_lon_val=found_lon
                            coords=t.get('location_search_coords',"Lat:{:.4f}, Lon:{:.4f}").format(found_lat, found_lon)
                            if service=="Nominatim": st.session_state.location_search_status_msg = f"{t.get('location_search_found','Found(N): {}').format(found_name)}\n({coords})"
                            elif service=="ArcGIS": st.session_state.location_search_status_msg = f"{t.get('location_search_found_fallback','Found(A): {}').format(found_name)}\n({coords})"
                            elif service=="Photon": st.session_state.location_search_status_msg = f"{t.get('location_search_found_fallback2','Found(P): {}').format(found_name)}\n({coords})"
                            status_ph.success(st.session_state.location_search_status_msg)
                            lat=found_lat; lon=found_lon; hgt=st.session_state.manual_height_val; loc_valid_tz=True; curr_loc_valid=True; st.session_state.location_is_valid_for_run=True
                        else:
                            st.session_state.location_search_success=False; st.session_state.searched_location_name=None
                            if err:
                                if isinstance(err, GeocoderTimedOut): st.session_state.location_search_status_msg = t.get('location_search_error_timeout',"Timeout.")
                                elif isinstance(err, GeocoderServiceError): st.session_state.location_search_status_msg = t.get('location_search_error_service',"Svc Err: {}").format(err)
                                else: st.session_state.location_search_status_msg = t.get('location_search_error_fallback2_failed',"All fail: {}").format(err)
                            else: st.session_state.location_search_status_msg = t.get('location_search_error_not_found',"Not found.")
                            status_ph.error(st.session_state.location_search_status_msg); curr_loc_valid=False; st.session_state.location_is_valid_for_run=False
                elif st.session_state.location_search_success:
                    lat=st.session_state.manual_lat_val; lon=st.session_state.manual_lon_val; hgt=st.session_state.manual_height_val; loc_valid_tz=True; curr_loc_valid=True; st.session_state.location_is_valid_for_run=True; status_ph.success(st.session_state.location_search_status_msg)
                else: curr_loc_valid=False; st.session_state.location_is_valid_for_run=False

            st.markdown("---"); tz_msg=""
            if loc_valid_tz and lat is not None and lon is not None:
                if tf:
                    try:
                        found_tz_val = tf.timezone_at(lng=lon, lat=lat)
                        if found_tz_val: pytz.timezone(found_tz_val); st.session_state.selected_timezone = found_tz_val; tz_msg=f"{t.get('timezone_auto_set_label','Detected TZ:')} **{found_tz_val}**"
                        else: st.session_state.selected_timezone='UTC'; tz_msg=f"{t.get('timezone_auto_fail_label','TZ:')} **UTC** ({t.get('timezone_auto_fail_msg','Failed')})"
                    except pytz.UnknownTimeZoneError:
                         st.session_state.selected_timezone='UTC'
                         invalid_tz_name = locals().get('found_tz_val', 'Unknown')
                         tz_msg = t.get('timezone_auto_fail_label','TZ:') + " **UTC** (Invalid: '{}')".format(invalid_tz_name)
                    except Exception as e: print(f"TZ Error: {e}"); st.session_state.selected_timezone='UTC'; tz_msg=f"{t.get('timezone_auto_fail_label','TZ:')} **UTC** (Error)"
                else: tz_msg=f"{t.get('timezone_auto_fail_label','TZ:')} **{INITIAL_TIMEZONE}** (N/A)"; st.session_state.selected_timezone=INITIAL_TIMEZONE
            else: tz_msg=f"{t.get('timezone_auto_fail_label','TZ:')} **{st.session_state.selected_timezone}** (Loc Invalid)"
            st.markdown(tz_msg, unsafe_allow_html=True)

        with st.expander(t.get('time_expander', "⏱️ Time"), expanded=False):
            time_opts = {'Now':t.get('time_option_now',"Now"), 'Specific':t.get('time_option_specific',"Specific Night")}
            st.radio(t.get('time_select_label',"Select Time"), list(time_opts.keys()), format_func=lambda k:time_opts[k], key="time_choice_exp", horizontal=True)
            is_now_time_choice = (st.session_state.time_choice_exp == "Now")
            if is_now_time_choice: st.caption(f"UTC: {Time.now().iso}")
            else: st.date_input(t.get('time_date_select_label',"Date:"), value=st.session_state.selected_date_widget, min_value=date.today()-timedelta(days=365*10), max_value=date.today()+timedelta(days=365*2), key='selected_date_widget')

        with st.expander(t.get('filters_expander', "✨ Filters"), expanded=False):
            st.markdown(t.get('mag_filter_header', "**Magnitude**"))
            mag_opts = {'Bortle Scale':t.get('mag_filter_option_bortle',"Bortle Scale"), 'Manual':t.get('mag_filter_option_manual',"Manual")}
            if st.session_state.mag_filter_mode_exp not in mag_opts: st.session_state.mag_filter_mode_exp = 'Bortle Scale'
            st.radio(t.get('mag_filter_method_label',"Method:"), list(mag_opts.keys()), format_func=lambda k:mag_opts[k], key="mag_filter_mode_exp", horizontal=True)
            st.slider(t.get('mag_filter_bortle_label',"Bortle:"), 1, 9, key='bortle_slider', help=t.get('mag_filter_bortle_help',"..."))
            if st.session_state.mag_filter_mode_exp == "Manual":
                st.slider(t.get('mag_filter_min_mag_label',"Min:"), -5.0, 20.0, step=0.5, format="%.1f", help=t.get('mag_filter_min_mag_help',"..."), key='manual_min_mag_slider')
                st.slider(t.get('mag_filter_max_mag_label',"Max:"), -5.0, 20.0, step=0.5, format="%.1f", help=t.get('mag_filter_max_mag_help',"..."), key='manual_max_mag_slider')
                if isinstance(st.session_state.manual_min_mag_slider,(int,float)) and isinstance(st.session_state.manual_max_mag_slider,(int,float)) and st.session_state.manual_min_mag_slider > st.session_state.manual_max_mag_slider: st.warning(t.get('mag_filter_warning_min_max',"Min > Max!"))

            st.markdown("---"); st.markdown(t.get('min_alt_header', "**Altitude**"))
            min_alt_filt, max_alt_filt = st.session_state.min_alt_slider, st.session_state.max_alt_slider
            if min_alt_filt > max_alt_filt: st.session_state.min_alt_slider = max_alt_filt; min_alt_filt = max_alt_filt
            st.slider(t.get('min_alt_label',"Min Alt (°):"), 0, 90, key='min_alt_slider', step=1)
            st.slider(t.get('max_alt_label',"Max Alt (°):"), 0, 90, key='max_alt_slider', step=1)
            if st.session_state.min_alt_slider > st.session_state.max_alt_slider: st.warning("Min > Max!")

            st.markdown("---"); st.markdown(t.get('moon_warning_header',"**Moon**"))
            st.slider(t.get('moon_warning_label',"Warn Moon > (%):"), 0, 100, key='moon_phase_slider', step=5)

            st.markdown("---"); st.markdown(t.get('object_types_header',"**Types**"))
            all_types_list = []
            if df_catalog_data is not None and not df_catalog_data.empty:
                try:
                    if 'Type' in df_catalog_data.columns: all_types_list = sorted(list(df_catalog_data['Type'].dropna().astype(str).unique()))
                    else: st.warning("Missing 'Type'.")
                except Exception as e: st.warning(f"{t.get('object_types_error_extract','Type Err')}: {e}")
            if all_types_list:
                curr_sel_types = [s for s in st.session_state.object_type_filter_exp if s in all_types_list]
                if curr_sel_types != st.session_state.object_type_filter_exp: st.session_state.object_type_filter_exp = curr_sel_types
                st.multiselect(t.get('object_types_label',"Filter Types:"), all_types_list, default=curr_sel_types, key="object_type_filter_exp")
            else: st.info("No types. Filter disabled."); st.session_state.object_type_filter_exp = []

            st.markdown("---"); st.markdown(t.get('size_filter_header',"**Size**"))
            size_ok = df_catalog_data is not None and 'MajAx' in df_catalog_data.columns and df_catalog_data['MajAx'].notna().any()
            size_disabled = not size_ok
            if size_ok:
                try:
                    vs = df_catalog_data['MajAx'].dropna(); min_s = max(0.1, float(vs.min())) if not vs.empty else 0.1; max_s = float(vs.max()) if not vs.empty else 120.0
                    min_st, max_st = st.session_state.size_arcmin_range; c_min = max(min_s, min(min_st, max_s)); c_max = min(max_s, max(max_st, min_s))
                    if c_min > c_max: c_min=c_max
                    if (c_min, c_max) != st.session_state.size_arcmin_range: st.session_state.size_arcmin_range = (c_min, c_max)
                    step = 0.1 if max_s <= 20 else (0.5 if max_s <= 100 else 1.0)
                    st.slider(
                        label=t.get('size_filter_label',"Size (arcmin):"),
                        min_value=min_s,
                        max_value=max_s,
                        step=step,
                        key='size_arcmin_range',
                        help=t.get('size_filter_help',"..."),
                        disabled=size_disabled
                    )
                except Exception as e: st.error(f"Size slider error: {e}"); size_disabled=True
            else: st.info("Size data missing. Filter disabled."); size_disabled=True
            if size_disabled: st.slider(t.get('size_filter_label',"Size (arcmin):"), 0.0, 1.0, value=(0.0,1.0), key='size_arcmin_range_disabled', disabled=True)

            st.markdown("---"); st.markdown(t.get('direction_filter_header',"**Direction**"))
            all_dir = t.get('direction_option_all',"All"); dir_disp = [all_dir] + CARDINAL_DIRECTIONS; dir_int = [ALL_DIRECTIONS_KEY] + CARDINAL_DIRECTIONS
            curr_dir = st.session_state.selected_peak_direction
            if curr_dir not in dir_int: curr_dir = ALL_DIRECTIONS_KEY; st.session_state.selected_peak_direction=curr_dir
            try: curr_idx = dir_int.index(curr_dir)
            except ValueError: curr_idx = 0
            sel_dir_disp = st.selectbox(t.get('direction_filter_label',"Culmination:"), dir_disp, index=curr_idx, key='direction_selectbox')
            sel_int = ALL_DIRECTIONS_KEY
            if sel_dir_disp != all_dir:
                try: sel_idx = dir_disp.index(sel_dir_disp); sel_int = dir_int[sel_idx]
                except ValueError: sel_int = ALL_DIRECTIONS_KEY
            if sel_int != st.session_state.selected_peak_direction: st.session_state.selected_peak_direction=sel_int

        with st.expander(t.get('results_options_expander',"⚙️ Results"), expanded=False):
            max_sl = len(df_catalog_data) if df_catalog_data is not None and not df_catalog_data.empty else 50; min_sl=5; actual_max=max(min_sl,max_sl); s_disabled=actual_max<=min_sl
            def_num = st.session_state.get('num_objects_slider',20); cl_def = max(min_sl, min(def_num, actual_max))
            if cl_def != def_num: st.session_state.num_objects_slider = cl_def
            st.slider(t.get('results_options_max_objects_label',"Max Objects:"), min_sl, actual_max, step=1, key='num_objects_slider', disabled=s_disabled)
            sort_map = {'Duration & Altitude':t.get('results_options_sort_duration',"Duration & Altitude"), 'Brightness':t.get('results_options_sort_magnitude',"Brightness")}
            if st.session_state.sort_method not in sort_map: st.session_state.sort_method='Duration & Altitude'
            st.radio(t.get('results_options_sort_method_label',"Sort By:"), list(sort_map.keys()), format_func=lambda k:sort_map[k], key='sort_method', horizontal=True)

        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**{t.get('bug_report', 'Found a bug?')}**") # Corrected call
        bug_email = "debrun2005@gmail.com"
        bug_subject = urllib.parse.quote("Bug Report: Advanced DSO Finder")
        bug_body = urllib.parse.quote(t.get('bug_report_body', "\n\n(Describe bug and steps to reproduce)"))
        bug_link = f"mailto:{bug_email}?subject={bug_subject}&body={bug_body}"
        st.sidebar.link_button(t.get('bug_report_button', '🐞 Report Issue'), bug_link)


    # --- Main Area ---
    st.subheader(t.get('search_params_header', "Search Parameters"))
    p1, p2 = st.columns(2)
    loc_disp = t.get('location_error', "Location Error: {}").format("Not Set"); observer_run = None
    if st.session_state.location_is_valid_for_run:
        lat, lon, hgt, tz = st.session_state.manual_lat_val, st.session_state.manual_lon_val, st.session_state.manual_height_val, st.session_state.selected_timezone
        try:
            observer_run = Observer(latitude=lat*u.deg, longitude=lon*u.deg, elevation=hgt*u.m, timezone=tz)
            if st.session_state.location_choice_key=="Manual": loc_disp = t.get('location_manual_display',"Manual ({:.4f},{:.4f})").format(lat,lon)
            elif st.session_state.searched_location_name: loc_disp = t.get('location_search_display',"Searched: {} ({:.4f},{:.4f})").format(st.session_state.searched_location_name,lat,lon)
            else: loc_disp = f"Lat:{lat:.4f}, Lon:{lon:.4f}"
        except Exception as e: loc_disp=t.get('location_error',"Loc Err: {}").format(f"Obs Err: {e}"); st.session_state.location_is_valid_for_run=False; observer_run=None
    p1.markdown(t.get('search_params_location',"📍 Loc: {}").format(loc_disp))
    time_disp = ""; is_now_main = (st.session_state.time_choice_exp == "Now")
    if is_now_main:
        ref_time = Time.now()
        try: local_now, tz_now = get_local_time_str(ref_time, st.session_state.selected_timezone); time_disp = t.get('search_params_time_now',"Now (from {} UTC)").format(f"{local_now} {tz_now}")
        except Exception: time_disp = t.get('search_params_time_now',"Now (from {} UTC)").format(ref_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M:%S') + " UTC")
    else: sel_date = st.session_state.selected_date_widget; ref_time = Time(datetime.combine(sel_date, time(12,0)), scale='utc'); time_disp = t.get('search_params_time_specific',"Night after {}").format(sel_date.strftime('%Y-%m-%d'))
    p1.markdown(t.get('search_params_time',"⏱️ Time: {}").format(time_disp))
    mag_disp = ""; min_mag_filt, max_mag_filt = -np.inf, np.inf
    if st.session_state.mag_filter_mode_exp == "Bortle Scale":
        max_mag_filt = astro_calculations.get_magnitude_limit(st.session_state.bortle_slider)
        mag_disp = t.get('search_params_filter_mag_bortle',"Bortle {} (<= {:.1f})").format(st.session_state.bortle_slider, max_mag_filt)
    else: min_mag_filt, max_mag_filt = st.session_state.manual_min_mag_slider, st.session_state.manual_max_mag_slider; mag_disp = t.get('search_params_filter_mag_manual',"Manual ({:.1f}-{:.1f})").format(min_mag_filt, max_mag_filt)
    p2.markdown(t.get('search_params_filter_mag',"✨ Mag: {}").format(mag_disp))
    min_alt_disp, max_alt_disp = st.session_state.min_alt_slider, st.session_state.max_alt_slider; sel_types = st.session_state.object_type_filter_exp
    types_str = ', '.join(sel_types) if sel_types else t.get('search_params_types_all',"All")
    p2.markdown(t.get('search_params_filter_alt_types',"🔭 Alt:{}-{}°,Type:{}").format(min_alt_disp, max_alt_disp, types_str))
    s_min, s_max = st.session_state.size_arcmin_range
    p2.markdown(t.get('search_params_filter_size',"📐 Size:{:.1f}-{:.1f}'").format(s_min, s_max))
    direction_disp = st.session_state.selected_peak_direction
    if direction_disp == ALL_DIRECTIONS_KEY: direction_disp = t.get('search_params_direction_all',"All")
    p2.markdown(t.get('search_params_filter_direction',"🧭 Dir@Max: {}").format(direction_disp))

    st.markdown("---")
    find_clicked = st.button(t.get('find_button_label', "🔭 Find Objects"), key="find_button", disabled=(df_catalog_data is None or not st.session_state.location_is_valid_for_run))
    if not st.session_state.location_is_valid_for_run and df_catalog_data is not None: st.warning(t.get('info_initial_prompt', "Set Location!"))

    results_ph = st.container()

    if find_clicked:
        st.session_state.find_button_pressed=True; st.session_state.show_plot=False; st.session_state.show_custom_plot=False; st.session_state.active_result_plot_data=None; st.session_state.custom_target_plot_data=None; st.session_state.last_results=[]; st.session_state.window_start_time=None; st.session_state.window_end_time=None
        if observer_run and df_catalog_data is not None:
            with st.spinner(t.get('spinner_searching',"Searching...")):
                try:
                    start_t, end_t, win_stat = astro_calculations.get_observable_window(observer_run, ref_time, is_now_main, t)
                    results_ph.info(win_stat); st.session_state.window_start_time=start_t; st.session_state.window_end_time=end_t
                    if start_t and end_t and start_t < end_t:
                        t_res = 5 * u.minute; obs_times = Time(np.arange(start_t.jd, end_t.jd, t_res.to(u.day).value), format='jd', scale='utc')
                        if len(obs_times) < 2: results_ph.warning("Win short.")
                        filt_df = df_catalog_data.copy()
                        filt_df = filt_df[(filt_df['Mag'] >= min_mag_filt) & (filt_df['Mag'] <= max_mag_filt)]
                        if sel_types: filt_df = filt_df[filt_df['Type'].isin(sel_types)]
                        size_ok = df_catalog_data is not None and 'MajAx' in df_catalog_data.columns and df_catalog_data['MajAx'].notna().any()
                        if size_ok: filt_df = filt_df.dropna(subset=['MajAx']); filt_df = filt_df[(filt_df['MajAx'] >= s_min) & (filt_df['MajAx'] <= s_max)]
                        if filt_df.empty: results_ph.warning(t.get('warning_no_objects_found',"No objects found.") + " (init filt)"); st.session_state.last_results=[]
                        else:
                            min_alt_search = st.session_state.min_alt_slider * u.deg
                            found = astro_calculations.find_observable_objects(observer_run.location, obs_times, min_alt_search, filt_df, t)
                            final = []
                            sel_dir = st.session_state.selected_peak_direction; max_alt_filter_loop = st.session_state.max_alt_slider
                            for obj in found:
                                if obj.get('Max Altitude (°)',-999) > max_alt_filter_loop: continue
                                if sel_dir != ALL_DIRECTIONS_KEY and obj.get('Direction at Max') != sel_dir: continue
                                final.append(obj)
                            sort_k = st.session_state.sort_method
                            if sort_k == 'Brightness': final.sort(key=lambda x: x.get('Magnitude', float('inf')) if x.get('Magnitude') is not None else float('inf'))
                            else: final.sort(key=lambda x: (x.get('Max Cont. Duration (h)',0), x.get('Max Altitude (°)',0)), reverse=True)
                            num_show = st.session_state.num_objects_slider; st.session_state.last_results = final[:num_show]
                            if not final: results_ph.warning(t.get('warning_no_objects_found',"No objects found."))
                            else: results_ph.success(t.get('success_objects_found',"{} found.").format(len(final))); sort_msg_key = 'info_showing_list_duration' if sort_k != 'Brightness' else 'info_showing_list_magnitude'; results_ph.info(t[sort_msg_key].format(len(st.session_state.last_results)))
                    else: results_ph.error(t.get('error_no_window',"No valid window.") + " Cannot search."); st.session_state.last_results=[]
                except Exception as e: results_ph.error(t.get('error_search_unexpected',"Error:") + f"\n```\n{e}\n```"); traceback.print_exc(); st.session_state.last_results=[]
        else:
            if df_catalog_data is None: results_ph.error("Err: No catalog.")
            if not observer_run: results_ph.error("Err: Invalid location.")
            st.session_state.last_results=[]

    if st.session_state.last_results:
        results_data = st.session_state.last_results; results_ph.subheader(t.get('results_list_header',"Results"))
        win_start = st.session_state.get('window_start_time'); win_end = st.session_state.get('window_end_time'); obs_exists = observer_run is not None
        if obs_exists and isinstance(win_start, Time) and isinstance(win_end, Time):
            mid_time = win_start + (win_end - win_start) / 2
            try:
                illum=moon_illumination(mid_time); moon_pct=illum*100; moon_svg=create_moon_phase_svg(illum, size=50)
                mc1, mc2 = results_ph.columns([1,3])
                with mc1: st.markdown(moon_svg, unsafe_allow_html=True)
                with mc2:
                    st.metric(label=t.get('moon_metric_label',"Moon Illum."), value=f"{moon_pct:.0f}%")
                    moon_thresh=st.session_state.moon_phase_slider
                    if moon_pct > moon_thresh:
                        st.warning(t.get('moon_warning_message',"Warn: Moon bright!").format(moon_pct, moon_thresh))
            except Exception as e: results_ph.warning(t.get('moon_phase_error',"Moon err: {}").format(e))
        elif st.session_state.find_button_pressed: results_ph.info("Cannot calc moon phase.")
        plot_map = {'Sky Path': t.get('graph_type_sky_path',"Sky Path"), 'Altitude Plot': t.get('graph_type_alt_time',"Altitude Plot")}
        if st.session_state.plot_type_selection not in plot_map: st.session_state.plot_type_selection='Sky Path'
        results_ph.radio(t.get('graph_type_label',"Plot Type:"), list(plot_map.keys()), format_func=lambda k:plot_map[k], key='plot_type_selection', horizontal=True)

        for i, obj in enumerate(results_data):
            # Get object details
            name = obj.get('Name','?')
            type_obj = obj.get('Type','?')
            mag = obj.get('Magnitude') # This might be None or a float

            # Create mag_s: Always a string, either formatted number or 'N/A'
            mag_s = f"{mag:.1f}" if mag is not None else t.get('magnitude_unknown', 'N/A')

            # Use a known-safe template directly to avoid issues with potentially
            # incorrect format specifiers in the translation file for 'results_expander_title'.
            safe_title_template = "{} ({}) - Mag: {}"
            title = safe_title_template.format(name, type_obj, mag_s)

            is_exp = (st.session_state.expanded_object_name == name)
            obj_c = results_ph.container()
            with obj_c.expander(title, expanded=is_exp):
                c1,c2,c3 = st.columns([2,2,1])
                c1.markdown(t.get('results_coords_header',"**Details:**")); c1.markdown(f"**{t.get('results_export_constellation','Const')}:** {obj.get('Constellation','?')}"); size=obj.get('Size (arcmin)'); c1.markdown(f"**{t.get('results_size_label','Size:')}** {t.get('results_size_value','{:.1f}\'').format(size) if size is not None else '?'}"); c1.markdown(f"**RA:** {obj.get('RA','?')}"); c1.markdown(f"**Dec:** {obj.get('Dec','?')}")
                c2.markdown(t.get('results_max_alt_header',"**Max Alt:**")); max_a_disp=obj.get('Max Altitude (°)',0); az=obj.get('Azimuth at Max (°)',0); direction=obj.get('Direction at Max','?')
                # --- Correction Start (Line 524 equivalent) ---
                # Use known-safe format strings directly for az_fmt and dir_fmt
                safe_az_fmt_template = "(Az:{:.1f}°)"
                safe_dir_fmt_template = ",Dir:{}"
                az_fmt = safe_az_fmt_template.format(az)
                dir_fmt = safe_dir_fmt_template.format(direction)
                # --- Correction End ---
                c2.markdown(f"**{max_a_disp:.1f}°** {az_fmt}{dir_fmt}") # Combine the formatted parts
                c2.markdown(t.get('results_best_time_header',"**Best Time (Loc):**")); peak_t=obj.get('Time at Max (UTC)'); local_t, local_tz = get_local_time_str(peak_t, st.session_state.selected_timezone); c2.markdown(f"{local_t} ({local_tz})")
                c2.markdown(t.get('results_cont_duration_header',"**Max Dur:**")); dur=obj.get('Max Cont. Duration (h)',0); c2.markdown(t.get('results_duration_value',"{:.1f}h").format(dur))
                gq=urllib.parse.quote_plus(f"{name} astronomy"); gu=f"https://google.com/search?q={gq}"; c3.markdown(f"[{t.get('google_link_text','Google')}]({gu})", unsafe_allow_html=True)
                sq=urllib.parse.quote_plus(name); su=f"http://simbad.cds.unistra.fr/simbad/sim-basic?Ident={sq}"; c3.markdown(f"[{t.get('simbad_link_text','SIMBAD')}]({su})", unsafe_allow_html=True)
                plot_key = f"plot_{name}_{i}"
                if st.button(t.get('results_graph_button',"📈 Plot"), key=plot_key): st.session_state.plot_object_name=name; st.session_state.active_result_plot_data=obj; st.session_state.show_plot=True; st.session_state.show_custom_plot=False; st.session_state.expanded_object_name=name; st.rerun()
                if st.session_state.show_plot and st.session_state.plot_object_name == name:
                    p_data=obj; min_a_plot=st.session_state.min_alt_slider; max_a_plot=st.session_state.max_alt_slider; st.markdown("---")
                    with st.spinner(t.get('results_spinner_plotting',"Plotting...")):
                        try: fig = create_plot(p_data, min_a_plot, max_a_plot, st.session_state.plot_type_selection, t)
                        except Exception as e: st.error(f"Plot Err:{e}"); traceback.print_exc(); fig=None
                        if fig:
                            st.pyplot(fig)
                            close_key=f"close_{name}_{i}"
                            if st.button(t.get('results_close_graph_button',"Close"), key=close_key):
                                st.session_state.show_plot=False
                                st.session_state.active_result_plot_data=None
                                st.session_state.expanded_object_name=None
                                st.rerun()
                        else: st.error(t.get('results_graph_not_created',"Plot failed."))
        if results_data:
            csv_ph = results_ph.container()
            try:
                export_data=[]
                for obj_csv in results_data:
                    peak_utc = obj_csv.get('Time at Max (UTC)')
                    local_t_csv, _ = get_local_time_str(peak_utc, st.session_state.selected_timezone)
                    export_data.append({
                        t.get('results_export_name', "Name"): obj_csv.get('Name', 'N/A'),
                        t.get('results_export_type', "Type"): obj_csv.get('Type', 'N/A'),
                        t.get('results_export_constellation', "Constellation"): obj_csv.get('Constellation', 'N/A'),
                        t.get('results_export_mag', "Magnitude"): obj_csv.get('Magnitude'),
                        t.get('results_export_size', "Size (arcmin)"): obj_csv.get('Size (arcmin)'),
                        t.get('results_export_ra', "RA"): obj_csv.get('RA', 'N/A'),
                        t.get('results_export_dec', "Dec"): obj_csv.get('Dec', 'N/A'),
                        t.get('results_export_max_alt', "Max Altitude (°)"): obj_csv.get('Max Altitude (°)', 0),
                        t.get('results_export_az_at_max', "Azimuth at Max (°)"): obj_csv.get('Azimuth at Max (°)', 0),
                        t.get('results_export_direction_at_max', "Direction at Max"): obj_csv.get('Direction at Max', 'N/A'),
                        t.get('results_export_time_max_utc', "Time at Max (UTC)"): peak_utc.iso if peak_utc else "N/A",
                        t.get('results_export_time_max_local', "Time at Max (Local TZ)"): local_t_csv,
                        t.get('results_export_cont_duration', "Max Cont Duration (h)"): obj_csv.get('Max Cont. Duration (h)', 0)
                    })
                df_exp = pd.DataFrame(export_data); dec_sep=',' if lang=='de' else '.'; csv=df_exp.to_csv(index=False, sep=';', encoding='utf-8-sig', decimal=dec_sep); now=datetime.now().strftime("%Y%m%d_%H%M"); fname=t.get('results_csv_filename',"dso_list_{}.csv").format(now)
                csv_ph.download_button(label=t.get('results_save_csv_button',"💾 CSV"), data=csv, file_name=fname, mime='text/csv', key='csv_download')
            except Exception as e: csv_ph.error(t.get('results_csv_export_error',"CSV Export Err: {}").format(e))

    elif st.session_state.find_button_pressed: results_ph.info(t.get('warning_no_objects_found',"No objects found."))

    # Custom Target Plotting
    st.markdown("---")
    with st.expander(t.get('custom_target_expander',"Plot Custom Target")):
        with st.form("custom_target"):
             st.text_input(t.get('custom_target_ra_label',"RA:"), key="custom_target_ra", placeholder=t.get('custom_target_ra_placeholder',"..."))
             st.text_input(t.get('custom_target_dec_label',"Dec:"), key="custom_target_dec", placeholder=t.get('custom_target_dec_placeholder',"..."))
             st.text_input(t.get('custom_target_name_label',"Name:"), key="custom_target_name", placeholder="Target")
             custom_submitted = st.form_submit_button(t.get('custom_target_button',"Plot"))
        err_ph = st.empty(); plot_area = st.container()
        if custom_submitted:
             st.session_state.show_plot=False; st.session_state.show_custom_plot=False; st.session_state.custom_target_plot_data=None; st.session_state.custom_target_error=""
             c_ra=st.session_state.custom_target_ra; c_dec=st.session_state.custom_target_dec; c_name=st.session_state.custom_target_name or t.get('custom_target_name_label',"Target").replace(":","")
             win_s=st.session_state.get('window_start_time'); win_e=st.session_state.get('window_end_time'); obs_exists=observer_run is not None
             if not c_ra or not c_dec: st.session_state.custom_target_error = t.get('custom_target_error_coords',"Invalid Coords."); err_ph.error(st.session_state.custom_target_error)
             elif not obs_exists or not isinstance(win_s, Time) or not isinstance(win_e, Time): st.session_state.custom_target_error = t.get('custom_target_error_window',"Invalid window/loc."); err_ph.error(st.session_state.custom_target_error)
             else:
                 try:
                     c_coord=SkyCoord(ra=c_ra, dec=c_dec, unit=(u.hourangle, u.deg))
                     if win_s < win_e: t_res=5*u.minute; obs_times_c = Time(np.arange(win_s.jd, win_e.jd, t_res.to(u.day).value),format='jd',scale='utc')
                     else: raise ValueError("Invalid window.")
                     if len(obs_times_c)<2: raise ValueError("Window too short.")
                     altaz_c = AltAz(obstime=obs_times_c, location=observer_run.location); c_altazs=c_coord.transform_to(altaz_c); c_alts=c_altazs.alt.to(u.deg).value; c_azs=c_altazs.az.to(u.deg).value
                     st.session_state.custom_target_plot_data = {'Name':c_name, 'altitudes':c_alts, 'azimuths':c_azs, 'times':obs_times_c}; st.session_state.show_custom_plot=True; st.session_state.custom_target_error=""; st.rerun()
                 except ValueError as e: st.session_state.custom_target_error=f"{t.get('custom_target_error_coords','Invalid Coords.')} ({e})"; err_ph.error(st.session_state.custom_target_error)
                 except Exception as e: st.session_state.custom_target_error=f"Custom Plot Err:{e}"; err_ph.error(st.session_state.custom_target_error); traceback.print_exc()
        if st.session_state.show_custom_plot and st.session_state.custom_target_plot_data:
            c_data=st.session_state.custom_target_plot_data; min_a_cust=st.session_state.min_alt_slider; max_a_cust=st.session_state.max_alt_slider
            with plot_area:
                 st.markdown("---")
                 with st.spinner(t.get('results_spinner_plotting',"Plotting...")):
                     try: fig=create_plot(c_data, min_a_cust, max_a_cust, st.session_state.plot_type_selection, t)
                     except Exception as e: st.error(f"Plot Err:{e}"); traceback.print_exc(); fig=None
                     # Corrected syntax for inner if block
                     if fig:
                         st.pyplot(fig)
                         if st.button(t.get('results_close_graph_button',"Close"), key="close_custom"):
                             st.session_state.show_custom_plot=False
                             st.session_state.custom_target_plot_data=None
                             st.rerun()
                     # Display error only if plot creation failed but form was submitted
                     elif custom_submitted:
                          st.error(t.get('results_graph_not_created',"Plot failed."))
        elif st.session_state.custom_target_error: err_ph.error(st.session_state.custom_target_error)

    # Donation Link (Integrated)
    st.markdown("---")
    st.markdown(t.get('donation_text', "Like the app? [Support the development on Ko-fi ☕](https://ko-fi.com/advanceddsofinder)"), unsafe_allow_html=True)


# Run App
if __name__ == "__main__":
    main()
