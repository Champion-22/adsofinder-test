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
import numpy as np # Import numpy

# --- Library Imports ---
try:
    from astropy.time import Time
    import astropy.units as u
    from astropy.coordinates import SkyCoord, AltAz
    from astroplan import Observer
    from astroplan.moon import moon_illumination
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pytz
    from timezonefinder import TimezoneFinder
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
    # Import specific functions or the whole module as needed
    # --- Correction Start: Import default cosmology parameters directly ---
    from astro_calculations import (
        CARDINAL_DIRECTIONS, calculate_lcdm_distances,
        convert_mpc_to_gly, format_large_number, get_lookback_comparison,
        get_comoving_comparison, convert_mpc_to_km, convert_km_to_ly,
        convert_km_to_au, convert_km_to_ls,
        H0_DEFAULT, OMEGA_M_DEFAULT, OMEGA_LAMBDA_DEFAULT # Import constants
    )
    # --- Correction End ---
    # Note: translations 't' will be passed as an argument
except ModuleNotFoundError as e:
    st.error(f"Module Not Found Error in UI module: {e}. Ensure astro_calculations.py is present.")
    st.stop()
except ImportError as e:
    st.error(f"Import Error from astro_calculations in UI module: {e}. Check function names.")
    st.stop()


# --- Constants ---
ALL_DIRECTIONS_KEY = 'All' # Define locally or import if defined elsewhere centrally

# --- UI Helper Functions (Plotting, Formatting, SVG) ---

def create_moon_phase_svg(illumination: float, size: int = 100) -> str:
    """Erstellt eine SVG-Darstellung der Mondphase."""
    if not 0 <= illumination <= 1: print(f"Warn: Invalid moon illum ({illumination}). Clamping."); illumination = max(0.0, min(1.0, illumination))
    radius = size / 2; cx = cy = radius
    # Farben aus Streamlit-Theme verwenden
    light_color = "var(--text-color, #e0e0e0)"; dark_color = "var(--secondary-background-color, #333333)"
    svg = f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
    # Dunkler Hintergrundkreis
    svg += f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{dark_color}"/>'

    if illumination < 0.01: # Neumond
        pass # Nur dunkler Kreis
    elif illumination > 0.99: # Vollmond
        svg += f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{light_color}"/>' # Heller Kreis
    else:
        # Berechne x-Position des Terminators
        x = radius * (illumination * 2 - 1)
        rx = abs(x) # Radius der Ellipse für den Terminator

        if illumination <= 0.5: # Zunehmende Sichel / Erstes Viertel
            # Pfad: Oben Mitte -> Bogen unten Mitte (Ellipse) -> Bogen oben Mitte (Kreis)
            large_arc_ellipse = 0; sweep_ellipse = 1 # Ellipse von oben nach unten rechts
            large_arc_circle = 0; sweep_circle = 1  # Kreisbogen für rechte Kante
            d=f"M {cx},{cy-radius} A {rx},{radius} 0 {large_arc_ellipse},{sweep_ellipse} {cx},{cy+radius} A {radius},{radius} 0 {large_arc_circle},{sweep_circle} {cx},{cy-radius} Z"
        else: # Zunehmender / Abnehmender Buckelmond
             # Pfad: Oben Mitte -> Bogen unten Mitte (Kreis) -> Bogen oben Mitte (Ellipse)
            large_arc_circle = 1; sweep_circle = 1  # Kreisbogen für linke Kante
            large_arc_ellipse = 0; sweep_ellipse = 1 # Ellipse von unten nach oben rechts
            d=f"M {cx},{cy-radius} A {radius},{radius} 0 {large_arc_circle},{sweep_circle} {cx},{cy+radius} A {rx},{radius} 0 {large_arc_ellipse},{sweep_ellipse} {cx},{cy-radius} Z"
        svg += f'<path d="{d}" fill="{light_color}"/>' # Beleuchteten Teil zeichnen
    svg += '</svg>'
    return svg

def get_local_time_str(utc_time: Time | None, timezone_str: str) -> tuple[str, str]:
    """Konvertiert ein UTC Time Objekt in einen lokalen Zeitstring, oder gibt "N/A" zurück."""
    if utc_time is None: return "N/A", "N/A"
    if not isinstance(utc_time, Time): print(f"Err: utc_time type {type(utc_time)}"); return "N/A", "N/A"
    if not isinstance(timezone_str, str) or not timezone_str: print(f"Err: tz type '{timezone_str}'"); return "N/A", "N/A"
    try:
        local_tz = pytz.timezone(timezone_str)
        utc_dt = utc_time.to_datetime(timezone.utc)
        local_dt = utc_dt.astimezone(local_tz)
        local_time_str = local_dt.strftime('%Y-%m-%d %H:%M:%S')
        tz_display_name = local_dt.tzname()
        if not tz_display_name: tz_display_name = local_tz.zone # Fallback, falls tzname() None ist
        return local_time_str, tz_display_name
    except pytz.exceptions.UnknownTimeZoneError:
        print(f"Err: Unknown TZ '{timezone_str}'.")
        return utc_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), "UTC (TZ Err)"
    except Exception as e:
        print(f"Err converting time: {e}")
        traceback.print_exc()
        return utc_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), "UTC (Conv Err)"

def create_plot(plot_data: dict, min_altitude_deg: float, max_altitude_deg: float, plot_type: str, t: dict) -> plt.Figure | None:
    """Erstellt entweder ein Höhen-Zeit-Diagramm oder ein Himmelspfad-Diagramm (Alt/Az)."""
    fig = None
    try:
        # --- Datenvalidierung ---
        if not isinstance(plot_data, dict):
            st.error(t.get('plot_error_invalid_data_type', "Plot Fehler: Ungültiger plot_data Typ."))
            return None
        times = plot_data.get('times')
        altitudes = plot_data.get('altitudes')
        azimuths = plot_data.get('azimuths') # Kann None sein für Höhen-Zeit
        obj_name = plot_data.get('Name', t.get('plot_object_default_name', 'Objekt'))

        if not isinstance(times, Time) or not isinstance(altitudes, np.ndarray):
            st.error(t.get('plot_error_missing_time_alt', "Plot Fehler: Fehlende oder ungültige Zeit/Höhen-Daten."))
            return None
        if plot_type == 'Sky Path' and not isinstance(azimuths, np.ndarray):
            st.error(t.get('plot_error_missing_azimuth', "Plot Fehler: Fehlende Azimut-Daten für Himmelspfad."))
            return None
        if len(times) != len(altitudes) or (azimuths is not None and len(times) != len(azimuths)):
            st.error(t.get('plot_error_mismatched_lengths', "Plot Fehler: Zeit-, Höhen- und Azimut-Arrays haben unterschiedliche Längen."))
            return None
        if len(times) < 1:
            st.error(t.get('plot_error_no_data_points', "Plot Fehler: Nicht genügend Datenpunkte zum Plotten."))
            return None

        plot_times = times.plot_date # Konvertiert Astropy Time für matplotlib

        # --- Theming ---
        try:
            theme_opts = st.get_option("theme.base")
            is_dark_theme = (theme_opts == "dark")
        except Exception:
            print("Warnung: Streamlit Theme nicht erkannt. Nehme helles Theme an.")
            is_dark_theme = False

        if is_dark_theme:
            plt.style.use('dark_background')
            fc = '#0E1117'; pc = 'deepskyblue'; gc = '#444'; lc = '#CCC'; tc = '#FFF'; lfc = '#262730'; min_c = 'tomato'; max_c = 'orange'; sc = '#555'
        else:
            plt.style.use('default')
            fc = '#FFFFFF'; pc = 'dodgerblue'; gc = 'darkgray'; lc = '#333'; tc = '#000'; lfc = '#F0F0F0'; min_c = 'red'; max_c = 'darkorange'; sc = '#888'

        # --- Plot Erstellung ---
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=fc, constrained_layout=True)
        ax.set_facecolor(fc)

        if plot_type == 'Altitude Plot':
            ax.plot(plot_times, altitudes, color=pc, alpha=0.9, lw=1.5, label=obj_name)
            ax.axhline(min_altitude_deg, color=min_c, linestyle='--', linewidth=1.2, label=t.get('graph_min_altitude_label', "Min Höhe ({:.0f}°)").format(min_altitude_deg), alpha=0.8)
            if max_altitude_deg < 90:
                 ax.axhline(max_altitude_deg, color=max_c, linestyle=':', linewidth=1.2, label=t.get('graph_max_altitude_label', "Max Höhe ({:.0f}°)").format(max_altitude_deg), alpha=0.8)

            ax.set_xlabel(t.get('graph_xlabel_time', "Zeit (UTC)"), color=lc, fontsize=11)
            ax.set_ylabel(t.get('graph_ylabel_alt', "Höhe (°)"), color=lc, fontsize=11)
            ax.set_title(t.get('graph_title_alt_time', "Höhe vs. Zeit: {}").format(obj_name), color=tc, fontsize=13, weight='bold')
            ax.set_ylim(0, 90)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            fig.autofmt_xdate(rotation=30)
            ax.grid(True, linestyle='-', alpha=0.5, color=gc)
            ax.tick_params(axis='x', colors=lc); ax.tick_params(axis='y', colors=lc)
            for spine in ax.spines.values(): spine.set_color(sc); spine.set_linewidth(0.5)

        elif plot_type == 'Sky Path':
            if azimuths is None:
                 st.error(t.get('plot_error_missing_azimuth', "Plot Fehler: Fehlende Azimut-Daten für Himmelspfad."))
                 plt.close(fig); return None

            ax.remove() # Standard-Achsen entfernen
            ax = fig.add_subplot(111, projection='polar', facecolor=fc)

            az_rad = np.deg2rad(azimuths)
            radius = 90 - altitudes # Zenitdistanz

            # Punkte nach Zeit einfärben
            time_delta = times.jd.max() - times.jd.min()
            time_normalized = (times.jd - times.jd.min()) / (time_delta + 1e-9) if time_delta > 0 else np.zeros_like(times.jd)
            colors = plt.cm.viridis(time_normalized)

            scatter = ax.scatter(az_rad, radius, c=colors, s=15, alpha=0.8, edgecolors='none', label=obj_name)
            ax.plot(az_rad, radius, color=pc, alpha=0.4, lw=0.8) # Punkte verbinden

            # Kreise für Min/Max Höhe
            ax.plot(np.linspace(0, 2 * np.pi, 100), np.full(100, 90 - min_altitude_deg), color=min_c, linestyle='--', linewidth=1.2, label=t.get('graph_min_altitude_label',"Min Höhe ({:.0f}°)").format(min_altitude_deg), alpha=0.8)
            if max_altitude_deg < 90:
                ax.plot(np.linspace(0, 2 * np.pi, 100), np.full(100, 90 - max_altitude_deg), color=max_c, linestyle=':', linewidth=1.2, label=t.get('graph_max_altitude_label',"Max Höhe ({:.0f}°)").format(max_altitude_deg), alpha=0.8)

            # Polardiagramm formatieren
            ax.set_theta_zero_location('N'); ax.set_theta_direction(-1) # Nord oben, Azimut im Uhrzeigersinn
            ax.set_yticks(np.arange(0, 91, 15)); ax.set_yticklabels([f"{90-a}°" for a in np.arange(0, 91, 15)], color=lc) # Höhenlinien
            ax.set_ylim(0, 90) # Radius 0 = Zenit, 90 = Horizont
            ax.set_title(t.get('graph_title_sky_path',"Himmelspfad: {}").format(obj_name), va='bottom', color=tc, fontsize=13, weight='bold', y=1.1)
            ax.grid(True, linestyle=':', alpha=0.5, color=gc)
            ax.spines['polar'].set_color(sc); ax.spines['polar'].set_linewidth(0.5)

            # Farbleiste für Zeit
            try:
                cbar = fig.colorbar(scatter, ax=ax, label=t.get('graph_colorbar_label', "Zeit (UTC)"), pad=0.1, shrink=0.7)
                cbar.set_ticks([0, 1])
                if len(times) > 0:
                    start_label = times[0].to_datetime(timezone.utc).strftime('%H:%M')
                    end_label = times[-1].to_datetime(timezone.utc).strftime('%H:%M')
                    cbar.ax.set_yticklabels([start_label, end_label])
                else: cbar.ax.set_yticklabels(['Start', 'Ende'])
                cbar.set_label(t.get('graph_colorbar_label', "Zeit (UTC)"), color=lc, fontsize=10)
                cbar.ax.yaxis.set_tick_params(color=lc, labelsize=9)
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=lc)
                cbar.outline.set_edgecolor(sc); cbar.outline.set_linewidth(0.5)
            except Exception as e: print(f"Warnung: Farbleiste konnte nicht erstellt werden: {e}")

        else:
            st.error(t.get('plot_error_unknown_type', "Plot Fehler: Unbekannter Plot-Typ: '{}'").format(plot_type))
            plt.close(fig); return None

        # --- Legende ---
        leg = ax.legend(loc='lower right', fontsize='small', facecolor=lfc, framealpha=0.8, edgecolor=sc)
        for text in leg.get_texts(): text.set_color(lc)

        return fig

    except Exception as e:
        st.error(t.get('plot_error_unexpected', "Plot Fehler: Unerwarteter Fehler bei Plot-Erstellung: {}").format(e))
        traceback.print_exc()
        if fig: plt.close(fig)
        return None


# --- Main UI Component Functions ---

def create_sidebar(t: dict, df_catalog_data: pd.DataFrame | None, tf: TimezoneFinder | None) -> None:
    """Erstellt die Sidebar UI Elemente."""
    with st.sidebar:
        st.header(t.get('settings_header', "Einstellungen"))

        # Katalog Status
        if 'catalog_status_msg' not in st.session_state: st.session_state.catalog_status_msg = ""
        if df_catalog_data is not None:
            new_msg = t.get('info_catalog_loaded', "Katalog geladen: {} Objekte.").format(len(df_catalog_data))
            if st.session_state.catalog_status_msg != new_msg:
                st.success(new_msg); st.session_state.catalog_status_msg = new_msg
        else:
            new_msg = t.get('error_catalog_load_failed', "Katalog konnte nicht geladen werden. Datei prüfen.")
            if st.session_state.catalog_status_msg != new_msg:
                st.error(new_msg); st.session_state.catalog_status_msg = new_msg

        # Sprachauswahl
        language_options = {'de': 'Deutsch', 'en': 'English', 'fr': 'Français'}
        lang_keys = list(language_options.keys())
        try:
            current_lang_key = st.session_state.language
            current_lang_idx = lang_keys.index(current_lang_key)
        except ValueError: current_lang_idx = 0

        selected_lang_key = st.radio(
            t.get('language_select_label', "Sprache"),
            lang_keys, format_func=lambda k: language_options[k],
            key='language_radio', index=current_lang_idx, horizontal=True
        )
        if selected_lang_key != st.session_state.language:
            st.session_state.language = selected_lang_key
            st.session_state.location_search_status_msg = "" # Suchmeldung zurücksetzen
            print(f"Sprache geändert zu: {selected_lang_key}. Rerun.")
            st.rerun()

        # --- Standort Expander ---
        with st.expander(t.get('location_expander', "📍 Standort"), expanded=True):
            loc_opts = {'Search': t.get('location_option_search', "Suche"), 'Manual': t.get('location_option_manual', "Manuell")}
            st.radio(t.get('location_select_label', "Standortmethode"), list(loc_opts.keys()), format_func=lambda k: loc_opts[k], key="location_choice_key", horizontal=True)

            lat, lon, hgt = None, None, None
            loc_valid_for_tz_lookup = False
            current_location_is_valid = False

            if st.session_state.location_choice_key == "Manual":
                st.number_input(t.get('location_lat_label', "Breite (°N)"), -90.0, 90.0, step=0.01, format="%.4f", key="manual_lat_val")
                st.number_input(t.get('location_lon_label', "Länge (°E)"), -180.0, 180.0, step=0.01, format="%.4f", key="manual_lon_val")
                st.number_input(t.get('location_elev_label', "Höhe (m)"), -500, step=10, format="%d", key="manual_height_val")
                lat = st.session_state.manual_lat_val; lon = st.session_state.manual_lon_val; hgt = st.session_state.manual_height_val
                if isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and isinstance(hgt, (int, float)):
                    loc_valid_for_tz_lookup = True; current_location_is_valid = True
                    st.session_state.location_is_valid_for_run = True
                    if st.session_state.location_search_success: # Reset search state if switching to manual
                        st.session_state.location_search_success = False; st.session_state.searched_location_name = None; st.session_state.location_search_status_msg = ""
                else:
                    st.warning(t.get('location_error_manual_invalid', "Manuelle Koordinaten oder Höhe ungültig."))
                    current_location_is_valid = False; st.session_state.location_is_valid_for_run = False

            elif st.session_state.location_choice_key == "Search":
                with st.form("loc_search_form"):
                    st.text_input(t.get('location_search_label', "Ort/Adresse suchen:"), key="location_search_query", placeholder=t.get('location_search_placeholder', "z.B. Berlin, Deutschland oder PLZ"))
                    st.number_input(t.get('location_elev_label', "Höhe (m)"), -500, step=10, format="%d", key="manual_height_val") # Höhe bleibt manuell
                    search_submitted = st.form_submit_button(t.get('location_search_submit_button', "Koordinaten finden"))

                status_placeholder = st.empty()
                if st.session_state.location_search_status_msg: # Vorherigen Status anzeigen
                    if st.session_state.location_search_success: status_placeholder.success(st.session_state.location_search_status_msg)
                    else: status_placeholder.error(st.session_state.location_search_status_msg)

                if search_submitted and st.session_state.location_search_query:
                    location_result = None; service_used = None; error_occurred = None
                    query = st.session_state.location_search_query; user_agent = f"AdvancedDSOFinder/{random.randint(1000,9999)}"
                    with st.spinner(t.get('spinner_geocoding', "Suche Standort...")):
                        # Geocoding Logik (Nominatim -> ArcGIS -> Photon)
                        try:
                            geolocator = Nominatim(user_agent=user_agent); location_result = geolocator.geocode(query, timeout=10); service_used = "Nominatim"; print("Nominatim success.")
                        except (GeocoderTimedOut, GeocoderServiceError) as e: print(f"Nominatim fail: {e}"); status_placeholder.info(t.get('location_search_info_fallback', "Nominatim fehlgeschlagen, versuche ArcGIS..."))
                        except Exception as e: print(f"Nominatim error: {e}"); status_placeholder.info(t.get('location_search_info_fallback', "Nominatim fehlgeschlagen, versuche ArcGIS...")); error_occurred = e
                        if not location_result:
                            try: fallback_geolocator = ArcGIS(timeout=15); location_result = fallback_geolocator.geocode(query, timeout=15); service_used = "ArcGIS"; print("ArcGIS success.")
                            except (GeocoderTimedOut, GeocoderServiceError) as e2: print(f"ArcGIS fail: {e2}"); status_placeholder.info(t.get('location_search_info_fallback2', "ArcGIS fehlgeschlagen, versuche Photon...")); error_occurred = e2 if not error_occurred else error_occurred
                            except Exception as e2: print(f"ArcGIS error: {e2}"); status_placeholder.info(t.get('location_search_info_fallback2', "ArcGIS fehlgeschlagen, versuche Photon...")); error_occurred = e2 if not error_occurred else error_occurred
                        if not location_result:
                             try: fallback_geolocator2 = Photon(user_agent=user_agent, timeout=15); location_result = fallback_geolocator2.geocode(query, timeout=15); service_used = "Photon"; print("Photon success.")
                             except (GeocoderTimedOut, GeocoderServiceError) as e3: print(f"Photon fail: {e3}"); error_occurred = e3 if not error_occurred else error_occurred
                             except Exception as e3: print(f"Photon error: {e3}"); error_occurred = e3 if not error_occurred else error_occurred

                        # Ergebnis verarbeiten
                        if location_result and service_used:
                            found_lat = location_result.latitude; found_lon = location_result.longitude; found_name = location_result.address
                            st.session_state.searched_location_name = found_name; st.session_state.location_search_success = True
                            st.session_state.manual_lat_val = found_lat; st.session_state.manual_lon_val = found_lon
                            coords_str = t.get('location_search_coords',"Lat:{:.4f}, Lon:{:.4f}").format(found_lat, found_lon)
                            service_map = {"Nominatim": "N", "ArcGIS": "A", "Photon": "P"}
                            service_tag = service_map.get(service_used, "?")
                            st.session_state.location_search_status_msg = f"{t.get(f'location_search_found_{service_used.lower()}',f'Gefunden ({service_tag}): {{}}').format(found_name)}\n({coords_str})"
                            status_placeholder.success(st.session_state.location_search_status_msg)
                            lat = found_lat; lon = found_lon; hgt = st.session_state.manual_height_val
                            loc_valid_for_tz_lookup = True; current_location_is_valid = True; st.session_state.location_is_valid_for_run = True
                        else: # Geocoding fehlgeschlagen
                            st.session_state.location_search_success = False; st.session_state.searched_location_name = None
                            if error_occurred:
                                if isinstance(error_occurred, GeocoderTimedOut): st.session_state.location_search_status_msg = t.get('location_search_error_timeout',"Zeitüberschreitung bei Geocoding.")
                                elif isinstance(error_occurred, GeocoderServiceError): st.session_state.location_search_status_msg = t.get('location_search_error_service',"Geocoding Dienstfehler: {}").format(error_occurred)
                                else: st.session_state.location_search_status_msg = t.get('location_search_error_fallback2_failed',"Alle Geocoding Dienste fehlgeschlagen: {}").format(error_occurred)
                            else: st.session_state.location_search_status_msg = t.get('location_search_error_not_found',"Standort nicht gefunden.")
                            status_placeholder.error(st.session_state.location_search_status_msg)
                            current_location_is_valid = False; st.session_state.location_is_valid_for_run = False

                elif st.session_state.location_search_success: # Frühere Suche war erfolgreich
                    lat = st.session_state.manual_lat_val; lon = st.session_state.manual_lon_val; hgt = st.session_state.manual_height_val
                    loc_valid_for_tz_lookup = True; current_location_is_valid = True; st.session_state.location_is_valid_for_run = True
                    status_placeholder.success(st.session_state.location_search_status_msg) # Status beibehalten
                else: # Keine Suche und kein vorheriger Erfolg
                    current_location_is_valid = False; st.session_state.location_is_valid_for_run = False

            # --- Zeitzonen-Erkennung ---
            st.markdown("---")
            timezone_message = ""
            if loc_valid_for_tz_lookup and lat is not None and lon is not None:
                if tf: # Prüfen, ob TimezoneFinder verfügbar ist
                    try:
                        found_timezone_val = tf.timezone_at(lng=lon, lat=lat)
                        if found_timezone_val:
                            pytz.timezone(found_timezone_val) # Validieren mit pytz
                            st.session_state.selected_timezone = found_timezone_val
                            timezone_message = f"{t.get('timezone_auto_set_label','Erkannte Zeitzone:')} **{found_timezone_val}**"
                        else: # tf.timezone_at gab None zurück
                            st.session_state.selected_timezone = 'UTC'
                            timezone_message = f"{t.get('timezone_auto_fail_label','Zeitzone:')} **UTC** ({t.get('timezone_auto_fail_msg','Erkennung fehlgeschlagen')})"
                    except pytz.UnknownTimeZoneError:
                         st.session_state.selected_timezone = 'UTC'
                         invalid_tz_name = locals().get('found_timezone_val', 'Unbekannt')
                         timezone_message = t.get('timezone_auto_fail_invalid_label','Zeitzone:') + f" **UTC** ({t.get('timezone_auto_fail_invalid_msg','Ungültiger Name:')} '{invalid_tz_name}')"
                    except Exception as e:
                        print(f"Fehler bei Zeitzonen-Suche: {e}")
                        st.session_state.selected_timezone = 'UTC'
                        timezone_message = f"{t.get('timezone_auto_fail_error_label','Zeitzone:')} **UTC** ({t.get('timezone_auto_fail_error_msg','Fehler')})"
                else: # TimezoneFinder (tf) nicht verfügbar
                    timezone_message = f"{t.get('timezone_finder_unavailable_label','Zeitzone:')} **{st.session_state.selected_timezone}** ({t.get('timezone_finder_unavailable_msg','Finder n.v.')})"
            else: # Standort ungültig für Zeitzonen-Suche
                timezone_message = f"{t.get('timezone_invalid_location_label','Zeitzone:')} **{st.session_state.selected_timezone}** ({t.get('timezone_invalid_location_msg','Standort ungültig')})"
            st.markdown(timezone_message, unsafe_allow_html=True)

        # --- Zeit Expander ---
        with st.expander(t.get('time_expander', "⏱️ Zeit"), expanded=False):
            time_opts = {'Now': t.get('time_option_now',"Jetzt"), 'Specific': t.get('time_option_specific',"Spezifische Nacht")}
            st.radio(t.get('time_select_label',"Zeitrahmen wählen"), list(time_opts.keys()), format_func=lambda k:time_opts[k], key="time_choice_exp", horizontal=True)
            is_now_time_choice = (st.session_state.time_choice_exp == "Now")
            if is_now_time_choice: st.caption(f"Aktuell UTC: {Time.now().iso}")
            else:
                st.date_input(
                    t.get('time_date_select_label',"Datum für Nacht wählen:"),
                    value=st.session_state.selected_date_widget,
                    min_value=date.today() - timedelta(days=365*10),
                    max_value=date.today() + timedelta(days=365*2),
                    key='selected_date_widget'
                )

        # --- Filter Expander ---
        with st.expander(t.get('filters_expander', "✨ Filter"), expanded=False):
            # Magnitude Filter
            st.markdown(t.get('mag_filter_header', "**Magnitude**"))
            mag_opts = {'Bortle Scale': t.get('mag_filter_option_bortle',"Bortle Skala"), 'Manual': t.get('mag_filter_option_manual',"Manuell")}
            if st.session_state.mag_filter_mode_exp not in mag_opts: st.session_state.mag_filter_mode_exp = 'Bortle Scale'
            st.radio(t.get('mag_filter_method_label',"Filter Methode:"), list(mag_opts.keys()), format_func=lambda k:mag_opts[k], key="mag_filter_mode_exp", horizontal=True)
            st.slider(t.get('mag_filter_bortle_label',"Bortle Skala:"), 1, 9, key='bortle_slider', help=t.get('mag_filter_bortle_help',"Bortle Klasse wählen (1=dunkel, 9=Stadt)"))
            if st.session_state.mag_filter_mode_exp == "Manual":
                st.slider(t.get('mag_filter_min_mag_label',"Min Magnitude:"), -5.0, 20.0, step=0.5, format="%.1f", help=t.get('mag_filter_min_mag_help',"Hellste Magnitude (kleinere Zahlen = heller)"), key='manual_min_mag_slider')
                st.slider(t.get('mag_filter_max_mag_label',"Max Magnitude:"), -5.0, 20.0, step=0.5, format="%.1f", help=t.get('mag_filter_max_mag_help',"Schwächste Magnitude (größere Zahlen = schwächer)"), key='manual_max_mag_slider')
                if isinstance(st.session_state.manual_min_mag_slider,(int,float)) and isinstance(st.session_state.manual_max_mag_slider,(int,float)) and st.session_state.manual_min_mag_slider > st.session_state.manual_max_mag_slider:
                    st.warning(t.get('mag_filter_warning_min_max',"Minimum Magnitude darf nicht größer als Maximum sein!"))

            # Höhen Filter
            st.markdown("---"); st.markdown(t.get('min_alt_header', "**Höhe**"))
            min_alt_filter = st.session_state.min_alt_slider; max_alt_filter = st.session_state.max_alt_slider
            if min_alt_filter > max_alt_filter: st.session_state.min_alt_slider = max_alt_filter; min_alt_filter = max_alt_filter # Korrigieren falls inkonsistent
            st.slider(t.get('min_alt_label',"Minimale Höhe (°):"), 0, 90, key='min_alt_slider', step=1, help=t.get('min_alt_help', "Objekte müssen über dieser Höhe sein."))
            st.slider(t.get('max_alt_label',"Maximale Höhe (°):"), 0, 90, key='max_alt_slider', step=1, help=t.get('max_alt_help', "Objekte müssen unter dieser Höhe sein (für Hindernisse)."))
            if st.session_state.min_alt_slider > st.session_state.max_alt_slider: st.warning(t.get('alt_filter_warning_min_max', "Minimale Höhe darf nicht größer als Maximale sein!"))

            # Mond Filter
            st.markdown("---"); st.markdown(t.get('moon_warning_header',"**Mond**"))
            st.slider(t.get('moon_warning_label',"Warnen wenn Mond > (%):"), 0, 100, key='moon_phase_slider', step=5, help=t.get('moon_warning_help', "Zeige Warnung in Ergebnissen, wenn Mond heller ist."))

            # Objekttyp Filter
            st.markdown("---"); st.markdown(t.get('object_types_header',"**Objekttypen**"))
            all_types_list = []
            if df_catalog_data is not None and not df_catalog_data.empty:
                try:
                    if 'Type' in df_catalog_data.columns: all_types_list = sorted(list(df_catalog_data['Type'].dropna().astype(str).unique()))
                    else: st.warning(t.get('object_types_warning_missing_column', "Katalog fehlt Spalte 'Type'."))
                except Exception as e: st.warning(f"{t.get('object_types_error_extract','Fehler beim Extrahieren der Typen')}: {e}")
            if all_types_list:
                current_selected_types = [s_type for s_type in st.session_state.object_type_filter_exp if s_type in all_types_list]
                if current_selected_types != st.session_state.object_type_filter_exp: st.session_state.object_type_filter_exp = current_selected_types
                st.multiselect(t.get('object_types_label',"Nach Objekttyp(en) filtern:"), all_types_list, default=current_selected_types, key="object_type_filter_exp", help=t.get('object_types_help',"Spezifische Typen wählen (leer lassen für alle)."))
            else: st.info(t.get('object_types_info_no_types', "Keine Objekttypen im Katalog verfügbar. Filter deaktiviert.")); st.session_state.object_type_filter_exp = []

            # Größen Filter
            st.markdown("---"); st.markdown(t.get('size_filter_header',"**Scheinbare Größe**"))
            size_data_available = (df_catalog_data is not None and 'MajAx' in df_catalog_data.columns and df_catalog_data['MajAx'].notna().any())
            size_filter_disabled = not size_data_available
            if size_data_available:
                try:
                    valid_sizes = df_catalog_data['MajAx'].dropna()
                    min_size_limit = max(0.1, float(valid_sizes.min())) if not valid_sizes.empty else 0.1
                    max_size_limit = float(valid_sizes.max()) if not valid_sizes.empty else 120.0
                    current_min_size, current_max_size = st.session_state.size_arcmin_range
                    clamped_min = max(min_size_limit, min(current_min_size, max_size_limit)); clamped_max = min(max_size_limit, max(current_max_size, min_size_limit))
                    if clamped_min > clamped_max: clamped_min = clamped_max
                    if (clamped_min, clamped_max) != st.session_state.size_arcmin_range: st.session_state.size_arcmin_range = (clamped_min, clamped_max)
                    step_size = 0.1 if max_size_limit <= 20 else (0.5 if max_size_limit <= 100 else 1.0)
                    st.slider(label=t.get('size_filter_label',"Größenbereich (Arcmin):"), min_value=min_size_limit, max_value=max_size_limit, value=st.session_state.size_arcmin_range, step=step_size, key='size_arcmin_range', help=t.get('size_filter_help',"Filter nach scheinbarer Größe der Hauptachse."), disabled=size_filter_disabled)
                except Exception as e: st.error(f"{t.get('size_filter_error', 'Fehler beim Einrichten des Größen-Sliders')}: {e}"); size_filter_disabled = True
            else: st.info(t.get('size_filter_info_missing', "Größen-Daten ('MajAx') fehlen. Filter deaktiviert.")); size_filter_disabled = True
            if size_filter_disabled: st.slider(t.get('size_filter_label',"Größenbereich (Arcmin):"), 0.0, 1.0, value=(0.0, 1.0), key='size_arcmin_range_disabled', disabled=True)

            # Richtungs Filter
            st.markdown("---"); st.markdown(t.get('direction_filter_header',"**Kulminationsrichtung**"))
            all_directions_text = t.get('direction_option_all',"Alle Richtungen")
            direction_display_options = [all_directions_text] + CARDINAL_DIRECTIONS
            direction_internal_keys = [ALL_DIRECTIONS_KEY] + CARDINAL_DIRECTIONS
            current_direction_key = st.session_state.selected_peak_direction
            if current_direction_key not in direction_internal_keys: current_direction_key = ALL_DIRECTIONS_KEY; st.session_state.selected_peak_direction=current_direction_key
            try: current_index = direction_internal_keys.index(current_direction_key)
            except ValueError: current_index = 0
            selected_direction_display = st.selectbox(t.get('direction_filter_label',"Nach Richtung bei max. Höhe filtern:"), direction_display_options, index=current_index, key='direction_selectbox', help=t.get('direction_filter_help',"Zeige nur Objekte, die in der gewählten Richtung kulminieren."))
            selected_internal_key = ALL_DIRECTIONS_KEY
            if selected_direction_display != all_directions_text:
                try: selected_index = direction_display_options.index(selected_direction_display); selected_internal_key = direction_internal_keys[selected_index]
                except ValueError: selected_internal_key = ALL_DIRECTIONS_KEY
            if selected_internal_key != st.session_state.selected_peak_direction: st.session_state.selected_peak_direction = selected_internal_key

        # --- Ergebnisoptionen Expander ---
        with st.expander(t.get('results_options_expander',"⚙️ Ergebnisoptionen"), expanded=False):
            # Max Objekte Slider
            max_possible_objects = len(df_catalog_data) if df_catalog_data is not None and not df_catalog_data.empty else 50
            min_objects_limit = 5; actual_max_limit = max(min_objects_limit, max_possible_objects)
            slider_disabled = (actual_max_limit <= min_objects_limit)
            default_num_objects = st.session_state.get('num_objects_slider', 20)
            clamped_default = max(min_objects_limit, min(default_num_objects, actual_max_limit))
            if clamped_default != default_num_objects: st.session_state.num_objects_slider = clamped_default
            st.slider(t.get('results_options_max_objects_label',"Max. anzuzeigende Objekte:"), min_value=min_objects_limit, max_value=actual_max_limit, value=st.session_state.num_objects_slider, step=1, key='num_objects_slider', disabled=slider_disabled, help=t.get('results_options_max_objects_help',"Begrenzt die Anzahl der Objekte in der Ergebnisliste."))

            # Sortiermethode Radio
            sort_method_map = {'Duration & Altitude': t.get('results_options_sort_duration',"Dauer & Höhe"), 'Brightness': t.get('results_options_sort_magnitude',"Helligkeit (Magnitude)")}
            if st.session_state.sort_method not in sort_method_map: st.session_state.sort_method = 'Duration & Altitude'
            st.radio(t.get('results_options_sort_method_label',"Ergebnisse sortieren nach:"), list(sort_method_map.keys()), format_func=lambda k: sort_method_map[k], key='sort_method', horizontal=True, help=t.get('results_options_sort_method_help',"Wählt die Sortierreihenfolge der gefundenen Objekte."))

        # --- Bug Report Link ---
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**{t.get('bug_report', 'Fehler gefunden oder Feedback?')}**")
        bug_email_address = "debrun2005@gmail.com"
        bug_email_subject = urllib.parse.quote(t.get('bug_report_subject', "Bug Report/Feedback: Advanced DSO Finder"))
        bug_email_body = urllib.parse.quote(t.get('bug_report_body', "Bitte beschreiben Sie das Problem oder den Vorschlag:\n\n(Bei Fehlern bitte Schritte zur Reproduktion angeben)\n\nApp Version/Datum: [Falls bekannt]\nBrowser/OS: [Falls relevant]"))
        mailto_link = f"mailto:{bug_email_address}?subject={bug_email_subject}&body={bug_email_body}"
        st.sidebar.link_button(t.get('bug_report_button', '🐞 Problem melden / Vorschlag machen'), mailto_link)

def display_search_parameters(t: dict, observer_run: Observer | None, ref_time: Time) -> tuple[float, float]:
    """Zeigt die Zusammenfassung der Suchparameter im Hauptbereich an."""
    st.subheader(t.get('search_params_header', "Zusammenfassung Suchparameter"))
    p1, p2 = st.columns(2)

    # Standort Anzeige
    location_display_text = t.get('location_not_set', "Standort nicht gesetzt oder ungültig.")
    if st.session_state.location_is_valid_for_run and observer_run:
        lat = observer_run.location.lat.deg; lon = observer_run.location.lon.deg
        if st.session_state.location_choice_key == "Manual": location_display_text = t.get('location_manual_display',"Manuell ({:.4f}°N, {:.4f}°E)").format(lat, lon)
        elif st.session_state.searched_location_name: location_display_text = t.get('location_search_display',"Gesucht: {} ({:.4f}°N, {:.4f}°E)").format(st.session_state.searched_location_name, lat, lon)
        else: location_display_text = f"Lat:{lat:.4f}, Lon:{lon:.4f}" # Fallback
    elif not st.session_state.location_is_valid_for_run: location_display_text = t.get('location_invalid_for_run', "Standort ungültig")
    p1.markdown(f"📍 **{t.get('search_params_location_label','Standort:')}** {location_display_text}")

    # Zeit Anzeige
    time_display_text = ""
    is_now_mode = (st.session_state.time_choice_exp == "Now")
    if is_now_mode:
        try: local_now_str, tz_now_name = get_local_time_str(ref_time, st.session_state.selected_timezone); time_display_text = t.get('search_params_time_now',"Jetzt (Lokal: {} {})").format(local_now_str, tz_now_name)
        except Exception: time_display_text = t.get('search_params_time_now_utc',"Jetzt (UTC: {})").format(ref_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))
    else: selected_date = st.session_state.selected_date_widget; time_display_text = t.get('search_params_time_specific',"Nacht vom {}").format(selected_date.strftime('%Y-%m-%d'))
    p1.markdown(f"⏱️ **{t.get('search_params_time_label','Zeit:')}** {time_display_text}")

    # Magnitude Filter Anzeige
    magnitude_display_text = ""; min_mag_filter, max_mag_filter = -np.inf, np.inf
    if st.session_state.mag_filter_mode_exp == "Bortle Scale":
        try:
            from astro_calculations import get_magnitude_limit # Import hier, falls nicht global
            max_mag_filter = get_magnitude_limit(st.session_state.bortle_slider)
            magnitude_display_text = t.get('search_params_filter_mag_bortle',"Bortle {} (≤ Mag {:.1f})").format(st.session_state.bortle_slider, max_mag_filter)
        except Exception as e: magnitude_display_text = t.get('search_params_filter_mag_bortle_error',"Bortle {} (Limit Fehler)").format(st.session_state.bortle_slider); print(f"Fehler bei Mag-Limit: {e}")
    else: # Manual mode
        min_mag_filter = st.session_state.manual_min_mag_slider; max_mag_filter = st.session_state.manual_max_mag_slider
        magnitude_display_text = t.get('search_params_filter_mag_manual',"Manuell (Mag {:.1f} bis {:.1f})").format(min_mag_filter, max_mag_filter)
    p2.markdown(f"✨ **{t.get('search_params_filter_mag_label','Magnitude:')}** {magnitude_display_text}")

    # Höhen Filter Anzeige
    min_alt_display = st.session_state.min_alt_slider; max_alt_display = st.session_state.max_alt_slider
    p2.markdown(f"🔭 **{t.get('search_params_filter_alt_label','Höhe:')}** {min_alt_display}° – {max_alt_display}°")

    # Objekttyp Filter Anzeige
    selected_types = st.session_state.object_type_filter_exp
    types_display_string = ', '.join(selected_types) if selected_types else t.get('search_params_types_all',"Alle")
    p2.markdown(f"**{t.get('search_params_filter_type_label','Typen:')}** {types_display_string}")

    # Größen Filter Anzeige
    size_min, size_max = st.session_state.size_arcmin_range
    p2.markdown(f"📐 **{t.get('search_params_filter_size_label','Größe:')}** {size_min:.1f}′ – {size_max:.1f}′")

    # Richtungs Filter Anzeige
    direction_key = st.session_state.selected_peak_direction
    direction_display_text = t.get('search_params_direction_all',"Alle") if direction_key == ALL_DIRECTIONS_KEY else direction_key
    p2.markdown(f"🧭 **{t.get('search_params_filter_direction_label','Kulm. Richtung:')}** {direction_display_text}")

    # Gibt die tatsächlichen Filterwerte zurück, die für die Berechnung verwendet werden
    return min_mag_filter, max_mag_filter


def display_results(t: dict, results_ph: st.container, observer_run: Observer | None) -> None:
    """Zeigt die Ergebnisliste, Plots, Download-Button und Kosmologie-Daten an."""
    results_data = st.session_state.last_results
    results_ph.subheader(t.get('results_list_header',"Ergebnisse"))

    window_start = st.session_state.get('window_start_time'); window_end = st.session_state.get('window_end_time')
    observer_exists = observer_run is not None

    # Mondphasen-Metrik und Warnung anzeigen
    if observer_exists and isinstance(window_start, Time) and isinstance(window_end, Time):
        mid_time = window_start + (window_end - window_start) / 2
        try:
            illumination = moon_illumination(mid_time); moon_percentage = illumination * 100
            moon_svg_icon = create_moon_phase_svg(illumination, size=50)
            mc1, mc2 = results_ph.columns([1, 3])
            with mc1: st.markdown(moon_svg_icon, unsafe_allow_html=True)
            with mc2:
                st.metric(label=t.get('moon_metric_label',"Mondbeleuchtung"), value=f"{moon_percentage:.0f}%")
                moon_warning_threshold = st.session_state.moon_phase_slider
                if moon_percentage > moon_warning_threshold:
                    st.warning(t.get('moon_warning_message',"Warnung: Mond ({:.0f}%) heller als Schwelle ({}%)!").format(moon_percentage, moon_warning_threshold))
        except Exception as e: results_ph.warning(t.get('moon_phase_error',"Mondphase konnte nicht berechnet werden: {}").format(e)); print(f"Fehler bei Mondphasenberechnung: {e}")
    elif st.session_state.find_button_pressed: results_ph.info(t.get('moon_phase_info_cannot_calc',"Mondphasenberechnung benötigt gültigen Standort und Zeitfenster."))

    # Plot-Typ Auswahl
    plot_type_map = {'Sky Path': t.get('graph_type_sky_path',"Himmelspfad (Polar)"), 'Altitude Plot': t.get('graph_type_alt_time',"Höhe vs. Zeit")}
    if st.session_state.plot_type_selection not in plot_type_map: st.session_state.plot_type_selection = 'Sky Path'
    results_ph.radio(t.get('graph_type_label',"Plot-Typ wählen:"), list(plot_type_map.keys()), format_func=lambda k: plot_type_map[k], key='plot_type_selection', horizontal=True)

    # Jedes Ergebnisobjekt in einem Expander anzeigen
    for i, obj_data in enumerate(results_data):
        obj_name = obj_data.get('Name', '?'); obj_type = obj_data.get('Type', '?'); obj_mag = obj_data.get('Magnitude')
        mag_str = f"{obj_mag:.1f}" if obj_mag is not None else t.get('magnitude_unknown', 'k.A.')
        expander_title_template = "{} ({}) - Mag: {}"; expander_title = expander_title_template.format(obj_name, obj_type, mag_str)
        is_expanded = (st.session_state.expanded_object_name == obj_name)
        obj_container = results_ph.container() # Eigener Container pro Objekt

        with obj_container.expander(expander_title, expanded=is_expanded):
            col1, col2, col3 = st.columns([2, 2, 1])
            # Spalte 1: Basisinfos & Koordinaten
            col1.markdown(f"**{t.get('results_details_header','Details:')}**")
            col1.markdown(f"**{t.get('results_export_constellation','Sternbild')}:** {obj_data.get('Constellation','?')}")
            size_arcmin = obj_data.get('Size (arcmin)'); size_display = t.get('results_size_value','{:.1f}′').format(size_arcmin) if size_arcmin is not None else '?'
            col1.markdown(f"**{t.get('results_size_label','Größe:')}** {size_display}")
            col1.markdown(f"**RA:** {obj_data.get('RA','?')}"); col1.markdown(f"**Dec:** {obj_data.get('Dec','?')}")

            # Spalte 2: Beobachtungsdaten
            col2.markdown(f"**{t.get('results_max_alt_header','Max. Höhe:')}**")
            max_alt_value = obj_data.get('Max Altitude (°)', 0); az_at_max = obj_data.get('Azimuth at Max (°)', 0); direction_at_max = obj_data.get('Direction at Max', '?')
            safe_az_fmt = "(Az:{:.1f}°)"; safe_dir_fmt = ", Dir:{}" # Sichere Formatstrings
            az_display = safe_az_fmt.format(az_at_max); dir_display = safe_dir_fmt.format(direction_at_max)
            col2.markdown(f"**{max_alt_value:.1f}°** {az_display}{dir_display}")
            col2.markdown(f"**{t.get('results_best_time_header','Beste Zeit (Lokal):')}**")
            peak_time_utc = obj_data.get('Time at Max (UTC)'); local_time_str, local_tz_name = get_local_time_str(peak_time_utc, st.session_state.selected_timezone)
            col2.markdown(f"{local_time_str} ({local_tz_name})")
            col2.markdown(f"**{t.get('results_cont_duration_header','Max. kont. Dauer:')}**")
            duration_hours = obj_data.get('Max Cont. Duration (h)', 0)
            col2.markdown(t.get('results_duration_value',"{:.1f} Stunden").format(duration_hours))

            # Spalte 3: Links & Buttons
            google_query = urllib.parse.quote_plus(f"{obj_name} astronomy"); google_url = f"https://google.com/search?q={google_query}"
            col3.markdown(f"[{t.get('google_link_text','Google')}]({google_url})", unsafe_allow_html=True)
            simbad_query = urllib.parse.quote_plus(obj_name); simbad_url = f"http://simbad.cds.unistra.fr/simbad/sim-basic?Ident={simbad_query}"
            col3.markdown(f"[{t.get('simbad_link_text','SIMBAD')}]({simbad_url})", unsafe_allow_html=True)

            # Plot Button
            plot_button_key = f"plot_{obj_name}_{i}"
            if st.button(t.get('results_graph_button',"📈 Plot"), key=plot_button_key):
                st.session_state.plot_object_name = obj_name; st.session_state.active_result_plot_data = obj_data
                st.session_state.show_plot = True; st.session_state.show_custom_plot = False
                st.session_state.expanded_object_name = obj_name; st.rerun()

            # --- Kosmologie Integration ---
            z_value = obj_data.get('z') # Rotverschiebungswert holen (Spaltenname ggf. anpassen!)
            show_cosmo_key = f"show_cosmo_{obj_name}_{i}" # Eindeutiger Schlüssel für den Zustand

            if z_value is not None and isinstance(z_value, (float, int)) and z_value > 0:
                if st.button(t('results_cosmology_button', "🌌 Kosmologie"), key=f"btn_cosmo_{obj_name}_{i}"):
                    # Zustand für dieses Objekt umschalten
                    current_state = st.session_state.get(show_cosmo_key, False)
                    st.session_state[show_cosmo_key] = not current_state
                    # Optional: Andere Kosmologie-Abschnitte schliessen
                    if not current_state: # Nur schliessen, wenn dieser geöffnet wird
                         for key in list(st.session_state.keys()): # Iterate over a copy of keys
                             if key.startswith("show_cosmo_") and key != show_cosmo_key:
                                 st.session_state[key] = False
                    st.rerun() # Rerun, um die Anzeige zu aktualisieren

                # Kosmologie-Daten anzeigen, wenn der Zustand True ist
                if st.session_state.get(show_cosmo_key, False):
                    cosmo_placeholder = st.container()
                    with cosmo_placeholder:
                        st.markdown("---") # Trennlinie
                        st.markdown(f"**{t('cosmology_results_header', 'Kosmologische Daten')} (z={z_value:.5f})**")
                        try:
                            # Berechnungsfunktion aufrufen (ist jetzt in astro_calculations)
                            # Standardparameter aus astro_calculations verwenden (oder hier definieren)
                            # --- Correction Start: Use imported constants directly ---
                            H0_COSMO = H0_DEFAULT
                            OMEGA_M_COSMO = OMEGA_M_DEFAULT
                            OMEGA_LAMBDA_COSMO = OMEGA_LAMBDA_DEFAULT
                            # --- Correction End ---

                            cosmo_results = calculate_lcdm_distances(z_value, H0_COSMO, OMEGA_M_COSMO, OMEGA_LAMBDA_COSMO)

                            cosmo_error_key = cosmo_results.get('error_msg')
                            if cosmo_error_key:
                                st.warning(t(cosmo_error_key, default="Kosmologie-Berechnungsfehler"))
                            else:
                                # Ergebnisse extrahieren
                                lookback_gyr_res = cosmo_results['lookback_gyr']
                                comoving_mpc_res = cosmo_results['comoving_mpc']
                                luminosity_mpc_res = cosmo_results['luminosity_mpc']
                                ang_diam_mpc_res = cosmo_results['ang_diam_mpc']

                                # Umrechnungen für die Anzeige
                                comoving_gly_res = convert_mpc_to_gly(comoving_mpc_res)
                                luminosity_gly_res = convert_mpc_to_gly(luminosity_mpc_res)
                                ang_diam_gly_res = convert_mpc_to_gly(ang_diam_mpc_res)
                                # comoving_km_res = convert_mpc_to_km(comoving_mpc_res) # Optional
                                # comoving_ly_res = convert_km_to_ly(comoving_km_res) # Optional

                                # Ergebnisse anzeigen (adaptiert von Redshift_Calculator)
                                st.metric(label=t("lookback_time"), value=f"{lookback_gyr_res:.3f}", delta=t("unit_Gyr"))
                                lookback_example_key = get_lookback_comparison(lookback_gyr_res)
                                st.caption(f"*{t(lookback_example_key)}*")

                                st.markdown(f"**{t('comoving_distance_title')}**")
                                st.text(f"  {comoving_mpc_res:,.3f} {t('unit_Mpc')}")
                                st.text(f"  {comoving_gly_res:,.3f} {t('unit_Gly')}")
                                comoving_example_key = get_comoving_comparison(comoving_mpc_res)
                                st.caption(f"*{t(comoving_example_key)}*")

                                st.markdown(f"**{t('luminosity_distance_title')}**")
                                st.text(f"  {luminosity_mpc_res:,.3f} {t('unit_Mpc')}")
                                st.text(f"  {luminosity_gly_res:,.3f} {t('unit_Gly')}")
                                st.caption(f"*{t('explanation_luminosity')}*")

                                st.markdown(f"**{t('angular_diameter_distance_title')}**")
                                st.text(f"  {ang_diam_mpc_res:,.3f} {t('unit_Mpc')}")
                                st.text(f"  {ang_diam_gly_res:,.3f} {t('unit_Gly')}")
                                st.caption(f"*{t('explanation_angular')}*")

                                # Integrationswarnung anzeigen, falls vorhanden
                                integration_warn_key = cosmo_results.get('integration_warning_key')
                                if integration_warn_key:
                                     integration_warn_args = cosmo_results.get('integration_warning_args', {})
                                     st.caption(t(integration_warn_key, **integration_warn_args))

                                st.caption(t("calculation_note")) # Hinweis auf Modell

                        except ImportError:
                             st.error(t('error_module_missing_cosmo', "Fehler: Funktion 'calculate_lcdm_distances' nicht in 'astro_calculations' gefunden."))
                        except Exception as e:
                            st.error(f"{t('error_calc_failed_cosmo', 'Kosmologie-Berechnung fehlgeschlagen')}: {e}")
                            traceback.print_exc() # Log full error

            elif z_value is not None: # z=0 oder z<0 (oder ungültiger Typ)
                 if isinstance(z_value, (float, int)) and z_value <= 0:
                      st.caption(t('cosmology_not_applicable', "Kosmologische Daten nicht anwendbar für z ≤ 0."))
                 else:
                      st.caption(t('cosmology_invalid_z', "Ungültiger Rotverschiebungswert im Katalog."))
            # --- Ende Kosmologie-Integration ---


            # --- Plot anzeigen, falls angefordert ---
            if st.session_state.show_plot and st.session_state.plot_object_name == obj_name:
                plot_data_to_use = st.session_state.active_result_plot_data
                min_alt_for_plot = st.session_state.min_alt_slider
                max_alt_for_plot = st.session_state.max_alt_slider
                st.markdown("---") # Trennlinie vor Plot

                with st.spinner(t.get('results_spinner_plotting',"Erstelle Plot...")):
                    try: figure = create_plot(plot_data_to_use, min_alt_for_plot, max_alt_for_plot, st.session_state.plot_type_selection, t)
                    except Exception as e: st.error(f"{t.get('plot_error_unexpected', 'Unerwarteter Plot-Fehler')}: {e}"); traceback.print_exc(); figure = None

                    if figure:
                        st.pyplot(figure)
                        close_button_key = f"close_{obj_name}_{i}"
                        if st.button(t.get('results_close_graph_button',"Plot schließen"), key=close_button_key):
                            st.session_state.show_plot = False; st.session_state.active_result_plot_data = None
                            st.session_state.expanded_object_name = None; st.rerun()
                    else: st.error(t.get('results_graph_not_created',"Plot konnte nicht erstellt werden."))

    # --- CSV Download Button ---
    if results_data: # Nur anzeigen, wenn Ergebnisse vorhanden sind
        csv_placeholder = results_ph.container()
        try:
            export_rows = []
            for obj_csv_data in results_data: # Daten für CSV vorbereiten
                peak_time_utc_csv = obj_csv_data.get('Time at Max (UTC)')
                local_time_csv, _ = get_local_time_str(peak_time_utc_csv, st.session_state.selected_timezone)
                export_rows.append({
                    t.get('results_export_name', "Name"): obj_csv_data.get('Name', 'N/A'),
                    t.get('results_export_type', "Typ"): obj_csv_data.get('Type', 'N/A'),
                    t.get('results_export_constellation', "Sternbild"): obj_csv_data.get('Constellation', 'N/A'),
                    t.get('results_export_mag', "Magnitude"): obj_csv_data.get('Magnitude'),
                    t.get('results_export_size', "Größe (arcmin)"): obj_csv_data.get('Size (arcmin)'),
                    t.get('results_export_ra', "RA"): obj_csv_data.get('RA', 'N/A'),
                    t.get('results_export_dec', "Dec"): obj_csv_data.get('Dec', 'N/A'),
                    t.get('results_export_max_alt', "Max Höhe (°)"): obj_csv_data.get('Max Altitude (°)', 0),
                    t.get('results_export_az_at_max', "Azimut bei Max (°)"): obj_csv_data.get('Azimuth at Max (°)', 0),
                    t.get('results_export_direction_at_max', "Richtung bei Max"): obj_csv_data.get('Direction at Max', 'N/A'),
                    t.get('results_export_time_max_utc', "Zeit bei Max (UTC)"): peak_time_utc_csv.iso if peak_time_utc_csv else "N/A",
                    t.get('results_export_time_max_local', "Zeit bei Max (Lokal)"): local_time_csv,
                    t.get('results_export_cont_duration', "Max kont. Dauer (h)"): obj_csv_data.get('Max Cont. Duration (h)', 0),
                    t.get('results_export_redshift', "Rotverschiebung (z)"): obj_csv_data.get('z') # Rotverschiebung hinzufügen
                })
            df_export = pd.DataFrame(export_rows)
            decimal_separator = ',' if st.session_state.language == 'de' else '.' # Dezimaltrennzeichen anpassen
            csv_string = df_export.to_csv(index=False, sep=';', encoding='utf-8-sig', decimal=decimal_separator)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M"); csv_filename = t.get('results_csv_filename',"dso_results_{}.csv").format(timestamp)
            csv_placeholder.download_button(label=t.get('results_save_csv_button',"💾 Ergebnisse als CSV speichern"), data=csv_string, file_name=csv_filename, mime='text/csv', key='csv_download_button')
        except Exception as e: csv_placeholder.error(t.get('results_csv_export_error',"Fehler beim Erstellen der CSV-Datei: {}").format(e)); print(f"CSV Export Fehler: {e}")


def create_custom_target_section(t: dict, results_ph: st.container, observer_run: Observer | None) -> None:
    """Erstellt den UI-Bereich zum Plotten eines benutzerdefinierten Ziels."""
    st.markdown("---") # Trennlinie
    with st.expander(t.get('custom_target_expander',"Eigenes RA/Dec Ziel plotten")):
        with st.form("custom_target_form"):
             st.text_input(t.get('custom_target_ra_label',"Rektaszension (RA):"), key="custom_target_ra", placeholder=t.get('custom_target_ra_placeholder',"z.B. 10h 08m 22.3s oder 10.1395h"))
             st.text_input(t.get('custom_target_dec_label',"Deklination (Dec):"), key="custom_target_dec", placeholder=t.get('custom_target_dec_placeholder',"z.B. +11d 58m 02s oder 11.9672d"))
             st.text_input(t.get('custom_target_name_label',"Zielname (Optional):"), key="custom_target_name", placeholder=t.get('custom_target_name_default_placeholder',"Eigenes Ziel"))
             custom_plot_submitted = st.form_submit_button(t.get('custom_target_button',"Eigenes Ziel plotten"))

        error_placeholder = st.empty(); plot_display_area = st.container()

        if custom_plot_submitted:
             st.session_state.show_plot = False; st.session_state.show_custom_plot = False
             st.session_state.custom_target_plot_data = None; st.session_state.custom_target_error = ""
             custom_ra_input = st.session_state.custom_target_ra; custom_dec_input = st.session_state.custom_target_dec
             custom_name_input = st.session_state.custom_target_name or t.get('custom_target_name_default',"Eigenes Ziel")
             window_start_time = st.session_state.get('window_start_time'); window_end_time = st.session_state.get('window_end_time')
             observer_exists = observer_run is not None

             if not custom_ra_input or not custom_dec_input: st.session_state.custom_target_error = t.get('custom_target_error_coords_missing',"RA und Dec Koordinaten benötigt."); error_placeholder.error(st.session_state.custom_target_error)
             elif not observer_exists or not isinstance(window_start_time, Time) or not isinstance(window_end_time, Time): st.session_state.custom_target_error = t.get('custom_target_error_window_invalid',"Gültiger Standort und Beobachtungsfenster benötigt (erst Suche durchführen)."); error_placeholder.error(st.session_state.custom_target_error)
             else:
                 try:
                     custom_skycoord = SkyCoord(ra=custom_ra_input, dec=custom_dec_input, unit=(u.hourangle, u.deg), frame='icrs')
                     if window_start_time >= window_end_time: raise ValueError(t.get('custom_target_error_window_order', "Fenster-Startzeit muss vor Endzeit liegen."))
                     time_resolution = 5 * u.minute; observation_times_custom = Time(np.arange(window_start_time.jd, window_end_time.jd, time_resolution.to(u.day).value), format='jd', scale='utc')
                     if len(observation_times_custom) < 2: raise ValueError(t.get('custom_target_error_window_short', "Beobachtungsfenster zu kurz zum Plotten."))
                     altaz_frame = AltAz(obstime=observation_times_custom, location=observer_run.location)
                     custom_altazs = custom_skycoord.transform_to(altaz_frame)
                     custom_altitudes = custom_altazs.alt.to(u.deg).value; custom_azimuths = custom_altazs.az.to(u.deg).value
                     st.session_state.custom_target_plot_data = {'Name': custom_name_input, 'altitudes': custom_altitudes, 'azimuths': custom_azimuths, 'times': observation_times_custom}
                     st.session_state.show_custom_plot = True; st.session_state.custom_target_error = ""; st.rerun()
                 except ValueError as e: st.session_state.custom_target_error = f"{t.get('custom_target_error_parsing','Fehler beim Parsen der Koordinaten:')} {e}"; error_placeholder.error(st.session_state.custom_target_error)
                 except Exception as e: st.session_state.custom_target_error = f"{t.get('custom_target_error_unexpected','Unerwarteter Fehler beim Plotten:')} {e}"; error_placeholder.error(st.session_state.custom_target_error); traceback.print_exc()

        if st.session_state.show_custom_plot and st.session_state.custom_target_plot_data:
            custom_plot_data = st.session_state.custom_target_plot_data
            min_alt_cust_plot = st.session_state.min_alt_slider; max_alt_cust_plot = st.session_state.max_alt_slider
            with plot_display_area:
                 st.markdown("---")
                 with st.spinner(t.get('results_spinner_plotting',"Erstelle Plot...")):
                     try: custom_figure = create_plot(custom_plot_data, min_alt_cust_plot, max_alt_cust_plot, st.session_state.plot_type_selection, t)
                     except Exception as e: st.error(f"{t.get('plot_error_unexpected','Plot Fehler')}: {e}"); traceback.print_exc(); custom_figure = None
                     if custom_figure:
                         st.pyplot(custom_figure)
                         if st.button(t.get('results_close_graph_button',"Plot schließen"), key="close_custom_plot_button"):
                             st.session_state.show_custom_plot = False; st.session_state.custom_target_plot_data = None; st.rerun()
                     elif custom_plot_submitted: st.error(t.get('results_graph_not_created',"Plot konnte nicht erstellt werden."))
        elif st.session_state.custom_target_error: error_placeholder.error(st.session_state.custom_target_error)


def display_donation_link(t: dict) -> None:
    """Zeigt den Ko-fi Spendenlink Button an."""
    st.markdown("---")
    kofi_url = "https://ko-fi.com/advanceddsofinder"
    kofi_text = t.get('donation_button_text', "Entwicklung unterstützen via Ko-fi ☕")
    st.link_button(kofi_text, kofi_url)
