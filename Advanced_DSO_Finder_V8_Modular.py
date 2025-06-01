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
# Assuming localization.py exists in the same directory and has a get_translation function
try:
    from localization import get_translation
except ImportError:
    # Fallback basic translation if localization.py is missing
    print("Warning: localization.py not found. Using basic fallback translations.")
    _translations = {
        'de': {
            "recessional_velocity": "Fluchtgeschwindigkeit",
            "unit_km_s": "km/s",
            "redshift_calculator_title": "Rotverschiebungsrechner",
            "input_params": "Eingabeparameter",
            "redshift_z": "Rotverschiebung (z)",
            "redshift_z_tooltip": "Kosmologische Rotverschiebung eingeben",
            "cosmo_params": "Kosmologische Parameter",
            "hubble_h0": "H₀ [km/s/Mpc]",
            "omega_m": "Ωm",
            "omega_lambda": "ΩΛ",
            "flat_universe_warning": "Ωm + ΩΛ ≉ 1. Flaches Universum wird angenommen.",
            "results_for": "Ergebnisse für z = {z:.5f}",
            "lookback_time": "Rückblickzeit",
            "unit_Gyr": "Gyr",
            "cosmo_distances": "Kosmologische Distanzen",
            "comoving_distance_title": "**Mitbewegte Distanz:**",
            "unit_Mpc": "Mpc",
            "unit_Gly": "Gly",
            "unit_km_sci": "km (wiss.)",
            "unit_km_full": "km (voll)",
            "unit_LJ": "Lj", # Lichtjahre
            "unit_AE": "AE", # Astronomische Einheiten
            "unit_Ls": "Ls", # Lichtsekunden
            "luminosity_distance_title": "**Leuchtkraftdistanz:**",
            "explanation_luminosity": "Relevant für die Helligkeit entfernter Objekte.",
            "angular_diameter_distance_title": "**Winkeldurchmesserdistanz:**",
            "explanation_angular": "Relevant für die scheinbare Größe entfernter Objekte.",
            "calculation_note": "Hinweis: Flaches ΛCDM-Modell angenommen.",
            "error_invalid_input": "Ungültige Eingabe für Rotverschiebungsberechnung.",
            "error_h0_positive": "H₀ muss positiv sein.",
            "error_omega_negative": "Ωm und ΩΛ dürfen nicht negativ sein.",
            "warn_blueshift": "Blueshift (z < 0): Objekt bewegt sich auf uns zu. Distanzen sind als 0 definiert, Rückblickzeit ist 0.",
            "warn_integration_accuracy": "Warnung: Integrationsgenauigkeit könnte gering sein (Fehler dc: {err_dc:.2e}, Fehler lt: {err_lt:.2e}).",
            "error_dep_scipy": "Fehler: Scipy wird für die Integration benötigt.",
            "error_calc_failed": "Berechnung fehlgeschlagen: {e}",
            "example_lookback_recent": "Praktisch die Gegenwart.",
            "example_lookback_humans": "Entspricht etwa der Zeitspanne seit der Entwicklung moderner Menschen.",
            "example_lookback_dinos": "Entspricht etwa der Zeitspanne seit dem Aussterben der Dinosaurier.",
            "example_lookback_multicellular": "Entspricht etwa der Entstehung von mehrzelligem Leben auf der Erde.",
            "example_lookback_earth": "Entspricht etwa dem Alter der Erde.",
            "example_lookback_early_univ": "Sehr frühes Universum.",
            "example_comoving_local": "Innerhalb der lokalen Gruppe von Galaxien.",
            "example_comoving_virgo": "Entfernung zum Virgo-Galaxienhaufen.",
            "example_comoving_coma": "Entfernung zum Coma-Galaxienhaufen.",
            "example_comoving_lss": "Großräumige Strukturen des Universums.",
            "example_comoving_quasars": "Typische Entfernungen zu fernen Quasaren.",
            "example_comoving_cmb": "Entfernung zur kosmischen Mikrowellenhintergrundstrahlung.",
            # Add other keys from the original script as needed for fallback
        },
        'en': {
            "recessional_velocity": "Recessional Velocity",
            "unit_km_s": "km/s",
            "redshift_calculator_title": "Redshift Calculator",
            "input_params": "Input Parameters",
            "redshift_z": "Redshift (z)",
            "redshift_z_tooltip": "Enter cosmological redshift",
            "cosmo_params": "Cosmological Parameters",
            "hubble_h0": "H₀ [km/s/Mpc]",
            "omega_m": "Ωm",
            "omega_lambda": "ΩΛ",
            "flat_universe_warning": "Ωm + ΩΛ ≉ 1. Assuming flat universe.",
            "results_for": "Results for z = {z:.5f}",
            "lookback_time": "Lookback Time",
            "unit_Gyr": "Gyr",
            "cosmo_distances": "Cosmological Distances",
            "comoving_distance_title": "**Comoving Distance:**",
            "unit_Mpc": "Mpc",
            "unit_Gly": "Gly",
            "unit_km_sci": "km (sci.)",
            "unit_km_full": "km (full)",
            "unit_LJ": "ly",
            "unit_AE": "AU",
            "unit_Ls": "Ls",
            "luminosity_distance_title": "**Luminosity Distance:**",
            "explanation_luminosity": "Relevant for the brightness of distant objects.",
            "angular_diameter_distance_title": "**Angular Diameter Distance:**",
            "explanation_angular": "Relevant for the apparent size of distant objects.",
            "calculation_note": "Note: Flat ΛCDM model assumed.",
            "error_invalid_input": "Invalid input for redshift calculation.",
            "error_h0_positive": "H₀ must be positive.",
            "error_omega_negative": "Ωm and ΩΛ must not be negative.",
            "warn_blueshift": "Blueshift (z < 0): Object is moving towards us. Distances are defined as 0, lookback time is 0.",
            "warn_integration_accuracy": "Warning: Integration accuracy might be low (error dc: {err_dc:.2e}, error lt: {err_lt:.2e}).",
            "error_dep_scipy": "Error: Scipy is required for integration.",
            "error_calc_failed": "Calculation failed: {e}",
            "example_lookback_recent": "Practically the present.",
            "example_lookback_humans": "Corresponds to roughly the time since modern humans evolved.",
            "example_lookback_dinos": "Corresponds to roughly the time since the dinosaurs went extinct.",
            "example_lookback_multicellular": "Corresponds to roughly the emergence of multicellular life on Earth.",
            "example_lookback_earth": "Corresponds to roughly the age of the Earth.",
            "example_lookback_early_univ": "Very early universe.",
            "example_comoving_local": "Within the Local Group of galaxies.",
            "example_comoving_virgo": "Distance to the Virgo Cluster.",
            "example_comoving_coma": "Distance to the Coma Cluster.",
            "example_comoving_lss": "Large-scale structures of the universe.",
            "example_comoving_quasars": "Typical distances to distant quasars.",
            "example_comoving_cmb": "Distance to the Cosmic Microwave Background.",
        },
        'fr': { # Basic French translations, can be improved
            "recessional_velocity": "Vitesse de Récession",
            "unit_km_s": "km/s",
            "redshift_calculator_title": "Calculateur de Décalage vers le Rouge",
            "input_params": "Paramètres d'Entrée",
            "redshift_z": "Décalage vers le Rouge (z)",
            "redshift_z_tooltip": "Entrez le décalage cosmologique vers le rouge",
            "cosmo_params": "Paramètres Cosmologiques",
            "hubble_h0": "H₀ [km/s/Mpc]",
            "omega_m": "Ωm",
            "omega_lambda": "ΩΛ",
            "flat_universe_warning": "Ωm + ΩΛ ≉ 1. Univers plat supposé.",
            "results_for": "Résultats pour z = {z:.5f}",
            "lookback_time": "Temps de Regard en Arrière",
            "unit_Gyr": "Ga", # Giga-années
            "cosmo_distances": "Distances Cosmologiques",
            "comoving_distance_title": "**Distance Comobile :**",
            "unit_Mpc": "Mpc",
            "unit_Gly": "Gal", # Giga-années-lumière
            "unit_km_sci": "km (sci.)",
            "unit_km_full": "km (complet)",
            "unit_LJ": "al", # Années-lumière
            "unit_AE": "UA", # Unités Astronomiques
            "unit_Ls": "sl", # Secondes-lumière
            "luminosity_distance_title": "**Distance de Luminosité :**",
            "explanation_luminosity": "Pertinent pour la luminosité des objets distants.",
            "angular_diameter_distance_title": "**Distance de Diamètre Angulaire :**",
            "explanation_angular": "Pertinent pour la taille apparente des objets distants.",
            "calculation_note": "Note : Modèle ΛCDM plat supposé.",
            "error_invalid_input": "Entrée invalide pour le calcul du décalage.",
            "error_h0_positive": "H₀ doit être positif.",
            "error_omega_negative": "Ωm et ΩΛ ne doivent pas être négatifs.",
            "warn_blueshift": "Décalage vers le bleu (z < 0) : L'objet se rapproche. Distances définies à 0, temps de regard en arrière à 0.",
            "warn_integration_accuracy": "Avertissement : Précision d'intégration faible (erreur dc : {err_dc:.2e}, erreur lt : {err_lt:.2e}).",
            "error_dep_scipy": "Erreur : Scipy requis pour l'intégration.",
            "error_calc_failed": "Échec du calcul : {e}",
            "example_lookback_recent": "Pratiquement le présent.",
            "example_lookback_humans": "Correspond approximativement à l'époque de l'évolution des humains modernes.",
            "example_lookback_dinos": "Correspond approximativement à l'extinction des dinosaures.",
            "example_lookback_multicellular": "Correspond approximativement à l'émergence de la vie multicellulaire.",
            "example_lookback_earth": "Correspond approximativement à l'âge de la Terre.",
            "example_lookback_early_univ": "Univers très primordial.",
            "example_comoving_local": "Au sein du Groupe Local de galaxies.",
            "example_comoving_virgo": "Distance à l'amas de la Vierge.",
            "example_comoving_coma": "Distance à l'amas de Coma.",
            "example_comoving_lss": "Structures à grande échelle de l'univers.",
            "example_comoving_quasars": "Distances typiques des quasars lointains.",
            "example_comoving_cmb": "Distance au fond diffus cosmologique.",
        }
    }
    def get_translation(lang, default_lang='en'):
        # Simplified fallback: return a dictionary for the requested language or default
        # In a real scenario, this would be more robust, handling missing keys etc.
        # The original script's get_translation likely returns an object with a .get() method.
        # For this example, we'll make it return the dict directly, and usage will be t[key] or t.get(key, default_val)
        
        # Make it behave more like the original by returning an object with a get method
        class TranslationWrapper:
            def __init__(self, lang_dict, fallback_dict):
                self.lang_dict = lang_dict
                self.fallback_dict = fallback_dict
            def get(self, key, default_value=None):
                val = self.lang_dict.get(key, self.fallback_dict.get(key, default_value if default_value is not None else key))
                # Ensure it returns a string if a format is expected
                if isinstance(val, str) and "{}" in val and default_value and "{}" not in default_value:
                     # This is a basic attempt to prevent errors if a format string is expected but not provided by fallback
                     # A more robust solution would be needed for complex cases.
                     return default_value
                return val

        _lang_dict = _translations.get(lang, _translations[default_lang])
        _fallback_dict = _translations[default_lang]
        return TranslationWrapper(_lang_dict, _fallback_dict)

# --- Page Config ---
st.set_page_config(page_title="Advanced DSO Finder", layout="wide")

# --- Global Configuration & Initial Values ---
INITIAL_LAT = 47.17
INITIAL_LON = 8.01
INITIAL_HEIGHT = 550
INITIAL_TIMEZONE = "Europe/Zurich"

# --- Path to Catalog File ---
try: APP_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError: APP_DIR = os.getcwd()
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
KM_PER_LS = C_KM_PER_S # Kilometers per light-second is c
GYR_PER_YR = 1e9 # Gigayears per year (used for Hubble time conversion)
H0_DEFAULT = 67.4
OMEGA_M_DEFAULT = 0.315
OMEGA_LAMBDA_DEFAULT = 0.685

# --- Initialize TimezoneFinder ---
@st.cache_resource
def get_timezone_finder():
    if TimezoneFinder:
        try: return TimezoneFinder(in_memory=True)
        except Exception as e: print(f"Error initializing TF: {e}"); st.warning(f"TF init failed: {e}. Auto TZ disabled."); return None
    return None
tf = get_timezone_finder()

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
    # (Unchanged from original)
    if not 0 <= illumination <= 1: print(f"Warn: Invalid moon illum ({illumination})."); illumination = max(0.0, min(1.0, illumination))
    radius = size / 2; cx = cy = radius
    light_color = "var(--text-color, #e0e0e0)"; dark_color = "var(--secondary-background-color, #333333)"
    svg = f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}"><circle cx="{cx}" cy="{cy}" r="{radius}" fill="{dark_color}"/>'
    if illumination < 0.01: pass
    elif illumination > 0.99: svg += f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{light_color}"/>'
    else:
        x_term = radius * (illumination * 2 - 1); rx_term = abs(x_term)
        if illumination <= 0.5: laf_e, sf_e, laf_c, sf_c = 0, 1, 0, 1
        else: laf_e, sf_e, laf_c, sf_c = 1, 1, 1, 1
        d = (f"M {cx},{cy - radius} A {rx_term},{radius} 0 {laf_e},{sf_e} {cx},{cy + radius} A {radius},{radius} 0 {laf_c},{sf_c} {cx},{cy - radius} Z")
        svg += f'<path d="{d}" fill="{light_color}"/>'
    return svg + '</svg>'

def load_ongc_data(catalog_path: str, lang: str) -> pd.DataFrame | None:
    # (Unchanged from original)
    t_load = get_translation(lang); required_cols = ['Name', 'RA', 'Dec', 'Type']; mag_cols = ['V-Mag', 'B-Mag', 'Mag']; size_col = 'MajAx'
    try:
        if not os.path.exists(catalog_path): st.error(f"{t_load.get('error_loading_catalog', 'Error:').split(':')[0]}: File not found"); return None
        df = pd.read_csv(catalog_path, sep=';', comment='#', low_memory=False)
        missing_req_cols = [col for col in required_cols if col not in df.columns]
        if missing_req_cols: st.error(f"Missing required columns: {', '.join(missing_req_cols)}"); return None
        df['RA_str'] = df['RA'].astype(str).str.strip(); df['Dec_str'] = df['Dec'].astype(str).str.strip()
        df.dropna(subset=['RA_str', 'Dec_str'], inplace=True); df = df[(df['RA_str'] != '') & (df['Dec_str'] != '')]
        mag_col_found = None
        for col in mag_cols:
            if col in df.columns:
                if pd.to_numeric(df[col], errors='coerce').notna().any(): mag_col_found = col; print(f"Using mag col: {mag_col_found}"); break
        if mag_col_found is None: st.error(f"No usable mag column ({', '.join(mag_cols)})"); return None
        df['Mag'] = pd.to_numeric(df[mag_col_found], errors='coerce'); df.dropna(subset=['Mag'], inplace=True)
        if size_col not in df.columns: st.warning(f"Size col '{size_col}' not found."); df[size_col] = np.nan
        else:
            df[size_col] = pd.to_numeric(df[size_col], errors='coerce')
            if not df[size_col].notna().any(): st.warning(f"No valid data in size col '{size_col}'."); df[size_col] = np.nan
        dso_types = ['Galaxy', 'Globular Cluster', 'Open Cluster', 'Nebula', 'Planetary Nebula', 'Supernova Remnant', 'HII', 'Emission Nebula',
                     'Reflection Nebula', 'Cluster + Nebula', 'Gal', 'GCl', 'Gx', 'OC', 'PN', 'SNR', 'Neb', 'EmN', 'RfN', 'C+N', 'Gxy', 'AGN', 'MWSC', 'OCl']
        type_pattern = '|'.join(dso_types)
        if 'Type' in df.columns: df_filtered = df[df['Type'].astype(str).str.contains(type_pattern, case=False, na=False)].copy()
        else: st.error("Missing 'Type' column."); return None
        final_cols = ['Name', 'RA_str', 'Dec_str', 'Mag', 'Type', size_col]; final_cols_exist = [col for col in final_cols if col in df_filtered.columns]
        df_final = df_filtered[final_cols_exist].copy()
        df_final.drop_duplicates(subset=['Name'], inplace=True, keep='first'); df_final.reset_index(drop=True, inplace=True)
        if not df_final.empty: print(f"Catalog loaded: {len(df_final)} objects."); return df_final
        else: st.warning(t_load.get('warning_catalog_empty', 'Catalog empty.')); return None
    except Exception as e: st.error(f"{t_load.get('error_loading_catalog', 'Catalog Error:')} {e}"); traceback.print_exc(); return None

def _get_fallback_window(reference_time: Time) -> tuple[Time, Time]:
    # (Unchanged from original)
    ref_dt = reference_time.to_datetime(timezone.utc); ref_date = ref_dt.date()
    start_dt = datetime.combine(ref_date, time(18, 0), tzinfo=timezone.utc); end_dt = datetime.combine(ref_date + timedelta(days=1), time(6, 0), tzinfo=timezone.utc)
    start_t = Time(start_dt, scale='utc'); end_t = Time(end_dt, scale='utc'); print(f"Using fallback window: {start_t.iso} to {end_t.iso}"); return start_t, end_t

def get_observable_window(observer: Observer, reference_time: Time, is_now: bool, lang: str) -> tuple[Time | None, Time | None, str]:
    # (Unchanged from original)
    t = get_translation(lang); status = ""; start_time, end_time = None, None; current_utc = Time.now()
    calc_base = reference_time
    if is_now:
        cur_dt = current_utc.to_datetime(timezone.utc); noon_today = datetime.combine(cur_dt.date(), time(12, 0), tzinfo=timezone.utc)
        calc_base = Time(noon_today - timedelta(days=1)) if cur_dt < noon_today else Time(noon_today)
    else: calc_base = Time(datetime.combine(reference_time.to_datetime(timezone.utc).date(), time(12, 0), tzinfo=timezone.utc), scale='utc')
    try:
        if not isinstance(observer, Observer): raise TypeError("Observer type error")
        set_t = observer.twilight_evening_astronomical(calc_base, which='next'); rise_t = observer.twilight_morning_astronomical(set_t if set_t else calc_base, which='next')
        if set_t is None or rise_t is None: raise ValueError("Cannot calc twilight")
        if rise_t <= set_t:
            try: # Polar check
                sun_ref = observer.sun_altaz(calc_base).alt; sun_12h = observer.sun_altaz(calc_base + 12*u.hour).alt
                if sun_ref < -18*u.deg and sun_12h < -18*u.deg: status = t.get('error_polar_night', "Polar night?"); start_time, end_time = _get_fallback_window(calc_base)
                elif sun_ref > -18*u.deg:
                    times_chk = calc_base + np.linspace(0, 24, 49)*u.hour; sun_alts_chk = observer.sun_altaz(times_chk).alt
                    if np.min(sun_alts_chk) > -18*u.deg: status = t.get('error_polar_day', "Polar day?"); start_time, end_time = _get_fallback_window(calc_base)
                if start_time: status += t.get('window_fallback_info', "\nFallback: {} to {} UTC").format(start_time.iso, end_time.iso); return start_time, end_time, status
            except Exception as check_e: print(f"Polar check err: {check_e}")
            raise ValueError("Rise <= Set twilight") # Raise if not polar & failed
        start_time, end_time = set_t, rise_t
        if is_now:
            if end_time < current_utc:
                status = t.get('window_already_passed', "Window passed.") + "\n"; next_noon = datetime.combine(current_utc.to_datetime(timezone.utc).date() + timedelta(days=1), time(12, 0), tzinfo=timezone.utc)
                set_next = observer.twilight_evening_astronomical(Time(next_noon), which='next'); rise_next = observer.twilight_morning_astronomical(set_next if set_next else Time(next_noon), which='next')
                if set_next is None or rise_next is None or rise_next <= set_next: raise ValueError("Cannot calc next twilight")
                start_time, end_time = set_next, rise_next
            elif start_time < current_utc: print(f"Adjust win start {start_time.iso} -> {current_utc.iso}"); start_time = current_utc
        start_f = start_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M %Z'); end_f = end_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M %Z')
        status += t.get('window_info_template', "Window: {} to {} UTC").format(start_f, end_f)
    except Exception as e:
        status = t.get('window_calc_error', "Win Err: {}\n{}").format(e, traceback.format_exc()); print(f"Win err: {e}")
        start_time, end_time = _get_fallback_window(calc_base)
        status += t.get('window_fallback_info', "\nFallback: {} to {} UTC").format(start_time.iso, end_time.iso)
    if start_time is None or end_time is None or end_time <= start_time: # Final fallback check
        if not status or "Error" not in status and "Fallback" not in status: status += ("\n" if status else "") + t.get('error_no_window', "No valid window.")
        start_fb, end_fb = _get_fallback_window(calc_base)
        if t.get('window_fallback_info', "\nFallback: {} to {} UTC").format(start_fb.iso, end_fb.iso) not in status: status += t.get('window_fallback_info', "\nFallback: {} to {} UTC").format(start_fb.iso, end_fb.iso)
        start_time, end_time = start_fb, end_fb
    return start_time, end_time, status

def find_observable_objects(observer_location: EarthLocation, observing_times: Time, min_altitude_limit: u.Quantity, catalog_df: pd.DataFrame, lang: str) -> list[dict]:
    # (Unchanged from original)
    t = get_translation(lang); observable_objects = []
    if not isinstance(observer_location, EarthLocation): st.error("Internal Error: observer_location type"); return []
    if not isinstance(observing_times, Time) or not observing_times.shape: st.error("Internal Error: observing_times type"); return []
    if not isinstance(min_altitude_limit, u.Quantity): st.error("Internal Error: min_altitude_limit type"); return []
    if not isinstance(catalog_df, pd.DataFrame): st.error("Internal Error: catalog_df type"); return []
    if catalog_df.empty: print("Input catalog_df empty."); return []
    if len(observing_times) < 2: st.warning("Obs window < 2 points.")
    altaz_frame = AltAz(obstime=observing_times, location=observer_location); min_alt_deg = min_altitude_limit.to(u.deg).value
    time_step_h = (observing_times[1] - observing_times[0]).sec / 3600.0 if len(observing_times) > 1 else 0
    for index, obj in catalog_df.iterrows():
        try:
            ra, dec, name, type, mag, size = obj.get('RA_str'), obj.get('Dec_str'), obj.get('Name', f"Obj {index}"), obj.get('Type', "?"), obj.get('Mag', np.nan), obj.get('MajAx', np.nan)
            if not ra or not dec: print(f"Skip '{name}': No RA/Dec."); continue
            try: coord = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))
            except ValueError as coord_e: print(f"Skip '{name}': Bad coords {coord_e}"); continue
            try: altazs = coord.transform_to(altaz_frame); alts = altazs.alt.to(u.deg).value; azs = altazs.az.to(u.deg).value
            except Exception as trans_e: print(f"Skip '{name}': Transform err {trans_e}"); continue
            max_alt = np.max(alts) if len(alts) > 0 else -999
            if max_alt >= min_alt_deg:
                peak_idx = np.argmax(alts); peak_alt = alts[peak_idx]; peak_time = observing_times[peak_idx]; peak_az = azs[peak_idx]; peak_dir = azimuth_to_direction(peak_az)
                try: const = get_constellation(coord)
                except Exception as const_e: print(f"Warn: Const fail {name} {const_e}"); const = "N/A"
                above_min = alts >= min_alt_deg; cont_dur_h = 0
                if time_step_h > 0 and np.any(above_min):
                    runs = np.split(np.arange(len(above_min)), np.where(np.diff(above_min))[0]+1); max_len = 0
                    for run in runs:
                        if run.size > 0 and above_min[run[0]]: max_len = max(max_len, len(run))
                    cont_dur_h = max_len * time_step_h
                result = {
                    'Name': name, 'Type': type, 'Constellation': const, 'Magnitude': mag if not np.isnan(mag) else None,
                    'Size (arcmin)': size if not np.isnan(size) else None, 'RA': ra, 'Dec': dec, 'Max Altitude (°)': peak_alt,
                    'Azimuth at Max (°)': peak_az, 'Direction at Max': peak_dir, 'Time at Max (UTC)': peak_time,
                    'Max Cont. Duration (h)': cont_dur_h, 'skycoord': coord, 'altitudes': alts, 'azimuths': azs, 'times': observing_times }
                observable_objects.append(result)
        except Exception as obj_e: print(t.get('error_processing_object', "Err proc {}: {}").format(obj.get('Name', f'Obj {index}'), obj_e))
    return observable_objects

def get_local_time_str(utc_time: Time | None, timezone_str: str) -> tuple[str, str]:
    # (Unchanged from original)
    if utc_time is None: return "N/A", "N/A"
    if not isinstance(utc_time, Time): print(f"Err: utc_time type {type(utc_time)}"); return "N/A", "N/A"
    if not isinstance(timezone_str, str) or not timezone_str: print(f"Err: tz_str type {timezone_str}"); return "N/A", "N/A"
    try:
        local_tz = pytz.timezone(timezone_str); utc_dt = utc_time.to_datetime(timezone.utc); local_dt = utc_dt.astimezone(local_tz)
        local_str = local_dt.strftime('%Y-%m-%d %H:%M:%S'); tz_name = local_dt.tzname(); tz_name = tz_name if tz_name else local_tz.zone
        return local_str, tz_name
    except pytz.exceptions.UnknownTimeZoneError: print(f"Err: Unknown TZ '{timezone_str}'."); return utc_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), "UTC (TZ Err)"
    except Exception as e: print(f"Err converting time: {e}"); traceback.print_exc(); return utc_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), "UTC (Conv Err)"

# --- Redshift Calculation Functions ---
def hubble_parameter_inv_integrand(z, omega_m, omega_lambda):
  # (Unchanged from original)
  epsilon = 1e-15; denom = np.sqrt(omega_m * (1 + z)**3 + omega_lambda + epsilon)
  return 1.0 / denom if denom >= epsilon else 0.0

def lookback_time_integrand(z, omega_m, omega_lambda):
  # (Unchanged from original)
  epsilon = 1e-15; term = omega_m * (1 + z)**3 + omega_lambda; term = max(term, 0)
  denom = (1 + z) * np.sqrt(term + epsilon)
  if math.isclose(z, 0): denom_zero = np.sqrt(omega_m + omega_lambda + epsilon); return 1.0/denom_zero if denom_zero >= epsilon else 0.0
  return 1.0 / denom if abs(denom) >= epsilon else 0.0

@st.cache_data
def calculate_lcdm_distances(redshift: float, h0: float, omega_m: float, omega_lambda: float) -> dict:
    """
    Calculates cosmological distances and related parameters for a given redshift
    in a flat Lambda-CDM model. Also calculates recessional velocity (v = z*c).
    """
    if not all(isinstance(v, (int, float)) for v in [redshift, h0, omega_m, omega_lambda]):
        return {'error_key': "error_invalid_input"}
    if h0 <= 0:
        return {'error_key': "error_h0_positive"}
    if omega_m < 0 or omega_lambda < 0: # Allow omega_k != 0 if not strictly flat
        return {'error_key': "error_omega_negative"}

    # Recessional velocity (v = z*c)
    # This is the classical Doppler formula, a good approximation for low z
    # and represents the apparent velocity due to Hubble expansion.
    recessional_velocity_km_s = redshift * C_KM_PER_S

    if redshift < 0: # Blueshift case
        return {
            'comoving_mpc': 0.0, 'luminosity_mpc': 0.0, 'ang_diam_mpc': 0.0,
            'lookback_gyr': 0.0, 'recessional_velocity_km_s': recessional_velocity_km_s,
            'error_key': "warn_blueshift" # Indicates it's a blueshift
        }
    if math.isclose(redshift, 0):
        return {
            'comoving_mpc': 0.0, 'luminosity_mpc': 0.0, 'ang_diam_mpc': 0.0,
            'lookback_gyr': 0.0, 'recessional_velocity_km_s': 0.0,
            'error_key': None
        }

    dh = C_KM_PER_S / h0  # Hubble distance in Mpc
    hubble_time_gyr = 1.0 / (h0 * (KM_PER_MPC / C_KM_PER_S) / (3600 * 24 * 365.25 * GYR_PER_YR)) # Hubble time in Gyr more directly

    try:
        # Comoving distance integral
        integral_dc, err_dc = quad(hubble_parameter_inv_integrand, 0, redshift, args=(omega_m, omega_lambda), limit=100)
        comoving_mpc = dh * integral_dc

        # Lookback time integral
        integral_lt, err_lt = quad(lookback_time_integrand, 0, redshift, args=(omega_m, omega_lambda), limit=100)
        lookback_gyr = hubble_time_gyr * integral_lt
        
        luminosity_mpc = comoving_mpc * (1 + redshift)
        ang_diam_mpc = comoving_mpc / (1 + redshift)

        warning_key, warning_args = None, {}
        if err_dc > 1e-5 or err_lt > 1e-5: # Check integration accuracy
            warning_key = "warn_integration_accuracy"
            warning_args = {'err_dc': err_dc, 'err_lt': err_lt}

        return {
            'comoving_mpc': comoving_mpc,
            'luminosity_mpc': luminosity_mpc,
            'ang_diam_mpc': ang_diam_mpc,
            'lookback_gyr': lookback_gyr,
            'recessional_velocity_km_s': recessional_velocity_km_s,
            'error_key': None,
            'warning_key': warning_key,
            'warning_args': warning_args
        }
    except ImportError: # Should be caught at startup, but as a safeguard
        return {'error_key': "error_dep_scipy"}
    except Exception as e:
        st.exception(e) # Log the full exception for debugging
        return {'error_key': "error_calc_failed", 'error_args': {'e': str(e)}}


# --- Redshift Unit Conversion & Formatting ---
def convert_mpc_to_km(d_mpc: float) -> float: return d_mpc * KM_PER_MPC
def convert_km_to_au(d_km: float) -> float: return 0.0 if d_km == 0 else d_km / KM_PER_AU
def convert_km_to_ly(d_km: float) -> float: return 0.0 if d_km == 0 else d_km / KM_PER_LY
def convert_km_to_ls(d_km: float) -> float: return 0.0 if d_km == 0 else d_km / KM_PER_LS # d_km / C_KM_PER_S

# === KORRIGIERTE FUNKTION (wie vom Nutzer bereitgestellt) ===
def convert_mpc_to_gly(d_mpc: float) -> float:
    """Converts Megaparsecs to Gigalightyears."""
    if d_mpc == 0:
        return 0.0
    # These assignments must be outside the if d_mpc == 0 block
    # KM_PER_LY is km per lightyear. km_per_gly is km per gigalightyear.
    km_per_gly = KM_PER_LY * 1e9 # 1 Gigalightyear = 1 billion lightyears
    dist_km = convert_mpc_to_km(d_mpc)
    # The check km_per_gly == 0 is technically unnecessary as KM_PER_LY > 0
    return dist_km / km_per_gly
# ===========================

def format_large_number(number: float | int) -> str:
    # (Unchanged from original)
    if number == 0: return "0";
    if not np.isfinite(number): return str(number)
    try: return f"{number:,.0f}".replace(",", " ") # Using space as thousands separator
    except (ValueError, TypeError): return str(number)

# --- Redshift Example Helpers ---
def get_lookback_comparison_key(gyr: float) -> str:
    # (Unchanged from original)
    if gyr < 0.001: return "example_lookback_recent"
    if gyr < 0.05: return "example_lookback_humans"
    if gyr < 0.3: return "example_lookback_dinos"
    if gyr < 1.0: return "example_lookback_multicellular"
    if gyr < 5.0: return "example_lookback_earth"
    return "example_lookback_early_univ"

def get_comoving_comparison_key(mpc: float) -> str:
    # (Unchanged from original)
    if mpc < 5: return "example_comoving_local"
    if mpc < 50: return "example_comoving_virgo"
    if mpc < 200: return "example_comoving_coma"
    if mpc < 1000: return "example_comoving_lss"
    if mpc < 8000: return "example_comoving_quasars"
    return "example_comoving_cmb"

# --- Plotting Function ---
def create_plot(plot_data: dict, min_altitude_deg: float, max_altitude_deg: float, plot_type: str, lang: str) -> plt.Figure | None:
    # (Unchanged from original)
    t = get_translation(lang); fig = None
    try:
        if not isinstance(plot_data, dict): st.error("Plot Err: Invalid data."); return None
        times, alts, azs = plot_data.get('times'), plot_data.get('altitudes'), plot_data.get('azimuths')
        name = plot_data.get('Name', 'Object')
        if not isinstance(times, Time) or not isinstance(alts, np.ndarray): st.error("Plot Err: Invalid times/alts."); return None
        if plot_type == 'Sky Path' and not isinstance(azs, np.ndarray): st.error("Plot Err: Invalid azs."); return None
        if len(times) != len(alts) or (azs is not None and len(times) != len(azs)): st.error("Plot Err: Mismatch arrays."); return None
        if len(times) < 1: st.error("Plot Err: No data."); return None
        plot_times = times.plot_date
        try: is_dark = (st.get_option("theme.base") == "dark")
        except Exception: is_dark = False # Fallback if theme option not accessible
        plt.style.use('dark_background' if is_dark else 'default')
        lbl_col = '#FAFAFA' if is_dark else '#333333'; title_col = '#FFFFFF' if is_dark else '#000000'; grid_col = '#444444' if is_dark else 'darkgray'
        prim_col = 'deepskyblue' if is_dark else 'dodgerblue'; min_col = 'tomato' if is_dark else 'red'; max_col = 'orange' if is_dark else 'darkorange'
        spine_col = '#AAAAAA' if is_dark else '#555555'; legend_face = '#262730' if is_dark else '#F0F0F0'; face_col = '#0E1117' if is_dark else '#FFFFFF'
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=face_col, constrained_layout=True); ax.set_facecolor(face_col)
        if plot_type == 'Altitude Plot':
            ax.plot(plot_times, alts, color=prim_col, alpha=0.9, lw=1.5, label=name)
            ax.axhline(min_altitude_deg, color=min_col, ls='--', lw=1.2, label=t.get('graph_min_altitude_label', "Min Alt ({:.0f}°)").format(min_altitude_deg), alpha=0.8)
            if max_altitude_deg < 90: ax.axhline(max_altitude_deg, color=max_col, ls=':', lw=1.2, label=t.get('graph_max_altitude_label', "Max Alt ({:.0f}°)").format(max_altitude_deg), alpha=0.8)
            ax.set(xlabel="Time (UTC)", ylabel=t.get('graph_ylabel', "Altitude (°)"), title=t.get('graph_title_alt_time', "Alt Plot for {}").format(name), ylim=(0, 90))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')); fig.autofmt_xdate(rotation=30)
        elif plot_type == 'Sky Path':
            if azs is None: st.error("Plot Err: Azimuths needed."); plt.close(fig); return None
            ax.remove(); ax = fig.add_subplot(111, projection='polar', facecolor=face_col)
            az_rad = np.deg2rad(azs); radius = 90 - alts
            time_delta = times.jd.max() - times.jd.min(); time_norm = (times.jd - times.jd.min()) / (time_delta + 1e-9); colors = plt.cm.plasma(time_norm)
            scatter = ax.scatter(az_rad, radius, c=colors, s=15, alpha=0.8, edgecolors='none', label=name)
            ax.plot(az_rad, radius, color=prim_col, alpha=0.4, lw=0.8)
            ax.plot(np.linspace(0, 2*np.pi, 100), np.full(100, 90 - min_altitude_deg), color=min_col, ls='--', lw=1.2, label=t.get('graph_min_altitude_label', "Min Alt ({:.0f}°)").format(min_altitude_deg), alpha=0.8)
            if max_altitude_deg < 90: ax.plot(np.linspace(0, 2*np.pi, 100), np.full(100, 90 - max_altitude_deg), color=max_col, ls=':', lw=1.2, label=t.get('graph_max_altitude_label', "Max Alt ({:.0f}°)").format(max_altitude_deg), alpha=0.8)
            ax.set_theta_zero_location('N'); ax.set_theta_direction(-1); ax.set_yticks(np.arange(0, 91, 15)); ax.set_yticklabels([f"{90-alt}°" for alt in np.arange(0, 91, 15)], color=lbl_col)
            ax.set(ylim=(0, 90), title=t.get('graph_title_sky_path', "Sky Path for {}").format(name)); ax.title.set(va='bottom', color=title_col, fontsize=13, weight='bold', y=1.1)
            try:
                cbar = fig.colorbar(scatter, ax=ax, label="Time (UTC)", pad=0.1, shrink=0.7)
                start_lbl, end_lbl = (times[0].to_datetime(timezone.utc).strftime('%H:%M'), times[-1].to_datetime(timezone.utc).strftime('%H:%M')) if len(times)>0 else ('Start', 'End')
                cbar.set_ticks([0, 1]); cbar.ax.set_yticklabels([start_lbl, end_lbl])
                cbar.set_label("Time (UTC)", color=lbl_col, fontsize=10); cbar.ax.yaxis.set_tick_params(color=lbl_col, labelsize=9)
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=lbl_col); cbar.outline.set_edgecolor(spine_col); cbar.outline.set_linewidth(0.5)
            except Exception as cbar_e: print(f"Warn: Cbar fail: {cbar_e}")
        else: st.error(f"Plot Err: Unknown type '{plot_type}'"); plt.close(fig); return None
        ax.grid(True, linestyle=':', alpha=0.5, color=grid_col); ax.tick_params(axis='x', colors=lbl_col); ax.tick_params(axis='y', colors=lbl_col)
        for spine in ax.spines.values(): spine.set_color(spine_col); spine.set_linewidth(0.5)
        legend = ax.legend(loc='lower right', fontsize='small', facecolor=legend_face, framealpha=0.8, edgecolor=spine_col)
        for text in legend.get_texts(): text.set_color(lbl_col)
        return fig
    except Exception as e: st.error(f"Plot Err: Unexpected: {e}"); traceback.print_exc(); plt.close(fig); return None

# --- Main App ---
def main():
    initialize_session_state()

    # Get Language and Translations
    lang = st.session_state.language
    t = get_translation(lang) # t is now an object with a .get() method or a dict
    actual_lang_keys = ['de', 'en', 'fr'] # Supported languages
    if lang not in actual_lang_keys:
        print(f"Info: Invalid lang '{lang}' in state, reset to 'de'.")
        st.session_state.language = 'de'; lang = 'de'; t = get_translation(lang)

    # Load Catalog Data
    @st.cache_data
    def cached_load_ongc_data(path, current_lang): return load_ongc_data(path, current_lang)
    df_catalog_data = cached_load_ongc_data(CATALOG_FILEPATH, lang)

    st.title(t.get("app_title", "Advanced DSO Finder")) # Assuming "app_title" key

    # Object Type Glossary (Unchanged from original)
    with st.expander(t.get('object_type_glossary_title', "Object Type Glossary")):
        glossary_items = t.get('object_type_glossary', {})
        if isinstance(glossary_items, dict) and glossary_items: # Check if dict and not empty
             col1, col2 = st.columns(2); sorted_items = sorted(glossary_items.items())
             for i, (abbr, name) in enumerate(sorted_items): (col1 if i % 2 == 0 else col2).markdown(f"**{abbr}:** {name}")
        else: st.info(t.get("glossary_not_available", "Glossary not available for this language or empty."))
    st.markdown("---")

    # --- Sidebar ---
    with st.sidebar:
        st.header(t.get('settings_header', "Settings"))
        # Catalog Status (Unchanged from original)
        if 'catalog_status_msg' not in st.session_state: st.session_state.catalog_status_msg = ""
        if df_catalog_data is not None: msg = t.get('info_catalog_loaded', "Cat: {} obj.").format(len(df_catalog_data)); msg_func = st.success
        else: msg = t.get('error_catalog_failed', "Catalog failed."); msg_func = st.error # Added translation key
        if st.session_state.catalog_status_msg != msg: msg_func(msg); st.session_state.catalog_status_msg = msg

        # Language Selector (Unchanged from original)
        lang_opts = {'de': 'Deutsch', 'en': 'English', 'fr': 'Français'}; lang_keys = list(lang_opts.keys())
        curr_idx = lang_keys.index(lang) if lang in lang_keys else 0
        sel_key = st.radio(t.get('language_select_label', "Language"), options=lang_keys, format_func=lang_opts.get, key='language_radio', index=curr_idx, horizontal=True)
        if sel_key != st.session_state.language: st.session_state.language = sel_key; st.session_state.location_search_status_msg = ""; st.rerun()

        # Location Settings (Largely unchanged, minor translation key additions for consistency)
        with st.expander(t.get('location_expander', "📍 Location"), expanded=True):
            loc_opts_map = {'Search': t.get('location_option_search', "Search"), 'Manual': t.get('location_option_manual', "Manual")}
            st.radio(t.get('location_select_label', "Method"), options=list(loc_opts_map.keys()), format_func=lambda k: loc_opts_map[k], key="location_choice_key", horizontal=True)
            lat_val, lon_val, h_val, loc_valid_tz, curr_loc_valid = None, None, None, False, False
            if st.session_state.location_choice_key == "Manual":
                st.number_input(t.get('location_lat_label', "Lat (°N)"), -90.0, 90.0, step=0.01, format="%.4f", key="manual_lat_val")
                st.number_input(t.get('location_lon_label', "Lon (°E)"), -180.0, 180.0, step=0.01, format="%.4f", key="manual_lon_val")
                st.number_input(t.get('location_elev_label', "Elev (m)"), -500, step=10, format="%d", key="manual_height_val")
                lat_val, lon_val, h_val = st.session_state.manual_lat_val, st.session_state.manual_lon_val, st.session_state.manual_height_val
                if isinstance(lat_val, (int, float)) and isinstance(lon_val, (int, float)) and isinstance(h_val, (int, float)):
                    loc_valid_tz, curr_loc_valid = True, True; st.session_state.location_is_valid_for_run = True
                    if st.session_state.location_search_success: st.session_state.update({'location_search_success': False, 'searched_location_name': None, 'location_search_status_msg': ""})
                else: st.warning(t.get('location_error_manual_none', "Manual fields invalid.")); curr_loc_valid = False; st.session_state.location_is_valid_for_run = False
            elif st.session_state.location_choice_key == "Search":
                with st.form("loc_search_form"):
                    st.text_input(t.get('location_search_label', "Loc Name:"), key="location_search_query", placeholder=t.get('location_search_placeholder', "..."))
                    st.number_input(t.get('location_elev_label', "Elev (m)"), -500, step=10, format="%d", key="manual_height_val") # Elevation still manual here
                    submitted = st.form_submit_button(t.get('location_search_submit_button', "Find"))
                status_ph = st.empty()
                if st.session_state.location_search_status_msg: (status_ph.success if st.session_state.location_search_success else status_ph.error)(st.session_state.location_search_status_msg)
                if submitted and st.session_state.location_search_query:
                    loc, svc, err = None, None, None; query = st.session_state.location_search_query; agent = f"AdvDSO/{random.randint(1000,9999)}"
                    with st.spinner(t.get('spinner_geocoding', "Searching...")):
                        try: print("Try Nomi..."); geo = Nominatim(user_agent=agent, timeout=10); loc = geo.geocode(query); svc = "Nominatim" if loc else None; print(f"Nomi: {svc}")
                        except (GeocoderTimedOut, GeocoderServiceError, Exception) as e_n: print(f"Nomi fail: {e_n}"); status_ph.info(t.get('location_search_info_fallback', "Trying fallback geocoder...")); err = e_n
                        if not loc:
                            try: print("Try Arc..."); geo_a = ArcGIS(timeout=15); loc = geo_a.geocode(query); svc = "ArcGIS" if loc else None; print(f"Arc: {svc}")
                            except (GeocoderTimedOut, GeocoderServiceError, Exception) as e_a: print(f"Arc fail: {e_a}"); status_ph.info(t.get('location_search_info_fallback2', "Trying another fallback...")); err = e_a if not err else err
                        if not loc:
                            try: print("Try Phot..."); geo_p = Photon(user_agent=agent, timeout=15); loc = geo_p.geocode(query); svc = "Photon" if loc else None; print(f"Phot: {svc}")
                            except (GeocoderTimedOut, GeocoderServiceError, Exception) as e_p: print(f"Phot fail: {e_p}"); err = e_p if not err else err
                        if loc and svc:
                            f_lat, f_lon, f_name = loc.latitude, loc.longitude, loc.address
                            st.session_state.update({'searched_location_name': f_name, 'location_search_success': True, 'manual_lat_val': f_lat, 'manual_lon_val': f_lon})
                            coord_str = t.get('location_search_coords', "Lat: {:.4f}, Lon: {:.4f}").format(f_lat, f_lon)
                            f_key = 'location_search_found' if svc=="Nominatim" else ('location_search_found_fallback' if svc=="ArcGIS" else 'location_search_found_fallback2')
                            st.session_state.location_search_status_msg = f"{t.get(f_key, 'Found: {} ({})').format(f_name, svc)}\n({coord_str})" # Added service name
                            status_ph.success(st.session_state.location_search_status_msg)
                            lat_val, lon_val, h_val = f_lat, f_lon, st.session_state.manual_height_val
                            loc_valid_tz, curr_loc_valid = True, True; st.session_state.location_is_valid_for_run = True
                        else:
                            st.session_state.update({'location_search_success': False, 'searched_location_name': None})
                            if err:
                                if isinstance(err, GeocoderTimedOut): e_key = 'location_search_error_timeout'; fmt_arg = None
                                elif isinstance(err, GeocoderServiceError): e_key = 'location_search_error_service'; fmt_arg = str(err)
                                else: e_key = 'location_search_error_fallback2_failed'; fmt_arg = str(err)
                                st.session_state.location_search_status_msg = t.get(e_key, "Geocoding Error: {}").format(fmt_arg) if fmt_arg else t.get(e_key, "Geocoding Error")
                            else: st.session_state.location_search_status_msg = t.get('location_search_error_not_found', "Location not found.")
                            status_ph.error(st.session_state.location_search_status_msg)
                            curr_loc_valid = False; st.session_state.location_is_valid_for_run = False
                elif st.session_state.location_search_success:
                    lat_val, lon_val, h_val = st.session_state.manual_lat_val, st.session_state.manual_lon_val, st.session_state.manual_height_val
                    loc_valid_tz, curr_loc_valid = True, True; st.session_state.location_is_valid_for_run = True
                    status_ph.success(st.session_state.location_search_status_msg)
                else: curr_loc_valid = False; st.session_state.location_is_valid_for_run = False
            st.markdown("---")
            tz_msg = "";
            if loc_valid_tz and lat_val is not None and lon_val is not None:
                if tf:
                    try: f_tz = tf.timezone_at(lng=lon_val, lat=lat_val)
                    except Exception as tz_e: print(f"TF err: {tz_e}"); f_tz = None
                    if f_tz:
                        try: pytz.timezone(f_tz); st.session_state.selected_timezone = f_tz; tz_msg = f"{t.get('timezone_auto_set_label', 'TZ:')} **{f_tz}**"
                        except pytz.UnknownTimeZoneError: st.session_state.selected_timezone = 'UTC'; tz_msg = f"{t.get('timezone_auto_fail_label', 'TZ:')} **UTC** ({t.get('timezone_error_invalid', 'Invalid')}: {f_tz})"
                    else: st.session_state.selected_timezone = 'UTC'; tz_msg = f"{t.get('timezone_auto_fail_label', 'TZ:')} **UTC** ({t.get('timezone_auto_fail_msg', 'Failed to determine')})"
                else: tz_msg = f"{t.get('timezone_auto_fail_label', 'TZ:')} **{INITIAL_TIMEZONE}** ({t.get('timezone_auto_na', 'Auto N/A')})"; st.session_state.selected_timezone = INITIAL_TIMEZONE
            else: tz_msg = f"{t.get('timezone_auto_fail_label', 'TZ:')} **{st.session_state.selected_timezone}** ({t.get('timezone_loc_invalid', 'Location Invalid')})"
            st.markdown(tz_msg, unsafe_allow_html=True)

        # Time Settings (Unchanged from original)
        with st.expander(t.get('time_expander', "⏱️ Time"), expanded=False):
            time_opts = {'Now': t.get('time_option_now', "Now"), 'Specific': t.get('time_option_specific', "Specific")}
            st.radio(t.get('time_select_label', "Time"), options=list(time_opts.keys()), format_func=lambda k: time_opts[k], key="time_choice_exp", horizontal=True)
            if st.session_state.time_choice_exp == "Now": st.caption(f"UTC: {Time.now().iso}")
            else: st.date_input(t.get('time_date_select_label', "Date:"), value=st.session_state.selected_date_widget, key='selected_date_widget')

        # Filter Settings (Unchanged from original)
        with st.expander(t.get('filters_expander', "✨ Filters"), expanded=False):
            st.markdown(t.get('mag_filter_header', "**Mag Filter**")); mag_opts = {'Bortle Scale': t.get('mag_filter_option_bortle', "Bortle"), 'Manual': t.get('mag_filter_option_manual', "Manual")}
            st.radio(t.get('mag_filter_method_label', "Method:"), options=list(mag_opts.keys()), format_func=lambda k: mag_opts[k], key="mag_filter_mode_exp", horizontal=True)
            st.slider(t.get('mag_filter_bortle_label', "Bortle:"), 1, 9, key='bortle_slider', help=t.get('mag_filter_bortle_help', "..."))
            if st.session_state.mag_filter_mode_exp == "Manual":
                st.slider(t.get('mag_filter_min_mag_label', "Min:"), -5.0, 20.0, step=0.5, format="%.1f", help=t.get('mag_filter_min_mag_help', "..."), key='manual_min_mag_slider')
                st.slider(t.get('mag_filter_max_mag_label', "Max:"), -5.0, 20.0, step=0.5, format="%.1f", help=t.get('mag_filter_max_mag_help', "..."), key='manual_max_mag_slider')
                if st.session_state.manual_min_mag_slider > st.session_state.manual_max_mag_slider: st.warning(t.get('mag_filter_warning_min_max', "Min > Max!"))
            st.markdown("---"); st.markdown(t.get('min_alt_header', "**Altitude**"))
            min_alt, max_alt = st.session_state.min_alt_slider, st.session_state.max_alt_slider;
            if min_alt > max_alt: st.session_state.min_alt_slider = max_alt; min_alt = max_alt # Ensure min_alt is not greater than max_alt
            st.slider(t.get('min_alt_label', "Min (°):"), 0, 90, key='min_alt_slider', step=1); st.slider(t.get('max_alt_label', "Max (°):"), 0, 90, key='max_alt_slider', step=1)
            if st.session_state.min_alt_slider > st.session_state.max_alt_slider: st.warning(t.get("alt_filter_warning_min_max", "Min Altitude > Max Altitude!")) # Added translation key
            st.markdown("---"); st.markdown(t.get('moon_warning_header', "**Moon**")); st.slider(t.get('moon_warning_label', "Warn > (%):"), 0, 100, key='moon_phase_slider', step=5)
            st.markdown("---"); st.markdown(t.get('object_types_header', "**Types**")); all_types = []
            if df_catalog_data is not None and 'Type' in df_catalog_data.columns:
                try: all_types = sorted(list(df_catalog_data['Type'].dropna().astype(str).unique()))
                except Exception as e: st.warning(f"{t.get('object_types_error_extract', 'Type Err')}: {e}")
            if all_types:
                sel = [s for s in st.session_state.object_type_filter_exp if s in all_types];
                if sel != st.session_state.object_type_filter_exp: st.session_state.object_type_filter_exp = sel
                st.multiselect(t.get('object_types_label', "Filter Types:"), options=all_types, default=sel, key="object_type_filter_exp")
            else: st.info(t.get("object_types_not_found", "No types found in catalog.")); st.session_state.object_type_filter_exp = [] # Added translation key
            st.markdown("---"); st.markdown(t.get('size_filter_header', "**Size**")); size_ok = df_catalog_data is not None and 'MajAx' in df_catalog_data.columns and df_catalog_data['MajAx'].notna().any(); size_disabled = not size_ok
            if size_ok:
                try:
                    valid_sz = df_catalog_data['MajAx'].dropna(); min_p = max(0.1, float(valid_sz.min())) if not valid_sz.empty else 0.1; max_p = float(valid_sz.max()) if not valid_sz.empty else 120.0
                    min_s, max_s = st.session_state.size_arcmin_range; c_min = max(min_p, min(min_s, max_p)); c_max = min(max_p, max(max_s, min_p))
                    if c_min > c_max: c_min = c_max
                    if (c_min, c_max) != st.session_state.size_arcmin_range: st.session_state.size_arcmin_range = (c_min, c_max)
                    step = 0.1 if max_p <= 20 else (0.5 if max_p <= 100 else 1.0)
                    st.slider(t.get('size_filter_label', "Size (arcmin):"), min_p, max_p, step=step, format="%.1f'", key='size_arcmin_range', help=t.get('size_filter_help', "..."), disabled=size_disabled)
                except Exception as sz_e: st.error(f"{t.get('size_slider_error', 'Size slider error')}: {sz_e}"); size_disabled = True # Added translation key
            else: st.info(t.get("size_data_not_available", "Size data not available in catalog.")); size_disabled = True # Added translation key
            if size_disabled: st.slider(t.get('size_filter_label', "Size (arcmin):"), 0.0, 1.0, (0.0, 1.0), key='size_disabled_placeholder', disabled=True) # Changed key to avoid conflict
            st.markdown("---"); st.markdown(t.get('direction_filter_header', "**Direction**")); all_str = t.get('direction_option_all', "All"); dir_disp = [all_str] + CARDINAL_DIRECTIONS; dir_int = [ALL_DIRECTIONS_KEY] + CARDINAL_DIRECTIONS
            curr_int = st.session_state.selected_peak_direction;
            if curr_int not in dir_int: curr_int = ALL_DIRECTIONS_KEY; st.session_state.selected_peak_direction = curr_int
            try: curr_idx_dir = dir_int.index(curr_int)
            except ValueError: curr_idx_dir = 0
            sel_disp_dir = st.selectbox(t.get('direction_filter_label', "Direction:"), options=dir_disp, index=curr_idx_dir, key='direction_sel')
            sel_int_dir = ALL_DIRECTIONS_KEY;
            if sel_disp_dir != all_str:
                try: sel_idx = dir_disp.index(sel_disp_dir); sel_int_dir = dir_int[sel_idx]
                except ValueError: sel_int_dir = ALL_DIRECTIONS_KEY
            if sel_int_dir != st.session_state.selected_peak_direction: st.session_state.selected_peak_direction = sel_int_dir

        # Result Options (Unchanged from original)
        with st.expander(t.get('results_options_expander', "⚙️ Results Opts"), expanded=False):
            max_sl = len(df_catalog_data) if df_catalog_data is not None else 50; min_sl=5; act_max=max(min_sl, max_sl); sl_dis=act_max<=min_sl
            def_num = st.session_state.get('num_objects_slider', 20); cl_def=max(min_sl, min(def_num, act_max))
            if cl_def != def_num: st.session_state.num_objects_slider = cl_def
            st.slider(t.get('results_options_max_objects_label', "Max Objs:"), min_sl, act_max, step=1, key='num_objects_slider', disabled=sl_dis)
            sort_opts = {'Duration & Altitude': t.get('results_options_sort_duration', "Duration"), 'Brightness': t.get('results_options_sort_magnitude', "Brightness")}
            st.radio(t.get('results_options_sort_method_label', "Sort By:"), options=list(sort_opts.keys()), format_func=lambda k: sort_opts[k], key='sort_method', horizontal=True)

        # Bug Report Button (Unchanged from original)
        st.sidebar.markdown("---"); bug_email="debrun2005@gmail.com"; bug_subj=urllib.parse.quote("Bug Report: Adv DSO Finder")
        bug_body=urllib.parse.quote(t.get('bug_report_body', "\n\n(Describe bug)")); bug_link=f"mailto:{bug_email}?subject={bug_subj}&body={bug_body}"
        st.sidebar.markdown(f"<a href='{bug_link}' target='_blank'>{t.get('bug_report_button', '🐞 Report Bug')}</a>", unsafe_allow_html=True)

    # --- Main Area ---
    st.subheader(t.get('search_params_header', "Search Parameters"))
    # Parameter Display (Unchanged from original)
    param_col1, param_col2 = st.columns(2)
    loc_disp = t.get('location_error', "Loc Err: {}").format(t.get('location_not_set', "Not Set")); observer_for_run = None # Added translation key
    if st.session_state.location_is_valid_for_run:
        lat, lon, h, tz = st.session_state.manual_lat_val, st.session_state.manual_lon_val, st.session_state.manual_height_val, st.session_state.selected_timezone
        try:
            observer_for_run = Observer(latitude=lat*u.deg, longitude=lon*u.deg, elevation=h*u.m, timezone=tz)
            if st.session_state.location_choice_key == "Manual": loc_disp = t.get('location_manual_display', "Manual ({:.4f}, {:.4f})").format(lat, lon)
            elif st.session_state.searched_location_name: loc_disp = t.get('location_search_display', "Searched: {} ({:.4f}, {:.4f})").format(st.session_state.searched_location_name, lat, lon)
            else: loc_disp = f"Lat: {lat:.4f}, Lon: {lon:.4f}"
        except Exception as obs_e: loc_disp = t.get('location_error', "Loc Err: {}").format(f"{t.get('observer_creation_failed', 'Observer fail')}: {obs_e}"); st.session_state.location_is_valid_for_run = False; observer_for_run = None # Added translation key
    param_col1.markdown(t.get('search_params_location', "📍 Loc: {}").format(loc_disp))
    time_disp = ""; is_now_main = (st.session_state.time_choice_exp == "Now")
    if is_now_main:
        ref_time_main = Time.now()
        try: loc_now, loc_tz_name = get_local_time_str(ref_time_main, st.session_state.selected_timezone); time_disp = t.get('search_params_time_now', "Now (from {} {})").format(loc_now, loc_tz_name) # Used loc_tz_name
        except Exception: time_disp = t.get('search_params_time_now_utc', "Now (from {} UTC)").format(f"{ref_time_main.to_datetime(timezone.utc):%Y-%m-%d %H:%M:%S}") # Added translation key
    else: sel_date = st.session_state.selected_date_widget; ref_time_main = Time(datetime.combine(sel_date, time(12,0)), scale='utc'); time_disp = t.get('search_params_time_specific', "Night after {}").format(f"{sel_date:%Y-%m-%d}")
    param_col1.markdown(t.get('search_params_time', "⏱️ Time: {}").format(time_disp))
    mag_disp = ""; min_mag_f, max_mag_f = -np.inf, np.inf
    if st.session_state.mag_filter_mode_exp == "Bortle Scale": max_mag_f = get_magnitude_limit(st.session_state.bortle_slider); mag_disp = t.get('search_params_filter_mag_bortle', "Bortle {} (<= {:.1f})").format(st.session_state.bortle_slider, max_mag_f)
    else: min_mag_f, max_mag_f = st.session_state.manual_min_mag_slider, st.session_state.manual_max_mag_slider; mag_disp = t.get('search_params_filter_mag_manual', "Manual ({:.1f}-{:.1f})").format(min_mag_f, max_mag_f)
    param_col2.markdown(t.get('search_params_filter_mag', "✨ Mag: {}").format(mag_disp))
    min_alt_d, max_alt_d = st.session_state.min_alt_slider, st.session_state.max_alt_slider; sel_types_d = st.session_state.object_type_filter_exp; types_s = ', '.join(sel_types_d) if sel_types_d else t.get('search_params_types_all', "All")
    param_col2.markdown(t.get('search_params_filter_alt_types', "🔭 Alt {}-{}°, Types: {}").format(min_alt_d, max_alt_d, types_s))
    size_min_d, size_max_d = st.session_state.size_arcmin_range; param_col2.markdown(t.get('search_params_filter_size', "📐 Size {:.1f}-{:.1f}'").format(size_min_d, size_max_d))
    dir_d = st.session_state.selected_peak_direction; dir_d = t.get('search_params_direction_all', "All") if dir_d == ALL_DIRECTIONS_KEY else dir_d; param_col2.markdown(t.get('search_params_filter_direction', "🧭 Dir @ Max: {}").format(dir_d))

    # Find Objects Button (Unchanged from original)
    st.markdown("---")
    find_clicked = st.button(t.get('find_button_label', "🔭 Find Objects"), key="find_button", disabled=(df_catalog_data is None or not st.session_state.location_is_valid_for_run))
    if not st.session_state.location_is_valid_for_run and df_catalog_data is not None: st.warning(t.get('info_initial_prompt', "Enter Coords or Search Loc..."))
    results_placeholder = st.container()

    # Processing Logic (Unchanged from original)
    if find_clicked:
        st.session_state.find_button_pressed = True; st.session_state.update({'show_plot': False, 'show_custom_plot': False, 'active_result_plot_data': None, 'custom_target_plot_data': None, 'last_results': [], 'window_start_time': None, 'window_end_time': None})
        if observer_for_run and df_catalog_data is not None:
            with st.spinner(t.get('spinner_searching', "Calculating...")):
                try:
                    start_t, end_t, win_stat = get_observable_window(observer_for_run, ref_time_main, is_now_main, lang); results_placeholder.info(win_stat)
                    st.session_state.window_start_time = start_t; st.session_state.window_end_time = end_t
                    if start_t and end_t and start_t < end_t:
                        obs_times = Time(np.arange(start_t.jd, end_t.jd, (5*u.min).to(u.day).value), format='jd', scale='utc')
                        if len(obs_times) < 2: results_placeholder.warning(t.get("warning_window_too_short", "Observation window too short for detailed calculation.")) # Added translation key
                        filt_df = df_catalog_data.copy(); filt_df = filt_df[(filt_df['Mag'] >= min_mag_f) & (filt_df['Mag'] <= max_mag_f)]
                        if sel_types_d: filt_df = filt_df[filt_df['Type'].isin(sel_types_d)]
                        size_ok_m = df_catalog_data is not None and 'MajAx' in df_catalog_data.columns and df_catalog_data['MajAx'].notna().any()
                        if size_ok_m: filt_df = filt_df.dropna(subset=['MajAx']); filt_df = filt_df[(filt_df['MajAx'] >= size_min_d) & (filt_df['MajAx'] <= size_max_d)]
                        if filt_df.empty: results_placeholder.warning(t.get('warning_no_objects_found_filters', "No objects found with current filters (initial pass).")); st.session_state.last_results = [] # Added translation key
                        else:
                            min_alt_s = st.session_state.min_alt_slider * u.deg
                            found_objs = find_observable_objects(observer_for_run.location, obs_times, min_alt_s, filt_df, lang)
                            final_objs = []
                            sel_dir_f = st.session_state.selected_peak_direction; max_alt_f = st.session_state.max_alt_slider
                            for obj in found_objs:
                                if obj.get('Max Altitude (°)', -999) > max_alt_f: continue
                                if sel_dir_f != ALL_DIRECTIONS_KEY and obj.get('Direction at Max') != sel_dir_f: continue
                                final_objs.append(obj)
                            sort_k = st.session_state.sort_method
                            if sort_k == 'Brightness': final_objs.sort(key=lambda x: x.get('Magnitude', float('inf')) if x.get('Magnitude') is not None else float('inf'))
                            else: final_objs.sort(key=lambda x: (x.get('Max Cont. Duration (h)', 0), x.get('Max Altitude (°)', 0)), reverse=True)
                            num_show = st.session_state.num_objects_slider; st.session_state.last_results = final_objs[:num_show]
                            if not final_objs: results_placeholder.warning(t.get('warning_no_objects_found', "No objects found..."))
                            else: results_placeholder.success(t.get('success_objects_found', "{} objs found.").format(len(final_objs))); sort_msg_key = 'info_showing_list_duration' if sort_k != 'Brightness' else 'info_showing_list_magnitude'; results_placeholder.info(t.get(sort_msg_key, "Showing {}...").format(len(st.session_state.last_results))) # Added translation key
                    else: results_placeholder.error(t.get('error_no_window', "No valid window...") + " " + t.get('error_cannot_search', "Cannot search.")); st.session_state.last_results = [] # Added translation key
                except Exception as search_e: results_placeholder.error(t.get('error_search_unexpected', "Search err:") + f"\n```\n{search_e}\n```"); traceback.print_exc(); st.session_state.last_results = []
        else:
             if df_catalog_data is None: results_placeholder.error(t.get("error_search_no_catalog", "Cannot search: Catalog missing.")) # Added translation key
             if not observer_for_run: results_placeholder.error(t.get("error_search_no_location", "Cannot search: Location invalid.")) # Added translation key
             st.session_state.last_results = []

    # Display Results Block (Largely unchanged, minor translation key consistency)
    if st.session_state.last_results:
        results_data = st.session_state.last_results
        results_placeholder.subheader(t.get('results_list_header', "Results"))
        win_start, win_end = st.session_state.get('window_start_time'), st.session_state.get('window_end_time'); obs_exists = observer_for_run is not None
        if obs_exists and isinstance(win_start, Time) and isinstance(win_end, Time):
            mid_t = win_start + (win_end - win_start) / 2
            try: illum = moon_illumination(mid_t); moon_pct = illum*100; moon_svg = create_moon_phase_svg(illum, 50); m_c1, m_c2 = results_placeholder.columns([1,3])
            except Exception as moon_e: results_placeholder.warning(t.get('moon_phase_error', "Moon Err: {}").format(moon_e)); moon_pct = -1; moon_svg = None
            if moon_svg: m_c1.markdown(moon_svg, unsafe_allow_html=True)
            if moon_pct >= 0:
                 with m_c2:
                    st.metric(label=t.get('moon_metric_label', "Moon Illum."), value=f"{moon_pct:.0f}%")
                    moon_thresh = st.session_state.moon_phase_slider
                    if moon_pct > moon_thresh: st.warning(t.get('moon_warning_message', "Warn: Moon >{:.0f}% (Threshold: {:.0f}%)!").format(moon_pct, moon_thresh)) # Clarified warning
        elif st.session_state.find_button_pressed: results_placeholder.info(t.get("moon_phase_not_available", "Moon phase information not available.")) # Added translation key
        plot_opts = {'Sky Path': t.get('graph_type_sky_path', "Sky Path"), 'Altitude Plot': t.get('graph_type_alt_time', "Alt Plot")}
        results_placeholder.radio(t.get('graph_type_label', "Graph:"), options=list(plot_opts.keys()), format_func=lambda k: plot_opts[k], key='plot_type_selection', horizontal=True)
        for i, obj_data in enumerate(results_data):
            name, type_obj = obj_data.get('Name','N/A'), obj_data.get('Type','N/A') # Renamed 'type' to 'type_obj' to avoid conflict
            obj_mag = obj_data.get('Magnitude')
            mag_s = f"{obj_mag:.1f}" if obj_mag is not None else "N/A"
            title_format_string = t.get('results_expander_title', "{} ({}) - Mag: {}")
            title = title_format_string.format(name, type_obj, mag_s)
            is_exp = (st.session_state.expanded_object_name == name)
            obj_cont = results_placeholder.container()
            with obj_cont.expander(title, expanded=is_exp):
                c1, c2, c3 = st.columns([2,2,1])
                c1.markdown(t.get('results_coords_header', "**Details:**")); c1.markdown(f"**{t.get('results_export_constellation', 'Const')}:** {obj_data.get('Constellation', 'N/A')}")
                size = obj_data.get('Size (arcmin)'); c1.markdown(f"**{t.get('results_size_label', 'Size:')}** {t.get('results_size_value', '{:.1f}\'').format(size) if size is not None else 'N/A'}")
                c1.markdown(f"**RA:** {obj_data.get('RA', 'N/A')}"); c1.markdown(f"**Dec:** {obj_data.get('Dec', 'N/A')}")
                c2.markdown(t.get('results_max_alt_header', "**Max Alt:**"))
                max_a = obj_data.get('Max Altitude (°)', 0); az_m = obj_data.get('Azimuth at Max (°)', 0); dir_m = obj_data.get('Direction at Max', 'N/A')
                # Using the corrected formatting logic from the original script for Azimuth and Direction
                az_fmt_str = t.get('results_azimuth_label', "(Az: {:.1f}°{})") 
                az_str = az_fmt_str.format(az_m, "") if isinstance(az_m, (int, float)) else t.get("azimuth_not_available", "(Az: N/A)") # Added fallback translation
                dir_fmt_str = t.get('results_direction_label', ", Dir: {}")
                dir_str = dir_fmt_str.format(dir_m) if dir_m != "N/A" else "" # Hide if N/A
                c2.markdown(f"**{max_a:.1f}°** {az_str}{dir_str}")
                c2.markdown(t.get('results_best_time_header', "**Best Time (Local):**"))
                peak_t = obj_data.get('Time at Max (UTC)'); loc_t, loc_tz_name = get_local_time_str(peak_t, st.session_state.selected_timezone); c2.markdown(f"{loc_t} ({loc_tz_name})")
                c2.markdown(t.get('results_cont_duration_header', "**Duration:**")); dur = obj_data.get('Max Cont. Duration (h)', 0); c2.markdown(t.get('results_duration_value', "{:.1f} hrs").format(dur))
                g_q = urllib.parse.quote_plus(f"{name} astronomy"); g_url = f"https://www.google.com/search?q={g_q}"; c3.markdown(f"[{t.get('google_link_text', 'Google')}]({g_url})", unsafe_allow_html=True)
                s_q = urllib.parse.quote_plus(name); s_url = f"http://simbad.u-strasbg.fr/simbad/sim-basic?Ident={s_q}"; c3.markdown(f"[{t.get('simbad_link_text', 'SIMBAD')}]({s_url})", unsafe_allow_html=True)
                plot_key = f"plot_{name}_{i}"
                if st.button(t.get('results_graph_button', "📈 Plot"), key=plot_key):
                    st.session_state.update({'plot_object_name': name, 'active_result_plot_data': obj_data, 'show_plot': True, 'show_custom_plot': False, 'expanded_object_name': name}); st.rerun()
                if st.session_state.show_plot and st.session_state.plot_object_name == name:
                    plot_d = st.session_state.active_result_plot_data; min_l, max_l = st.session_state.min_alt_slider, st.session_state.max_alt_slider; st.markdown("---")
                    with st.spinner(t.get('results_spinner_plotting', "Plotting...")):
                        try:
                            fig_p = create_plot(plot_d, min_l, max_l, st.session_state.plot_type_selection, lang)
                            if fig_p:
                                st.pyplot(fig_p); close_key = f"close_{name}_{i}"
                                if st.button(t.get('results_close_graph_button', "Close Plot"), key=close_key): st.session_state.update({'show_plot': False, 'active_result_plot_data': None, 'expanded_object_name': None}); st.rerun()
                            else: st.error(t.get('results_graph_not_created', "Plot fail."))
                        except Exception as plt_e: st.error(t.get('results_graph_error', "Plot Err: {}").format(plt_e)); traceback.print_exc()
        if results_data:
            csv_ph = results_placeholder.empty()
            try:
                export_d = []; tz_csv = st.session_state.selected_timezone
                for obj in results_data:
                    peak_utc_csv = obj.get('Time at Max (UTC)'); loc_t_csv, _ = get_local_time_str(peak_utc_csv, tz_csv)
                    export_d.append({ t.get('results_export_name',"Name"): obj.get('Name'), t.get('results_export_type',"Type"): obj.get('Type'), t.get('results_export_constellation',"Const"): obj.get('Constellation'),
                        t.get('results_export_mag',"Mag"): obj.get('Magnitude'), t.get('results_export_size',"Size'"): obj.get('Size (arcmin)'), t.get('results_export_ra',"RA"): obj.get('RA'),
                        t.get('results_export_dec',"Dec"): obj.get('Dec'), t.get('results_export_max_alt',"MaxAlt"): obj.get('Max Altitude (°)', np.nan), # Ensure float for CSV
                        t.get('results_export_az_at_max',"Az@Max"): obj.get('Azimuth at Max (°)', np.nan), # Ensure float for CSV
                        t.get('results_export_direction_at_max',"Dir@Max"): obj.get('Direction at Max'), t.get('results_export_time_max_utc',"TimeMaxUTC"): peak_utc_csv.iso if peak_utc_csv else 'N/A',
                        t.get('results_export_time_max_local',"TimeMaxLoc"): loc_t_csv, t.get('results_export_cont_duration',"Dur(h)"): obj.get('Max Cont. Duration (h)', np.nan) }) # Ensure float for CSV
                df_ex = pd.DataFrame(export_d); dec_char = ',' if lang == 'de' else '.'; csv_s = df_ex.to_csv(index=False, sep=';', encoding='utf-8-sig', decimal=dec_char)
                now_s = datetime.now().strftime("%Y%m%d_%H%M"); csv_fn = t.get('results_csv_filename', "dso_list_{}.csv").format(now_s)
                csv_ph.download_button(label=t.get('results_save_csv_button', "💾 Save CSV"), data=csv_s, file_name=csv_fn, mime='text/csv', key='csv_dl')
            except Exception as csv_e: csv_ph.error(t.get('results_csv_export_error', "CSV Err: {}").format(csv_e))
    elif st.session_state.find_button_pressed: results_placeholder.info(t.get('warning_no_objects_found_after_search', "No objects found based on your criteria.")) # Added translation key

    # Custom Target Plotting (Unchanged from original)
    st.markdown("---")
    with st.expander(t.get('custom_target_expander', "Plot Custom Target")):
        with st.form("custom_form"):
             st.text_input(t.get('custom_target_ra_label', "RA:"), key="custom_target_ra", placeholder=t.get('custom_target_ra_placeholder', "..."))
             st.text_input(t.get('custom_target_dec_label', "Dec:"), key="custom_target_dec", placeholder=t.get('custom_target_dec_placeholder', "..."))
             st.text_input(t.get('custom_target_name_label', "Name (Opt):"), key="custom_target_name", placeholder=t.get("custom_target_name_placeholder", "My Comet")) # Added translation key
             custom_submitted = st.form_submit_button(t.get('custom_target_button', "Create Plot"))
        custom_err_ph = st.empty(); custom_plot_ph = st.empty()
        if custom_submitted:
             st.session_state.update({'show_plot': False, 'show_custom_plot': False, 'custom_target_plot_data': None, 'custom_target_error': ""})
             cust_ra, cust_dec = st.session_state.custom_target_ra, st.session_state.custom_target_dec; cust_name = st.session_state.custom_target_name or t.get('custom_target_default_name', "Custom Target") # Added translation key
             win_s_c, win_e_c = st.session_state.get('window_start_time'), st.session_state.get('window_end_time'); obs_ex_c = observer_for_run is not None
             if not cust_ra or not cust_dec: st.session_state.custom_target_error = t.get('custom_target_error_coords', "Invalid RA/Dec."); custom_err_ph.error(st.session_state.custom_target_error)
             elif not obs_ex_c or not isinstance(win_s_c, Time) or not isinstance(win_e_c, Time): st.session_state.custom_target_error = t.get('custom_target_error_window', "Invalid window/loc."); custom_err_ph.error(st.session_state.custom_target_error)
             else:
                 try:
                     cust_coord = SkyCoord(ra=cust_ra, dec=cust_dec, unit=(u.hourangle, u.deg))
                     if win_s_c < win_e_c: obs_times_c = Time(np.arange(win_s_c.jd, win_e_c.jd, (5*u.min).to(u.day).value), format='jd', scale='utc')
                     else: raise ValueError(t.get("custom_target_error_invalid_window_order", "Invalid time window order.")) # Added translation key
                     if len(obs_times_c) < 2: raise ValueError(t.get("custom_target_error_window_short", "Time window too short for custom plot.")) # Added translation key
                     altaz_fr_c = AltAz(obstime=obs_times_c, location=observer_for_run.location); cust_altazs = cust_coord.transform_to(altaz_fr_c)
                     st.session_state.custom_target_plot_data = {'Name': cust_name, 'altitudes': cust_altazs.alt.to(u.deg).value, 'azimuths': cust_altazs.az.to(u.deg).value, 'times': obs_times_c}
                     st.session_state.show_custom_plot = True; st.session_state.custom_target_error = ""; st.rerun()
                 except ValueError as cust_coord_e: st.session_state.custom_target_error = f"{t.get('custom_target_error_coords', 'Invalid RA/Dec.')} ({cust_coord_e})"; custom_err_ph.error(st.session_state.custom_target_error)
                 except Exception as cust_e: st.session_state.custom_target_error = f"{t.get('custom_target_error_general', 'Custom plot error')}: {cust_e}"; custom_err_ph.error(st.session_state.custom_target_error); traceback.print_exc() # Added translation key
        if st.session_state.show_custom_plot and st.session_state.custom_target_plot_data:
            cust_plot_d = st.session_state.custom_target_plot_data; min_a_c, max_a_c = st.session_state.min_alt_slider, st.session_state.max_alt_slider
            with custom_plot_ph.container():
                 st.markdown("---");
                 with st.spinner(t.get('results_spinner_plotting', "Plotting...")):
                     try:
                         fig_c = create_plot(cust_plot_d, min_a_c, max_a_c, st.session_state.plot_type_selection, lang)
                         if fig_c:
                             st.pyplot(fig_c);
                             if st.button(t.get('results_close_graph_button', "Close Plot"), key="close_custom"): st.session_state.update({'show_custom_plot': False, 'custom_target_plot_data': None}); st.rerun()
                         else: st.error(t.get('results_graph_not_created', "Plot fail."))
                     except Exception as plt_e_c: st.error(t.get('results_graph_error', "Plot Err: {}").format(plt_e_c)); traceback.print_exc()
        elif st.session_state.custom_target_error: custom_err_ph.error(st.session_state.custom_target_error)

    # --- Redshift Calculator ---
    st.markdown("---")
    with st.expander(t.get("redshift_calculator_title", "Redshift Calculator"), expanded=False):
        # Input Parameters
        st.subheader(t.get("input_params", "Input Parameters"))
        rc_z = st.number_input(
            label=t.get("redshift_z", "Redshift (z)"),
            min_value=-0.999,  # Allow slightly more negative for extreme blueshifts if needed
            value=st.session_state.redshift_z_input,
            step=0.01, # Adjusted step for finer control
            format="%.5f", # More precision for z
            key="redshift_z_input",
            help=t.get("redshift_z_tooltip", "Enter cosmological redshift (z). Negative for blueshift.")
        )

        # Cosmological Parameters
        st.subheader(t.get("cosmo_params", "Cosmological Parameters"))
        col_cosmo1, col_cosmo2, col_cosmo3 = st.columns(3)
        with col_cosmo1:
            rc_h0 = st.number_input(
                label=t.get("hubble_h0", "H₀ [km/s/Mpc]"),
                min_value=1.0,
                value=st.session_state.redshift_h0_input,
                step=0.1,
                format="%.1f",
                key="redshift_h0_input"
            )
        with col_cosmo2:
            rc_om = st.number_input(
                label=t.get("omega_m", "Ωm (Matter Density)"), # Clarified label
                min_value=0.0,
                max_value=2.0, # Max usually around 1, but allow flexibility
                value=st.session_state.redshift_omega_m_input,
                step=0.001, # Finer step
                format="%.3f",
                key="redshift_omega_m_input"
            )
        with col_cosmo3:
            rc_ol = st.number_input(
                label=t.get("omega_lambda", "ΩΛ (Dark Energy Density)"), # Clarified label
                min_value=0.0,
                max_value=2.0, # Max usually around 1, but allow flexibility
                value=st.session_state.redshift_omega_lambda_input,
                step=0.001, # Finer step
                format="%.3f",
                key="redshift_omega_lambda_input"
            )
        
        # Flatness check (Omega_k = 1 - Omega_m - Omega_Lambda)
        omega_k = 1.0 - rc_om - rc_ol
        if not math.isclose(omega_k, 0.0, abs_tol=1e-3):
            st.info(
                t.get("non_flat_universe_info", "Note: Ωm + ΩΛ = {sum_omega:.3f}. This implies a non-flat universe (Ωk = {omega_k:.3f}). Calculations assume this geometry.")
                .format(sum_omega=(rc_om + rc_ol), omega_k=omega_k)
            )
        else:
             st.caption(t.get("flat_universe_assumed", "Assuming a flat universe (Ωk ≈ 0)."))


        st.markdown("---")
        st.subheader(t.get("results_for", "Results for z = {z:.5f}").format(z=rc_z))

        # Perform calculation
        rc_results = calculate_lcdm_distances(rc_z, rc_h0, rc_om, rc_ol)
        rc_error_key = rc_results.get('error_key')

        if rc_error_key and rc_error_key != "warn_blueshift": # Handle errors first, but allow blueshift warning to proceed
            rc_error_args = rc_results.get('error_args', {})
            rc_error_text = t.get(rc_error_key, "Calculation Error: {}").format(**rc_error_args) # Default error message
            st.error(rc_error_text)
        else:
            # Display warning for blueshift if present, then proceed with results
            if rc_error_key == "warn_blueshift":
                st.warning(t.get("warn_blueshift", "Blueshift (z < 0): Object is moving towards us. Distances are defined as 0, lookback time is 0."))
            
            # Display other warnings (e.g., integration accuracy)
            rc_warning_key = rc_results.get('warning_key')
            if rc_warning_key:
                rc_warning_args = rc_results.get('warning_args', {})
                st.info(t.get(rc_warning_key, "Calculation Warning: {}").format(**rc_warning_args)) # Default warning message

            # Unpack results
            rc_vel_km_s = rc_results.get('recessional_velocity_km_s', 0.0)
            rc_lookback_gyr = rc_results.get('lookback_gyr', 0.0)
            rc_comov_mpc = rc_results.get('comoving_mpc', 0.0)
            rc_lum_mpc = rc_results.get('luminosity_mpc', 0.0)
            rc_angd_mpc = rc_results.get('ang_diam_mpc', 0.0)

            # Derived values for display
            rc_comov_gly = convert_mpc_to_gly(rc_comov_mpc)
            rc_lum_gly = convert_mpc_to_gly(rc_lum_mpc)
            rc_angd_gly = convert_mpc_to_gly(rc_angd_mpc)
            
            rc_comov_km = convert_mpc_to_km(rc_comov_mpc)
            rc_comov_ly = convert_km_to_ly(rc_comov_km)
            rc_comov_au = convert_km_to_au(rc_comov_km)
            rc_comov_ls = convert_km_to_ls(rc_comov_km)
            rc_comov_km_fmt = format_large_number(rc_comov_km)

            # Display Metrics: Velocity and Lookback Time
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(
                    label=t.get("recessional_velocity", "Recessional Velocity"),
                    value=f"{rc_vel_km_s:,.1f}", # Format with comma and one decimal place
                    delta=t.get("unit_km_s", "km/s")
                )
                if rc_z > 0:
                    st.caption(t.get("velocity_positive_caption", "Positive: Moving away (redshift)"))
                elif rc_z < 0:
                    st.caption(t.get("velocity_negative_caption", "Negative: Moving towards (blueshift)"))
                else:
                    st.caption(t.get("velocity_zero_caption", "Zero: No significant cosmological motion relative to us"))


            with res_col2:
                st.metric(
                    label=t.get("lookback_time", "Lookback Time"),
                    value=f"{rc_lookback_gyr:.4f}", # Standard formatting
                    delta=t.get("unit_Gyr", "Gyr")
                )
                lb_ex_key = get_lookback_comparison_key(rc_lookback_gyr)
                st.caption(f"*{t.get(lb_ex_key, 'Contextual example for lookback time...')}*") # Default example text

            st.markdown("---")
            st.subheader(t.get("cosmo_distances", "Cosmological Distances"))
            
            dist_col1, dist_col2, dist_col3 = st.columns(3)
            with dist_col1:
                st.markdown(t.get("comoving_distance_title", "**Comoving Distance:**"))
                st.text(f"  {rc_comov_mpc:,.4f} {t.get('unit_Mpc', 'Mpc')}")
                st.text(f"  {rc_comov_gly:,.4f} {t.get('unit_Gly', 'Gly')}")
                cv_ex_key = get_comoving_comparison_key(rc_comov_mpc)
                st.caption(f"*{t.get(cv_ex_key, 'Contextual example for comoving distance...')}*") # Default example text
                with st.expander(t.get("comoving_other_units_expander", "Other Units")):
                    st.text(f"  {rc_comov_km_fmt} {t.get('unit_km_full', 'km')}")
                    st.text(f"  {rc_comov_ly:,.2e} {t.get('unit_LJ', 'ly')}") # Scientific notation for ly
                    st.text(f"  {rc_comov_au:,.2e} {t.get('unit_AE', 'AU')}") # Scientific notation for AU
                    st.text(f"  {rc_comov_ls:,.2e} {t.get('unit_Ls', 'Ls')}") # Scientific notation for Ls
            
            with dist_col2:
                st.markdown(t.get("luminosity_distance_title", "**Luminosity Distance:**"))
                st.text(f"  {rc_lum_mpc:,.4f} {t.get('unit_Mpc', 'Mpc')}")
                st.text(f"  {rc_lum_gly:,.4f} {t.get('unit_Gly', 'Gly')}")
                st.caption(f"*{t.get('explanation_luminosity', 'Relevant for the observed brightness of standard candles.')}*")

            with dist_col3:
                st.markdown(t.get("angular_diameter_distance_title", "**Angular Diameter Distance:**"))
                st.text(f"  {rc_angd_mpc:,.4f} {t.get('unit_Mpc', 'Mpc')}")
                st.text(f"  {rc_angd_gly:,.4f} {t.get('unit_Gly', 'Gly')}")
                st.caption(f"*{t.get('explanation_angular', 'Relevant for the observed angular size of standard rulers.')}*")

            st.caption(t.get("calculation_note", "Note: Calculations based on the flat ΛCDM cosmological model by default. Non-flat models adjust Ωk based on inputs."))

    # --- Footer Links ---
    st.markdown("---")
    # Original DSO Finder donation link
    st.caption(t.get('donation_text', "Like the DSO Finder? [Consider supporting its development!](https://www.buymeacoffee.com/SkyObserver)"), unsafe_allow_html=True)
    # Removed Redshift Calculator specific donation link as per original script structure

# --- Run the App ---
if __name__ == "__main__":
    main()
