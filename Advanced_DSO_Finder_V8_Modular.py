# -*- coding: utf-8 -*-
# --- Basic Imports ---
import os
import traceback
import threading
from datetime import datetime, date, time, timedelta, timezone
import pandas as pd
import math
import numpy as np
import webbrowser
import json # Added for saving/loading settings

# --- Kivy Imports ---
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty, StringProperty, NumericProperty, ListProperty, BooleanProperty, DictProperty
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.recycleview import RecyclerView
from kivy.lang import Builder
from kivy.clock import Clock
# Optional: Für Matplotlib-Integration
# from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
# import matplotlib.pyplot as plt

# --- Library Imports (Backend) ---
try:
    from astropy.time import Time
    import astropy.units as u
    from astropy.coordinates import EarthLocation, SkyCoord, get_sun, AltAz, get_constellation
    from astroplan import Observer
    from astroplan.moon import moon_illumination
    import pytz
    from timezonefinder import TimezoneFinder
    from geopy.geocoders import Nominatim, ArcGIS, Photon
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    from scipy.integrate import quad # Needed for Redshift Calc
except ImportError as e:
    print(f"ERROR: Missing backend libraries: {e}")
    # exit()

# --- Localization Import ---
try:
    from localization import get_translation
except ImportError:
    print("ERROR: localization.py not found.")
    def get_translation(lang):
        print(f"Warning: localization.py not found, using dummy translation for lang '{lang}'.")
        return lambda key, default="": default
    # exit()

# --- Backend Functions ---
# TODO: Hier übernimmst du deine bestehenden Backend-Funktionen

try:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    APP_DIR = os.getcwd()
CATALOG_FILENAME = "ongc.csv"
CATALOG_FILEPATH = os.path.join(APP_DIR, CATALOG_FILENAME)

# --- Dummy Backend Funktionen (ersetzen!) ---
def load_ongc_data(path, lang):
    print(f"Dummy: Lade Katalog von {path} in Sprache {lang}")
    try:
        df = pd.read_csv(path, sep=';', comment='#', low_memory=False, nrows=100)
        print(f"Dummy: Katalog geladen mit {len(df)} Zeilen (max 100). Spalten: {df.columns.tolist()}")
        if 'Name' not in df.columns or 'RA' not in df.columns or 'Dec' not in df.columns:
             print("Dummy Warning: Wichtige Spalten fehlen im Katalog!")
        df['Mag'] = 9.9
        df['Type'] = 'Galaxy'
        df['RA_str'] = df['RA']
        df['Dec_str'] = df['Dec']
        return df[['Name', 'RA_str', 'Dec_str', 'Mag', 'Type']].dropna().head(50)
    except Exception as e:
        print(f"Dummy Error: Fehler beim Laden des Katalogs: {e}")
        return None

def get_observable_window(observer, ref_time, is_now, lang):
     print("Dummy: Berechne Beobachtungsfenster...")
     start = Time(datetime.combine(ref_time.to_datetime(timezone.utc).date(), time(20,0)), scale='utc')
     end = start + timedelta(hours=8)
     return start, end, "Dummy: Fenster 20:00-04:00 UTC"

def find_observable_objects(observer_location, obs_times, min_alt_limit, catalog_df, lang):
    print("Dummy: Suche sichtbare Objekte...")
    if catalog_df is None or catalog_df.empty: return []
    results = []
    for i, row in catalog_df.head(5).iterrows():
        results.append({
            'Name': row.get('Name', f'DummyObj {i}'), 'Type': row.get('Type', 'DummyType'),
            'Constellation': 'DummyConst', 'Magnitude': row.get('Mag', 9.9),
            'Size (arcmin)': 10.0, 'RA': row.get('RA_str', '00h00m00s'),
            'Dec': row.get('Dec_str', '+00d00m00s'), 'Max Altitude (°)': 45.0 + i*5,
            'Azimuth at Max (°)': 180.0, 'Direction at Max': 'S',
            'Time at Max (UTC)': obs_times[len(obs_times)//2] + timedelta(minutes=i*10),
            'Max Cont. Duration (h)': 2.0 + i*0.5, 'skycoord': None,
            'altitudes': np.linspace(10, 45.0 + i*5, len(obs_times)),
            'azimuths': np.linspace(90, 270, len(obs_times)), 'times': obs_times
        })
    return results

def get_local_time_str(utc_time: Time | None, timezone_str: str) -> tuple[str, str]:
     if utc_time is None: return "N/A", "N/A"
     if 'pytz' not in globals():
         print("Warning: pytz not available for get_local_time_str")
         try: return utc_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), "UTC (pytz N/A)"
         except: return "N/A", "N/A"
     try:
         local_tz = pytz.timezone(timezone_str); utc_dt = utc_time.to_datetime(timezone.utc);
         local_dt = utc_dt.astimezone(local_tz); local_str = local_dt.strftime('%Y-%m-%d %H:%M:%S');
         tz_name = local_dt.tzname(); tz_name = tz_name if tz_name else local_tz.zone
         return local_str, tz_name
     except Exception as e:
         print(f"Error in get_local_time_str: {e}")
         try: return utc_time.to_datetime(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'), "UTC (TZ Err)"
         except: return "N/A", "N/A"

def calculate_lcdm_distances(redshift, h0, omega_m, omega_lambda):
     print("Dummy: Berechne Redshift-Distanzen...")
     return {'comoving_mpc': 100.0 * redshift, 'luminosity_mpc': 110.0 * redshift,
             'ang_diam_mpc': 90.0 * redshift, 'lookback_gyr': 1.0 * redshift,
             'error_key': None}
# --- Ende Dummy Backend Funktionen ---

# Globale Konstanten
INITIAL_LAT = 47.17
INITIAL_LON = 8.01
INITIAL_HEIGHT = 550 # Höhe in Metern
INITIAL_TIMEZONE = "Europe/Zurich" # Standard-Zeitzone
H0_DEFAULT = 67.4
OMEGA_M_DEFAULT = 0.315
OMEGA_LAMBDA_DEFAULT = 0.685
CARDINAL_DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
ALL_DIRECTIONS_KEY = 'All'
USER_SETTINGS_FILE = "dso_finder_settings.json" # Dateiname für Benutzereinstellungen

# --- Definition der Klasse für RecyclerView-Einträge ---
class ResultItem(RecycleDataViewBehavior, BoxLayout):
    index = None
    data = DictProperty({})
    name = StringProperty('')
    obj_type = StringProperty('')
    magnitude = StringProperty('')
    max_alt = StringProperty('')
    best_time = StringProperty('')

    def refresh_view_attrs(self, rv, index, data):
        self.index = index
        self.data = data
        self.name = data.get('Name', 'N/A')
        self.obj_type = data.get('Type', 'N/A')
        mag = data.get('Magnitude')
        self.magnitude = f"{mag:.1f}" if mag is not None else "N/A"
        alt = data.get('Max Altitude (°)')
        self.max_alt = f"{alt:.1f}" if alt is not None else "N/A"
        peak_t = data.get('Time at Max (UTC)')
        try:
            app = App.get_running_app()
            current_tz = app.current_timezone if app else 'UTC'
            local_time_str, tz_name = get_local_time_str(peak_t, current_tz)
            self.best_time = f"{local_time_str} ({tz_name})" if local_time_str != "N/A" else "N/A"
        except Exception as e:
             print(f"Error formatting time in ResultItem: {e}")
             self.best_time = "N/A"
        return super(ResultItem, self).refresh_view_attrs(rv, index, data)

# --- Kivy UI Definition (via Kv Language) ---
try:
    kv_file_path = os.path.join(os.path.dirname(__file__), 'dsofindergui.kv')
    Builder.load_file(kv_file_path)
except Exception as e:
     print(f"FATAL ERROR: Could not load KV file 'dsofindergui.kv': {e}")
     print("Ensure the file exists in the same directory as the script and has valid Kivy syntax.")
     exit()

class MainScreen(Screen):
    status_message = StringProperty("Status: Bereit.")
    results_list = ListProperty([])
    search_in_progress = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.app = App.get_running_app()
        # Die Initialisierung der UI-Felder mit gespeicherten Werten erfolgt in _populate_fields_from_settings
        Clock.schedule_once(self._trigger_load_initial_data_and_settings)

    def _trigger_load_initial_data_and_settings(self, dt):
        # Zuerst Katalog laden
        self.status_message = self.app.t('status_loading_catalog', "Lade Katalog...")
        load_thread = threading.Thread(target=self._load_initial_data_thread)
        load_thread.daemon = True
        load_thread.start()
        # Dann UI Felder mit geladenen oder Standardwerten befüllen
        # Dies wird nun nach dem Laden des Katalogs in _update_after_load aufgerufen

    def _populate_fields_from_settings(self):
        """Populates input fields with values loaded from app settings or defaults."""
        try:
            self.ids.latitude_input.text = str(self.app.user_latitude)
            self.ids.longitude_input.text = str(self.app.user_longitude)
            self.ids.elevation_input.text = str(self.app.user_elevation)
            # Zeitzone wird über app.current_timezone gehandhabt und könnte auch gespeichert werden
            # self.ids.timezone_spinner.text = self.app.current_timezone # Wenn es ein Timezone-Spinner-Widget gäbe
            print(f"UI populated with Lat: {self.app.user_latitude}, Lon: {self.app.user_longitude}, Elev: {self.app.user_elevation}")
        except AttributeError as e:
            print(f"Warning: Could not populate all fields from settings. Widget ID missing? {e}")
            self.status_message = self.app.t('error_setting_ui_fields', "Fehler beim Setzen der UI-Felder aus Einstellungen.")
        except Exception as e:
            print(f"Error populating UI fields from settings: {e}")
            self.status_message = self.app.t('error_setting_ui_fields_general', "Allg. Fehler beim Setzen der UI-Felder.")


    def _load_initial_data_thread(self):
        catalog = None
        status = ""
        try:
            catalog = load_ongc_data(CATALOG_FILEPATH, self.app.language) # Echte Funktion nutzen!
            if catalog is None or catalog.empty:
                status = self.app.t('error_loading_catalog', "Fehler: Katalog konnte nicht geladen oder ist leer.")
            else:
                status = self.app.t('info_catalog_loaded', "Katalog geladen: {} Objekte.").format(len(catalog))
        except Exception as e:
            status = self.app.t('error_loading_catalog_exc', "Fehler beim Katalogladen: {}").format(e)
            traceback.print_exc()
        Clock.schedule_once(lambda dt: self._update_after_load(catalog, status))

    def _update_after_load(self, catalog, status):
         self.app.df_catalog = catalog
         self.status_message = status
         self._populate_fields_from_settings() # Jetzt UI-Felder befüllen
         # TODO: Update UI filters based on catalog

    def trigger_find_objects(self):
        if self.search_in_progress:
            self.status_message = self.app.t('status_search_running', "Suche läuft bereits...")
            return
        if self.app.df_catalog is None or self.app.df_catalog.empty:
            self.status_message = self.app.t('status_catalog_not_loaded', "Katalog nicht geladen.")
            return
        try:
            lat = float(self.ids.latitude_input.text)
            lon = float(self.ids.longitude_input.text)
            height = int(self.ids.elevation_input.text) # Höhe ist üblicherweise eine Ganzzahl
            min_alt = self.ids.min_alt_slider.value
            max_alt = self.ids.max_alt_slider.value
            
            selected_timezone = self.app.current_timezone # Wird von App-Ebene geholt
            time_mode_text_now = self.app.t('time_option_now', "Now")
            is_now = self.ids.time_mode_spinner.text == time_mode_text_now
            selected_date_str = self.ids.date_input.text if not is_now else date.today().isoformat()

            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("Ungültige Koordinaten")
            if not (0 <= min_alt <= 90 and 0 <= max_alt <= 90 and min_alt <= max_alt):
                raise ValueError("Ungültige Altitude-Werte")
            if not is_now:
                datetime.strptime(selected_date_str, '%Y-%m-%d') # Validiert das Datumformat

        except ValueError as e:
            self.status_message = self.app.t('error_invalid_input', "Fehler: Ungültige Eingabe: {}").format(e)
            traceback.print_exc()
            return
        except KeyError as e: # Falls ein UI-Element (id) nicht gefunden wird
             translator = self.app.t if self.app and callable(self.app.t) else (lambda k, d="": d)
             self.status_message = translator('error_missing_widget_id', "Fehler: Widget-ID nicht gefunden: {}. Prüfe .kv.").format(e)
             traceback.print_exc()
             return
        except Exception as e: # Andere unerwartete Fehler
             translator = self.app.t if self.app and callable(self.app.t) else (lambda k, d="": d)
             self.status_message = translator('error_unknown_param_read', "Unbekannter Fehler beim Lesen der Parameter: {}").format(e)
             traceback.print_exc()
             return

        self.search_in_progress = True
        translator = self.app.t if self.app and callable(self.app.t) else (lambda k, d="": d)
        self.status_message = translator('status_searching', "Suche läuft...")
        self.results_list = []
        
        thread_args = (lat, lon, height, min_alt, max_alt, selected_timezone, is_now, selected_date_str)
        search_thread = threading.Thread(target=self._find_objects_thread, args=thread_args)
        search_thread.daemon = True
        search_thread.start()

    def _find_objects_thread(self, lat, lon, height, min_alt, max_alt, tz_str, is_now, date_str):
        results = []
        status = ""
        translator = self.app.t if self.app and callable(self.app.t) else (lambda k, d="": d)
        try:
            if 'Observer' not in globals():
                raise ImportError("astroplan.Observer not available")
            
            observer = Observer(latitude=lat*u.deg, longitude=lon*u.deg, elevation=height*u.m, timezone=tz_str)
            
            ref_dt = date.fromisoformat(date_str)
            ref_time = Time.now() if is_now else Time(datetime.combine(ref_dt, time(12,0)), scale='utc') # Mitte des Tages als Referenz
            
            start_t, end_t, win_stat = get_observable_window(observer, ref_time, is_now, self.app.language)
            status += win_stat + "\n"

            if self.app.df_catalog is None:
                raise ValueError("Katalog nicht geladen")
            
            filtered_df = self.app.df_catalog.copy()
            # TODO: Apply real filters based on args from UI (Magnitude, Type, Size, Direction)
            print(f"Info: Kataloggröße vor Filterung: {len(filtered_df)}")
            # Beispiel-Filter (muss durch echte UI-Filter ersetzt werden)
            # filtered_df = filtered_df[filtered_df['Mag'] < float(self.ids.mag_limit_input.text)] 
            filtered_df = filtered_df[filtered_df['Mag'] < 15.0] # Placeholder
            print(f"Info: Kataloggröße nach Filterung: {len(filtered_df)}")

            if start_t and end_t and start_t < end_t and not filtered_df.empty:
                # Erzeuge Zeitpunkte im Beobachtungsfenster (z.B. alle 5 Minuten)
                obs_times = Time(np.arange(start_t.jd, end_t.jd, (5*u.min).to(u.day).value), format='jd', scale='utc')
                
                if len(obs_times) < 2:
                    status += translator('warning_window_too_short', "Warnung: Beobachtungsfenster zu kurz.")
                    results = []
                else:
                    min_alt_q = min_alt * u.deg
                    if 'find_observable_objects' not in globals():
                        raise NameError("find_observable_objects not defined")
                    
                    found_objs = find_observable_objects(observer.location, obs_times, min_alt_q, filtered_df, self.app.language)
                    
                    # Post-Filterung (z.B. maximale Höhe, Himmelsrichtung) und Sortierung
                    final_objs = [obj for obj in found_objs if obj.get('Max Altitude (°)', -99) <= max_alt]
                    # TODO: Implement direction filter if needed
                    
                    final_objs.sort(key=lambda x: x.get('Max Altitude (°)', 0), reverse=True) # Sort by max altitude
                    
                    max_results = 50 # Begrenze die Anzahl der Ergebnisse
                    results = final_objs[:max_results]

                    if not results:
                        status += translator('warning_no_objects_match_filters', "Keine Objekte entsprechen allen Filtern.")
                    else:
                        status += translator('success_objects_found', "{} Objekte gefunden.").format(len(results))
            elif filtered_df.empty:
                status += translator('warning_no_objects_pass_initial_filters', "Keine Objekte entsprechen den Primärfiltern.")
            else: # Kein gültiges Fenster
                status += translator('error_no_window', "Kein gültiges Beobachtungsfenster gefunden.") + f" Start: {start_t}, End: {end_t}"

        except Exception as e:
            status = translator('error_search_unexpected', "Fehler bei der Suche: {}").format(e)
            traceback.print_exc()
        finally:
            Clock.schedule_once(lambda dt: self._update_ui_after_search(results, status))

    def _update_ui_after_search(self, results, status):
        self.search_in_progress = False
        self.status_message = status
        self.results_list = results

    def show_plot(self, object_data):
         name = object_data.get('Name', 'Unknown')
         translator = self.app.t if self.app and callable(self.app.t) else (lambda k, d="": d)
         self.status_message = translator('plot_requested', "Plot für {} angefordert (noch nicht implementiert).").format(name)
         print(f"Plot requested for: {object_data.get('Name')}")
         # TODO: Implement plotting logic using Matplotlib or Kivy Garden

    def open_link(self, url):
        try:
            webbrowser.open(url)
        except Exception as e:
            self.status_message = f"Could not open link: {e}"
            print(f"Error opening link {url}: {e}")

class DSOFinderApp(App):
    language = StringProperty('de')
    t = ObjectProperty(lambda key, default="": default)
    df_catalog = ObjectProperty(None)
    
    # Benutzereinstellungen für Standort und Höhe
    user_latitude = NumericProperty(INITIAL_LAT)
    user_longitude = NumericProperty(INITIAL_LON)
    user_elevation = NumericProperty(INITIAL_HEIGHT) # Höhe als NumericProperty
    current_timezone = StringProperty(INITIAL_TIMEZONE) # Behält die aktuelle Zeitzone

    redshift_z = NumericProperty(0.1)
    redshift_h0 = NumericProperty(H0_DEFAULT)
    redshift_results = DictProperty({})
    # TODO: Add OmegaM, OmegaL properties if they need to be configurable and saved

    def build(self):
        lang_dict = get_translation(self.language)
        self.t = lang_dict.get if isinstance(lang_dict, dict) else (lambda key, default="": default)
        self.title = self.t('app_title', "Advanced DSO Finder")
        sm = ScreenManager()
        sm.add_widget(MainScreen(name='main'))
        return sm

    def on_start(self):
        """Wird nach build() aufgerufen. Lade hier Benutzereinstellungen."""
        self.load_user_settings()
        # Die UI-Felder werden in MainScreen._populate_fields_from_settings aktualisiert,
        # nachdem der Katalog geladen wurde und die App-Eigenschaften (user_latitude etc.) gesetzt sind.

    def on_stop(self):
        """Wird aufgerufen, wenn die App geschlossen wird. Speichere hier Benutzereinstellungen."""
        self.save_user_settings()

    def load_user_settings(self):
        """Lädt Benutzereinstellungen (Lat, Lon, Höhe) aus einer JSON-Datei."""
        settings_path = os.path.join(self.user_data_dir, USER_SETTINGS_FILE)
        try:
            if os.path.exists(settings_path):
                with open(settings_path, 'r') as f:
                    data = json.load(f)
                    self.user_latitude = float(data.get('latitude', INITIAL_LAT))
                    self.user_longitude = float(data.get('longitude', INITIAL_LON))
                    self.user_elevation = int(data.get('elevation', INITIAL_HEIGHT))
                    # self.current_timezone = data.get('timezone', INITIAL_TIMEZONE) # Optional: Zeitzone auch speichern
                    print(f"User settings loaded from {settings_path}")
            else:
                print(f"Settings file not found ({settings_path}). Using default values.")
                # Standardwerte sind bereits in den Properties gesetzt, hier nichts zu tun.
        except ValueError as e:
            print(f"Error decoding settings from {settings_path}. Using default values. Error: {e}")
            # Bei Fehler auf Standardwerte zurückfallen (sind schon gesetzt)
        except Exception as e:
            print(f"Failed to load user settings from {settings_path}. Using default values. Error: {e}")
            # Bei Fehler auf Standardwerte zurückfallen (sind schon gesetzt)

    def save_user_settings(self):
        """Speichert aktuelle Benutzereinstellungen (Lat, Lon, Höhe) in eine JSON-Datei."""
        settings_path = os.path.join(self.user_data_dir, USER_SETTINGS_FILE)
        main_screen = self.root.get_screen('main') if self.root and self.root.has_screen('main') else None

        if not main_screen:
            print("Error saving settings: Main screen not available.")
            return
            
        data_to_save = {}
        try:
            data_to_save['latitude'] = float(main_screen.ids.latitude_input.text)
            data_to_save['longitude'] = float(main_screen.ids.longitude_input.text)
            data_to_save['elevation'] = int(main_screen.ids.elevation_input.text)
            # data_to_save['timezone'] = self.current_timezone # Optional: Zeitzone auch speichern
            
            with open(settings_path, 'w') as f:
                json.dump(data_to_save, f, indent=4)
            print(f"User settings saved to {settings_path}")
        except ValueError:
            print(f"Warning: Could not save settings due to invalid number format in input fields.")
        except AttributeError:
             print(f"Warning: Could not save settings. One or more input fields (latitude_input, etc.) not found on main screen.")
        except Exception as e:
            print(f"Failed to save user settings to {settings_path}. Error: {e}")


    def on_language(self, instance, lang):
        lang_dict = get_translation(lang)
        self.t = lang_dict.get if isinstance(lang_dict, dict) else (lambda key, default="": default)
        print(f"Language changed to: {lang}.")
        self.title = self.t('app_title', "Advanced DSO Finder")
        try: 
            if self.root and hasattr(self.root, 'has_screen') and self.root.has_screen('main'):
                 self.root.get_screen('main').status_message = self.t('language_changed_status', "Sprache geändert.")
        except Exception as e:
            print(f"Error updating UI text on language change: {e}")

    def calculate_redshift(self):
        print("Trigger calculate_redshift (not fully implemented)")
        # TODO: Read OmegaM, OmegaL from UI Properties/Widgets if they are configurable
        omega_m_val = OMEGA_M_DEFAULT 
        omega_l_val = OMEGA_LAMBDA_DEFAULT
        translator = self.t if callable(self.t) else (lambda k, d="": d)
        try:
            if 'calculate_lcdm_distances' not in globals():
                raise NameError("calculate_lcdm_distances not defined")
            results = calculate_lcdm_distances(self.redshift_z, self.redshift_h0, omega_m_val, omega_l_val)
            self.redshift_results = results
            print(f"Redshift results: {results}")
        except Exception as e:
            print(f"Error calculating redshift: {e}")
            error_msg = translator('error_calc_failed', "Redshift calculation failed: {}").format(e)
            self.redshift_results = {'error_key': 'error_calc_failed', 'error_message': error_msg}
            # TODO: Show error in Redshift UI

# --- App Start ---
if __name__ == '__main__':
    # import multiprocessing # Only needed if you use the multiprocessing module directly
    # multiprocessing.freeze_support() # Typically for Windows freezing
    DSOFinderApp().run()
