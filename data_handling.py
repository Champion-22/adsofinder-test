# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
import numpy as np
import os
import io # Used for handling potential byte strings in coordinates

def parse_coord_string(coord_str: str | bytes | None) -> float | None:
    """
    Versucht, einen Koordinatenstring (RA oder Dec) in Grad umzuwandeln.
    Behandelt verschiedene Formate (z.B. '10 08 22.3', '+11 58 02', Bytes).
    Gibt None zurück, wenn die Umwandlung fehlschlägt.
    """
    if coord_str is None:
        return None
    # Wenn es Bytes sind, versuchen zu dekodieren
    if isinstance(coord_str, bytes):
        try:
            coord_str = coord_str.decode('utf-8')
        except UnicodeDecodeError:
            try:
                coord_str = coord_str.decode('latin-1') # Fallback-Encoding
            except UnicodeDecodeError:
                print(f"Warnung: Konnte Koordinaten-Bytes nicht dekodieren: {coord_str!r}")
                return None

    # Sicherstellen, dass es ein String ist
    if not isinstance(coord_str, str):
         # Manchmal sind es vielleicht schon Zahlen (obwohl die Spalte als Objekt geladen wird)
        if isinstance(coord_str, (int, float)) and np.isfinite(coord_str):
            return float(coord_str)
        return None

    coord_str = coord_str.strip()
    if not coord_str:
        return None

    # Einfachste Annahme: Bereits eine Zahl (als String)
    try:
        val = float(coord_str)
        if np.isfinite(val): # Nur endliche Zahlen zurückgeben
             return val
        else:
             return None # NaN oder Inf ignorieren
    except ValueError:
        pass # Weiter zu komplexeren Formaten

    # Format mit Leerzeichen (z.B. Stunden/Grad, Minuten, Sekunden)
    # Ersetze gängige Trennzeichen durch Leerzeichen und teile auf
    parts = coord_str.replace(':', ' ').replace('h', ' ').replace('m', ' ').replace('s', ' ').replace('d', ' ').replace('\'', ' ').replace('"', ' ').split()

    # Entferne leere Strings, die durch mehrere Leerzeichen entstehen könnten
    parts = [p for p in parts if p]

    if len(parts) == 3:
        try:
            # Vorzeichen behandeln (wichtig für Deklination)
            sign = -1 if parts[0].startswith('-') else 1
            # Teile in Zahlen umwandeln (ignoriere Vorzeichen beim ersten Teil für abs)
            # Behandle mögliche '+' Zeichen
            part0_val = parts[0].replace('+', '')
            h_or_d = abs(float(part0_val))
            m = float(parts[1])
            s = float(parts[2])

            # Umrechnung basierend auf Vorzeichen (RA vs Dec)
            # RA (typischerweise positiv): Stunden -> Grad
            # Dec (kann negativ sein): Grad direkt
            # Annahme: Wenn das erste Vorzeichen '+' oder keins ist UND der Wert <= 24 ist, ist es RA.
            is_likely_ra = (sign == 1 and h_or_d <= 24)

            if is_likely_ra:
                degrees = (h_or_d + m / 60.0 + s / 3600.0) * 15.0 # Stunden zu Grad
            else: # Wahrscheinlich Dec in Grad (oder RA > 24h?)
                degrees = sign * (h_or_d + m / 60.0 + s / 3600.0) # Grad direkt

            # Überprüfen, ob das Ergebnis eine gültige Zahl ist
            if np.isfinite(degrees):
                return degrees
            else:
                print(f"Warnung: Ergebnis der Koordinatenberechnung ist ungültig (NaN/Inf): {degrees} from {parts}")
                return None
        except ValueError:
            # Fehler beim Umwandeln der Teile in Zahlen
            print(f"Warnung: Konnte Koordinaten-Teile nicht in Zahlen umwandeln: {parts} from '{coord_str}'")
            return None
    # Optional: Behandle Format mit nur 2 Teilen (Grad/Stunden und Minuten?)
    # elif len(parts) == 2: ...
    else: # Unbekanntes oder ungültiges Format
        # Versuche nicht, einzelne Zahlen hier erneut zu parsen, das wurde oben abgedeckt
        print(f"Warnung: Unbekanntes oder unvollständiges Koordinatenformat: '{coord_str}' (Geparste Teile: {parts})")
        return None


def load_ongc_data(filepath: str) -> pd.DataFrame | None:
    """
    Lädt die ONGC Katalogdaten aus einer CSV-Datei und verarbeitet sie vor.

    Args:
        filepath: Der Pfad zur CSV-Datei (ongc.csv).

    Returns:
        Ein Pandas DataFrame mit den verarbeiteten Katalogdaten oder None bei Fehlern.
        Stellt sicher, dass RA/Dec als numerische Gradwerte vorhanden sind ('RA_deg', 'Dec_deg').
        Konvertiert 'Mag', 'MajAx', 'MinAx', 'PosAng', 'B-Mag', 'V-Mag', 'J-Mag', 'H-Mag', 'K-Mag', 'z' zu numerischen Typen.
    """
    print(f"Versuche Katalog zu laden von: {filepath}")
    if not os.path.exists(filepath):
        print(f"Fehler: Katalogdatei nicht gefunden unter {filepath}")
        return None

    try:
        # Versuche, die CSV zu laden, mit ';' als Trennzeichen und Header in Zeile 1 (index 0)
        # low_memory=False kann bei gemischten Typen helfen
        # encoding='latin-1' oder 'iso-8859-1' ist oft robuster für ältere Astro-Kataloge
        try:
            df = pd.read_csv(filepath, sep=';', header=0, low_memory=False, encoding='utf-8')
            print("Katalog erfolgreich mit UTF-8 geladen.")
        except UnicodeDecodeError:
            print("UTF-8 Dekodierung fehlgeschlagen, versuche latin-1...")
            df = pd.read_csv(filepath, sep=';', header=0, low_memory=False, encoding='latin-1')
            print("Katalog erfolgreich mit latin-1 geladen.")

        print(f"Katalog geladen. {len(df)} Zeilen, Spalten: {df.columns.tolist()}")

        # --- Spaltennamen prüfen und ggf. anpassen ---
        # Der ONGC-Katalog hat normalerweise Spalten 'RA' und 'Dec' für die String-Repräsentation
        ra_col_str = 'RA'
        dec_col_str = 'Decl' # Im ONGC heisst die Spalte oft 'Decl'

        if ra_col_str not in df.columns:
            print(f"Fehler: Benötigte Spalte '{ra_col_str}' nicht im Katalog gefunden.")
            return None
        if dec_col_str not in df.columns:
            print(f"Fehler: Benötigte Spalte '{dec_col_str}' nicht im Katalog gefunden.")
            # Versuch mit alternativem Namen 'Dec'
            if 'Dec' in df.columns:
                 dec_col_str = 'Dec'
                 print("Warnung: Spalte 'Decl' nicht gefunden, verwende stattdessen 'Dec'.")
            else:
                 print(f"Fehler: Weder '{dec_col_str}' noch 'Dec' im Katalog gefunden.")
                 return None

        # --- Koordinaten parsen ---
        # Wende die parse_coord_string Funktion auf die RA/Dec Spalten an
        # Erstelle neue Spalten für die numerischen Gradwerte
        print("Parse RA Koordinaten...")
        df['RA_deg'] = df[ra_col_str].apply(parse_coord_string)
        print("Parse Dec Koordinaten...")
        df['Dec_deg'] = df[dec_col_str].apply(parse_coord_string)

        # Überprüfe, wie viele Koordinaten erfolgreich geparst wurden
        parsed_ra_count = df['RA_deg'].notna().sum()
        parsed_dec_count = df['Dec_deg'].notna().sum()
        total_count = len(df)
        print(f"RA Parsing erfolgreich für {parsed_ra_count}/{total_count} Einträge.")
        print(f"Dec Parsing erfolgreich für {parsed_dec_count}/{total_count} Einträge.")

        # Optional: Filtere Einträge ohne gültige Koordinaten heraus
        original_len = len(df)
        df = df.dropna(subset=['RA_deg', 'Dec_deg'])
        filtered_len = len(df)
        if original_len > filtered_len:
            print(f"Info: {original_len - filtered_len} Einträge wegen fehlender/ungültiger Koordinaten entfernt.")

        if df.empty:
            print("Fehler: Nach Koordinaten-Parsing sind keine gültigen Einträge mehr vorhanden.")
            return None

        # --- Numerische Spalten konvertieren ---
        # Liste der Spalten, die numerisch sein sollten (füge 'z' hinzu)
        numeric_cols = ['Mag', 'MajAx', 'MinAx', 'PosAng', 'B-Mag', 'V-Mag', 'J-Mag', 'H-Mag', 'K-Mag', 'z']
        for col in numeric_cols:
            if col in df.columns:
                # pd.to_numeric mit errors='coerce' wandelt ungültige Werte in NaN um
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Optional: NaN-Werte in numerischen Spalten behandeln (z.B. mit 0 oder Median füllen),
                # aber oft ist es besser, sie als NaN zu belassen und später zu filtern.
                # Beispiel: df[col] = df[col].fillna(np.nan) # Stellt sicher, dass es NaN ist
                print(f"Spalte '{col}' zu numerisch konvertiert (ungültige Werte -> NaN).")
            else:
                print(f"Warnung: Optionale numerische Spalte '{col}' nicht im Katalog gefunden.")
                # Optional: Fehlende Spalte als NaN hinzufügen
                # df[col] = np.nan

        # --- Datentypen optimieren (optional, spart Speicher) ---
        # Beispiel: Wenn Magnitude immer im Bereich -10 bis 30 liegt, könnte float32 reichen
        # for col in ['Mag', 'B-Mag', 'V-Mag', 'J-Mag', 'H-Mag', 'K-Mag']:
        #     if col in df.columns:
        #         df[col] = df[col].astype(pd.Float32Dtype()) # Nullable Float32
        # for col in ['MajAx', 'MinAx', 'PosAng', 'z']:
        #      if col in df.columns:
        #         df[col] = df[col].astype(pd.Float64Dtype()) # Nullable Float64 für höhere Präzision bei z

        # --- Index zurücksetzen (optional, falls durch dropna Lücken entstanden sind) ---
        df = df.reset_index(drop=True)

        print(f"Katalogverarbeitung abgeschlossen. {len(df)} gültige Einträge.")
        return df

    except FileNotFoundError:
        print(f"Fehler: Katalogdatei nicht gefunden unter {filepath}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Fehler: Katalogdatei ist leer: {filepath}")
        return None
    except pd.errors.ParserError:
        print(f"Fehler: Katalogdatei konnte nicht geparst werden (Formatproblem?): {filepath}")
        return None
    except Exception as e:
        print(f"Unerwarteter Fehler beim Laden/Verarbeiten des Katalogs: {e}")
        import traceback
        traceback.print_exc()
        return None

# Beispielaufruf (nur zum Testen des Moduls, wenn es direkt ausgeführt wird)
if __name__ == "__main__":
    # Finde den Pfad zur Beispieldatei relativ zum Skriptverzeichnis
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Gehe ein Verzeichnis nach oben, wenn die Module in Unterordnern liegen
    # base_dir = os.path.dirname(script_dir)
    # catalog_path = os.path.join(base_dir, "ongc.csv") # Annahme: ongc.csv liegt im Hauptverzeichnis
    catalog_path = os.path.join(script_dir, "ongc.csv") # Annahme: ongc.csv liegt im selben Verzeichnis

    print(f"Teste Ladevorgang für: {catalog_path}")
    df_loaded = load_ongc_data(catalog_path)

    if df_loaded is not None:
        print("\nKatalog erfolgreich geladen und verarbeitet.")
        print("Erste 5 Zeilen:")
        print(df_loaded.head())
        print("\nInfo:")
        df_loaded.info()
        print("\nStatistik für numerische Spalten:")
        # Wähle nur Spalten aus, die wahrscheinlich numerisch sind
        numeric_cols_info = ['Mag', 'MajAx', 'MinAx', 'PosAng', 'B-Mag', 'V-Mag', 'J-Mag', 'H-Mag', 'K-Mag', 'z', 'RA_deg', 'Dec_deg']
        print(df_loaded[[col for col in numeric_cols_info if col in df_loaded.columns]].describe())
        # Prüfe auf NaN-Werte in wichtigen Spalten
        print("\nNaN-Werte in Schlüsselspalten:")
        print(df_loaded[['Name', 'RA_deg', 'Dec_deg', 'Mag']].isna().sum())
    else:
        print("\nKatalog konnte nicht geladen werden.")
