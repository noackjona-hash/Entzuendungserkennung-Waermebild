import cv2
import numpy as np
import math
import logging
import json
import csv
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ==========================================
# DATENSTRUKTUREN (Data Classes)
# ==========================================

@dataclass
class TemperatureStats:
    """
    Kapselt alle statistischen Temperaturdaten einer Entzündung.
    Wichtig für die spätere wissenschaftliche Auswertung.
    """
    min_val: float = 0.0
    max_val: float = 0.0
    mean_val: float = 0.0
    median_val: float = 0.0
    std_dev: float = 0.0
    variance: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "min": round(self.min_val, 2),
            "max": round(self.max_val, 2),
            "mean": round(self.mean_val, 2),
            "median": round(self.median_val, 2),
            "std_dev": round(self.std_dev, 2),
            "variance": round(self.variance, 2)
        }

@dataclass
class Entzuendung:
    """
    Erweiterte Datenstruktur zur Speicherung der Informationen einer gefundenen Entzündung.
    """
    gelenk_name: str
    groesse_px: float       
    staerke: float # Abwärtskompatibilität für main.py (entspricht mean_val)
    zentrum: Tuple[int, int] 
    kontur: np.ndarray # Hauptkontur für main.py
    
    # Neue, detaillierte Eigenschaften
    stats_raw: TemperatureStats = field(default_factory=TemperatureStats)
    stats_celsius: TemperatureStats = field(default_factory=TemperatureStats)
    konturen_ebenen: Dict[str, np.ndarray] = field(default_factory=dict) # 'core', 'mid', 'outer'
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0) # x, y, w, h
    roi_bild: Optional[np.ndarray] = None # Ausgeschnittener Bereich (Region of Interest)

# ==========================================
# HAUPTKLASSE DER ANALYSE
# ==========================================

class ThermalAnalyzer:
    """
    Wissenschaftliche Klasse zur Analyse von Wärmebildern. 
    Beinhaltet Preprocessing, Adaptive Thresholding, Multi-Level Segmentation,
    statistische Auswertung und Daten-Export.
    """
    
    def __init__(self, bild_pfad: str, segmente: List[dict]):
        """
        Initialisiert den Analyzer und bereitet das Bild vor.
        """
        self.bild_pfad = bild_pfad
        self.segmente = segmente
        self.ausgabe_ordner = os.path.dirname(self.bild_pfad)
        self.basis_dateiname = os.path.splitext(os.path.basename(self.bild_pfad))[0]
        
        # Setup Logging
        self._setup_logger()
        self.logger.info(f"Initialisiere ThermalAnalyzer für Bild: {bild_pfad}")
        
        # Lade Bild (Umlaut-sicher)
        self.original_bild = self._lade_bild_sicher(self.bild_pfad)
        
        # Vorverarbeitung (Preprocessing)
        self.graustufen_bild = cv2.cvtColor(self.original_bild, cv2.COLOR_BGR2GRAY)
        self.vorverarbeitetes_bild = self._preprocess_image(self.graustufen_bild)
        
        # Speicher für Ergebnisse
        self.gefundene_entzuendungen: List[Entzuendung] = []
        
        # Kalibrierungswerte für Pseudo-Celsius-Umrechnung (anpassbar)
        # Bsp: Pixel 0 = 20°C (Raumtemp), Pixel 255 = 40°C (Starke Entzündung)
        self.temp_min_celsius = 20.0
        self.temp_max_celsius = 42.0

    def _setup_logger(self):
        """Richtet ein professionelles Logging-System ein."""
        self.logger = logging.getLogger("ThermalAnalyzer")
        self.logger.setLevel(logging.DEBUG)
        
        # Verhindere doppelte Logs bei mehrmaligem Aufruf
        if not self.logger.handlers:
            log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            
            # File Handler
            log_datei = os.path.join(self.ausgabe_ordner, f"{self.basis_dateiname}_analyse.log")
            fh = logging.FileHandler(log_datei, encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(log_format)
            self.logger.addHandler(fh)

    def _lade_bild_sicher(self, pfad: str) -> np.ndarray:
        """Lädt ein Bild sicher über numpy, um Windows-Pfadprobleme mit Umlauten zu umgehen."""
        try:
            with open(pfad, 'rb') as f:
                bytes_data = f.read()
            array = np.frombuffer(bytes_data, dtype=np.uint8)
            bild = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if bild is None:
                raise ValueError("cv2.imdecode hat None zurückgegeben.")
            return bild
        except Exception as e:
            self.logger.error(f"Fehler beim Laden des Bildes {pfad}: {e}")
            raise FileNotFoundError(f"Bild konnte nicht geladen werden: {pfad}")

    def _preprocess_image(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Wendet Bildverbesserungsalgorithmen an (Denoising & Kontrastverstärkung).
        Wichtig für verrauschte Wärmebildkameras (wie FLIR).
        """
        self.logger.debug("Starte Preprocessing...")
        # 1. Rauschunterdrückung (Non-Local Means Denoising)
        # Glättet Rauschen, behält aber harte Kanten (wichtig für Entzündungsränder)
        denoised = cv2.fastNlMeansDenoising(gray_image, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # 2. Kontrastverstärkung (CLAHE - Contrast Limited Adaptive Histogram Equalization)
        # Erhöht den Kontrast lokal, um Hitzepunkte besser vom Hintergrund zu trennen
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        return enhanced

    def _pixel_zu_celsius(self, pixel_wert: float) -> float:
        """Rechnet einen 8-Bit Pixelwert (0-255) linear in einen Celsius-Wert um."""
        prozent = pixel_wert / 255.0
        celsius = self.temp_min_celsius + (prozent * (self.temp_max_celsius - self.temp_min_celsius))
        return celsius

    def _berechne_statistiken(self, maske: np.ndarray) -> Tuple[TemperatureStats, TemperatureStats]:
        """Berechnet detaillierte statistische Daten für einen maskierten Bereich."""
        # Extrahiere alle Pixelwerte im maskierten Bereich
        pixel_werte = self.vorverarbeitetes_bild[maske == 255]
        
        if len(pixel_werte) == 0:
            return TemperatureStats(), TemperatureStats()
            
        # Berechne Roh-Statistiken (0-255)
        raw_stats = TemperatureStats(
            min_val=float(np.min(pixel_werte)),
            max_val=float(np.max(pixel_werte)),
            mean_val=float(np.mean(pixel_werte)),
            median_val=float(np.median(pixel_werte)),
            std_dev=float(np.std(pixel_werte)),
            variance=float(np.var(pixel_werte))
        )
        
        # Berechne Celsius-Statistiken
        celsius_werte = np.array([self._pixel_zu_celsius(p) for p in pixel_werte])
        celsius_stats = TemperatureStats(
            min_val=float(np.min(celsius_werte)),
            max_val=float(np.max(celsius_werte)),
            mean_val=float(np.mean(celsius_werte)),
            median_val=float(np.median(celsius_werte)),
            std_dev=float(np.std(celsius_werte)),
            variance=float(np.var(celsius_werte))
        )
        
        return raw_stats, celsius_stats

    def analysiere(self, temperatur_schwellenwert: int = 210, max_distanz: int = 60) -> List[Entzuendung]:
        """
        Hauptalgorithmus: Findet und analysiert Entzündungen an den markierten Segmenten.
        Nutzt ein Multi-Level Region Growing für detaillierte Zonen.
        """
        self.logger.info(f"Starte Analyse mit Schwellenwert {temperatur_schwellenwert} und Radius {max_distanz}")
        self.gefundene_entzuendungen = []
        
        for seg in self.segmente:
            s, e = seg['start'], seg['end']
            mX, mY = (s[0] + e[0]) // 2, (s[1] + e[1]) // 2
            self.logger.debug(f"Analysiere Segment '{seg['name']}' bei ({mX}, {mY})")
            
            # Lokalen Suchbereich maskieren
            lokale_maske = np.zeros(self.vorverarbeitetes_bild.shape, dtype="uint8")
            cv2.circle(lokale_maske, (mX, mY), max_distanz, 255, -1)
            lokaler_bereich = cv2.bitwise_and(self.vorverarbeitetes_bild, self.vorverarbeitetes_bild, mask=lokale_maske)
            
            # Hitzepunkt (Peak) im lokalen Bereich finden
            _, max_val, _, max_loc = cv2.minMaxLoc(lokaler_bereich, mask=lokale_maske)
            
            if max_val >= temperatur_schwellenwert:
                self.logger.info(f"Entzündung an '{seg['name']}' erkannt! Peak: {max_val}")
                
                # Multi-Level Thresholding für 3 Zonen (Kern, Mitte, Außen)
                schwellen = {
                    'core': max(180, int(max_val) - 15), # Heißester Kern
                    'mid': max(160, int(max_val) - 40),  # Mittlere Ausbreitung
                    'outer': max(140, int(max_val) - 75) # Äußerer Wärmehof
                }
                
                konturen_dict = {}
                haupt_kontur = None
                
                kernel = np.ones((5, 5), np.uint8)
                
                # Iteriere durch die Schwellenwerte, um Zonen zu finden
                for ebene, schwelle in schwellen.items():
                    _, thresh = cv2.threshold(lokaler_bereich, schwelle, 255, cv2.THRESH_BINARY)
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                    
                    konturen, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if konturen:
                        groesste_kontur = max(konturen, key=cv2.contourArea)
                        if cv2.contourArea(groesste_kontur) >= 20:
                            konturen_dict[ebene] = groesste_kontur
                            if ebene == 'outer':
                                haupt_kontur = groesste_kontur
                                
                if haupt_kontur is None and 'core' in konturen_dict:
                    haupt_kontur = konturen_dict['core']
                    
                if haupt_kontur is not None:
                    # Zentrum berechnen
                    M = cv2.moments(haupt_kontur)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        cX, cY = max_loc
                        
                    # Statistiken für den Gesamten betroffenen Bereich (Outer Contour) berechnen
                    entz_maske = np.zeros(self.vorverarbeitetes_bild.shape, dtype="uint8")
                    cv2.drawContours(entz_maske, [haupt_kontur], -1, 255, -1)
                    
                    raw_stats, celsius_stats = self._berechne_statistiken(entz_maske)
                    
                    # Bounding Box berechnen (Für ROI Extraktion)
                    x, y, w, h = cv2.boundingRect(haupt_kontur)
                    
                    # ROI (Region of Interest) ausschneiden (mit etwas Rand)
                    padding = 20
                    y1 = max(0, y - padding)
                    y2 = min(self.original_bild.shape[0], y + h + padding)
                    x1 = max(0, x - padding)
                    x2 = min(self.original_bild.shape[1], x + w + padding)
                    roi = self.original_bild[y1:y2, x1:x2].copy()
                    
                    # Entzündungsobjekt erstellen und speichern
                    entz = Entzuendung(
                        gelenk_name=seg['name'],
                        groesse_px=cv2.contourArea(haupt_kontur),
                        staerke=raw_stats.mean_val, # Kompatibilität mit main.py
                        zentrum=(cX, cY),
                        kontur=haupt_kontur,
                        stats_raw=raw_stats,
                        stats_celsius=celsius_stats,
                        konturen_ebenen=konturen_dict,
                        bounding_box=(x, y, w, h),
                        roi_bild=roi
                    )
                    self.gefundene_entzuendungen.append(entz)
                    
        self.logger.info(f"Analyse beendet. {len(self.gefundene_entzuendungen)} Entzündungen gefunden.")
        
        # Datenexport triggern
        self._exportiere_daten()
        self._speichere_rois()
        
        return self.gefundene_entzuendungen

    def _exportiere_daten(self):
        """Exportiert alle Ergebnisse als JSON und CSV für die wissenschaftliche Auswertung."""
        json_pfad = os.path.join(self.ausgabe_ordner, f"{self.basis_dateiname}_daten.json")
        csv_pfad = os.path.join(self.ausgabe_ordner, f"{self.basis_dateiname}_daten.csv")
        
        export_daten = []
        for e in self.gefundene_entzuendungen:
            data = {
                "gelenk": e.gelenk_name,
                "groesse_px": round(e.groesse_px, 2),
                "zentrum_x": e.zentrum[0],
                "zentrum_y": e.zentrum[1],
                "rohdaten": e.stats_raw.to_dict(),
                "celsius": e.stats_celsius.to_dict()
            }
            export_daten.append(data)
            
        # 1. Speichere JSON
        try:
            with open(json_pfad, 'w', encoding='utf-8') as f:
                json.dump(export_daten, f, indent=4, ensure_ascii=False)
            self.logger.debug(f"JSON exportiert nach {json_pfad}")
        except Exception as e:
            self.logger.error(f"Fehler beim JSON Export: {e}")

        # 2. Speichere CSV
        try:
            if export_daten:
                with open(csv_pfad, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, delimiter=';')
                    # Header
                    writer.writerow(['Gelenk', 'Groesse_px', 'Zentrum_X', 'Zentrum_Y', 
                                     'Temp_Max_C', 'Temp_Mean_C', 'Temp_StdDev_C'])
                    # Data
                    for e in export_daten:
                        writer.writerow([
                            e['gelenk'], e['groesse_px'], e['zentrum_x'], e['zentrum_y'],
                            e['celsius']['max'], e['celsius']['mean'], e['celsius']['std_dev']
                        ])
                self.logger.debug(f"CSV exportiert nach {csv_pfad}")
        except Exception as e:
            self.logger.error(f"Fehler beim CSV Export: {e}")

    def _speichere_rois(self):
        """Speichert die isolierten Entzündungsbereiche (ROIs) als separate Bilder."""
        roi_ordner = os.path.join(self.ausgabe_ordner, f"{self.basis_dateiname}_ROIs")
        if self.gefundene_entzuendungen and not os.path.exists(roi_ordner):
            os.makedirs(roi_ordner)
            
        for e in self.gefundene_entzuendungen:
            if e.roi_bild is not None:
                # Säubere Gelenkname für Dateisystem (z.B. Leerzeichen ersetzen)
                sicherer_name = str(e.gelenk_name).replace(" ", "_").replace("/", "_")
                roi_pfad = os.path.join(roi_ordner, f"{sicherer_name}.png")
                erfolg, buffer = cv2.imencode('.png', e.roi_bild)
                if erfolg:
                    with open(roi_pfad, 'wb') as f:
                        f.write(buffer)

    def render_output(self, output_pfad: str):
        """
        Rendert das wissenschaftliche finale Ergebnisbild mit detaillierten Overlays.
        Zeigt Konturen in verschiedenen Farben je nach Hitzezone.
        """
        ausgabe = self.original_bild.copy()
        
        # 1. Zeichne Segmente (Basis-Linien)
        for seg in self.segmente:
            cv2.line(ausgabe, seg['start'], seg['end'], (255, 150, 0), 1)
            cv2.circle(ausgabe, seg['start'], 3, (255, 255, 255), -1)
            cv2.circle(ausgabe, seg['end'], 3, (255, 255, 255), -1)

        # 2. Zeichne Entzündungs-Details
        for entz in self.gefundene_entzuendungen:
            
            # Zeichne Bounding Box (Kasten um die Entzündung) leicht transparent oder fein
            x, y, w, h = entz.bounding_box
            cv2.rectangle(ausgabe, (x, y), (x+w, y+h), (100, 100, 100), 1, cv2.LINE_AA)
            
            # Zeichne Multi-Level Konturen
            # Outer = Gelb, Mid = Orange, Core = Rot
            farben = {'outer': (0, 255, 255), 'mid': (0, 165, 255), 'core': (0, 0, 255)}
            dicken = {'outer': 1, 'mid': 2, 'core': -1} # Core wird ausgefüllt (-1) oder dick gezeichnet
            
            for ebene, kontur in entz.konturen_ebenen.items():
                farbe = farben.get(ebene, (255, 255, 255))
                dicke = dicken.get(ebene, 1)
                cv2.drawContours(ausgabe, [kontur], -1, farbe, dicke, cv2.LINE_AA)
                
            # Zentrum markieren
            cv2.drawMarker(ausgabe, entz.zentrum, (0, 0, 0), cv2.MARKER_CROSS, 10, 2)

            # Professionelles Text-Label mit berechneter Celsius-Temperatur
            label_name = f"{entz.gelenk_name}"
            label_temp = f"Max: {entz.stats_celsius.max_val:.1f}C"
            
            # Text-Hintergrund für Name
            t_size1, _ = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            t_size2, _ = cv2.getTextSize(label_temp, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            max_width = max(t_size1[0], t_size2[0])
            
            bg_rect_start = (x, y - 35)
            bg_rect_end = (x + max_width + 5, y - 5)
            
            # Halbtransparenter Hintergrund wäre aufwendiger in CV2, wir nehmen Schwarz
            cv2.rectangle(ausgabe, bg_rect_start, bg_rect_end, (0, 0, 0), -1)
            
            # Text schreiben
            cv2.putText(ausgabe, label_name, (x + 2, y - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(ausgabe, label_temp, (x + 2, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                        
        # UMLAUT-SICHERES SPEICHERN
        erfolg, buffer = cv2.imencode('.png', ausgabe)
        if erfolg:
            with open(output_pfad, 'wb') as f:
                f.write(buffer)
            self.logger.info(f"Ergebnisbild erfolgreich gespeichert unter: {output_pfad}")
        else:
            self.logger.error("Fehler beim Encodieren des Ergebnisbildes.")
            raise IOError("Fehler: Das Ergebnisbild konnte nicht kodiert werden.")