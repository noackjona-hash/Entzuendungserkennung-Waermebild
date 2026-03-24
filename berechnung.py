import cv2
import numpy as np
import math
import logging
import json
import csv
import os
import base64
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

# ==============================================================================
# KONFIGURATION & SETUP
# ==============================================================================

@dataclass
class ThermalConfig:
    """
    Zentrale Konfigurationsklasse für alle Algorithmus-Parameter.
    Erlaubt das einfache Feintuning der gesamten Pipeline.
    """
    # Temperatur-Kalibrierung (Pixel zu Celsius)
    temp_min_celsius: float = 20.0
    temp_max_celsius: float = 42.0
    
    # Erkennungs-Parameter
    basis_schwellenwert_pixel: int = 210  # Fallback, falls dynamische Berechnung fehlschlägt
    suchradius_pixel: int = 70            # Radius um den Zeh/Finger
    min_kontur_flaeche: int = 30          # Minimale Pixelanzahl für eine Entzündung
    
    # Scoring-Gewichte (Für die Heuristik)
    score_gewicht_absolut: float = 0.4    # Gewichtung der absoluten Temperatur
    score_gewicht_kontrast: float = 0.3   # Gewichtung des lokalen Kontrasts
    score_gewicht_asymmetrie: float = 0.2 # Gewichtung der Links/Rechts Differenz
    score_gewicht_form: float = 0.1       # Gewichtung der Zirkularität (runde Form)
    
    # JUGEND FORSCHT UPDATE: Strikter Schwellenwert (vorher 65.0, jetzt 85.0)
    # So werden nur die wirklich sicheren, extremen Entzündungen angezeigt!
    min_confidence_score: float = 85.0    
    
    # NMS (Non-Maximum Suppression) Parameter
    nms_overlap_distanz: int = 40         # Wenn zwei Zentren näher als X Pixel sind, sind sie Duplikate

# ==============================================================================
# DATENSTRUKTUREN (Data Classes) - WISSENSCHAFTLICHES LEVEL
# ==============================================================================

@dataclass
class TemperatureStats:
    """Kapselt alle statistischen Temperaturdaten eines Bereichs."""
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
class MorphologyFeatures:
    """Kapselt geometrische Eigenschaften einer gefundenen Entzündung."""
    flaeche: float = 0.0
    umfang: float = 0.0
    zirkularitaet: float = 0.0       # 1.0 ist ein perfekter Kreis
    aspect_ratio: float = 0.0        # Verhältnis Breite/Höhe der Bounding Box
    solidity: float = 0.0            # Verhältnis Fläche zu Convex Hull Fläche
    
    def to_dict(self) -> dict:
        return {
            "flaeche_px": round(self.flaeche, 2),
            "zirkularitaet": round(self.zirkularitaet, 3),
            "aspect_ratio": round(self.aspect_ratio, 3),
            "solidity": round(self.solidity, 3)
        }

@dataclass
class HeuristicScore:
    """Speichert die detaillierte Zusammensetzung der Konfidenz-Bewertung."""
    absolut_score: float = 0.0
    kontrast_score: float = 0.0
    asymmetrie_score: float = 0.0
    form_score: float = 0.0
    total_confidence: float = 0.0    # 0.0 bis 100.0 Prozent
    is_valid: bool = False           # True, wenn total_confidence >= min_confidence_score

@dataclass
class ThermalProfile:
    """Speichert den Temperatur-Gradienten entlang eines markierten Segments."""
    distanzen: List[float] = field(default_factory=list)
    temperaturen_c: List[float] = field(default_factory=list)
    max_temp_c: float = 0.0
    mean_temp_c: float = 0.0

@dataclass
class Entzuendung:
    """
    Haupt-Datenstruktur. Beinhaltet alle berechneten Daten einer Anomalie.
    """
    gelenk_name: str
    groesse_px: float       
    staerke: float 
    zentrum: Tuple[int, int] 
    kontur: np.ndarray 
    
    stats_raw: TemperatureStats = field(default_factory=TemperatureStats)
    stats_celsius: TemperatureStats = field(default_factory=TemperatureStats)
    morphology: MorphologyFeatures = field(default_factory=MorphologyFeatures)
    score: HeuristicScore = field(default_factory=HeuristicScore)
    profil: ThermalProfile = field(default_factory=ThermalProfile)
    
    roi_bild: Optional[np.ndarray] = None 

# ==============================================================================
# HAUPTKLASSE DER ANALYSE
# ==============================================================================

class ThermalAnalyzer:
    """
    Professionelle Pipeline zur Analyse medizinischer Wärmebilder.
    Beinhaltet: Preprocessing, Körper-Isolation, Asymmetrie-Prüfung,
    Multilevel-Segmentation, Heuristik-Scoring, NMS. (Optimiert für In-Memory API)
    """
    
    def __init__(self, bild_pfad: Optional[str] = None, segmente: List[dict] = None, bild_bytes: Optional[bytes] = None):
        self.bild_pfad = bild_pfad
        self.segmente = segmente if segmente else []
        self.ausgabe_ordner = os.path.dirname(self.bild_pfad) if self.bild_pfad else "in_memory"
        self.basis_dateiname = os.path.splitext(os.path.basename(self.bild_pfad))[0] if self.bild_pfad else "api_upload"
        self.config = ThermalConfig()
        
        self.analyse_protokoll: List[str] = []
        
        self._setup_logger()
        self.logger.info("="*60)
        self.logger.info(f"START THERMAL ANALYSIS PIPELINE v2.6 (In-Memory / Cloud Optimized)")
        
        # 1. Bild sicher laden (von Pfad ODER direkt aus dem RAM)
        if bild_bytes:
            self.original_bild = self._lade_bild_aus_bytes(bild_bytes)
            self.logger.info("Bild erfolgreich direkt aus dem Arbeitsspeicher geladen.")
        elif self.bild_pfad:
            self.original_bild = self._lade_bild_sicher(self.bild_pfad)
        else:
            raise ValueError("Es muss entweder bild_pfad oder bild_bytes übergeben werden.")
            
        self.bild_hoehe, self.bild_breite = self.original_bild.shape[:2]
        
        self.graustufen_bild = cv2.cvtColor(self.original_bild, cv2.COLOR_BGR2GRAY)
        self.vorverarbeitetes_bild = self._preprocess_image(self.graustufen_bild)
        
        # 3. Globale Körperstatistiken berechnen (Ignoriert Hintergrund)
        self._berechne_globale_koerper_statistiken()
        
        # 4. Asymmetrie-Analyse vorbereiten (Links vs. Rechts)
        self._bereite_symmetrie_analyse_vor()
        
        # Speicher für Ergebnisse
        self.gefundene_entzuendungen: List[Entzuendung] = []
        self.alle_kandidaten: List[Entzuendung] = []

    def _setup_logger(self):
        self.logger = logging.getLogger("ThermalAnalyzer")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            log_format = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
            # Cloud-Optimierung: Nur noch Console-Log, keine lokalen .log Dateien mehr!
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(log_format)
            self.logger.addHandler(ch)

    def _lade_bild_aus_bytes(self, bild_bytes: bytes) -> np.ndarray:
        try:
            array = np.frombuffer(bild_bytes, dtype=np.uint8)
            bild = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if bild is None: raise ValueError("Decode-Fehler.")
            return bild
        except Exception as e:
            self.logger.critical(f"Konnte Bild nicht aus RAM laden: {e}")
            raise ValueError("Bild-Bytes korrupt oder ungültiges Format.")

    def _lade_bild_sicher(self, pfad: str) -> np.ndarray:
        try:
            with open(pfad, 'rb') as f:
                bytes_data = f.read()
            array = np.frombuffer(bytes_data, dtype=np.uint8)
            bild = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if bild is None: raise ValueError("Decode-Fehler.")
            return bild
        except Exception as e:
            self.logger.critical(f"Konnte Bild nicht laden: {e}")
            raise FileNotFoundError(f"Bildladefehler: {pfad}")

    def _preprocess_image(self, gray_image: np.ndarray) -> np.ndarray:
        self.logger.debug("Führe Preprocessing durch (Denoising + CLAHE)...")
        denoised = cv2.fastNlMeansDenoising(gray_image, None, h=10, templateWindowSize=7, searchWindowSize=21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(denoised)

    def _pixel_zu_celsius(self, pixel_wert: float) -> float:
        prozent = max(0.0, min(1.0, pixel_wert / 255.0))
        return self.config.temp_min_celsius + (prozent * (self.config.temp_max_celsius - self.config.temp_min_celsius))

    def _berechne_globale_koerper_statistiken(self):
        _, koerper_maske = cv2.threshold(self.vorverarbeitetes_bild, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        koerper_pixel = self.vorverarbeitetes_bild[koerper_maske == 255]
        
        if len(koerper_pixel) > 0:
            self.global_mean_temp_raw = float(np.mean(koerper_pixel))
            self.global_std_temp_raw = float(np.std(koerper_pixel))
            sorted_pixels = np.sort(koerper_pixel)
            top_5_percent_idx = int(len(sorted_pixels) * 0.95)
            self.global_hot_threshold = float(sorted_pixels[top_5_percent_idx])
            
            self.logger.info(f"Globale Körper-Statistik -> Mean: {self.global_mean_temp_raw:.1f}, "
                             f"StdDev: {self.global_std_temp_raw:.1f}, Top 5% Schwelle: {self.global_hot_threshold:.1f}")
            self.analyse_protokoll.append(f"🔍 Globale Körper-Temperatur ermittelt (Ø {self._pixel_zu_celsius(self.global_mean_temp_raw):.1f}°C).")
        else:
            self.global_mean_temp_raw = 128.0
            self.global_std_temp_raw = 30.0
            self.global_hot_threshold = 200.0

    def _bereite_symmetrie_analyse_vor(self):
        if not self.segmente: return
        alle_x = [(s['start'][0] + s['end'][0]) / 2 for s in self.segmente]
        bild_mitte_x = sum(alle_x) / len(alle_x) if alle_x else self.bild_breite / 2
        
        self.segmente_links, self.segmente_rechts = [], []
        temp_sum_links, count_links = 0.0, 0
        temp_sum_rechts, count_rechts = 0.0, 0
        
        for seg in self.segmente:
            s, e = seg['start'], seg['end']
            mX, mY = (s[0] + e[0]) // 2, (s[1] + e[1]) // 2
            
            maske = np.zeros(self.vorverarbeitetes_bild.shape, dtype="uint8")
            cv2.circle(maske, (mX, mY), self.config.suchradius_pixel, 255, -1)
            lokal = cv2.bitwise_and(self.vorverarbeitetes_bild, self.vorverarbeitetes_bild, mask=maske)
            _, max_val, _, _ = cv2.minMaxLoc(lokal, mask=maske)
            
            seg['pre_max_temp'] = max_val
            
            if mX < bild_mitte_x:
                self.segmente_links.append(seg)
                temp_sum_links += max_val
                count_links += 1
            else:
                self.segmente_rechts.append(seg)
                temp_sum_rechts += max_val
                count_rechts += 1
                
        self.mean_temp_links = temp_sum_links / count_links if count_links > 0 else 0
        self.mean_temp_rechts = temp_sum_rechts / count_rechts if count_rechts > 0 else 0
        self.asymmetrie_faktor = abs(self.mean_temp_links - self.mean_temp_rechts)
        self.heissere_seite = "links" if self.mean_temp_links > self.mean_temp_rechts else "rechts"
        
        if self.asymmetrie_faktor > 15:
            self.analyse_protokoll.append(f"⚖️ Starke Asymmetrie erkannt: Die {self.heissere_seite} Seite ist signifikant heißer.")

    def _berechne_statistiken(self, maske: np.ndarray) -> Tuple[TemperatureStats, TemperatureStats]:
        pixel_werte = self.vorverarbeitetes_bild[maske == 255]
        if len(pixel_werte) == 0: return TemperatureStats(), TemperatureStats()
            
        raw = TemperatureStats(min_val=float(np.min(pixel_werte)), max_val=float(np.max(pixel_werte)), mean_val=float(np.mean(pixel_werte)), median_val=float(np.median(pixel_werte)), std_dev=float(np.std(pixel_werte)), variance=float(np.var(pixel_werte)))
        celsius_werte = np.array([self._pixel_zu_celsius(p) for p in pixel_werte])
        celsius = TemperatureStats(min_val=float(np.min(celsius_werte)), max_val=float(np.max(celsius_werte)), mean_val=float(np.mean(celsius_werte)), median_val=float(np.median(celsius_werte)), std_dev=float(np.std(celsius_werte)), variance=float(np.var(celsius_werte)))
        return raw, celsius

    def _berechne_morphologie(self, kontur: np.ndarray) -> MorphologyFeatures:
        flaeche = cv2.contourArea(kontur)
        umfang = cv2.arcLength(kontur, True)
        zirkularitaet = (4 * math.pi * flaeche) / (umfang * umfang) if umfang > 0 else 0.0
        x, y, w, h = cv2.boundingRect(kontur)
        hull_flaeche = cv2.contourArea(cv2.convexHull(kontur))
        solidity = float(flaeche) / hull_flaeche if hull_flaeche > 0 else 0.0
        
        return MorphologyFeatures(flaeche=flaeche, umfang=umfang, zirkularitaet=zirkularitaet, aspect_ratio=float(w)/h if h>0 else 0.0, solidity=solidity)

    def _erstelle_thermisches_profil(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> ThermalProfile:
        x1, y1 = p1
        x2, y2 = p2
        length = int(np.hypot(x2-x1, y2-y1))
        
        if length == 0: return ThermalProfile()
            
        x_indices = np.linspace(x1, x2, length)
        y_indices = np.linspace(y1, y2, length)
        distanzen, temperaturen_c = [], []
        
        for i in range(length):
            xi, yi = int(round(x_indices[i])), int(round(y_indices[i]))
            if 0 <= xi < self.bild_breite and 0 <= yi < self.bild_hoehe:
                raw_val = float(self.vorverarbeitetes_bild[yi, xi])
                temp_c = self._pixel_zu_celsius(raw_val)
                distanzen.append(float(i))
                temperaturen_c.append(temp_c)
                
        if not temperaturen_c: return ThermalProfile()
        return ThermalProfile(distanzen=distanzen, temperaturen_c=temperaturen_c, max_temp_c=max(temperaturen_c), mean_temp_c=sum(temperaturen_c)/len(temperaturen_c))

    def _bewerte_kandidat(self, seg_name: str, peak_raw: float, lokal_mean: float, morph: MorphologyFeatures, is_linke_seite: bool) -> HeuristicScore:
        baseline = max(150.0, self.global_mean_temp_raw)
        abs_score = 0.0 if peak_raw <= baseline else min(100.0, ((peak_raw - baseline) / (255.0 - baseline)) * 100.0)
            
        delta = peak_raw - lokal_mean
        kontrast_score = min(100.0, max(0.0, (delta / 50.0) * 100.0))
        
        asym_score = 100.0
        if self.asymmetrie_faktor > 15.0: 
            if (is_linke_seite and self.heissere_seite == "rechts") or (not is_linke_seite and self.heissere_seite == "links"):
                asym_score = 30.0 
                
        form_score = 0.0 if morph.zirkularitaet < 0.2 else min(100.0, morph.zirkularitaet * 100.0)
        
        total = (abs_score * self.config.score_gewicht_absolut) + (kontrast_score * self.config.score_gewicht_kontrast) + (asym_score * self.config.score_gewicht_asymmetrie) + (form_score * self.config.score_gewicht_form)
        is_valid = total >= self.config.min_confidence_score
        
        # Protokoll für Explainable AI
        if is_valid:
            self.analyse_protokoll.append(f"🟥 {seg_name}: Auffällig! Score {total:.1f}% (Erforderlich: {self.config.min_confidence_score}%).")
        else:
            self.analyse_protokoll.append(f"🟩 {seg_name}: Sicher. Score {total:.1f}% ist zu niedrig für eine Diagnose.")
                          
        return HeuristicScore(absolut_score=abs_score, kontrast_score=kontrast_score, asymmetrie_score=asym_score, form_score=form_score, total_confidence=total, is_valid=is_valid)

    def _non_maximum_suppression(self, kandidaten: List[Entzuendung]) -> List[Entzuendung]:
        if not kandidaten: return []
        kandidaten.sort(key=lambda x: x.score.total_confidence, reverse=True)
        gefiltert: List[Entzuendung] = []
        for kand in kandidaten:
            is_duplicate = False
            for etabliert in gefiltert:
                distanz = math.hypot(kand.zentrum[0] - etabliert.zentrum[0], kand.zentrum[1] - etabliert.zentrum[1])
                if distanz < self.config.nms_overlap_distanz:
                    is_duplicate = True
                    self.logger.info(f"NMS: Unterdrücke '{kand.gelenk_name}' wegen Überlappung mit '{etabliert.gelenk_name}'.")
                    self.analyse_protokoll.append(f"ℹ️ {kand.gelenk_name} ignoriert: Überlappt mit dem stärkeren Befund bei {etabliert.gelenk_name}.")
                    break
            if not is_duplicate:
                gefiltert.append(kand)
        return gefiltert

    def analysiere(self, temperatur_schwellenwert: Optional[int] = None, max_distanz: Optional[int] = None, **kwargs) -> List[Entzuendung]:
        if temperatur_schwellenwert is not None: self.config.basis_schwellenwert_pixel = temperatur_schwellenwert
        if max_distanz is not None: self.config.suchradius_pixel = max_distanz
            
        self.alle_kandidaten = []
        self.analyse_protokoll.append(f"🚀 Beginne Detail-Analyse von {len(self.segmente)} Segmenten...")
        
        for seg in self.segmente:
            s, e = seg['start'], seg['end']
            mX, mY = (s[0] + e[0]) // 2, (s[1] + e[1]) // 2
            is_links = mX < (self.bild_breite / 2)
            
            profil = self._erstelle_thermisches_profil(s, e)
            
            lokale_maske = np.zeros(self.vorverarbeitetes_bild.shape, dtype="uint8")
            cv2.circle(lokale_maske, (mX, mY), self.config.suchradius_pixel, 255, -1)
            lokaler_bereich = cv2.bitwise_and(self.vorverarbeitetes_bild, self.vorverarbeitetes_bild, mask=lokale_maske)
            
            _, max_val, _, max_loc = cv2.minMaxLoc(lokaler_bereich, mask=lokale_maske)
            lokal_mean = cv2.mean(self.vorverarbeitetes_bild, mask=lokale_maske)[0]
            
            lokaler_schwelle = max(self.global_mean_temp_raw + 10, int(max_val) - 30)
            
            schwellen = {
                'core': max(lokaler_schwelle + 20, int(max_val) - 10),
                'mid': max(lokaler_schwelle + 10, int(max_val) - 20),
                'outer': lokaler_schwelle
            }
            
            konturen_dict = {}
            haupt_kontur = None
            kernel = np.ones((3, 3), np.uint8)
            
            for ebene, schwelle in schwellen.items():
                _, thresh = cv2.threshold(lokaler_bereich, schwelle, 255, cv2.THRESH_BINARY)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                
                konturen, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if konturen:
                    gueltige_konturen = [cnt for cnt in konturen if cv2.contourArea(cnt) >= self.config.min_kontur_flaeche]
                    if gueltige_konturen:
                        groesste_kontur = max(gueltige_konturen, key=cv2.contourArea)
                        konturen_dict[ebene] = groesste_kontur
                        if ebene == 'outer': haupt_kontur = groesste_kontur
                            
            if haupt_kontur is None and 'core' in konturen_dict:
                haupt_kontur = konturen_dict['core']
                
            if haupt_kontur is not None:
                M = cv2.moments(haupt_kontur)
                cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else max_loc[0]
                cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else max_loc[1]
                x, y, w, h = cv2.boundingRect(haupt_kontur)
                
                morph = self._berechne_morphologie(haupt_kontur)
                entz_maske = np.zeros(self.vorverarbeitetes_bild.shape, dtype="uint8")
                cv2.drawContours(entz_maske, [haupt_kontur], -1, 255, -1)
                raw_stats, celsius_stats = self._berechne_statistiken(entz_maske)
                
                score = self._bewerte_kandidat(seg['name'], max_val, lokal_mean, morph, is_links)
                
                padding = 15
                y1 = max(0, y - padding)
                y2 = min(self.bild_hoehe, y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(self.bild_breite, x + w + padding)
                roi = self.original_bild[y1:y2, x1:x2].copy()
                
                kandidat = Entzuendung(
                    gelenk_name=seg['name'], groesse_px=morph.flaeche, staerke=raw_stats.mean_val,
                    zentrum=(cX, cY), kontur=haupt_kontur, stats_raw=raw_stats, stats_celsius=celsius_stats,
                    morphology=morph, score=score, profil=profil, konturen_ebenen=konturen_dict,
                    bounding_box=(x, y, w, h), roi_bild=roi
                )
                if score.is_valid:
                    self.alle_kandidaten.append(kandidat)
            else:
                self.analyse_protokoll.append(f"🟩 {seg['name']}: Keine relevante Wärme-Ausdehnung gefunden.")
                    
        self.gefundene_entzuendungen = self._non_maximum_suppression(self.alle_kandidaten)
        self.analyse_protokoll.append(f"🏁 Analyse beendet. {len(self.gefundene_entzuendungen)} finale Entzündung(en) verifiziert.")
        
        # In-Memory Optimierung: Wir speichern keine Dateien mehr ab! (Kein JSON, CSV, ROI)
        
        return self.gefundene_entzuendungen

    # ==========================================================================
    # BILD AUSGABE (RENDERING)
    # ==========================================================================

    def render_base64(self) -> str:
        """
        Rendert das Ergebnisbild direkt im Arbeitsspeicher und gibt es als Base64-String zurück.
        Perfekt für Cloud-APIs, da absolut keine Festplatten-Zugriffe nötig sind.
        """
        ausgabe = self.original_bild.copy()
        
        for seg in self.segmente:
            cv2.line(ausgabe, seg['start'], seg['end'], (255, 100, 0), 1, cv2.LINE_AA)
            cv2.circle(ausgabe, seg['start'], 2, (150, 150, 150), -1)
            cv2.circle(ausgabe, seg['end'], 2, (150, 150, 150), -1)

        farben = {'outer': (0, 255, 255), 'mid': (0, 165, 255), 'core': (0, 0, 255)}
        
        for entz in self.gefundene_entzuendungen:
            x, y, w, h = entz.bounding_box
            cv2.rectangle(ausgabe, (x, y), (x+w, y+h), (255, 255, 255), 1, cv2.LINE_AA)
            
            for ebene, kontur in entz.konturen_ebenen.items():
                farbe = farben.get(ebene, (255, 255, 255))
                dicke = -1 if ebene == 'core' else 2
                cv2.drawContours(ausgabe, [kontur], -1, farbe, dicke, cv2.LINE_AA)
                
            cv2.drawMarker(ausgabe, entz.zentrum, (0, 0, 0), cv2.MARKER_CROSS, 15, 2)
            cv2.drawMarker(ausgabe, entz.zentrum, (255, 255, 255), cv2.MARKER_CROSS, 15, 1)

            label_name = f"{entz.gelenk_name}"
            label_temp = f"Temp: {entz.stats_celsius.max_val:.1f}C"
            label_conf = f"Conf: {entz.score.total_confidence:.1f}%"
            
            t1, _ = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            t2, _ = cv2.getTextSize(label_temp, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            t3, _ = cv2.getTextSize(label_conf, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            max_w = max(t1[0], t2[0], t3[0])
            
            bg_rect_start = (x, max(0, y - 50))
            bg_rect_end = (x + max_w + 6, max(0, y - 2))
            
            cv2.rectangle(ausgabe, bg_rect_start, bg_rect_end, (0, 0, 0), -1)
            
            text_x = x + 3
            text_y_base = y - 40
            cv2.putText(ausgabe, label_name, (text_x, text_y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(ausgabe, label_temp, (text_x, text_y_base + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            
            conf_color = (0, 255, 0) if entz.score.total_confidence > 80 else (0, 255, 255)
            cv2.putText(ausgabe, label_conf, (text_x, text_y_base + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, conf_color, 1, cv2.LINE_AA)
                        
        erfolg, buffer = cv2.imencode('.png', ausgabe)
        if erfolg:
            self.logger.info("Ergebnisbild erfolgreich im RAM encodiert.")
            return base64.b64encode(buffer).decode("utf-8")
        else:
            self.logger.error("Fehler beim Encodieren des Ergebnisbildes.")
            raise IOError("Fehler: Das Ergebnisbild konnte nicht kodiert werden.")

    def render_output(self, output_pfad: str):
        ausgabe = self.original_bild.copy()
        
        for seg in self.segmente:
            cv2.line(ausgabe, seg['start'], seg['end'], (255, 100, 0), 1, cv2.LINE_AA)
            cv2.circle(ausgabe, seg['start'], 2, (150, 150, 150), -1)
            cv2.circle(ausgabe, seg['end'], 2, (150, 150, 150), -1)

        farben = {'outer': (0, 255, 255), 'mid': (0, 165, 255), 'core': (0, 0, 255)}
        
        for entz in self.gefundene_entzuendungen:
            x, y, w, h = entz.bounding_box
            cv2.rectangle(ausgabe, (x, y), (x+w, y+h), (255, 255, 255), 1, cv2.LINE_AA)
            
            for ebene, kontur in entz.konturen_ebenen.items():
                farbe = farben.get(ebene, (255, 255, 255))
                dicke = -1 if ebene == 'core' else 2
                cv2.drawContours(ausgabe, [kontur], -1, farbe, dicke, cv2.LINE_AA)
                
            cv2.drawMarker(ausgabe, entz.zentrum, (0, 0, 0), cv2.MARKER_CROSS, 15, 2)
            cv2.drawMarker(ausgabe, entz.zentrum, (255, 255, 255), cv2.MARKER_CROSS, 15, 1)

            label_name = f"{entz.gelenk_name}"
            label_temp = f"Temp: {entz.stats_celsius.max_val:.1f}C"
            label_conf = f"Conf: {entz.score.total_confidence:.1f}%"
            
            t1, _ = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            t2, _ = cv2.getTextSize(label_temp, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            t3, _ = cv2.getTextSize(label_conf, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            max_w = max(t1[0], t2[0], t3[0])
            
            bg_rect_start = (x, max(0, y - 50))
            bg_rect_end = (x + max_w + 6, max(0, y - 2))
            
            cv2.rectangle(ausgabe, bg_rect_start, bg_rect_end, (0, 0, 0), -1)
            
            text_x = x + 3
            text_y_base = y - 40
            cv2.putText(ausgabe, label_name, (text_x, text_y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(ausgabe, label_temp, (text_x, text_y_base + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            
            conf_color = (0, 255, 0) if entz.score.total_confidence > 80 else (0, 255, 255)
            cv2.putText(ausgabe, label_conf, (text_x, text_y_base + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, conf_color, 1, cv2.LINE_AA)
                        
        erfolg, buffer = cv2.imencode('.png', ausgabe)
        if erfolg:
            with open(output_pfad, 'wb') as f:
                f.write(buffer)
            self.logger.info(f"Finales Ergebnisbild gespeichert unter: {output_pfad}")
        else:
            self.logger.error("Fehler beim Encodieren des Ergebnisbildes.")
            raise IOError("Fehler: Das Ergebnisbild konnte nicht kodiert werden.")