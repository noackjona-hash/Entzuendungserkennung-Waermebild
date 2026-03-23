import cv2
import numpy as np
import math
import logging
import json
import csv
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

# ==============================================================================
# KONFIGURATION & SETUP
# ==============================================================================

@dataclass
class ThermalConfig:
    temp_min_celsius: float = 20.0
    temp_max_celsius: float = 42.0
    
    basis_schwellenwert_pixel: int = 210  
    suchradius_pixel: int = 70            
    min_kontur_flaeche: int = 30          
    
    score_gewicht_absolut: float = 0.4    
    score_gewicht_kontrast: float = 0.3   
    score_gewicht_asymmetrie: float = 0.2 
    score_gewicht_form: float = 0.1       
    
    # JUGEND FORSCHT UPDATE: Strikter Schwellenwert (vorher 65.0, jetzt 85.0)
    # So werden nur die wirklich sicheren, extremen Entzündungen angezeigt!
    min_confidence_score: float = 85.0    
    
    nms_overlap_distanz: int = 40         

# ==============================================================================
# DATENSTRUKTUREN (Data Classes)
# ==============================================================================

@dataclass
class TemperatureStats:
    min_val: float = 0.0
    max_val: float = 0.0
    mean_val: float = 0.0
    median_val: float = 0.0
    std_dev: float = 0.0
    variance: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "min": round(self.min_val, 2), "max": round(self.max_val, 2),
            "mean": round(self.mean_val, 2), "median": round(self.median_val, 2),
            "std_dev": round(self.std_dev, 2), "variance": round(self.variance, 2)
        }

@dataclass
class MorphologyFeatures:
    flaeche: float = 0.0
    umfang: float = 0.0
    zirkularitaet: float = 0.0       
    aspect_ratio: float = 0.0        
    solidity: float = 0.0            
    
    def to_dict(self) -> dict:
        return {
            "flaeche_px": round(self.flaeche, 2), "zirkularitaet": round(self.zirkularitaet, 3),
            "aspect_ratio": round(self.aspect_ratio, 3), "solidity": round(self.solidity, 3)
        }

@dataclass
class HeuristicScore:
    absolut_score: float = 0.0
    kontrast_score: float = 0.0
    asymmetrie_score: float = 0.0
    form_score: float = 0.0
    total_confidence: float = 0.0    
    is_valid: bool = False           

@dataclass
class ThermalProfile:
    distanzen: List[float] = field(default_factory=list)
    temperaturen_c: List[float] = field(default_factory=list)
    max_temp_c: float = 0.0
    mean_temp_c: float = 0.0

@dataclass
class Entzuendung:
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
    
    konturen_ebenen: Dict[str, np.ndarray] = field(default_factory=dict) 
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0) 
    roi_bild: Optional[np.ndarray] = None 

# ==============================================================================
# HAUPTKLASSE DER ANALYSE
# ==============================================================================

class ThermalAnalyzer:
    
    def __init__(self, bild_pfad: str, segmente: List[dict]):
        self.bild_pfad = bild_pfad
        self.segmente = segmente
        self.ausgabe_ordner = os.path.dirname(self.bild_pfad)
        self.basis_dateiname = os.path.splitext(os.path.basename(self.bild_pfad))[0]
        self.config = ThermalConfig()
        
        # NEU: Der Erklär-Log für das Frontend (Explainable AI)
        self.analyse_protokoll: List[str] = []
        
        self._setup_logger()
        self.original_bild = self._lade_bild_sicher(self.bild_pfad)
        self.bild_hoehe, self.bild_breite = self.original_bild.shape[:2]
        
        self.graustufen_bild = cv2.cvtColor(self.original_bild, cv2.COLOR_BGR2GRAY)
        self.vorverarbeitetes_bild = self._preprocess_image(self.graustufen_bild)
        
        self._berechne_globale_koerper_statistiken()
        self._bereite_symmetrie_analyse_vor()
        
        self.gefundene_entzuendungen: List[Entzuendung] = []
        self.alle_kandidaten: List[Entzuendung] = []

    def _setup_logger(self):
        self.logger = logging.getLogger("ThermalAnalyzer")
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            log_format = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
            log_datei = os.path.join(self.ausgabe_ordner, f"{self.basis_dateiname}_analyse.log")
            fh = logging.FileHandler(log_datei, encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(log_format)
            self.logger.addHandler(fh)

    def _lade_bild_sicher(self, pfad: str) -> np.ndarray:
        try:
            with open(pfad, 'rb') as f:
                bytes_data = f.read()
            array = np.frombuffer(bytes_data, dtype=np.uint8)
            bild = cv2.imdecode(array, cv2.IMREAD_COLOR)
            if bild is None: raise ValueError("Decode-Fehler.")
            return bild
        except Exception as e:
            raise FileNotFoundError(f"Bildladefehler: {pfad}")

    def _preprocess_image(self, gray_image: np.ndarray) -> np.ndarray:
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
        else:
            self.global_mean_temp_raw = 128.0
            self.global_std_temp_raw = 30.0
            self.global_hot_threshold = 200.0
            
        self.analyse_protokoll.append(f"🔍 Globale Körper-Temperatur ermittelt (Ø {self._pixel_zu_celsius(self.global_mean_temp_raw):.1f}°C).")

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
            
            if mX < bild_mitte_x:
                self.segmente_links.append(seg); temp_sum_links += max_val; count_links += 1
            else:
                self.segmente_rechts.append(seg); temp_sum_rechts += max_val; count_rechts += 1
                
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
        
        # Protokoll schreiben für das Frontend
        if is_valid:
            self.analyse_protokoll.append(f"🟥 {seg_name}: Auffällig! Score {total:.1f}% (Erforderlich: {self.config.min_confidence_score}%).")
        else:
            self.analyse_protokoll.append(f"🟩 {seg_name}: Sicher. Score {total:.1f}% ist zu niedrig für eine Diagnose.")
            
        return HeuristicScore(absolut_score=abs_score, kontrast_score=kontrast_score, asymmetrie_score=asym_score, form_score=form_score, total_confidence=total, is_valid=is_valid)

    def analysiere(self, temperatur_schwellenwert: Optional[int] = None, max_distanz: Optional[int] = None, **kwargs) -> List[Entzuendung]:
        if temperatur_schwellenwert is not None: self.config.basis_schwellenwert_pixel = temperatur_schwellenwert
        if max_distanz is not None: self.config.suchradius_pixel = max_distanz
            
        self.alle_kandidaten = []
        self.analyse_protokoll.append(f"🚀 Beginne Detail-Analyse von {len(self.segmente)} Segmenten...")
        
        for seg in self.segmente:
            s, e = seg['start'], seg['end']
            mX, mY = (s[0] + e[0]) // 2, (s[1] + e[1]) // 2
            is_links = mX < (self.bild_breite / 2)
            
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
                    gueltige = [cnt for cnt in konturen if cv2.contourArea(cnt) >= self.config.min_kontur_flaeche]
                    if gueltige:
                        groesste = max(gueltige, key=cv2.contourArea)
                        konturen_dict[ebene] = groesste
                        if ebene == 'outer': haupt_kontur = groesste
                            
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
                
                # Scoring berechnen (und Protokoll schreiben)
                score = self._bewerte_kandidat(seg['name'], max_val, lokal_mean, morph, is_links)
                
                if score.is_valid:
                    self.alle_kandidaten.append(Entzuendung(
                        gelenk_name=seg['name'], groesse_px=morph.flaeche, staerke=raw_stats.mean_val,
                        zentrum=(cX, cY), kontur=haupt_kontur, stats_raw=raw_stats, stats_celsius=celsius_stats,
                        morphology=morph, score=score, profil=ThermalProfile(), konturen_ebenen=konturen_dict,
                        bounding_box=(x, y, w, h)
                    ))
            else:
                self.analyse_protokoll.append(f"🟩 {seg['name']}: Keine relevante Wärme-Ausdehnung gefunden.")
                    
        # NMS
        if self.alle_kandidaten:
            self.alle_kandidaten.sort(key=lambda x: x.score.total_confidence, reverse=True)
            gefiltert = []
            for kand in self.alle_kandidaten:
                is_duplicate = False
                for etabliert in gefiltert:
                    dist = math.hypot(kand.zentrum[0] - etabliert.zentrum[0], kand.zentrum[1] - etabliert.zentrum[1])
                    if dist < self.config.nms_overlap_distanz:
                        is_duplicate = True
                        self.analyse_protokoll.append(f"ℹ️ {kand.gelenk_name} ignoriert: Überlappt mit dem stärkeren Befund bei {etabliert.gelenk_name}.")
                        break
                if not is_duplicate:
                    gefiltert.append(kand)
            self.gefundene_entzuendungen = gefiltert
        else:
            self.gefundene_entzuendungen = []
            
        self.analyse_protokoll.append(f"🏁 Analyse beendet. {len(self.gefundene_entzuendungen)} finale Entzündung(en) verifiziert.")
        return self.gefundene_entzuendungen

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
            
            text_x, text_y_base = x + 3, y - 40
            cv2.putText(ausgabe, label_name, (text_x, text_y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(ausgabe, label_temp, (text_x, text_y_base + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(ausgabe, label_conf, (text_x, text_y_base + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                        
        erfolg, buffer = cv2.imencode('.png', ausgabe)
        if erfolg:
            with open(output_pfad, 'wb') as f:
                f.write(buffer)
        else:
            raise IOError("Fehler beim Speichern des Bildes.")