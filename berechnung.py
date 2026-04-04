import cv2
import numpy as np
import math
import logging
import time
import uuid
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

class SeverityLevel(Enum):
    NORMAL = auto()
    MILD = auto()
    MODERATE = auto()
    SEVERE = auto()
    CRITICAL = auto()

@dataclass
class ThermalConfig:
    temp_min_celsius: float = 20.0
    temp_max_celsius: float = 42.0
    
    basis_schwellenwert_pixel: int = 210  
    suchradius_pixel: int = 80 # Standardwert, kann überschrieben werden          
    min_kontur_flaeche: int = 12          
    
    score_gewicht_absolut: float = 0.40    
    score_gewicht_kontrast: float = 0.30   
    score_gewicht_asymmetrie: float = 0.20 
    score_gewicht_form: float = 0.10       
    
    min_confidence_score: float = 75.0 # Leicht gesenkt für bessere Sensitivität bei anderen Körperteilen   
    nms_overlap_distanz: int = 45         

@dataclass
class TemperatureStats:
    min_val: float = 0.0
    max_val: float = 0.0
    mean_val: float = 0.0
    median_val: float = 0.0
    std_dev: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "min": round(self.min_val, 2), 
            "max": round(self.max_val, 2),
            "mean": round(self.mean_val, 2), 
            "median": round(self.median_val, 2),
            "std_dev": round(self.std_dev, 2)
        }

@dataclass
class MorphologyFeatures:
    flaeche: float = 0.0
    umfang: float = 0.0
    zirkularitaet: float = 0.0       
    
    def to_dict(self) -> dict:
        return {
            "flaeche_px": round(self.flaeche, 2), 
            "zirkularitaet": round(self.zirkularitaet, 3)
        }

@dataclass
class HeuristicScore:
    absolut_score: float = 0.0
    kontrast_score: float = 0.0
    asymmetrie_score: float = 0.0
    form_score: float = 0.0
    total_confidence: float = 0.0    
    is_valid: bool = False           
    severity: SeverityLevel = SeverityLevel.NORMAL

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
    
    konturen_ebenen: Dict[str, np.ndarray] = field(default_factory=dict) 
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0) 
    
    def get_severity_color(self) -> Tuple[int, int, int]:
        if self.score.severity == SeverityLevel.CRITICAL: return (0, 0, 255) # BGR: Rot
        elif self.score.severity == SeverityLevel.SEVERE: return (0, 100, 255) # Orange
        elif self.score.severity == SeverityLevel.MODERATE: return (0, 165, 255) 
        else: return (0, 255, 255) # Gelb

class MathUtils:
    @staticmethod
    def extract_hu_moments(contour: np.ndarray) -> List[float]:
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments)
        return [-1 * math.copysign(1.0, h[0]) * math.log10(abs(h[0])) if h[0] != 0 else 0.0 for h in hu_moments]

class SecurityContext:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.creation_time = time.time()
        self.is_secured = True
        
    def verify_integrity(self) -> bool:
        if time.time() - self.creation_time > 300: 
            self.is_secured = False
        return self.is_secured

class ThermalAnalyzer:
    def __init__(self, bild_bytes: bytes = None, bild_pfad: str = None, messpunkte: List[dict] = None, suchradius: int = 80):
        self.security = SecurityContext()
        self.messpunkte = messpunkte or []
        self.config = ThermalConfig()
        self.config.suchradius_pixel = suchradius # Dynamischer Radius
        self.analyse_protokoll: List[str] = []
        
        self.logger = logging.getLogger(f"ThermalAnalyzer_{self.security.session_id}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(asctime)s | RAM-SECURE | %(message)s'))
            self.logger.addHandler(ch)
            
        if bild_bytes:
            self.original_bild = self._lade_bild_aus_bytes(bild_bytes)
        elif bild_pfad:
            self.original_bild = cv2.imread(bild_pfad)
            if self.original_bild is None: raise ValueError(f"Bild nicht gefunden: {bild_pfad}")
        else:
            raise ValueError("Weder Bild-Bytes noch Bild-Pfad angegeben.")
            
        self.bild_hoehe, self.bild_breite = self.original_bild.shape[:2]
        self.graustufen_bild = cv2.cvtColor(self.original_bild, cv2.COLOR_BGR2GRAY)
        
        # Vorverarbeitung
        denoised = cv2.bilateralFilter(self.graustufen_bild, d=5, sigmaColor=50, sigmaSpace=50)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.vorverarbeitetes_bild = clahe.apply(denoised)
        
        self._berechne_globale_koerper_statistiken()
        self._bereite_symmetrie_analyse_vor()
        
        self.gefundene_entzuendungen: List[Entzuendung] = []
        self.alle_kandidaten: List[Entzuendung] = []

    def _lade_bild_aus_bytes(self, bild_bytes: bytes) -> np.ndarray:
        if not self.security.verify_integrity(): raise RuntimeError("Security Context abgelaufen.")
        array = np.frombuffer(bild_bytes, dtype=np.uint8)
        bild = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if bild is None: raise ValueError("Matrix Dimensionen korrupt.")
        return bild

    def _pixel_zu_celsius(self, pixel_wert: float) -> float:
        prozent = max(0.0, min(1.0, pixel_wert / 255.0))
        return self.config.temp_min_celsius + (prozent * (self.config.temp_max_celsius - self.config.temp_min_celsius))

    def _berechne_globale_koerper_statistiken(self):
        _, koerper_maske = cv2.threshold(self.vorverarbeitetes_bild, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        koerper_pixel = self.vorverarbeitetes_bild[koerper_maske == 255]
        
        if len(koerper_pixel) > 0:
            self.global_mean_temp_raw = float(np.mean(koerper_pixel))
        else:
            self.global_mean_temp_raw = 128.0
            
        temp_c = self._pixel_zu_celsius(self.global_mean_temp_raw)
        self.analyse_protokoll.append(f"🔍 Physiologische Baseline ermittelt (Ø {temp_c:.1f}°C).")

    def _bereite_symmetrie_analyse_vor(self):
        if not self.messpunkte: 
            self.asymmetrie_faktor = 0
            self.heissere_seite = "none"
            return
            
        alle_x = [mp['punkt'][0] for mp in self.messpunkte]
        bild_mitte_x = sum(alle_x) / len(alle_x) if alle_x else self.bild_breite / 2
        
        temp_sum_links, count_links = 0.0, 0
        temp_sum_rechts, count_rechts = 0.0, 0
        
        for mp in self.messpunkte:
            mX, mY = mp['punkt']
            maske = np.zeros(self.vorverarbeitetes_bild.shape, dtype="uint8")
            cv2.circle(maske, (mX, mY), self.config.suchradius_pixel, 255, -1)
            lokal = cv2.bitwise_and(self.vorverarbeitetes_bild, self.vorverarbeitetes_bild, mask=maske)
            _, max_val, _, _ = cv2.minMaxLoc(lokal, mask=maske)
            
            if mX < bild_mitte_x:
                temp_sum_links += max_val
                count_links += 1
            else:
                temp_sum_rechts += max_val
                count_rechts += 1
                
        mean_links = temp_sum_links / count_links if count_links > 0 else 0
        mean_rechts = temp_sum_rechts / count_rechts if count_rechts > 0 else 0
        self.asymmetrie_faktor = abs(mean_links - mean_rechts)
        self.heissere_seite = "links" if mean_links > mean_rechts else "rechts"

    def _berechne_statistiken(self, maske: np.ndarray) -> Tuple[TemperatureStats, TemperatureStats]:
        pixel_werte = self.vorverarbeitetes_bild[maske == 255]
        if len(pixel_werte) == 0: return TemperatureStats(), TemperatureStats()
            
        raw = TemperatureStats(
            min_val=float(np.min(pixel_werte)), max_val=float(np.max(pixel_werte)), 
            mean_val=float(np.mean(pixel_werte)), median_val=float(np.median(pixel_werte)), 
            std_dev=float(np.std(pixel_werte))
        )
        
        celsius_werte = np.array([self._pixel_zu_celsius(p) for p in pixel_werte])
        celsius = TemperatureStats(
            min_val=float(np.min(celsius_werte)), max_val=float(np.max(celsius_werte)), 
            mean_val=float(np.mean(celsius_werte)), median_val=float(np.median(celsius_werte)), 
            std_dev=float(np.std(celsius_werte))
        )
        return raw, celsius

    def _berechne_morphologie(self, kontur: np.ndarray) -> MorphologyFeatures:
        flaeche = cv2.contourArea(kontur)
        umfang = cv2.arcLength(kontur, True)
        zirkularitaet = (4 * math.pi * flaeche) / (umfang * umfang) if umfang > 0 else 0.0
        return MorphologyFeatures(flaeche=flaeche, umfang=umfang, zirkularitaet=zirkularitaet)

    def _bewerte_kandidat(self, seg_name: str, peak_raw: float, lokal_mean: float, morph: MorphologyFeatures, is_linke_seite: bool) -> HeuristicScore:
        baseline = max(130.0, self.global_mean_temp_raw)
        abs_score = 0.0 if peak_raw <= baseline else min(100.0, ((peak_raw - baseline) / (255.0 - baseline)) * 100.0)
        
        delta = peak_raw - lokal_mean
        kontrast_score = min(100.0, max(0.0, (delta / 40.0) * 100.0))
        
        asym_score = 100.0
        if self.asymmetrie_faktor > 15.0: 
            if (is_linke_seite and self.heissere_seite == "rechts") or (not is_linke_seite and self.heissere_seite == "links"):
                asym_score = 40.0 
                
        form_score = 0.0 if morph.zirkularitaet < 0.1 else min(100.0, morph.zirkularitaet * 100.0)
        
        total = (
            (abs_score * self.config.score_gewicht_absolut) + 
            (kontrast_score * self.config.score_gewicht_kontrast) + 
            (asym_score * self.config.score_gewicht_asymmetrie) + 
            (form_score * self.config.score_gewicht_form)
        )
        is_valid = total >= self.config.min_confidence_score
        
        severity = SeverityLevel.NORMAL
        if is_valid:
            if total >= 92.0: severity = SeverityLevel.CRITICAL
            elif total >= 85.0: severity = SeverityLevel.SEVERE
            elif total >= 78.0: severity = SeverityLevel.MODERATE
            else: severity = SeverityLevel.MILD
        
        if is_valid:
            self.analyse_protokoll.append(f"🟥 {seg_name}: Akuter Befund! Konfidenz: {total:.1f}% ({severity.name}).")
        else:
            self.analyse_protokoll.append(f"🟩 {seg_name}: Unauffällig. Konfidenz {total:.1f}%.")
            
        return HeuristicScore(absolut_score=abs_score, kontrast_score=kontrast_score, asymmetrie_score=asym_score, form_score=form_score, total_confidence=total, is_valid=is_valid, severity=severity)

    def analysiere(self) -> List[Entzuendung]:
        self.alle_kandidaten = []
        self.analyse_protokoll.append(f"🚀 Initialisiere Segmentierung mit Radius {self.config.suchradius_pixel}px...")
        
        for mp in self.messpunkte:
            mX, mY = mp['punkt']
            is_links = mX < (self.bild_breite / 2)
            
            lokale_maske = np.zeros(self.vorverarbeitetes_bild.shape, dtype="uint8")
            cv2.circle(lokale_maske, (mX, mY), self.config.suchradius_pixel, 255, -1)
            lokaler_bereich = cv2.bitwise_and(self.vorverarbeitetes_bild, self.vorverarbeitetes_bild, mask=lokale_maske)
            
            _, max_val, _, max_loc = cv2.minMaxLoc(lokaler_bereich, mask=lokale_maske)
            lokal_mean = cv2.mean(self.vorverarbeitetes_bild, mask=lokale_maske)[0]
            
            lokaler_schwelle = max(self.global_mean_temp_raw + 5, int(max_val) - 35)
            
            schwellen = {
                'core': max(lokaler_schwelle + 20, int(max_val) - 10),
                'mid': max(lokaler_schwelle + 10, int(max_val) - 20),
                'outer': lokaler_schwelle                              
            }
            
            konturen_dict = {}
            haupt_kontur = None
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            
            for ebene, schwelle in schwellen.items():
                _, thresh = cv2.threshold(lokaler_bereich, schwelle, 255, cv2.THRESH_BINARY)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
                konturen, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if konturen:
                    gueltige = [cnt for cnt in konturen if cv2.contourArea(cnt) >= self.config.min_kontur_flaeche]
                    if gueltige:
                        groesste = max(gueltige, key=cv2.contourArea)
                        konturen_dict[ebene] = groesste
                        if ebene == 'outer': haupt_kontur = groesste
                            
            if haupt_kontur is None:
                for ebene in ['outer', 'mid', 'core']:
                    if ebene in konturen_dict:
                        haupt_kontur = konturen_dict[ebene]
                        break
                
            if haupt_kontur is not None:
                M = cv2.moments(haupt_kontur)
                cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else max_loc[0]
                cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else max_loc[1]
                x, y, w, h = cv2.boundingRect(haupt_kontur)
                
                morph = self._berechne_morphologie(haupt_kontur)
                entz_maske = np.zeros(self.vorverarbeitetes_bild.shape, dtype="uint8")
                cv2.drawContours(entz_maske, [haupt_kontur], -1, 255, -1)
                raw_stats, celsius_stats = self._berechne_statistiken(entz_maske)
                
                score = self._bewerte_kandidat(mp['name'], max_val, lokal_mean, morph, is_links)
                
                if score.is_valid:
                    self.alle_kandidaten.append(Entzuendung(
                        gelenk_name=mp['name'], groesse_px=morph.flaeche, staerke=raw_stats.mean_val,
                        zentrum=(cX, cY), kontur=haupt_kontur, stats_raw=raw_stats, 
                        stats_celsius=celsius_stats, morphology=morph, score=score, 
                        konturen_ebenen=konturen_dict, bounding_box=(x, y, w, h)
                    ))
                    
        # NMS Filter
        self.gefundene_entzuendungen = []
        if self.alle_kandidaten:
            self.alle_kandidaten.sort(key=lambda x: x.score.total_confidence, reverse=True)
            for kand in self.alle_kandidaten:
                is_duplicate = False
                for etabliert in self.gefundene_entzuendungen:
                    dist = math.hypot(kand.zentrum[0] - etabliert.zentrum[0], kand.zentrum[1] - etabliert.zentrum[1])
                    if dist < self.config.nms_overlap_distanz:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    self.gefundene_entzuendungen.append(kand)
                    
        self.analyse_protokoll.append(f"🏁 Analysezyklus beendet. {len(self.gefundene_entzuendungen)} pathologische Befunde gesichert.")
        return self.gefundene_entzuendungen

    def render_image_to_file(self, pfad: str):
        """Erzeugt das Overlay-Bild für die lokale GUI und speichert es auf die Festplatte."""
        ausgabe = self.original_bild.copy()
        
        for mp in self.messpunkte:
            cv2.drawMarker(ausgabe, mp['punkt'], (255, 255, 255), cv2.MARKER_CROSS, 10, 1)

        for entz in self.gefundene_entzuendungen:
            x, y, w, h = entz.bounding_box
            color_severity = entz.get_severity_color()
            
            cv2.rectangle(ausgabe, (x, y), (x+w, y+h), color_severity, 1, cv2.LINE_AA)
            
            farben_isothermen = {'outer': (0, 255, 255), 'mid': (0, 165, 255), 'core': (0, 0, 255)}
            for ebene, kontur in entz.konturen_ebenen.items():
                farbe = farben_isothermen.get(ebene, (255, 255, 255))
                dicke = -1 if ebene == 'core' else 2
                cv2.drawContours(ausgabe, [kontur], -1, farbe, dicke, cv2.LINE_AA)
                
            cv2.drawMarker(ausgabe, entz.zentrum, (0, 0, 0), cv2.MARKER_CROSS, 15, 2)
            cv2.drawMarker(ausgabe, entz.zentrum, (255, 255, 255), cv2.MARKER_CROSS, 15, 1)

            label_name = f"{entz.gelenk_name}"
            label_temp = f"T-Max: {entz.stats_celsius.max_val:.1f}C"
            label_conf = f"Conf: {entz.score.total_confidence:.1f}%"
            
            t1, _ = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            t2, _ = cv2.getTextSize(label_temp, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            t3, _ = cv2.getTextSize(label_conf, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            max_w = max(t1[0], t2[0], t3[0])
            
            bg_rect_start = (x, max(0, y - 50))
            bg_rect_end = (x + max_w + 6, max(0, y - 2))
            
            overlay = ausgabe.copy()
            cv2.rectangle(overlay, bg_rect_start, bg_rect_end, (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, ausgabe, 0.3, 0, ausgabe)
            
            text_x, text_y_base = x + 3, y - 40
            cv2.putText(ausgabe, label_name, (text_x, text_y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(ausgabe, label_temp, (text_x, text_y_base + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(ausgabe, label_conf, (text_x, text_y_base + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_severity, 1, cv2.LINE_AA)
                        
        cv2.imwrite(pfad, ausgabe)