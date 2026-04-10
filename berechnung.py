import cv2
import numpy as np
import math
import logging
import time
import json
import os
import datetime
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional

# =============================================================================
# DEFINITIONEN & ENUMS
# =============================================================================

class SeverityLevel(Enum):
    """Klassifizierung der Entzündungsstärke nach medizinischen Parametern."""
    NORMAL = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3
    CRITICAL = 4

@dataclass
class ThermalConfig:
    """Zentrale Konfiguration für die Bildverarbeitung und Heuristik."""
    temp_min_celsius: float = 20.0
    temp_max_celsius: float = 42.0
    
    # Segmentierung
    basis_schwellenwert_pixel: int = 180  
    suchradius_pixel: int = 80          
    min_kontur_flaeche: int = 15          
    
    # Scoring-Gewichtung (Heuristik)
    score_gewicht_absolut: float = 0.35    # Absolute Hitze
    score_gewicht_kontrast: float = 0.25   # Delta zur Umgebung
    score_gewicht_asymmetrie: float = 0.30 # Seitenvergleich (NEU: Stärker gewichtet)
    score_gewicht_form: float = 0.10       # Zirkularität
    
    min_confidence_score: float = 70.0    
    nms_overlap_distanz: int = 40         

@dataclass
class TemperatureStats:
    min_val: float = 0.0
    max_val: float = 0.0
    mean_val: float = 0.0
    
    def to_dict(self) -> dict:
        return {"min": round(self.min_val, 2), "max": round(self.max_val, 2), "mean": round(self.mean_val, 2)}

@dataclass
class Entzuendung:
    """Repräsentiert einen pathologischen Befund."""
    gelenk_name: str
    zentrum: Tuple[int, int]
    konturen_ebenen: Dict[str, np.ndarray]
    bounding_box: Tuple[int, int, int, int]
    stats_celsius: TemperatureStats
    score_total: float = 0.0
    severity: SeverityLevel = SeverityLevel.NORMAL
    symmetrie_alarm: bool = False
    delta_t_gegenseite: float = 0.0

# =============================================================================
# TREND-MANAGEMENT (THERAPIEMONITORING)
# =============================================================================

class TrendManager:
    """Verwaltet die Speicherung von Scans zur Verlaufsanalyse."""
    DATABASE_FILE = "patient_history.json"

    @staticmethod
    def save_scan(file_hash: str, ergebnisse: List[Entzuendung]):
        history = TrendManager.load_history()
        
        scan_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "anomalien_count": len(ergebnisse),
            "max_temp": max([e.stats_celsius.max_val for e in ergebnisse]) if ergebnisse else 0,
            "details": [
                {"name": e.gelenk_name, "temp": e.stats_celsius.max_val, "score": e.score_total}
                for e in ergebnisse
            ]
        }
        
        if file_hash not in history:
            history[file_hash] = []
        
        history[file_hash].append(scan_entry)
        
        with open(TrendManager.DATABASE_FILE, "w") as f:
            json.dump(history, f, indent=4)

    @staticmethod
    def load_history() -> dict:
        if not os.path.exists(TrendManager.DATABASE_FILE):
            return {}
        try:
            with open(TrendManager.DATABASE_FILE, "r") as f:
                return json.load(f)
        except:
            return {}

# =============================================================================
# THERMAL ANALYZER ENGINE
# =============================================================================

class ThermalAnalyzer:
    """
    Hauptklasse zur Analyse von Wärmebildern. 
    Implementiert Multilevel-Segmentierung und asymmetrische Validierung.
    """
    def __init__(self, bild_cv: np.ndarray, messpunkte: List[dict], suchradius: int = 80):
        self.original_bild = bild_cv
        self.messpunkte = messpunkte
        self.config = ThermalConfig(suchradius_pixel=suchradius)
        self.analyse_protokoll: List[str] = []
        
        self.h, self.w = bild_cv.shape[:2]
        self.gray = cv2.cvtColor(bild_cv, cv2.COLOR_BGR2GRAY)
        
        # Hochwertige Vorverarbeitung
        self.denoised = cv2.bilateralFilter(self.gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.processed = clahe.apply(self.denoised)
        
        self.global_mean = np.mean(self.processed)
        self.gefundene_befunde: List[Entzuendung] = []

    def _pixel_zu_celsius(self, val: float) -> float:
        return self.config.temp_min_celsius + (val / 255.0) * (self.config.temp_max_celsius - self.config.temp_min_celsius)

    def _analyze_point(self, name: str, x: int, y: int) -> Optional[Entzuendung]:
        """Analysiert einen spezifischen Punkt auf thermische Anomalien."""
        # ROI Maskierung
        mask = np.zeros(self.processed.shape, dtype="uint8")
        cv2.circle(mask, (x, y), self.config.suchradius_pixel, 255, -1)
        roi = cv2.bitwise_and(self.processed, self.processed, mask=mask)
        
        _, max_val, _, max_loc = cv2.minMaxLoc(roi, mask=mask)
        local_mean = cv2.mean(self.processed, mask=mask)[0]
        
        # Multilevel-Thresholding (Core, Mid, Outer)
        ebenen = {}
        main_cnt = None
        thresholds = {'core': max_val - 15, 'mid': max_val - 30, 'outer': max_val - 45}
        
        for k, th in thresholds.items():
            _, binarized = cv2.threshold(roi, th, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                best = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(best) >= self.config.min_kontur_flaeche:
                    ebenen[k] = best
                    if k == 'mid': main_cnt = best
        
        if main_cnt is None: return None
        
        # Morphologie & Stats
        m_x, m_y, m_w, m_h = cv2.boundingRect(main_cnt)
        temp_stats = TemperatureStats(
            max_val=self._pixel_zu_celsius(max_val),
            mean_val=self._pixel_zu_celsius(cv2.mean(self.processed, mask=binarized)[0])
        )
        
        # Initialer Score (Ohne Symmetrie)
        score = min(100.0, ((max_val - self.global_mean) / 50.0) * 100.0)
        
        return Entzuendung(
            gelenk_name=name, zentrum=(x, y), konturen_ebenen=ebenen,
            bounding_box=(m_x, m_y, m_w, m_h), stats_celsius=temp_stats,
            score_total=score
        )

    def _check_pair_symmetry(self):
        """
        Vergleicht linke und rechte Körperseite. 
        Ein Delta > 1.0°C zwischen Paaren gilt als hochgradig verdächtig.
        """
        paare = {}
        for b in self.gefundene_befunde:
            # Extrahiere Identifikator (z.B. "Zeh 1")
            parts = b.gelenk_name.split(" - ")
            if len(parts) > 1:
                key = parts[1]
                if key not in paare: paare[key] = []
                paare[key].append(b)
        
        for key, members in paare.items():
            if len(members) == 2:
                t1 = members[0].stats_celsius.max_val
                t2 = members[1].stats_celsius.max_val
                delta = abs(t1 - t2)
                
                if delta > 1.0:
                    self.analyse_protokoll.append(f"⚖️ Symmetrie-Delta bei {key}: {delta:.1f}°C")
                    for m in members:
                        m.symmetrie_alarm = True
                        m.delta_t_gegenseite = delta
                        # Score-Boost durch Asymmetrie-Bestätigung
                        m.score_total = min(100.0, m.score_total + (delta * 10))

    def analysiere(self) -> List[Entzuendung]:
        self.analyse_protokoll.append(f"🚀 Starte Deep-Scan (Baseline: {self._pixel_zu_celsius(self.global_mean):.1f}°C)")
        
        raw_candidates = []
        for mp in self.messpunkte:
            res = self._analyze_point(mp['name'], mp['punkt'][0], mp['punkt'][1])
            if res: raw_candidates.append(res)
            
        # NMS & Symmetrie
        self.gefundene_befunde = raw_candidates # In Realität NMS-Filtering hier
        self._check_pair_symmetry()
        
        # Severity-Mapping
        for b in self.gefundene_befunde:
            if b.score_total > 90: b.severity = SeverityLevel.CRITICAL
            elif b.score_total > 80: b.severity = SeverityLevel.SEVERE
            elif b.score_total > 70: b.severity = SeverityLevel.MODERATE
            else: b.severity = SeverityLevel.MILD
            
        self.analyse_protokoll.append(f"🏁 Analyse beendet. {len(self.gefundene_befunde)} Befunde gesichert.")
        return self.gefundene_befunde