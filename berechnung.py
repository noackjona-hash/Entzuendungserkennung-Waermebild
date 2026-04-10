import cv2
import numpy as np
import json
import os
import datetime
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# =============================================================================
# ENTERPRISE DEFINITIONEN & KLASSIFIZIERUNGEN
# =============================================================================

class SeverityLevel(Enum):
    """Klinische Klassifizierung der thermischen Anomalie."""
    NORMAL = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3
    CRITICAL = 4

@dataclass
class ThermalConfig:
    """Zentrale Konfiguration für die Computer Vision Engine."""
    temp_min_celsius: float = 20.0
    temp_max_celsius: float = 42.0
    suchradius_pixel: int = 80          
    min_kontur_flaeche: int = 15          
    score_gewicht_absolut: float = 0.35    
    score_gewicht_kontrast: float = 0.25   
    score_gewicht_asymmetrie: float = 0.30 
    score_gewicht_form: float = 0.10       
    min_confidence_score: float = 65.0    
    nms_overlap_distanz: int = 40         

@dataclass
class TemperatureStats:
    min_val: float = 0.0
    max_val: float = 0.0
    mean_val: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "min": round(self.min_val, 1), 
            "max": round(self.max_val, 1), 
            "mean": round(self.mean_val, 1)
        }

@dataclass
class Entzuendung:
    """Repräsentiert einen pathologischen Befund im Wärmebild."""
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
# KLINISCHES THERAPIE-MONITORING (TREND-MANAGER)
# =============================================================================

class TrendManager:
    """Verwaltet Langzeitdatenbank für das Therapie-Monitoring pro Patient."""
    DATABASE_FILE = "clinical_database.json"

    @staticmethod
    def save_scan(patient_id: str, ergebnisse: List[Entzuendung]) -> List[dict]:
        history = TrendManager.load_history()
        
        # Rundung für saubere Speicherung
        max_t = round(max([e.stats_celsius.max_val for e in ergebnisse]), 1) if ergebnisse else 0.0
        
        scan_entry = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "anomalien_count": len(ergebnisse),
            "max_temp": max_t,
            "severity_peak": max([e.severity.value for e in ergebnisse]) if ergebnisse else 0
        }
        
        if patient_id not in history:
            history[patient_id] = []
            
        history[patient_id].append(scan_entry)
        
        with open(TrendManager.DATABASE_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
            
        return history[patient_id]

    @staticmethod
    def load_history() -> dict:
        if not os.path.exists(TrendManager.DATABASE_FILE):
            return {}
        try:
            with open(TrendManager.DATABASE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

# =============================================================================
# CORE VISION ENGINE (THERMAL ANALYZER)
# =============================================================================

class ThermalAnalyzer:
    """
    Hauptklasse zur Analyse. Nutzt Adaptive Morphologie und Symmetrie-Prüfung.
    """
    def __init__(self, bild_cv: np.ndarray, messpunkte: List[dict], suchradius: int = 80):
        self.original_bild = bild_cv
        self.messpunkte = messpunkte
        self.config = ThermalConfig(suchradius_pixel=suchradius)
        self.analyse_protokoll: List[str] = []
        
        self.h, self.w = bild_cv.shape[:2]
        self.gray = cv2.cvtColor(bild_cv, cv2.COLOR_BGR2GRAY)
        
        # High-End Rauschunterdrückung & Kontrastverstärkung
        self.denoised = cv2.bilateralFilter(self.gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.processed = clahe.apply(self.denoised)
        
        self.global_mean = float(np.mean(self.processed))
        self.gefundene_befunde: List[Entzuendung] = []

    def _pixel_zu_celsius(self, val: float) -> float:
        return self.config.temp_min_celsius + (val / 255.0) * (self.config.temp_max_celsius - self.config.temp_min_celsius)

    def _analyze_point(self, name: str, x: int, y: int) -> Optional[Entzuendung]:
        """Prüft eine Region of Interest (ROI) auf Anomalien."""
        mask = np.zeros(self.processed.shape, dtype="uint8")
        cv2.circle(mask, (x, y), self.config.suchradius_pixel, 255, -1)
        roi = cv2.bitwise_and(self.processed, self.processed, mask=mask)
        
        _, max_val, _, _ = cv2.minMaxLoc(roi, mask=mask)
        
        ebenen = {}
        main_cnt = None
        thresholds = {'core': max_val - 10, 'mid': max_val - 25, 'outer': max_val - 40}
        
        for k, th in thresholds.items():
            if th < 50: continue # Ignoriere extrem kalte Bereiche
            _, binarized = cv2.threshold(roi, th, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                best = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(best) >= self.config.min_kontur_flaeche:
                    ebenen[k] = best
                    if k == 'mid': main_cnt = best
        
        if main_cnt is None: return None
        
        m_x, m_y, m_w, m_h = cv2.boundingRect(main_cnt)
        temp_stats = TemperatureStats(
            max_val=self._pixel_zu_celsius(max_val),
            mean_val=self._pixel_zu_celsius(cv2.mean(self.processed, mask=binarized)[0])
        )
        
        # Heuristische Konfidenzberechnung
        score = min(100.0, ((max_val - self.global_mean) / 60.0) * 100.0)
        if score < self.config.min_confidence_score: return None

        return Entzuendung(
            gelenk_name=name, zentrum=(x, y), konturen_ebenen=ebenen,
            bounding_box=(m_x, m_y, m_w, m_h), stats_celsius=temp_stats, score_total=score
        )

    def _check_pair_symmetry(self):
        """Klinischer Seitenvergleich (Asymmetrie = Entzündungsindikator)."""
        paare = {}
        for b in self.gefundene_befunde:
            parts = b.gelenk_name.split(" - ")
            if len(parts) > 1:
                key = parts[1]
                if key not in paare: paare[key] = []
                paare[key].append(b)
        
        for key, members in paare.items():
            if len(members) >= 2:
                # Vergleiche die beiden stärksten Befunde dieses Gelenktyps
                members = sorted(members, key=lambda x: x.stats_celsius.max_val, reverse=True)
                t1 = members[0].stats_celsius.max_val
                t2 = members[1].stats_celsius.max_val
                delta = abs(t1 - t2)
                
                if delta >= 0.8: # Ab 0.8°C klinisch relevant
                    self.analyse_protokoll.append(f"⚖️ SYMMETRIE-ALARM {key}: Δ {delta:.1f}°C")
                    for m in members[:2]:
                        m.symmetrie_alarm = True
                        m.delta_t_gegenseite = delta
                        m.score_total = min(100.0, m.score_total + (delta * 12))

    def analysiere(self) -> List[Entzuendung]:
        self.analyse_protokoll.append(f"🚀 Starte Deep-Scan (Baseline: {self._pixel_zu_celsius(self.global_mean):.1f}°C)")
        
        raw_candidates = []
        for mp in self.messpunkte:
            res = self._analyze_point(mp['name'], mp['punkt'][0], mp['punkt'][1])
            if res: raw_candidates.append(res)
            
        self.gefundene_befunde = raw_candidates
        self._check_pair_symmetry()
        
        # Zuweisung des Härtegrads
        for b in self.gefundene_befunde:
            if b.score_total >= 90: b.severity = SeverityLevel.CRITICAL
            elif b.score_total >= 80: b.severity = SeverityLevel.SEVERE
            elif b.score_total >= 70: b.severity = SeverityLevel.MODERATE
            else: b.severity = SeverityLevel.MILD
            
        # Nach Konfidenz sortieren
        self.gefundene_befunde.sort(key=lambda x: x.score_total, reverse=True)
        self.analyse_protokoll.append(f"🏁 Analyse beendet. {len(self.gefundene_befunde)} pathologische Muster detektiert.")
        return self.gefundene_befunde