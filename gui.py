import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import os
import datetime
from PIL import Image, ImageTk

# ReportLab Platypus Engine für High-End PDFs
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm

from berechnung import ThermalAnalyzer, TrendManager
from fussfinder import FootFinder

class ClinicalDesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ThermoAI Vision - Clinical Workstation (v15)")
        self.root.geometry("1200x850")
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#0f172a')
        style.configure('TButton', font=('Inter', 10, 'bold'), background='#06b6d4', foreground='white', borderwidth=0, padding=6)
        style.map('TButton', background=[('active', '#0891b2')])
        style.configure('TLabel', background='#0f172a', foreground='#f8fafc', font=('Inter', 10))
        self.root.configure(bg='#0f172a')

        self.img_cv = None
        self.ergebnisse = []
        self.patient_id = "PAT-2026-001"
        self.setup_ui()

    def setup_ui(self):
        # Top Bar
        top = ttk.Frame(self.root, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Label(top, text="ThermoAI Vision", font=('Inter', 18, 'bold'), foreground='#38bdf8').pack(side=tk.LEFT, padx=10)
        ttk.Label(top, text="Lead: Jona Noack | JF 2026", font=('Inter', 10), foreground='#94a3b8').pack(side=tk.LEFT, padx=10)
        
        self.btn_export = ttk.Button(top, text="📄 Medizinischen Report (PDF) erstellen", command=self.export_pdf, state=tk.DISABLED)
        self.btn_export.pack(side=tk.RIGHT, padx=10)
        ttk.Button(top, text="🔬 Auto-Scan (Füße)", command=self.run_analysis).pack(side=tk.RIGHT, padx=5)
        ttk.Button(top, text="📸 Wärmebild laden", command=self.load_image).pack(side=tk.RIGHT, padx=5)

        # Canvas Area
        self.canvas_frame = tk.Frame(self.root, bg="#020617", bd=2, relief=tk.SUNKEN)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="#020617", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if not path: return
        self.img_path = path
        self.img_cv = cv2.imread(path)
        self.ergebnisse = []
        self.btn_export.config(state=tk.DISABLED)
        self.render_canvas()

    def run_analysis(self):
        if self.img_cv is None:
            return messagebox.showwarning("Fehler", "Bitte zuerst ein Bild laden.")
            
        pts = FootFinder.find_toes(self.img_cv)
        if not pts:
            return messagebox.showerror("Fehler", "Keine anatomischen Anker (Zehen) gefunden.")
            
        analyzer = ThermalAnalyzer(self.img_cv, messpunkte=pts, suchradius=80)
        self.ergebnisse = analyzer.analysiere()
        TrendManager.save_scan(self.patient_id, self.ergebnisse)
        
        self.render_canvas(overlay_results=True)
        self.btn_export.config(state=tk.NORMAL)
        messagebox.showinfo("Scan abgeschlossen", f"Deep-Scan beendet.\nEs wurden {len(self.ergebnisse)} thermische Anomalien detektiert.")

    def render_canvas(self, overlay_results=False):
        if self.img_cv is None: return
        
        display_img = self.img_cv.copy()
        
        if overlay_results:
            for e in self.ergebnisse:
                x, y, w, h = e.bounding_box
                color = (0, 0, 255) if e.score_total > 85 else (0, 165, 255)
                cv2.rectangle(display_img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(display_img, f"{e.stats_celsius.max_val:.1f}C", (x, max(15, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w = display_img.shape[:2]
        
        # Responsive Scaling
        c_w = self.canvas.winfo_width() or 1000
        c_h = self.canvas.winfo_height() or 700
        scale = min(c_w/w, c_h/h)
        
        if scale < 1:
            display_img = cv2.resize(display_img, (int(w*scale), int(h*scale)))
            
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(display_img))
        self.canvas.delete("all")
        self.canvas.create_image(c_w//2, c_h//2, anchor=tk.CENTER, image=self.tk_img)

    def export_pdf(self):
        """Erstellt ein extrem professionelles PDF mit ReportLab Platypus."""
        if not self.ergebnisse: return
        save_path = filedialog.asksaveasfilename(defaultextension=".pdf", initialfile=f"Befund_{self.patient_id}_{datetime.datetime.now().strftime('%Y%m%d')}.pdf")
        if not save_path: return

        # Dokument aufbauen
        doc = SimpleDocTemplate(save_path, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
        elements = []
        styles = getSampleStyleSheet()
        
        # Eigene Styles
        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=22, textColor=colors.HexColor("#0f172a"), spaceAfter=10)
        subtitle_style = ParagraphStyle('Sub', parent=styles['Normal'], fontName='Helvetica', fontSize=10, textColor=colors.gray, spaceAfter=20)
        heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=14, spaceBefore=20, spaceAfter=10)

        # Header
        elements.append(Paragraph("Radiologischer Thermografie-Befund", title_style))
        elements.append(Paragraph(f"ThermoAI Vision v15 | Lead Researcher: Jona Noack | Jugend Forscht 2026", subtitle_style))
        elements.append(Paragraph(f"<b>Patienten-ID:</b> {self.patient_id} &nbsp;&nbsp;&nbsp; <b>Datum:</b> {datetime.datetime.now().strftime('%d.%m.%Y %H:%M')}", styles['Normal']))
        elements.append(Spacer(1, 0.5*cm))

        # Bild einbetten
        temp_img_path = "temp_export.jpg"
        cv2.imwrite(temp_img_path, self.img_cv)
        img = RLImage(temp_img_path, width=12*cm, height=9*cm)
        elements.append(img)
        elements.append(Spacer(1, 0.5*cm))

        # Befunde Tabelle
        elements.append(Paragraph("1. Detektierte Anomalien & Messwerte", heading_style))
        
        table_data = [["Gelenk / Region", "T-Max (°C)", "T-Mean (°C)", "Konfidenz", "Symmetrie-Warnung"]]
        for e in self.ergebnisse:
            sym_text = f"JA (Δ {e.delta_t_gegenseite:.1f}°C)" if e.symmetrie_alarm else "Nein"
            row = [e.gelenk_name, f"{e.stats_celsius.max_val:.1f}", f"{e.stats_celsius.mean_val:.1f}", f"{e.score_total:.1f}%", sym_text]
            table_data.append(row)
            
        t = Table(table_data, colWidths=[4.5*cm, 2.5*cm, 2.5*cm, 3*cm, 3.5*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#06b6d4")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f8fafc")),
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1"))
        ]))
        elements.append(t)

        # Therapie Verlauf (History)
        elements.append(Paragraph("2. Historischer Therapieverlauf (Trend)", heading_style))
        history = TrendManager.load_history().get(self.patient_id, [])
        if history:
            hist_data = [["Datum", "Anzahl Befunde", "Höchste Temperatur"]]
            for h in history[-5:]: # Letzte 5 Scans
                hist_data.append([h['timestamp'], str(h['anomalien_count']), f"{h['max_temp']} °C"])
            
            ht = Table(hist_data, colWidths=[5*cm, 4*cm, 4*cm])
            ht.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#334155")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
            ]))
            elements.append(ht)
        else:
            elements.append(Paragraph("Keine historischen Daten verfügbar.", styles['Normal']))

        # PDF Generieren
        doc.build(elements)
        if os.path.exists(temp_img_path): os.remove(temp_img_path)
        messagebox.showinfo("Erfolg", "Medizinischer PDF-Report wurde generiert und gespeichert!")

if __name__ == "__main__":
    root = tk.Tk()
    app = ClinicalDesktopApp(root)
    root.mainloop()