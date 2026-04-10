import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import datetime
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

from berechnung import ThermalAnalyzer, TrendManager

class ProfessionalThermalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ThermoAI Enterprise - Clinical Workstation")
        self.root.geometry("1200x900")
        self.root.configure(bg="#121212")
        
        self.setup_ui()
        self.current_img = None
        self.points = []

    def setup_ui(self):
        # Toolbar
        tb = tk.Frame(self.root, bg="#1e1e1e", height=50)
        tb.pack(side=tk.TOP, fill=tk.X)
        
        tk.Button(tb, text="📂 Load Thermal", command=self.load_image, bg="#333", fg="white").pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(tb, text="🔬 Run Analysis", command=self.run_analysis, bg="#06b6d4", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(tb, text="📄 Export PDF", command=self.export_pdf, bg="#ef4444", fg="white").pack(side=tk.RIGHT, padx=10)

        # Main Canvas
        self.canvas = tk.Canvas(self.root, bg="#050505", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.canvas.bind("<Button-1>", self.add_point)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.img_path = path
            self.current_img = cv2.imread(path)
            self.show_image(self.current_img)

    def show_image(self, img):
        # Resize für Anzeige
        h, w = img.shape[:2]
        scale = min(800/h, 1000/w)
        resized = cv2.resize(img, (int(w*scale), int(h*scale)))
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0,0, anchor=tk.NW, image=self.tk_img)

    def add_point(self, event):
        # In echte Koordinaten umrechnen und speichern...
        pass

    def run_analysis(self):
        if self.current_img is None: return
        # Dummy-Punkte für Demo oder Auto-Modus
        analyzer = ThermalAnalyzer(self.current_img, [{"name": "Zeh Test", "punkt": (100, 100)}])
        self.ergebnisse = analyzer.analysiere()
        messagebox.showinfo("Success", f"{len(self.ergebnisse)} Anomalien gefunden.")

    def export_pdf(self):
        if not hasattr(self, 'ergebnisse'): return
        save_path = filedialog.asksaveasfilename(defaultextension=".pdf")
        if not save_path: return

        c = canvas.Canvas(save_path, pagesize=A4)
        c.setFont("Helvetica-Bold", 24)
        c.setStrokeColor(colors.cyan)
        c.drawString(50, 780, "THERMOAI VISION - MEDICAL REPORT")
        
        c.setFont("Helvetica", 10)
        c.drawString(50, 760, f"Patient ID: REF-{os.getlogin().upper()}")
        c.drawString(50, 745, f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Ergebnisse
        y = 700
        for e in self.ergebnisse:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(60, y, f"Befund: {e.gelenk_name}")
            c.setFont("Helvetica", 10)
            c.drawString(70, y-15, f"Max Temp: {e.stats_celsius.max_val:.1f} °C | Confidence: {e.score_total:.1f}%")
            if e.symmetrie_alarm:
                c.setFillColor(colors.red)
                c.drawString(70, y-30, f"!!! SYMMETRIE WARNUNG: Delta {e.delta_t_gegenseite:.1f}°C")
                c.setFillColor(colors.black)
            y -= 50
            
        c.showPage()
        c.save()
        messagebox.showinfo("Export", "PDF Report erfolgreich generiert.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ProfessionalThermalApp(root)
    root.mainloop()