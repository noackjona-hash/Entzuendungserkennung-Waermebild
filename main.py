import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
from datetime import datetime
from fpdf import FPDF

from modules.loader import load_and_preprocess
from modules.geometry import find_both_feet, extract_toes_with_ai
from modules.analysis import perform_deep_analysis

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class IgniteApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("IGNITE - Clinical Thermography Pro")
        self.geometry("1500x950")
        self.minsize(1200, 800)

        self.current_image_path = None
        self.cv_original = None
        self.cv_gray = None
        self.foot_mask = None
        self.final_cv_image = None
        self.left_toes_cache = []
        self.right_toes_cache = []
        self.analysis_results = []
        self.manual_mode = False
        self.manual_clicks = []
        
        self.setup_ui()

    def setup_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # --- SEITENLEISTE ---
        self.sidebar_frame = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        ctk.CTkLabel(self.sidebar_frame, text="IGNITE", font=ctk.CTkFont(size=36, weight="bold"), text_color="#00ccff").grid(row=0, column=0, padx=20, pady=(30, 0))
        ctk.CTkLabel(self.sidebar_frame, text="Pro Diagnostics Edition", font=ctk.CTkFont(size=14)).grid(row=1, column=0, padx=20, pady=(0, 30))

        self.btn_load = ctk.CTkButton(self.sidebar_frame, text="📁 Wärmebild laden", command=self.load_image, height=45, font=ctk.CTkFont(size=14))
        self.btn_load.grid(row=2, column=0, padx=20, pady=10)

        self.btn_analyze = ctk.CTkButton(self.sidebar_frame, text="🧠 Deep Scan (KI)", command=self.run_analysis, height=45, font=ctk.CTkFont(size=14, weight="bold"), state="disabled")
        self.btn_analyze.grid(row=3, column=0, padx=20, pady=10)

        self.btn_manual = ctk.CTkButton(self.sidebar_frame, text="🖱️ Manuelle Messung", command=self.start_manual_mode, height=45, font=ctk.CTkFont(size=14), state="disabled", fg_color="#d4a017", hover_color="#b38600")
        self.btn_manual.grid(row=4, column=0, padx=20, pady=10)

        self.btn_pdf = ctk.CTkButton(self.sidebar_frame, text="📄 Klinischen Report (PDF)", command=self.generate_pdf_report, height=45, font=ctk.CTkFont(size=14, weight="bold"), state="disabled", fg_color="#b32400", hover_color="#801a00")
        self.btn_pdf.grid(row=5, column=0, padx=20, pady=40)

        # --- HAUPTBEREICH (Tabs) ---
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.tabview.add("Visualisierung")
        self.tabview.add("Detail-Analyse")
        self.tabview.add("Einstellungen")

        self.setup_visual_tab()
        self.setup_details_tab()
        self.setup_settings_tab()

    def setup_visual_tab(self):
        tab = self.tabview.tab("Visualisierung")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        import tkinter as tk
        self.canvas_frame = ctk.CTkFrame(tab)
        self.canvas_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.canvas = tk.Canvas(self.canvas_frame, bg="#0d0d0d", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(expand=True, fill=tk.BOTH, padx=2, pady=2)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.status_label = ctk.CTkLabel(tab, text="System initialisiert. Lade ein Bild.", text_color="#00ffcc", font=ctk.CTkFont(size=14))
        self.status_label.grid(row=1, column=0, sticky="w", padx=15, pady=5)

    def setup_details_tab(self):
        tab = self.tabview.tab("Detail-Analyse")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)
        
        self.scroll_frame = ctk.CTkScrollableFrame(tab, fg_color="#1a1a1a")
        self.scroll_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(self.scroll_frame, text="KLINISCHE ROHDATEN & METRIKEN", font=ctk.CTkFont(size=20, weight="bold"), text_color="#00ccff").pack(pady=20)
        
        self.result_text = ctk.CTkTextbox(self.scroll_frame, font=ctk.CTkFont(family="Consolas", size=14), height=600, wrap="none", fg_color="#121212")
        self.result_text.pack(expand=True, fill="both", padx=20, pady=10)
        self.result_text.configure(state="disabled")

    def setup_settings_tab(self):
        tab = self.tabview.tab("Einstellungen")
        
        ctk.CTkLabel(tab, text="Klinische Schwellenwerte (TDI %)", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(20, 10), anchor="w", padx=30)
        
        self.lbl_warn = ctk.CTkLabel(tab, text="Verdacht ab (TDI %): 8.0")
        self.lbl_warn.pack(anchor="w", padx=30)
        self.slider_warn = ctk.CTkSlider(tab, from_=2.0, to=20.0, number_of_steps=36, command=self.on_slider_change)
        self.slider_warn.set(8.0)
        self.slider_warn.pack(fill="x", padx=30, pady=(0, 20))

        self.lbl_severe = ctk.CTkLabel(tab, text="Schwer ab (TDI %): 15.0")
        self.lbl_severe.pack(anchor="w", padx=30)
        self.slider_severe = ctk.CTkSlider(tab, from_=10.0, to=40.0, number_of_steps=60, command=self.on_slider_change)
        self.slider_severe.set(15.0)
        self.slider_severe.pack(fill="x", padx=30, pady=(0, 20))

    def on_slider_change(self, value):
        self.lbl_warn.configure(text=f"Verdacht ab (TDI %): {self.slider_warn.get():.1f}")
        self.lbl_severe.configure(text=f"Schwer ab (TDI %): {self.slider_severe.get():.1f}")
        if self.final_cv_image is not None and not self.manual_mode:
            self.update_live_analysis()

    # --- LOGIK ---
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Thermal Images", "*.jpg *.jpeg *.png")])
        if file_path:
            self.current_image_path = file_path
            self.cv_original, self.cv_gray = load_and_preprocess(file_path)
            self.left_toes_cache, self.right_toes_cache = [], []
            
            if self.cv_original is not None:
                self.display_image(self.cv_original)
                self.btn_analyze.configure(state="normal")
                self.btn_manual.configure(state="normal")
                self.btn_pdf.configure(state="disabled")
                self.status_label.configure(text="Bereit für den Deep Scan.")

    def run_analysis(self):
        if self.cv_original is None: return
        self.status_label.configure(text="Extrahierte Merkmale... KI-Modell analysiert...")
        self.update()
        
        left_contour, right_contour, foot_mask = find_both_feet(self.cv_gray)
        self.foot_mask = foot_mask
        
        if left_contour is not None and right_contour is not None:
            self.left_toes_cache = extract_toes_with_ai(left_contour, self.cv_gray, foot_mask)
            self.right_toes_cache = extract_toes_with_ai(right_contour, self.cv_gray, foot_mask)
            
            if self.left_toes_cache and len(self.left_toes_cache) == 5:
                self.update_live_analysis()
                self.btn_pdf.configure(state="normal")
                self.status_label.configure(text="Erfolg: Umfassende Diagnose abgeschlossen.")
            else:
                messagebox.showwarning("Hinweis", "KI-Zuordnung unsicher. Nutze die manuelle Messung.")
        else:
            messagebox.showerror("Fehler", "Füße nicht segmentierbar.")

    def update_live_analysis(self):
        if not self.left_toes_cache or not self.right_toes_cache: return
        warn_th, severe_th = self.slider_warn.get(), self.slider_severe.get()
        
        img_copy = self.cv_original.copy()
        if self.foot_mask is None: self.foot_mask = np.ones_like(self.cv_gray)*255
        
        final_image, results = perform_deep_analysis(img_copy, self.left_toes_cache, self.right_toes_cache, self.cv_gray, self.foot_mask, warn_th, severe_th)
        
        self.final_cv_image = final_image
        self.analysis_results = results
        self.display_image(final_image)
        self.populate_details_tab(results)

    def start_manual_mode(self):
        if self.cv_original is None: return
        self.manual_mode = True
        self.manual_clicks = []
        self.left_toes_cache, self.right_toes_cache = [], []
        self.display_image(self.cv_original)
        self.status_label.configure(text="Manuell: Markiere alle 10 Zehen (von links nach rechts).")

    def on_canvas_click(self, event):
        if not self.manual_mode or self.cv_original is None: return
        self.manual_clicks.append((event.x, event.y))
        cv2.drawMarker(self.cv_original, (event.x, event.y), (0, 255, 255), cv2.MARKER_CROSS, 10, 2)
        self.display_image(self.cv_original)
        if len(self.manual_clicks) == 10:
            self.manual_mode = False
            self.process_manual_clicks()

    def process_manual_clicks(self):
        _, _, self.foot_mask = find_both_feet(self.cv_gray)
        if self.foot_mask is None: self.foot_mask = np.ones_like(self.cv_gray) * 255 
        r = 20
        for i, (x, y) in enumerate(self.manual_clicks):
            x_s, x_e = max(0, x-r), min(self.cv_gray.shape[1], x+r)
            y_s, y_e = max(0, y-r), min(self.cv_gray.shape[0], y+r)
            roi_gray = self.cv_gray[y_s:y_e, x_s:x_e]
            roi_mask = self.foot_mask[y_s:y_e, x_s:x_e]
            
            if roi_gray.size > 0:
                _, max_val, _, max_loc = cv2.minMaxLoc(roi_gray, mask=roi_mask)
                temp, meas_pt = int(max_val), (x_s + max_loc[0], y_s + max_loc[1])
            else:
                temp, meas_pt = int(self.cv_gray[y, x]), (x, y)

            data = {"tip": (x, y), "temp": temp, "sensor": meas_pt}
            if i < 5: self.left_toes_cache.append(data)
            else: self.right_toes_cache.append(data)

        self.update_live_analysis()
        self.btn_pdf.configure(state="normal")
        self.status_label.configure(text="Manuelle Messung abgeschlossen.")

    def populate_details_tab(self, results):
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        
        fai = results[0]["fai"] if results else 0
        
        report = f"=================================================================================\n"
        report += f" IGNITE CLINICAL THERMOGRAPHY REPORT\n"
        report += f" Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"=================================================================================\n\n"
        
        report += f"GLOBALE METRIKEN:\n"
        report += f"-----------------\n"
        report += f"Fuss-Asymmetrie-Index (FAI): {fai}%\n"
        report += f"(Ein FAI > 10% deutet auf systematische Durchblutungsstoerungen hin)\n\n"
        
        report += f"LOKALE ZEHEN-ANALYSE:\n"
        report += f"{'-'*85}\n"
        report += f"{'Zeh':<10} | {'Status':<10} | {'TDI':<7} | {'Temp L':<7} | {'Temp R':<7} | {'Area L/R (px)':<15} | {'Grad L/R'}\n"
        report += f"{'-'*85}\n"
        
        toe_names = ["Kl. Zeh", "Zeh 4", "Mittelzeh", "Zeh 2", "Gr. Zeh"]

        for res in results:
            name = toe_names[res["toe_index"]]
            area_str = f"{res['area_l']}/{res['area_r']}"
            grad_str = f"{res['grad_l']}/{res['grad_r']}"
            
            report += f"{name:<10} | {res['status']:<10} | {res['tdi']:>6.2f}% | {res['t_l']:<7} | {res['t_r']:<7} | {area_str:<15} | {grad_str}\n"

        report += f"\n\nERKLÄRUNG DER METRIKEN:\n"
        report += f"TDI (Thermal Divergence Index): Prozentuale Abweichung der Temperatur zwischen Links und Rechts.\n"
        report += f"Area (Hotspot Fläche): Anzahl der stark erhitzten Pixel im Messbereich.\n"
        report += f"Gradient (Abfall): Temperaturdifferenz vom Zentrum zum Rand (hoher Wert = akute, scharfe Entzündung).\n"

        self.result_text.insert("end", report)
        self.result_text.configure(state="disabled")

    def generate_pdf_report(self):
        if not self.analysis_results or self.final_cv_image is None: return
        
        save_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")], initialfile=f"Ignite_Report_{datetime.now().strftime('%Y%m%d')}.pdf")
        if not save_path: return

        # Temporaeres Bild speichern fuer PDF
        temp_img_path = "temp_report_img.jpg"
        cv2.imwrite(temp_img_path, self.final_cv_image)

        try:
            pdf = FPDF()
            pdf.add_page()
            
            # Header
            pdf.set_font("Arial", 'B', 20)
            pdf.cell(200, 10, txt="IGNITE DIAGNOSTICS - THERMAL REPORT", ln=True, align='C')
            pdf.set_font("Arial", '', 10)
            pdf.cell(200, 10, txt=f"Erstellt am: {datetime.now().strftime('%d.%m.%Y %H:%M')}", ln=True, align='C')
            pdf.ln(10)

            # Bild einfuegen
            pdf.image(temp_img_path, x=15, w=180)
            pdf.ln(120) # Platz lassen

            # Tabelle / Daten
            pdf.set_font("Courier", '', 10)
            
            # Hole den exakten Text aus dem Text-Feld
            report_text = self.result_text.get("1.0", "end-1c")
            
            for line in report_text.split('\n'):
                # Umlaute fuer FPDF bereinigen
                safe_line = line.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('Ä', 'Ae').replace('Ö', 'Oe').replace('Ü', 'Ue').replace('ß', 'ss')
                pdf.cell(200, 5, txt=safe_line, ln=True)

            pdf.output(save_path)
            messagebox.showinfo("Erfolg", f"Hochprofessioneller Medizin-Report wurde gespeichert unter:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Fehler", f"Konnte PDF nicht erstellen: {str(e)}")
        finally:
            if os.path.exists(temp_img_path): os.remove(temp_img_path)

    def display_image(self, cv_img):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.tk_img = ImageTk.PhotoImage(image=img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

if __name__ == "__main__":
    app = IgniteApp()
    app.mainloop()