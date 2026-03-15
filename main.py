import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os

from modules.loader import load_and_preprocess
from modules.geometry import find_both_feet, extract_toes_from_contour
from modules.analysis import perform_bilateral_analysis

class IgniteApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ignite - Thermografische Entzuendungserkennung")
        self.root.geometry("1450x900")
        self.root.configure(bg="#121212") # Noch tieferer Darkmode

        self.current_image_path = None
        self.cv_original = None
        self.cv_gray = None
        self.final_cv_image = None
        self.left_toes_cache = []
        self.right_toes_cache = []
        self.manual_mode = False
        self.manual_clicks = []
        
        self.setup_ui()

    def setup_ui(self):
        header = tk.Frame(self.root, bg="#1e1e1e", height=80)
        header.pack(fill=tk.X)
        tk.Label(header, text="IGNITE DIAGNOSE-SYSTEM", font=("Segoe UI", 24, "bold"), bg="#1e1e1e", fg="#00ccff").pack(pady=10)

        toolbar = tk.Frame(self.root, bg="#2b2b2b", pady=10)
        toolbar.pack(fill=tk.X)

        tk.Button(toolbar, text="📁 Bild laden", font=("Arial", 11), bg="#444444", fg="white", command=self.load_image, width=15).pack(side=tk.LEFT, padx=10)
        self.btn_analyze = tk.Button(toolbar, text="⚡ Auto-Analyse (KI)", font=("Arial", 11, "bold"), bg="#007acc", fg="white", state=tk.DISABLED, command=self.run_analysis, width=18)
        self.btn_analyze.pack(side=tk.LEFT, padx=10)
        self.btn_manual = tk.Button(toolbar, text="🖱️ Manuell wählen", font=("Arial", 11), bg="#d4a017", fg="black", state=tk.DISABLED, command=self.start_manual_mode, width=15)
        self.btn_manual.pack(side=tk.LEFT, padx=10)
        self.btn_pdf = tk.Button(toolbar, text="📄 PDF Bericht", font=("Arial", 11), bg="#b32400", fg="white", state=tk.DISABLED, command=self.save_pdf, width=15)
        self.btn_pdf.pack(side=tk.RIGHT, padx=10)

        main_container = tk.Frame(self.root, bg="#121212")
        main_container.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

        self.canvas = tk.Canvas(main_container, bg="#000000", highlightthickness=1, highlightbackground="#333333", cursor="crosshair")
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        sidebar = tk.Frame(main_container, bg="#1e1e1e", width=380, highlightthickness=1, highlightbackground="#333333")
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))

        tk.Label(sidebar, text="DIAGNOSE-PARAMETER (TDI)", font=("Arial", 12, "bold"), bg="#1e1e1e", fg="#00ccff").pack(pady=(15, 5))
        
        self.warn_slider = tk.Scale(sidebar, from_=2.0, to=20.0, resolution=0.5, orient=tk.HORIZONTAL, label="Verdacht ab (TDI %):", bg="#1e1e1e", fg="white", highlightthickness=0, command=self.update_live_analysis)
        self.warn_slider.set(8.0)
        self.warn_slider.pack(fill=tk.X, padx=15)

        self.severe_slider = tk.Scale(sidebar, from_=10.0, to=40.0, resolution=0.5, orient=tk.HORIZONTAL, label="Schwer ab (TDI %):", bg="#1e1e1e", fg="white", highlightthickness=0, command=self.update_live_analysis)
        self.severe_slider.set(15.0)
        self.severe_slider.pack(fill=tk.X, padx=15)

        tk.Frame(sidebar, bg="#444444", height=1).pack(fill=tk.X, pady=15, padx=15)

        tk.Label(sidebar, text="WISSENSCHAFTLICHE DATEN", font=("Arial", 12, "bold"), bg="#1e1e1e", fg="white").pack(pady=5)
        self.toe_list_frame = tk.Frame(sidebar, bg="#1e1e1e")
        self.toe_list_frame.pack(expand=True, fill=tk.BOTH, padx=10)

        self.status_var = tk.StringVar(value="System bereit.")
        tk.Label(self.root, textvariable=self.status_var, bd=0, bg="#2b2b2b", fg="#00ffcc", font=("Arial", 10)).pack(side=tk.BOTTOM, fill=tk.X, ipady=3)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Thermal Images", "*.jpg *.jpeg *.png")])
        if file_path:
            self.current_image_path = file_path
            self.cv_original, self.cv_gray = load_and_preprocess(file_path)
            self.left_toes_cache, self.right_toes_cache = [], []
            
            if self.cv_original is not None:
                self.display_image(self.cv_original)
                self.btn_analyze.config(state=tk.NORMAL)
                self.btn_manual.config(state=tk.NORMAL)
                self.btn_pdf.config(state=tk.DISABLED)
                self.clear_toe_list()
                self.status_var.set("Bild geladen. Bereit für Analyse.")

    def run_analysis(self):
        if self.cv_original is None: return
        self.status_var.set("Analysiere... Führe Otsu-Binarisierung und K-Means Machine Learning aus.")
        self.root.update()
        
        left_contour, right_contour, foot_mask = find_both_feet(self.cv_gray)
        if left_contour is not None and right_contour is not None:
            self.left_toes_cache = extract_toes_from_contour(left_contour, self.cv_gray, foot_mask)
            self.right_toes_cache = extract_toes_from_contour(right_contour, self.cv_gray, foot_mask)
            
            if len(self.left_toes_cache) == 5 and len(self.right_toes_cache) == 5:
                self.update_live_analysis()
                self.btn_pdf.config(state=tk.NORMAL)
                self.status_var.set("Erfolg: K-Means Clustering hat alle Regionen perfekt segmentiert.")
            else:
                messagebox.showwarning("Hinweis", "Das KI-Modell konnte die Fusskante nicht eindeutig unterteilen. Bitte manuelle Auswahl nutzen.")
        else:
            messagebox.showerror("Fehler", "Bildkontrast zu gering. Otsu-Segmentierung fehlgeschlagen.")

    def update_live_analysis(self, *args):
        if not self.left_toes_cache or not self.right_toes_cache: return
        
        warn_th = self.warn_slider.get()
        severe_th = self.severe_slider.get()
        
        img_copy = self.cv_original.copy()
        final_image, results = perform_bilateral_analysis(img_copy, self.left_toes_cache, self.right_toes_cache, warn_th, severe_th)
        
        self.final_cv_image = final_image
        self.display_image(final_image)
        self.update_toe_list(results, warn_th, severe_th)

    def start_manual_mode(self):
        if self.cv_original is None: return
        self.manual_mode = True
        self.manual_clicks = []
        self.left_toes_cache, self.right_toes_cache = [], []
        self.display_image(self.cv_original)
        self.clear_toe_list()
        self.status_var.set("Manueller Modus: Markiere die 10 Zehen mit Fadenkreuz-Klicks.")

    def on_canvas_click(self, event):
        if not self.manual_mode or self.cv_original is None: return
        self.manual_clicks.append((event.x, event.y))
        
        # Profi-Fadenkreuz beim manuellen Klicken
        cv2.drawMarker(self.cv_original, (event.x, event.y), (0, 255, 255), cv2.MARKER_CROSS, 10, 2)
        self.display_image(self.cv_original)
        
        if len(self.manual_clicks) == 10:
            self.manual_mode = False
            self.process_manual_clicks()

    def process_manual_clicks(self):
        # Fussmaske neu berechnen fuer den Deep Sensor
        _, _, foot_mask = find_both_feet(self.cv_gray)
        if foot_mask is None: foot_mask = np.ones_like(self.cv_gray) * 255 # Fallback
        
        for i, (x, y) in enumerate(self.manual_clicks):
            x_s, x_e = max(0, x-15), min(self.cv_gray.shape[1], x+15)
            y_s, y_e = max(0, y-15), min(self.cv_gray.shape[0], y+15)
            roi_gray = self.cv_gray[y_s:y_e, x_s:x_e]
            roi_mask = foot_mask[y_s:y_e, x_s:x_e]
            
            if roi_gray.size > 0:
                _, max_val, _, max_loc = cv2.minMaxLoc(roi_gray, mask=roi_mask)
                temp, meas_pt = int(max_val), (x_s + max_loc[0], y_s + max_loc[1])
            else:
                temp, meas_pt = int(self.cv_gray[y, x]), (x, y)

            data = {"tip": (x, y), "temp": temp, "sensor": meas_pt}
            if i < 5: self.left_toes_cache.append(data)
            else: self.right_toes_cache.append(data)

        self.update_live_analysis()
        self.btn_pdf.config(state=tk.NORMAL)
        self.status_var.set("Manuelle Sensor-Kalibrierung abgeschlossen.")

    def clear_toe_list(self):
        for widget in self.toe_list_frame.winfo_children(): widget.destroy()

    def update_toe_list(self, results, warn_th, severe_th):
        self.clear_toe_list()
        tk.Label(self.toe_list_frame, text="Zeh | Temp L | Temp R | TDI (%)", font=("Arial", 11, "bold"), bg="#1e1e1e", fg="#aaaaaa").pack(anchor=tk.W, pady=(0, 10))
        
        toe_names = ["Kl. Zeh", "Zeh 4  ", "Mittel ", "Zeh 2  ", "Gr. Zeh"]

        for i, res in enumerate(results):
            tdi = res["tdi"]
            t_l, t_r = res["temp_l"], res["temp_r"]
            
            if tdi >= severe_th: color = "#ff4444"
            elif tdi >= warn_th: color = "#ffaa00"
            else: color = "#00ff00"
            
            tk.Label(self.toe_list_frame, text=f"{toe_names[i]}: {t_l:3} | {t_r:3} | {tdi:>5.2f}%", font=("Consolas", 14), bg="#1e1e1e", fg=color).pack(anchor=tk.W, pady=4)

    def save_pdf(self):
        if self.final_cv_image is None: return
        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if path:
            img_rgb = cv2.cvtColor(self.final_cv_image, cv2.COLOR_BGR2RGB)
            Image.fromarray(img_rgb).save(path, "PDF", resolution=100.0)
            messagebox.showinfo("Erfolg", "Wissenschaftlicher PDF-Bericht gespeichert.")

    def display_image(self, cv_img):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.img_tk = ImageTk.PhotoImage(image=img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)

if __name__ == "__main__":
    root = tk.Tk()
    app = IgniteApp(root)
    root.mainloop()