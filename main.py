import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

from modules.loader import load_and_preprocess
from modules.geometry import find_both_feet, extract_toes_from_contour
from modules.analysis import perform_bilateral_analysis

class IgniteApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ignite - Thermografische Entzuendungserkennung")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1e1e1e")

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
        # Header
        header = tk.Frame(self.root, bg="#2b2b2b", height=80)
        header.pack(fill=tk.X)
        tk.Label(header, text="IGNITE DIAGNOSE-SYSTEM", font=("Segoe UI", 24, "bold"), bg="#2b2b2b", fg="#00ccff").pack(pady=10)

        # Toolbar
        toolbar = tk.Frame(self.root, bg="#333333", pady=10)
        toolbar.pack(fill=tk.X)

        tk.Button(toolbar, text="📁 Bild laden", font=("Arial", 11), bg="#444444", fg="white", command=self.load_image, width=15).pack(side=tk.LEFT, padx=10)
        self.btn_analyze = tk.Button(toolbar, text="⚡ Auto-Analyse", font=("Arial", 11), bg="#007acc", fg="white", state=tk.DISABLED, command=self.run_analysis, width=15)
        self.btn_analyze.pack(side=tk.LEFT, padx=10)
        self.btn_manual = tk.Button(toolbar, text="🖱️ Manuell wählen", font=("Arial", 11), bg="#d4a017", fg="black", state=tk.DISABLED, command=self.start_manual_mode, width=15)
        self.btn_manual.pack(side=tk.LEFT, padx=10)
        self.btn_pdf = tk.Button(toolbar, text="📄 PDF Bericht", font=("Arial", 11), bg="#b32400", fg="white", state=tk.DISABLED, command=self.save_pdf, width=15)
        self.btn_pdf.pack(side=tk.RIGHT, padx=10)

        # Main Layout
        main_container = tk.Frame(self.root, bg="#1e1e1e")
        main_container.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

        self.canvas = tk.Canvas(main_container, bg="#000000", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Rechte Seitenleiste (Messwerte & Parameter)
        sidebar = tk.Frame(main_container, bg="#2b2b2b", width=300)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))

        # Dynamische Parameter Slider
        tk.Label(sidebar, text="KLINISCHE PARAMETER", font=("Arial", 12, "bold"), bg="#2b2b2b", fg="#00ccff").pack(pady=(10, 5))
        
        self.warn_slider = tk.Scale(sidebar, from_=5, to=30, orient=tk.HORIZONTAL, label="Verdacht ab (\u0394 T):", bg="#2b2b2b", fg="white", highlightthickness=0, command=self.update_live_analysis)
        self.warn_slider.set(15)
        self.warn_slider.pack(fill=tk.X, padx=10)

        self.severe_slider = tk.Scale(sidebar, from_=20, to=80, orient=tk.HORIZONTAL, label="Schwer ab (\u0394 T):", bg="#2b2b2b", fg="white", highlightthickness=0, command=self.update_live_analysis)
        self.severe_slider.set(30)
        self.severe_slider.pack(fill=tk.X, padx=10)

        tk.Frame(sidebar, bg="#555555", height=2).pack(fill=tk.X, pady=15, padx=10)

        # Tabellen-Bereich
        tk.Label(sidebar, text="MESSWERTE", font=("Arial", 12, "bold"), bg="#2b2b2b", fg="white").pack(pady=5)
        self.toe_list_frame = tk.Frame(sidebar, bg="#2b2b2b")
        self.toe_list_frame.pack(expand=True, fill=tk.BOTH, padx=10)

        self.status_var = tk.StringVar(value="System bereit.")
        tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#2b2b2b", fg="#aaaaaa").pack(side=tk.BOTTOM, fill=tk.X)

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
                self.status_var.set("Bild geladen.")

    def run_analysis(self):
        if self.cv_original is None: return
        
        left_contour, right_contour, foot_mask = find_both_feet(self.cv_gray)
        if left_contour is not None and right_contour is not None:
            self.left_toes_cache = extract_toes_from_contour(left_contour, self.cv_gray, foot_mask)
            self.right_toes_cache = extract_toes_from_contour(right_contour, self.cv_gray, foot_mask)
            
            if len(self.left_toes_cache) == 5 and len(self.right_toes_cache) == 5:
                self.update_live_analysis()
                self.btn_pdf.config(state=tk.NORMAL)
                self.status_var.set("Auto-Analyse erfolgreich.")
            else:
                messagebox.showwarning("Hinweis", "Zehen nicht perfekt erkannt. Bitte manuelle Auswahl nutzen.")
        else:
            messagebox.showerror("Fehler", "Füße konnten nicht getrennt werden.")

    def update_live_analysis(self, *args):
        # Wird ausgefuehrt, wenn die Slider bewegt werden (Live-Update)
        if not self.left_toes_cache or not self.right_toes_cache: return
        
        warn_th = self.warn_slider.get()
        severe_th = self.severe_slider.get()
        
        img_copy = self.cv_original.copy()
        final_image, delta_ts = perform_bilateral_analysis(img_copy, self.left_toes_cache, self.right_toes_cache, warn_th, severe_th)
        
        self.final_cv_image = final_image
        self.display_image(final_image)
        self.update_toe_list(delta_ts, warn_th, severe_th)

    def start_manual_mode(self):
        if self.cv_original is None: return
        self.manual_mode = True
        self.manual_clicks = []
        self.left_toes_cache, self.right_toes_cache = [], []
        self.display_image(self.cv_original)
        self.clear_toe_list()
        self.status_var.set("Manueller Modus: Klicke nacheinander auf alle 10 Zehen (von links nach rechts).")

    def on_canvas_click(self, event):
        if not self.manual_mode or self.cv_original is None: return
        self.manual_clicks.append((event.x, event.y))
        self.canvas.create_oval(event.x-4, event.y-4, event.x+4, event.y+4, fill="#ffcc00", outline="white")
        
        if len(self.manual_clicks) == 10:
            self.manual_mode = False
            self.process_manual_clicks()

    def process_manual_clicks(self):
        for i, (x, y) in enumerate(self.manual_clicks):
            x_s, x_e = max(0, x-10), min(self.cv_gray.shape[1], x+10)
            y_s, y_e = max(0, y-10), min(self.cv_gray.shape[0], y+10)
            roi = self.cv_gray[y_s:y_e, x_s:x_e]
            
            if roi.size > 0:
                _, max_val, _, max_loc = cv2.minMaxLoc(roi)
                temp, meas_pt = int(max_val), (x_s + max_loc[0], y_s + max_loc[1])
            else:
                temp, meas_pt = int(self.cv_gray[y, x]), (x, y)

            data = {"tip": (x, y), "temp": temp, "sensor": meas_pt}
            if i < 5: self.left_toes_cache.append(data)
            else: self.right_toes_cache.append(data)

        self.update_live_analysis()
        self.btn_pdf.config(state=tk.NORMAL)
        self.status_var.set("Manuelle Analyse abgeschlossen.")

    def clear_toe_list(self):
        for widget in self.toe_list_frame.winfo_children(): widget.destroy()

    def update_toe_list(self, delta_ts, warn_th, severe_th):
        self.clear_toe_list()
        tk.Label(self.toe_list_frame, text="Zeh | Temp L | Temp R | \u0394 T", font=("Arial", 10, "bold"), bg="#2b2b2b", fg="#aaaaaa").pack(anchor=tk.W, pady=(0, 5))
        
        toe_names = ["Kl. Zeh", "Zeh 4  ", "Mittel ", "Zeh 2  ", "Gr. Zeh"]
        r_toes_disp = list(reversed(self.right_toes_cache))

        for i in range(5):
            t_l = self.left_toes_cache[i]["temp"]
            t_r = r_toes_disp[i]["temp"]
            dt = delta_ts[i]
            abs_dt = abs(dt)
            
            if abs_dt >= severe_th: color = "#ff4444"
            elif abs_dt >= warn_th: color = "#ffa500"
            else: color = "white"
            
            tk.Label(self.toe_list_frame, text=f"{toe_names[i]}: {t_l:3} | {t_r:3} | {dt:+3}", font=("Consolas", 12), bg="#2b2b2b", fg=color).pack(anchor=tk.W, pady=2)

    def save_pdf(self):
        if self.final_cv_image is None: return
        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if path:
            img_rgb = cv2.cvtColor(self.final_cv_image, cv2.COLOR_BGR2RGB)
            Image.fromarray(img_rgb).save(path, "PDF", resolution=100.0)
            messagebox.showinfo("Erfolg", "Bericht wurde gespeichert.")

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