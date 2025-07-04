import json
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import threading
import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.outliers_influence import OLSInfluence
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import scipy.stats as stats
import pandas as pd
import mysql.connector
from mysql.connector import Error
from statsmodels.stats.outliers_influence import OLSInfluence

class DBClient:
    def __init__(self, host="localhost", user="root", password="", database=None):
        self.cfg = dict(host=host, user=user, password=password, autocommit=True)
        self.database = database
        self.conn = None

    def connect(self):
        if self.conn is None or not self.conn.is_connected():
            self.conn = mysql.connector.connect(**self.cfg)

    def close(self):
        if self.conn and self.conn.is_connected():
            self.conn.close()

    def create_database(self, dbname):
        self.connect()
        cursor = self.conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{dbname}`")
        self.conn.database = dbname  # Cambia la base de datos activa
        self.database = dbname
        cursor.close()

    def create_measurements_table(self):
        self.connect()
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS mediciones (
                id INT AUTO_INCREMENT PRIMARY KEY,
                tiempo FLOAT,
                radio FLOAT,
                track_id INT
            )
        """)
        cursor.close()

    def insert_raw_measurement(self, tiempo, radio, track_id):
        self.connect()
        cursor = self.conn.cursor()
        query = "INSERT INTO mediciones (tiempo, radio, track_id) VALUES (%s, %s, %s)"
        cursor.execute(query, (tiempo, radio, track_id))
        cursor.close()

    def delete_all_measurements(self):
        self.connect()
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM mediciones")
        cursor.close()

    def drop_measurements_table(self):
        self.connect()
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS mediciones")
        cursor.close()

    def drop_database(self, dbname):
        self.connect()
        cursor = self.conn.cursor()
        cursor.execute(f"DROP DATABASE IF EXISTS `{dbname}`")
        cursor.close()

    def update_measurement(self, record_id, tiempo=None, radio=None, track_id=None):
        self.connect()
        cursor = self.conn.cursor()
        updates = []
        params = []

        if tiempo is not None:
            updates.append("tiempo = %s")
            params.append(tiempo)
        if radio is not None:
            updates.append("radio = %s")
            params.append(radio)
        if track_id is not None:
            updates.append("track_id = %s")
            params.append(track_id)

        if updates:
            query = f"UPDATE mediciones SET {', '.join(updates)} WHERE id = %s"
            params.append(record_id)
            cursor.execute(query, params)
        cursor.close()




class ExperimentTab:

    def __init__(self, notebook, app_ref, tab_name="Nuevo Experimento"):
        self.app_ref = app_ref
        self.notebook = notebook
        self.tab_name = tab_name
        self.frame = ttk.Frame(self.notebook)
        self.notebook.add(self.frame, text=tab_name)
        self.notebook.select(self.frame)

        # Toolbar
        toolbar = ttk.Frame(self.frame)
        toolbar.pack(fill=tk.X, pady=3)
        ttk.Button(toolbar, text="＋ Nueva Pestaña", command=lambda: self.app_ref.add_new_tab()).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Pruebas de Diagnóstico", command=self.show_diagnostic_tests).pack(side=tk.LEFT, padx=5)

        # Panel superior de controles
        ctrl_frame = ttk.Frame(self.frame)
        ctrl_frame.pack(fill=tk.X, pady=5)
        ttk.Label(ctrl_frame, text="Video: ").pack(side=tk.LEFT, padx=2)
        self.video_entry = ttk.Entry(ctrl_frame, width=30)
        self.video_entry.pack(side=tk.LEFT)
        ttk.Button(ctrl_frame, text="Cargar Video", command=self.browse_video).pack(side=tk.LEFT, padx=2)

        ttk.Label(ctrl_frame, text="Máscara: ").pack(side=tk.LEFT, padx=10)
        self.mask_entry = ttk.Entry(ctrl_frame, width=30)
        self.mask_entry.pack(side=tk.LEFT)
        ttk.Button(ctrl_frame, text="Cargar Máscara", command=self.browse_mask).pack(side=tk.LEFT, padx=2)

        ttk.Label(ctrl_frame, text="Escala (m/px): ").pack(side=tk.LEFT, padx=10)
        self.scale_entry = ttk.Entry(ctrl_frame, width=10)
        self.scale_entry.insert(0, "0,0")
        self.scale_entry.pack(side=tk.LEFT)

        ttk.Button(ctrl_frame, text="Procesar", command=self.process_video).pack(side=tk.LEFT, padx=10)
        ttk.Button(ctrl_frame, text="Definir Escala Visual", command=self.definir_escala_visual).pack(side=tk.LEFT, padx=10)
        ttk.Button(ctrl_frame, text="Actualizar Diagnóstico", command=lambda: (
            self.residuals_panel.update_residual_plots(),
            self.graph_plotter.update_plot(),
            self.show_model_summary()
        )).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="Importar CSV", command=self.importar_csv).pack(side=tk.LEFT, padx=5)

        ttk.Button(ctrl_frame, text="Guardar en MySQL", command=self.guardar_en_mysql).pack(side=tk.LEFT, padx=5)

        # Panel general
        main_container = ttk.Frame(self.frame)
        main_container.pack(fill=tk.BOTH, expand=True)
        top_split = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        top_split.pack(fill=tk.BOTH, expand=True)

        # Tabla
        table_frame = ttk.Frame(top_split)
        self.tree = ttk.Treeview(table_frame, columns=("Tiempo", "Radio"), show="headings")
        self.tree.heading("Tiempo", text="Tiempo (s)")
        self.tree.heading("Radio", text="r (m)")
        self.tree.pack(fill=tk.BOTH, expand=True)
        top_split.add(table_frame, weight=1)

        # Gráfico
        graph_frame = ttk.Frame(top_split)
        self.graph_plotter = GraphPlotter(parent=graph_frame, table=self.tree)
        self.graph_plotter.update_plot()
        top_split.add(graph_frame, weight=1)

        # Modelo
        self.model_summary_frame = ttk.LabelFrame(main_container, text="Resumen del Modelo Cuadrático")
        self.model_summary_frame.pack(fill=tk.X, padx=5, pady=5)
        self.model_tree = ttk.Treeview(
            self.model_summary_frame,
            columns=("Parámetro", "Estimación", "Error estándar", "t-valor", "p-valor"),
            show='headings', height=4
        )
        for col in ("Parámetro", "Estimación", "Error estándar", "t-valor", "p-valor"):
            self.model_tree.heading(col, text=col)
            self.model_tree.column(col, width=100)
        self.model_tree.pack(fill=tk.X)

        # Residuales
        diag_frame = ttk.LabelFrame(self.frame, text="Diagnósticos Residuales")
        diag_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.residuals_panel = ResidualsPlotter(parent=diag_frame, table=self.tree)
        self.residuals_panel.update_residual_plots()

    def analizar_datos_de_tabla(self):
        data = [self.tree.item(c)['values'] for c in self.tree.get_children()]
        if not data:
            messagebox.showwarning("Análisis", "No hay datos cargados en la tabla.")
            return

        try:
            self.graph_plotter.update_plot()
            self.residuals_panel.update_residual_plots()
            self.show_model_summary()
        except Exception as e:
            messagebox.showerror("Error en análisis", f"No se pudo analizar los datos.\n{e}")

    def importar_csv(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo CSV",
            filetypes=[("CSV files", "*.csv"), ("Todos los archivos", "*.*")]
        )
        if not file_path:
            return

        try:
            # Leer primera línea para detectar delimitador y encabezado
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                first_line = f.readline().strip()
                delimiter = ';' if ';' in first_line else ','

            # Releer archivo completo
            df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8-sig')

            # Normalizar encabezados
            df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]

            # Asegurar columnas correctas
            if not {'Tiempo', 'Radio'}.issubset(df.columns):
                # Quizás es un archivo SIN encabezado, intenta cargar manualmente
                df = pd.read_csv(file_path, delimiter=delimiter, header=None, names=["Tiempo", "Radio"])
                df.columns = ["Tiempo", "Radio"]

            # Validar de nuevo
            if not {'Tiempo', 'Radio'}.issubset(df.columns):
                raise ValueError("El archivo debe tener columnas 'Tiempo' y 'Radio'.")

            # Limpiar tabla
            self.tree.delete(*self.tree.get_children())

            # Insertar datos
            for _, row in df.iterrows():
                tiempo = float(row["Tiempo"])
                radio = float(row["Radio"])
                self.tree.insert('', 'end', values=(f"{tiempo:.6f}", f"{radio:.6f}"))

            messagebox.showinfo("Importación exitosa", "Archivo CSV cargado correctamente.")
            self.frame.after(100, self.analizar_datos_de_tabla)

        except Exception as e:
            messagebox.showerror("Error al importar", str(e))

    def guardar_en_mysql(self):
        dbname = self.tab_name.lower().replace(" ", "_")  # Por ejemplo: "Exp 1" → "exp_1"
        client = DBClient()
        client.create_database(dbname)
        client.create_measurements_table()

        for item in self.tree.get_children():
            tiempo, radio, tid = self.tree.item(item)["values"]
            client.insert_raw_measurement(float(tiempo), float(radio), int(tid))

        messagebox.showinfo("Guardado exitoso", f"Datos guardados en base '{dbname}'")

    def show_model_summary(self):
        data = [self.tree.item(c)['values'] for c in self.tree.get_children()]
        if not data:
            messagebox.showwarning("Resumen", "No hay datos para analizar.")
            return

        t = np.array([float(r[0]) for r in data])
        r = np.array([float(r[1]) for r in data])
        X = np.column_stack((np.ones_like(t), t, t ** 2))
        model = sm.OLS(r, X).fit()

        df_res = pd.DataFrame({
            'Parámetro': ['Intercepto', 't', 't^2'],
            'Estimación': model.params,
            'Error estándar': model.bse,
            't-valor': model.tvalues,
            'p-valor': model.pvalues
        })

        self.model_tree.delete(*self.model_tree.get_children())

        for _, row in df_res.iterrows():
            self.model_tree.insert('', 'end', values=list(row))

    def show_diagnostic_tests(self):
        data = [self.tree.item(c)['values'] for c in self.tree.get_children()]
        if not data:
            messagebox.showwarning("Diagnóstico", "No hay datos para analizar.")
            return

        t = np.array([float(r[0]) for r in data])
        r = np.array([float(r[1]) for r in data])
        X = np.column_stack((np.ones_like(t), t, t ** 2))
        model = sm.OLS(r, X).fit()
        resid = model.resid
        infl = OLSInfluence(model)
        lev = infl.hat_matrix_diag
        cooks_d = infl.cooks_distance[0]

        from scipy.stats import shapiro, anderson
        from statsmodels.stats.diagnostic import het_breuschpagan
        from statsmodels.stats.stattools import durbin_watson

        # Validaciones
        try:
            sw_stat, sw_p = shapiro(resid[:5000]) if len(resid) > 5000 else shapiro(resid)
            sw_row = ("Shapiro-Wilk", f"{sw_stat:.4f}", f"{sw_p:.4f}")
        except Exception as e:
            sw_row = ("Shapiro-Wilk", "N/A", f"Error: {str(e)}")

        try:
            ad_result = anderson(resid)
            ad_row = (
            "Anderson-Darling", f"{ad_result.statistic:.4f}", f"crítico 5%: {ad_result.critical_values[2]:.4f}")
        except Exception as e:
            ad_row = ("Anderson-Darling", "N/A", f"Error: {str(e)}")

        try:
            bp_stat, bp_p, _, _ = het_breuschpagan(resid, model.model.exog)
            bp_row = ("Breusch-Pagan", f"{bp_stat:.4f}", f"{bp_p:.4f}")
        except Exception as e:
            bp_row = ("Breusch-Pagan", "N/A", f"Error: {str(e)}")

        try:
            dw_stat = durbin_watson(resid)
            dw_row = ("Durbin-Watson", f"{dw_stat:.4f}", "-")
        except Exception as e:
            dw_row = ("Durbin-Watson", "N/A", f"Error: {str(e)}")

        # Influencia
        infl = OLSInfluence(model)
        cooks_d = infl.cooks_distance[0]
        lev = infl.hat_matrix_diag
        cooks_row = ("Cook's D (máx.)", f"{np.max(cooks_d):.4f}", "umbral: 0.3636")
        lev_row = ("Leverage (máx.)", f"{np.max(lev):.4f}", "umbral: 0.4286")

        diag_data = [sw_row, ad_row, bp_row, dw_row, cooks_row, lev_row]

        top = tk.Toplevel(self.frame)
        top.title("Resultados de Pruebas de Diagnóstico de Residuales")
        tree = ttk.Treeview(top, columns=("Prueba", "Valor", "p-valor / umbral"), show="headings")
        for col in ("Prueba", "Valor", "p-valor / umbral"):
            tree.heading(col, text=col)
            tree.column(col, width=160)
        for row in diag_data:
            tree.insert('', 'end', values=row)
        tree.pack(fill=tk.BOTH, expand=True)

    def definir_escala_visual(self):
        if not hasattr(self, "video_path") or not self.video_path:
            messagebox.showerror("Error", "Primero cargue un video.")
            return

        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))  # Forzar resolución
        cap.release()

        if not ret:
            messagebox.showerror("Error", "No se pudo leer el primer frame del video.")
            return

        puntos = []

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                puntos.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
                cv2.imshow("Seleccione 2 puntos", frame)

                if len(puntos) == 2:
                    cv2.line(frame, puntos[0], puntos[1], (255, 0, 0), 2)
                    cv2.imshow("Seleccione 2 puntos", frame)
                    d_px = np.linalg.norm(np.array(puntos[0]) - np.array(puntos[1]))
                    d_real = simpledialog.askfloat("Distancia real", "Ingrese distancia real entre puntos (en metros):")

                    if d_real:
                        self.pt1 = puntos[0]
                        self.pt2 = puntos[1]
                        self.distancia_real = d_real
                        self.escala = d_real / d_px
                        self.scale_entry.delete(0, tk.END)
                        self.scale_entry.insert(0, f"{self.escala:.5f}")
                    cv2.destroyWindow("Seleccione 2 puntos")

        import tkinter.simpledialog as simpledialog
        cv2.imshow("Seleccione 2 puntos", frame)
        cv2.setMouseCallback("Seleccione 2 puntos", click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def browse_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi"), ("All files", "*.*")])
        if path:
            self.video_path = path
            self.video_entry.delete(0, tk.END)
            self.video_entry.insert(0, path)

    def browse_mask(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg"), ("All files", "*.*")])
        if path:
            self.mask_path = path
            self.mask_entry.delete(0, tk.END)
            self.mask_entry.insert(0, path)

    def process_video(self):
        # 1) Leer y validar escala
        try:
            pts = [float(p) for p in self.scale_entry.get().split(",")]
            self.scale_points = pts
        except ValueError:
            messagebox.showerror("Error", "Puntos de escala inválidos.")
            return

        # 2) Validar video cargado
        vid_path = self.video_entry.get().strip()
        if not vid_path:
            messagebox.showerror("Error", "Primero cargue un video.")
            return

        # 3) Crear/configurar VideoProcessor
        if not hasattr(self, "processor"):
            self.processor = VideoProcessor(self.tree)
        self.processor.set_video(vid_path)

        # 4) (Opcional) máscara
        mask_path = getattr(self, "mask_path", "")
        if mask_path:
            self.processor.set_mask(mask_path)

        # 5) Validar que ya definiste los dos puntos y distancia real
        if not (hasattr(self, "pt1") and hasattr(self, "pt2") and hasattr(self, "distancia_real")):
            messagebox.showerror("Error", "Define primero la escala visual (botón “Definir Escala Visual”).")
            return
        self.processor.set_points(self.pt1, self.pt2, self.distancia_real)

        # 6) Limpiar tabla antes de empezar
        for iid in self.tree.get_children():
            self.tree.delete(iid)

        # 7) Ejecutar el procesamiento en un hilo para no bloquear la GUI
        threading.Thread(target=self.processor.process_video, daemon=True).start()

    def get_state(self):
        table_data = [self.tree.item(i)["values"] for i in self.tree.get_children()]
        return {
            "video_path": self.video_path,
            "mask_path": self.mask_path,
            "scale_points": self.scale_points,
            "table_data": table_data
        }

    @classmethod
    def from_state(cls, notebook, state):
        tab = cls(notebook, app_ref=None, tab_name="Loaded Experiment")
        tab.video_path = state.get("video_path", "")
        tab.mask_path = state.get("mask_path", "")
        tab.scale_points = state.get("scale_points", [0, 0])
        tab.video_entry.insert(0, tab.video_path)
        tab.mask_entry.insert(0, tab.mask_path)
        tab.scale_entry.delete(0, tk.END)
        tab.scale_entry.insert(0, f"{tab.scale_points[0]},{tab.scale_points[1]}")
        for row in state.get("table_data", []):
            tab.tree.insert("", "end", values=row)
        return tab

class App:
    def __init__(self, root):
        root.title("Multi-Tab Experiment App")
        self.notebook = ttk.Notebook(root); self.notebook.pack(fill=tk.BOTH, expand=True)
        self.tabs = []
        self.add_new_tab()

        # Menú con opciones Nuevo, Cerrar, Guardar, Cargar
        menu_bar = tk.Menu(root); file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Nuevo Experimento", command=self.add_new_tab)
        file_menu.add_command(label="Cerrar Experimento", command=self.close_current_tab)
        file_menu.add_separator()
        file_menu.add_command(label="Guardar Experimento", command=self.save_experiment_state)
        file_menu.add_command(label="Cargar Experimento", command=self.load_experiment_state)
        menu_bar.add_cascade(label="Archivo", menu=file_menu)
        root.config(menu=menu_bar)

    def add_new_tab(self):
        tab = ExperimentTab(
            self.notebook,
            app_ref=self,
            tab_name=f"Exp {len(self.tabs) + 1}"
        )
        self.tabs.append(tab)

    def get_current_tab(self):
        current = self.notebook.select()
        for tab in self.tabs:
            if str(tab.frame) == current:  # comparar el widget identificador
                return tab
        return None

    def close_current_tab(self):
        if len(self.tabs) <= 1:
            messagebox.showwarning("Aviso", "No se puede cerrar la última pestaña.")
            return
        if messagebox.askyesno("Cerrar Experimento", "¿Desea cerrar esta pestaña?"):
            tab = self.get_current_tab()
            if tab:
                self.notebook.forget(tab.frame)
                self.tabs.remove(tab)

    def save_experiment_state(self):
        tab = self.get_current_tab()
        if not tab: return
        state = tab.get_state()
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
        if file_path:
            try:
                with open(file_path, "w") as f:
                    json.dump(state, f, indent=4)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar: {e}")

    def load_experiment_state(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON","*.json")])
        if not file_path: return
        try:
            with open(file_path, "r") as f:
                state = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer: {e}")
            return
        tab = ExperimentTab.from_state(self.notebook, state)
        self.tabs.append(tab)


class VideoProcessor:
    def __init__(self, table):
        self.model = YOLO("wheel.pt")
        self.tracker = Sort()
        self.video_path = None
        self.mask = None
        self.pt1 = None
        self.pt2 = None
        self.distancia_real = 1.0
        self.escala = 1.0
        self.table = table

    def set_video(self, path):
        self.video_path = path

    def set_mask(self, mask_path):
        self.mask = cv2.imread(mask_path, 0)  # Grayscale

    def set_points(self, pt1, pt2, distancia_real):
        self.pt1 = pt1
        self.pt2 = pt2
        d_px = np.linalg.norm(np.array(pt2) - np.array(pt1))
        self.distancia_real = distancia_real
        self.escala = distancia_real / d_px

    def process_video(self):
        if not self.video_path or not self.pt1 or not self.pt2:
            print("Faltan datos necesarios para procesar el video.")
            return

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = 1280
        height = 720

        out = cv2.VideoWriter(
            "output.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

        font = cv2.FONT_HERSHEY_SIMPLEX
        frame_count = 0

        self.table.delete(*self.table.get_children())

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (width, height))

            if self.mask is not None:
                mask_resized = cv2.resize(self.mask, (width, height))
                frame = cv2.bitwise_and(frame, frame, mask=mask_resized)

            results = self.model(frame)
            boxes = []
            for res in results:
                if res.boxes.xyxy is not None:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy().flatten()
                    mask = confs > 0.25
                    boxes = xyxy[mask].astype(int)

            if len(boxes) > 0:
                tracks = self.tracker.update(boxes)
            else:
                tracks = self.tracker.update(np.empty((0, 4)))

            for track in tracks:
                x1, y1, x2, y2, tid = track.astype(int)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                dx = (cx - self.pt1[0]) * self.escala
                dy = (cy - self.pt1[1]) * self.escala
                r = math.sqrt(dx**2 + dy**2)
                tiempo = frame_count / fps

                self.table.insert('', 'end', values=(f"{tiempo:.2f}", f"{r:.4f}", int(tid)))

                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"ID {tid}", (x1, y1 - 10), font, 0.6, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            origin = self.pt1
            axis_len = 150
            # Dibuja ejes X e Y
            cv2.arrowedLine(frame, origin, (origin[0] + axis_len, origin[1]), (255, 0, 0), 2)
            cv2.arrowedLine(frame, origin, (origin[0], origin[1] - axis_len), (0, 0, 255), 2)
            cv2.putText(frame, 'X', (origin[0] + axis_len + 5, origin[1] + 5), font, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, 'Y', (origin[0] - 15, origin[1] - axis_len - 5), font, 0.7, (0, 0, 255), 2)

            # Etiqueta SRI y distancia real junto al origen
            cv2.putText(
                frame,
                f"SRI ({origin[0]}, {origin[1]})",  # muestra coordenadas del origen
                (origin[0] + 10, origin[1] + 25),  # ligeramente desplazado
                font,
                0.6,
                (0, 255, 255),  # color amarillo
                2
            )
            cv2.putText(
                frame,
                f"{self.distancia_real:.2f} m",  # distancia real en metros
                (origin[0] + 10, origin[1] + 45),
                font,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.line(frame, self.pt1, self.pt2, (0, 255, 255), 2)
            cv2.putText(frame, f"{self.distancia_real:.2f} m", (self.pt1[0], self.pt1[1] - 10), font, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Escala: {self.escala:.5f} m/px", (10, 30), font, 0.7, (255, 255, 255), 2)

            cv2.imshow("Resultado", frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()


class GraphPlotter:
    def __init__(self, parent, table):
        self.table = table
        self.frame = tk.Frame(parent)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.figure, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_plot(self):
        self.ax.clear()
        data = [self.table.item(child)["values"] for child in self.table.get_children()]
        if not data:
            self.ax.set_title("Sin datos para graficar")
            self.canvas.draw()
            return

        t = np.array([float(row[0]) for row in data])
        r = np.array([float(row[1]) for row in data])

        X = np.column_stack((t**2, t, np.ones_like(t)))
        model = sm.OLS(r, X).fit()
        t_line = np.linspace(t.min(), t.max(), 200)
        X_pred = np.column_stack((t_line**2, t_line, np.ones_like(t_line)))
        y_pred = model.predict(X_pred)

        prstd, iv_l, iv_u = wls_prediction_std(model, exog=X_pred, alpha=0.05)

        self.ax.scatter(t, r, label="Datos", color="blue", s=2)
        self.ax.plot(t_line, y_pred, color="black", label="Ajuste cuadrático")
        self.ax.fill_between(t_line, iv_l, iv_u, color='orange', alpha=0.3, label="Predicción ±95%")

        self.ax.set_xlabel("Tiempo (s)")
        self.ax.set_ylabel("r (m)")
        self.ax.legend()
        self.ax.grid(True)
        self.figure.tight_layout()
        self.canvas.draw()




class ResidualsPlotter:
    def __init__(self, parent, table):
        self.table = table
        self.frame = tk.Frame(parent)
        self.frame.pack(fill=tk.BOTH, expand=True)

        from matplotlib.figure import Figure
        self.fig = Figure(figsize=(10, 12), constrained_layout=True)
        self.axes = self.fig.subplots(4, 2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _extract_data_from_table(self):
        data = [self.table.item(child)["values"] for child in self.table.get_children()]
        if not data:
            return None, None
        t = np.array([float(row[0]) for row in data])
        r = np.array([float(row[1]) for row in data])
        return t, r

    def _prepare_model(self, t, r):
        X = np.column_stack((t**2, t, np.ones_like(t)))
        model = sm.OLS(r, X).fit()
        influence = OLSInfluence(model)
        d = np.linalg.norm(X - X.mean(axis=0), axis=1)
        return model, influence, d

    def update_residual_plots(self):
        self.fig.clf()
        self.axes = self.fig.subplots(4, 2)

        t, r = self._extract_data_from_table()
        if t is None:
            return

        model, influence, d = self._prepare_model(t, r)

        residuals = model.resid
        fitted = model.fittedvalues
        studentized = influence.resid_studentized_external
        standardized = influence.resid_studentized_internal
        leverage = influence.hat_matrix_diag
        cooks_d = influence.cooks_distance[0]

        axs = self.axes.flatten()

        # 1. QQ-Plot
        qqplot(studentized, line='45', ax=axs[0], markersize=1)
        axs[0].set_title("QQ-Plot de residuales studentizados")

        # 2. Histograma
        axs[1].hist(residuals, bins=20, density=True, alpha=0.6)
        x_vals = np.linspace(*axs[1].get_xlim(), 100)
        pdf = stats.norm.pdf(x_vals, residuals.mean(), residuals.std())
        axs[1].plot(x_vals, pdf, '--')
        axs[1].set_title("Histograma de residuales")

        # 3. Boxplot
        axs[2].boxplot(residuals, vert=False)
        axs[2].set_title("Boxplot de residuales")

        # 4. Std resid vs fitted
        axs[3].scatter(fitted, standardized, s=1)
        axs[3].axhline(0, linestyle='--')
        axs[3].set_title("Residuales estandarizados vs ajustados")
        axs[3].set_xlabel("Valores ajustados")
        axs[3].set_ylabel("Residuales estandarizados")

        # 5. Std resid vs distancia
        axs[4].scatter(d, standardized, s=1)
        axs[4].axhline(0, linestyle='--')
        axs[4].set_title("Residuales estandarizados vs distancia d")
        axs[4].set_xlabel("Distancia d")

        # 6. Resid vs orden
        axs[5].scatter(range(len(residuals)), residuals, s=1)
        axs[5].set_title("Residuales vs orden de observación")

        # 7. Std resid vs leverage
        axs[6].scatter(leverage, studentized, s=1)
        axs[6].axhline(0, linestyle='--')
        axs[6].set_title("Residuales estandarizados vs leverage")
        axs[6].set_xlabel("Leverage")

        # 8. Std resid vs Cook's
        axs[7].scatter(cooks_d, standardized, s=1)
        axs[7].axhline(0, linestyle='--')
        axs[7].set_title("Residuales estandarizados vs Distancia de Cook")
        axs[7].set_xlabel("Distancia de Cook")

        self.canvas.draw()



if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()