# src/gui/comparison_view.py
"""
Panel de comparación: UNA grabación -> los tres modelos -> resultados lado a lado.

Este panel vive DENTRO de la ventana principal (misma página, mismo scroll).
No contiene lógica de reconocimiento. Solo:
  - graba el audio (src.data.mic_recorder)
  - se lo entrega a MultiModelRunner (src.models.multi_model_runner), que ejecuta los tres modelos
  - muestra lo que devuelve: tabla individual, comparación general y gráficas

Las gráficas leen results/evaluation_summary.csv y results/evaluation_by_mic.csv,
generados por:  python -m src.models.evaluate_all
Si esos archivos no existen, se avisa y no se dibuja nada inventado.

La ventana principal le entrega tres funciones (así el panel no depende de ella):
  get_microphone()        -> (id_del_dispositivo, etiqueta)   micrófono elegido arriba
  can_start()             -> True si la app no está grabando ni cargando
  on_busy_change(ocupado) -> avisa a la app para bloquear/desbloquear sus controles
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import threading
import traceback
import tkinter as tk

import pandas as pd
import customtkinter as ctk

from src.data.mic_recorder import record_audio_to_wav
from src.models.model_factory import available_models

# ---------------------------------------------------------------
# Paleta (la misma de app.py)
# ---------------------------------------------------------------
BG = "#0D0F16"
CARD = "#161923"
BORDER = "#252A38"
ACCENT = "#6EA8FF"
TEXT = "#E8EBF3"
MUTED = "#9AA3B5"
OK = "#34D399"
BAD = "#F87171"
WARN = "#F59E0B"
SERIES_COLORS = ["#6EA8FF", "#34D399", "#F59E0B", "#C084FC"]

BTN_TEXT = "Grabar y comparar los 3 modelos   ·   4.0 s"
RESULTS_DIR = "results"
CHART_H = 250

ROW_NAMES = [
    ("speaker", "Hablante identificado"),
    ("score", "Similitud coseno"),
    ("threshold", "Umbral usado"),
    ("seconds", "Tiempo de procesamiento"),
    ("status", "Estado de reconocimiento"),
]


# ---------------------------------------------------------------
# Lectura de la evaluación real
# ---------------------------------------------------------------
def load_evaluation(results_dir=RESULTS_DIR):
    """Devuelve (resumen, por_microfono). Si no existen los CSV, devuelve (None, None)."""
    summary_path = os.path.join(results_dir, "evaluation_summary.csv")
    mic_path = os.path.join(results_dir, "evaluation_by_mic.csv")
    if not os.path.exists(summary_path):
        return None, None
    try:
        summary = pd.read_csv(summary_path)
        by_mic = pd.read_csv(mic_path) if os.path.exists(mic_path) else None
        return summary, by_mic
    except Exception as e:
        print(f"[Comparación] No se pudieron leer los resultados de evaluación: {e}")
        return None, None


# ---------------------------------------------------------------
# Gráfica de barras agrupadas (dibujada con tkinter, sin dependencias nuevas)
# ---------------------------------------------------------------
def draw_grouped_bars(canvas, title, categories, series, decimals=1, width=640, height=CHART_H):
    """
    categories: nombres del eje horizontal (los modelos).
    series: {nombre_de_serie: [valor por categoría]}  (NaN se omite)
    """
    canvas.delete("all")
    pad_l, pad_r, pad_t, pad_b = 14, 14, 38, 56
    canvas.create_text(pad_l, 16, text=title, anchor="w", fill=TEXT, font=("Segoe UI", 11, "bold"))

    values = [v for vals in series.values() for v in vals if v == v]
    if not categories or not values:
        canvas.create_text(width / 2, height / 2, text="Sin datos", fill=MUTED, font=("Segoe UI", 10))
        return

    vmax = max(values) * 1.18 if max(values) > 0 else 1.0
    plot_h = height - pad_t - pad_b
    plot_w = width - pad_l - pad_r
    base_y = height - pad_b
    n_cat, n_ser = len(categories), len(series)
    group_w = plot_w / n_cat
    bar_w = min(36, group_w * 0.8 / n_ser)

    canvas.create_line(pad_l, base_y, width - pad_r, base_y, fill=BORDER)
    for ci, cat in enumerate(categories):
        center_x = pad_l + ci * group_w + group_w / 2
        start_x = center_x - bar_w * n_ser / 2
        for si, vals in enumerate(series.values()):
            v = vals[ci]
            if v != v:          # NaN
                continue
            h = plot_h * (v / vmax)
            x0 = start_x + si * bar_w
            color = SERIES_COLORS[si % len(SERIES_COLORS)]
            canvas.create_rectangle(x0 + 2, base_y - h, x0 + bar_w - 2, base_y, fill=color, outline="")
            canvas.create_text(x0 + bar_w / 2, base_y - h - 8, text=f"{v:.{decimals}f}",
                               fill=TEXT, font=("Segoe UI", 8))
        canvas.create_text(center_x, base_y + 14, text=cat, fill=MUTED, font=("Segoe UI", 9))

    if n_ser > 1:               # leyenda solo si hay varias series
        legend_x = pad_l
        for si, name in enumerate(series):
            color = SERIES_COLORS[si % len(SERIES_COLORS)]
            canvas.create_rectangle(legend_x, height - 20, legend_x + 10, height - 10, fill=color, outline="")
            canvas.create_text(legend_x + 14, height - 15, text=name, anchor="w", fill=MUTED, font=("Segoe UI", 8))
            legend_x += 14 + 7 * len(name) + 14


# ---------------------------------------------------------------
# Panel de comparación (se coloca dentro de la página principal)
# ---------------------------------------------------------------
class ComparisonPanel(ctk.CTkFrame):
    def __init__(self, master, get_microphone, can_start, on_busy_change):
        super().__init__(master, fg_color="transparent")
        self.get_microphone = get_microphone
        self.can_start = can_start
        self.on_busy_change = on_busy_change

        self.model_names = available_models()
        self.runner = None           # se carga la primera vez que se compara
        self.is_busy = False
        self.cells = {}              # (clave_de_fila, modelo) -> etiqueta de la tabla
        self.chart_specs = []        # lo que dibuja cada gráfica (para redibujar al cambiar el ancho)

        self._build_ui()
        self._refresh_charts()

    # ----- Construcción de la interfaz -----
    def _card(self):
        return ctk.CTkFrame(self, fg_color=CARD, corner_radius=16, border_width=1, border_color=BORDER)

    def _section(self, parent, text):
        return ctk.CTkLabel(parent, text=text.upper(), text_color=MUTED,
                            font=ctk.CTkFont(size=11, weight="bold"))

    def _build_ui(self):
        ctk.CTkLabel(self, text="Comparación de los tres modelos", text_color=TEXT,
                     font=ctk.CTkFont(size=20, weight="bold")).pack(anchor="w", pady=(0, 2))
        ctk.CTkLabel(self, text="Una sola grabación, procesada por los tres modelos. "
                                "Usa el micrófono elegido arriba.",
                     text_color=MUTED, font=ctk.CTkFont(size=13), wraplength=640,
                     justify="left").pack(anchor="w")
        self.lbl_status = ctk.CTkLabel(self, text="Pulsa el botón para grabar y comparar.",
                                       text_color=MUTED, font=ctk.CTkFont(size=13),
                                       wraplength=640, justify="left")
        self.lbl_status.pack(anchor="w", pady=(6, 10))

        self.btn_record = ctk.CTkButton(self, text=BTN_TEXT, height=54, corner_radius=27,
                                        fg_color=ACCENT, hover_color="#8FBBFF", text_color=BG,
                                        text_color_disabled="#2B3350",
                                        font=ctk.CTkFont(size=16, weight="bold"),
                                        command=self._start_comparison)
        self.btn_record.pack(fill="x", pady=(0, 14))

        # A. Resultados individuales (tabla)
        card_table = self._card()
        card_table.pack(fill="x", pady=(0, 14))
        self._section(card_table, "A · Resultados individuales (misma grabación)").pack(anchor="w", padx=24, pady=(18, 8))
        grid = ctk.CTkFrame(card_table, fg_color="transparent")
        grid.pack(fill="x", padx=24, pady=(0, 18))
        for col in range(len(self.model_names) + 1):
            grid.grid_columnconfigure(col, weight=1, minsize=150)
        for col, name in enumerate(self.model_names, start=1):
            ctk.CTkLabel(grid, text=name, text_color=ACCENT,
                         font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=col, padx=6, pady=(0, 8), sticky="w")
        for r, (key, label) in enumerate(ROW_NAMES, start=1):
            ctk.CTkLabel(grid, text=label, text_color=MUTED, wraplength=150, justify="left",
                         font=ctk.CTkFont(size=13)).grid(row=r, column=0, padx=6, pady=6, sticky="w")
            for col, name in enumerate(self.model_names, start=1):
                cell = ctk.CTkLabel(grid, text="—", text_color=TEXT, wraplength=150, justify="left",
                                    font=ctk.CTkFont(size=13, weight="bold"))
                cell.grid(row=r, column=col, padx=6, pady=6, sticky="w")
                self.cells[(key, name)] = cell

        # B. Comparación general
        card_general = self._card()
        card_general.pack(fill="x", pady=(0, 14))
        self._section(card_general, "B · Comparación general").pack(anchor="w", padx=24, pady=(18, 6))
        self.lbl_general = ctk.CTkLabel(card_general, text="Graba una frase para comparar los modelos.",
                                        text_color=TEXT, wraplength=620, justify="left",
                                        font=ctk.CTkFont(size=14))
        self.lbl_general.pack(anchor="w", padx=24, pady=(0, 18))

        # C. Gráficas de la evaluación completa
        card_charts = self._card()
        card_charts.pack(fill="x")
        head = ctk.CTkFrame(card_charts, fg_color="transparent")
        head.pack(fill="x", padx=24, pady=(18, 4))
        self._section(head, "C · Gráficas (evaluación completa con datos reales)").pack(side="left")
        ctk.CTkButton(head, text="⟳  Actualizar gráficas", width=150, height=28, corner_radius=8,
                      fg_color="#222737", hover_color="#2C3242", text_color=TEXT,
                      font=ctk.CTkFont(size=12), command=self._refresh_charts).pack(side="right")
        self.lbl_charts_info = ctk.CTkLabel(card_charts, text="", text_color=MUTED, wraplength=620,
                                            justify="left", font=ctk.CTkFont(size=12))
        self.lbl_charts_info.pack(anchor="w", padx=24, pady=(0, 8))
        self.canvases = []
        for i in range(4):
            canvas = tk.Canvas(card_charts, height=CHART_H, bg=CARD, highlightthickness=1,
                               highlightbackground=BORDER)
            canvas.pack(fill="x", padx=24, pady=(0, 12))
            canvas.bind("<Configure>", lambda event, idx=i: self._draw_chart(idx))   # redibuja al cambiar el ancho
            self.canvases.append(canvas)
        ctk.CTkFrame(card_charts, fg_color="transparent", height=6).pack()

    # ----- Estado -----
    def _set_status(self, text, color=MUTED):
        self.lbl_status.configure(text=text, text_color=color)

    # ----- Grabar y comparar -----
    def _start_comparison(self):
        if self.is_busy:
            return
        if not self.can_start():
            self._set_status("Espera a que termine la carga del modelo o la otra grabación.", WARN)
            return
        device_id, mic_label = self.get_microphone()
        if device_id is None:
            self._set_status("Selecciona un micrófono válido arriba.", BAD)
            return

        self.is_busy = True
        self.on_busy_change(True)            # la app bloquea sus propios controles
        self.btn_record.configure(state="disabled", text="Preparando...")
        for cell in self.cells.values():
            cell.configure(text="—", text_color=TEXT)
        self.lbl_general.configure(text="Procesando...")
        threading.Thread(target=self._comparison_task, args=(device_id, mic_label), daemon=True).start()

    def _comparison_task(self, device_id, mic_label):
        try:
            # Los tres modelos se cargan solo la primera vez que se compara
            if self.runner is None:
                self.after(0, lambda: self._set_status(
                    "Cargando los 3 modelos (solo la primera vez, puede tardar)...", WARN))
                from src.models.multi_model_runner import MultiModelRunner   # import diferido
                self.runner = MultiModelRunner()

            self.after(0, lambda: self.btn_record.configure(text="Grabando... ¡Habla ahora!"))
            self.after(0, lambda: self._set_status("Capturando 4.0 segundos...", ACCENT))
            wav_path = record_audio_to_wav(output_path="temp_mic_capture_compare.wav",
                                           duration_sec=4.0, device_id=device_id)

            self.after(0, lambda: self.btn_record.configure(text="Analizando con los 3 modelos..."))
            result = self.runner.run(wav_path, mic=mic_label)      # los tres, misma grabación
            self.after(0, lambda: self._show_result(result))
        except Exception as e:
            traceback.print_exc()
            msg = str(e)
            self.after(0, lambda: self._finish(f"Error: {msg}", BAD))

    def _finish(self, status_text, color):
        self.is_busy = False
        self.btn_record.configure(state="normal", text=BTN_TEXT)
        self._set_status(status_text, color)
        self.on_busy_change(False)           # la app desbloquea sus controles

    # ----- Mostrar resultados -----
    def _show_result(self, result):
        if not result.audio_ok:
            self.lbl_general.configure(text="La grabación no pasó la validación; no se ejecutó ningún modelo.")
            self._finish(f"Audio inválido: {result.audio_error}", BAD)
            return

        for name in self.model_names:
            r = result.results.get(name)
            if r is not None:
                self._fill_column(name, r)

        self.lbl_general.configure(text=self._general_text(result))
        if self.runner.load_errors:
            failed = "; ".join(f"{k}: {v[:80]}" for k, v in self.runner.load_errors.items())
            self._finish(f"Comparación terminada, pero algunos modelos no cargaron: {failed}", WARN)
        else:
            self._finish("Comparación terminada. Puedes grabar de nuevo.", OK)

    def _set(self, key, name, text, color=TEXT):
        self.cells[(key, name)].configure(text=text, text_color=color)

    def _fill_column(self, name, r):
        if r.status == "identificado":
            self._set("speaker", name, r.identified_as, OK)
            self._set("status", name, "✔ Identificado", OK)
        elif r.status == "rechazado":
            self._set("speaker", name, f"Desconocido\n(más cercano: {r.candidate})", BAD)
            self._set("status", name, "✖ Rechazado", BAD)
        elif r.status == "error":
            self._set("speaker", name, "—")
            self._set("status", name, f"⚠ Error: {r.error[:70]}", WARN)
        else:   # no disponible
            self._set("speaker", name, "—")
            self._set("status", name, f"⚠ No disponible: {r.error[:70]}", WARN)

        if r.status in ("identificado", "rechazado"):
            self._set("score", name, f"{r.score:.4f}")
            matcher = self.runner.matchers.get(name)
            self._set("threshold", name, f"{matcher.threshold:.2f}" if matcher else "—")
            self._set("seconds", name, f"{r.seconds:.3f} s")
        elif r.status == "error":
            self._set("seconds", name, f"{r.seconds:.3f} s")

    def _general_text(self, result):
        lines = []
        agreement = result.agreement()
        identified = result.identified_models()
        if agreement == "misma_persona":
            lines.append(f"• Los modelos que identificaron coinciden: todos dicen que es «{identified[0].identified_as}».")
        elif agreement == "identificaciones_distintas":
            detail = ", ".join(f"{r.model_name} → {r.identified_as}" for r in identified)
            lines.append(f"• Hay diferencias entre las identificaciones: {detail}.")
        elif agreement == "parcial":
            lines.append(f"• Los que identificaron coinciden («{identified[0].identified_as}»), pero otros rechazaron la grabación.")
        elif agreement == "todos_rechazan":
            lines.append("• Ningún modelo identificó a nadie: los tres rechazaron la grabación.")
        else:
            lines.append("• Ningún modelo pudo llegar a una decisión.")

        best = result.highest_valid_similarity()
        lines.append(f"• Mayor similitud válida: {best.model_name} ({best.score:.4f}). "
                     "Ojo: cada modelo puntúa en una escala distinta, esto no indica que sea el mejor."
                     if best else "• Mayor similitud válida: ningún modelo identificó a alguien.")
        fast = result.fastest()
        lines.append(f"• Menor tiempo en esta grabación: {fast.model_name} ({fast.seconds:.3f} s)."
                     if fast else "• Menor tiempo: sin datos.")
        rejected = result.rejected_models()
        lines.append(f"• Rechazaron la grabación: {', '.join(rejected) if rejected else 'ninguno'}.")
        lines.append("\nEsto es UNA sola grabación. Qué modelo rinde mejor se decide con las gráficas "
                     "de abajo (aciertos, falsas aceptaciones y falsos rechazos de la evaluación completa), "
                     "no con una similitud.")
        return "\n".join(lines)

    # ----- Gráficas (evaluación real) -----
    def _draw_chart(self, index):
        """Dibuja una gráfica con el ancho actual de su lienzo (se llama al cambiar el tamaño)."""
        if index >= len(self.chart_specs):
            return
        canvas = self.canvases[index]
        width = canvas.winfo_width()
        if width < 120:          # aún no se ha mostrado en pantalla
            return
        title, categories, series, decimals = self.chart_specs[index]
        draw_grouped_bars(canvas, title, categories, series, decimals=decimals, width=width, height=CHART_H)

    def _set_chart_specs(self, specs):
        self.chart_specs = specs
        for i in range(len(self.canvases)):
            self._draw_chart(i)

    def _refresh_charts(self):
        summary, by_mic = load_evaluation()
        if summary is None:
            self._set_chart_specs([("Sin resultados de evaluación", [], {}, 1)] * 4)
            self.lbl_charts_info.configure(
                text="Aún no hay resultados reales. Ejecuta:  python -m src.models.evaluate_all  "
                     "y pulsa «Actualizar gráficas».")
            return

        models = list(summary["modelo"])
        pct = lambda col: [100.0 * float(v) for v in summary[col]]
        n_known = int(summary["pruebas_conocidos"].iloc[0])
        n_unknown = int(summary["pruebas_desconocidos"].iloc[0])
        self.lbl_charts_info.configure(
            text=f"Prueba final con {n_known} audios de hablantes registrados y {n_unknown} de desconocidos, "
                 "iguales para los tres modelos (results/evaluation_summary.csv).")

        mic_series = {}
        if by_mic is not None:
            for mic in sorted(by_mic["mic"].unique()):
                values = []
                for model in models:
                    row = by_mic[(by_mic["modelo"] == model) & (by_mic["mic"] == mic)]
                    values.append(100.0 * float(row["id_correcta"].iloc[0]) if len(row) else float("nan"))
                mic_series[mic] = values

        self._set_chart_specs([
            ("Precisión de identificación (%)", models,
             {"Exactitud cerrada": pct("acc_cerrada"), "Identificación correcta": pct("id_correcta")}, 1),
            ("Tiempo medio por audio (ms)", models,
             {"Tiempo": [float(v) for v in summary["ms_media"]]}, 1),
            ("Falsas identificaciones y rechazos (%)", models,
             {"FAR (desconocidos aceptados)": pct("far"), "FRR (falsos rechazos)": pct("frr"),
              "Falsa identificación": pct("falsa_identificacion")}, 1),
            ("Identificación correcta por micrófono (%)", models, mic_series, 1),
        ])