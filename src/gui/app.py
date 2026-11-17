# src/gui/app.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import threading
import customtkinter as ctk
from src.data.mic_recorder import list_input_devices, record_audio_to_wav
from src.models.speaker_matcher import SpeakerMatcher

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ---------------------------------------------------------------
# PASO 1: Paleta de colores (todos en un solo lugar).
# Si quieres cambiar el look, solo modificas estos valores.
# ---------------------------------------------------------------
BG = "#0D0F16"        # fondo de la ventana
CARD = "#161923"      # fondo de las tarjetas
BORDER = "#252A38"    # borde de las tarjetas
ACCENT = "#6EA8FF"    # azul principal
TEXT = "#E8EBF3"      # texto normal
MUTED = "#9AA3B5"     # texto secundario
OK = "#34D399"        # verde (identificado)
BAD = "#F87171"       # rojo (desconocido / error)
WARN = "#F59E0B"      # ámbar (cargando)
TRACK = "#242A3A"     # fondo de barras y slider

# Colores (fondo, texto) para la "píldora" de estado de arriba
STATUS_STYLES = {
    "loading": ("#2A2111", WARN),
    "ok": ("#12261F", "#6EE7B7"),
    "error": ("#2A1214", BAD),
}

# Colores (fondo, texto) para la etiqueta del veredicto
BADGE_STYLES = {
    "wait": ("#222737", "#B4BCCC"),
    "listen": ("#1B2440", ACCENT),
    "ok": ("#12261F", "#6EE7B7"),
    "bad": ("#2A1214", BAD),
}

BTN_TEXT = "Grabar e identificar   ·   4.0 s"


class SpeakerRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sistema Biométrico de Identificación de Hablante (1:N)")
        self.geometry("820x700")
        self.resizable(False, False)
        self.configure(fg_color=BG)

        self.matcher = None
        self.devices_map = {}
        self.is_recording = False

        self._build_ui()
        self._load_devices()
        # Cargar el modelo ECAPA-TDNN en segundo plano al abrir la app
        threading.Thread(target=self._init_model, daemon=True).start()

    # ---------------------------------------------------------------
    # Pequeñas ayudas para no repetir código
    # ---------------------------------------------------------------
    def _section_label(self, parent, text):
        """Etiqueta pequeña en mayúsculas para titular cada sección."""
        return ctk.CTkLabel(
            parent, text=text.upper(), text_color=MUTED,
            font=ctk.CTkFont(size=11, weight="bold")
        )

    def _make_card(self):
        """Tarjeta con esquinas redondeadas y borde fino."""
        return ctk.CTkFrame(
            self, fg_color=CARD, corner_radius=16,
            border_width=1, border_color=BORDER
        )

    def _set_status(self, text, kind):
        """Cambia la píldora de estado de arriba (loading / ok / error)."""
        bg, fg = STATUS_STYLES[kind]
        self.lbl_status.configure(text=f"●  {text}", fg_color=bg, text_color=fg)

    def _set_badge(self, text, kind):
        """Cambia la etiqueta del veredicto (wait / listen / ok / bad)."""
        bg, fg = BADGE_STYLES[kind]
        self.lbl_badge.configure(text=f"  {text}  ", fg_color=bg, text_color=fg)

    # ---------------------------------------------------------------
    # PASO 2: Construcción de la interfaz
    # ---------------------------------------------------------------
    def _build_ui(self):
        # ----- Encabezado: logo + título a la izquierda, estado a la derecha
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=40, pady=(28, 0))

        left = ctk.CTkFrame(header, fg_color="transparent")
        left.pack(side="left")

        logo = ctk.CTkFrame(left, width=44, height=44, corner_radius=12, fg_color=ACCENT)
        logo.pack(side="left", padx=(0, 14))
        logo.pack_propagate(False)
        ctk.CTkLabel(logo, text="🎙", text_color=BG, font=ctk.CTkFont(size=22)).place(
            relx=0.5, rely=0.5, anchor="center"
        )

        titles = ctk.CTkFrame(left, fg_color="transparent")
        titles.pack(side="left")
        ctk.CTkLabel(
            titles, text="Identificación por Voz", text_color=TEXT,
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(anchor="w")
        ctk.CTkLabel(
            titles, text="Biometría de hablante · ECAPA-TDNN · 1:N",
            text_color=MUTED, font=ctk.CTkFont(size=13)
        ).pack(anchor="w")

        self.lbl_status = ctk.CTkLabel(
            header, text="", height=32, corner_radius=16,
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.lbl_status.pack(side="right")
        self._set_status("Cargando modelo y centroides...", "loading")

        # ----- Tarjeta de configuración (micrófono + umbral)
        card_cfg = self._make_card()
        card_cfg.pack(fill="x", padx=40, pady=(22, 0))

        self._section_label(card_cfg, "Micrófono").pack(anchor="w", padx=24, pady=(20, 6))

        mic_row = ctk.CTkFrame(card_cfg, fg_color="transparent")
        mic_row.pack(fill="x", padx=24)

        self.combo_mic = ctk.CTkComboBox(
            mic_row, height=40, corner_radius=10, state="readonly",
            fg_color=BG, border_color="#2C3242", button_color="#2C3242",
            button_hover_color="#3A4258", text_color=TEXT,
            dropdown_fg_color=CARD, dropdown_text_color=TEXT,
            dropdown_hover_color="#222737"
        )
        self.combo_mic.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.btn_refresh_mic = ctk.CTkButton(
            mic_row, text="⟳  Actualizar", width=120, height=40, corner_radius=10,
            fg_color="#222737", hover_color="#2C3242", text_color=TEXT,
            command=self._load_devices
        )
        self.btn_refresh_mic.pack(side="left")

        # Umbral: título a la izquierda y valor en una "chip" a la derecha
        tau_row = ctk.CTkFrame(card_cfg, fg_color="transparent")
        tau_row.pack(fill="x", padx=24, pady=(22, 6))
        self._section_label(tau_row, "Umbral de decisión (Tau)").pack(side="left")
        self.lbl_thresh = ctk.CTkLabel(
            tau_row, text="0.59", height=26, corner_radius=8,
            fg_color="#1B2440", text_color=ACCENT,
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.lbl_thresh.pack(side="right")

        self.slider_thresh = ctk.CTkSlider(
            card_cfg, from_=30, to=85, number_of_steps=55,
            fg_color=TRACK, progress_color=ACCENT,
            button_color=ACCENT, button_hover_color="#8FBBFF",
            command=self._on_threshold_change
        )
        # Valor inicial 0.59 (el slider trabaja con enteros: 59 = 0.59)
        self.slider_thresh.set(59)
        self.slider_thresh.pack(fill="x", padx=24)

        hint_row = ctk.CTkFrame(card_cfg, fg_color="transparent")
        hint_row.pack(fill="x", padx=24, pady=(4, 18))
        ctk.CTkLabel(hint_row, text="Más permisivo", text_color=MUTED,
                     font=ctk.CTkFont(size=12)).pack(side="left")
        ctk.CTkLabel(hint_row, text="Más estricto", text_color=MUTED,
                     font=ctk.CTkFont(size=12)).pack(side="right")

        # ----- Botón principal de grabación (grande y redondeado)
        self.btn_record = ctk.CTkButton(
            self, text=BTN_TEXT, height=64, width=360, corner_radius=32,
            fg_color=ACCENT, hover_color="#8FBBFF",
            text_color=BG, text_color_disabled="#2B3350",
            font=ctk.CTkFont(size=17, weight="bold"),
            state="disabled", command=self._start_recognition_thread
        )
        self.btn_record.pack(pady=24)

        # ----- Tarjeta de resultados
        card_res = self._make_card()
        card_res.pack(fill="both", expand=True, padx=40, pady=(0, 28))

        top = ctk.CTkFrame(card_res, fg_color="transparent")
        top.pack(fill="x", padx=24, pady=(20, 0))
        self._section_label(top, "Veredicto").pack(side="left")
        self.lbl_badge = ctk.CTkLabel(
            top, text="", height=28, corner_radius=14,
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.lbl_badge.pack(side="right")
        self._set_badge("En espera", "wait")

        # Candidato (izquierda) y similitud (derecha), en grande
        values = ctk.CTkFrame(card_res, fg_color="transparent")
        values.pack(fill="x", padx=24, pady=(14, 0))

        col_left = ctk.CTkFrame(values, fg_color="transparent")
        col_left.pack(side="left")
        ctk.CTkLabel(col_left, text="Candidato más cercano", text_color=MUTED,
                     font=ctk.CTkFont(size=13)).pack(anchor="w")
        self.lbl_candidate = ctk.CTkLabel(
            col_left, text="—", text_color=TEXT,
            font=ctk.CTkFont(size=30, weight="bold")
        )
        self.lbl_candidate.pack(anchor="w")

        col_right = ctk.CTkFrame(values, fg_color="transparent")
        col_right.pack(side="right")
        ctk.CTkLabel(col_right, text="Similitud coseno", text_color=MUTED,
                     font=ctk.CTkFont(size=13)).pack(anchor="e")
        self.lbl_score = ctk.CTkLabel(
            col_right, text="0.0000", text_color=ACCENT,
            font=ctk.CTkFont(size=30, weight="bold")
        )
        self.lbl_score.pack(anchor="e")

        # Barra de similitud + marca vertical que indica dónde está el umbral
        bar_box = ctk.CTkFrame(card_res, fg_color="transparent", height=56)
        bar_box.pack(fill="x", padx=24, pady=(16, 0))
        bar_box.pack_propagate(False)

        self.progress_score = ctk.CTkProgressBar(
            bar_box, height=10, corner_radius=5,
            fg_color=TRACK, progress_color=ACCENT
        )
        self.progress_score.set(0.0)
        self.progress_score.pack(fill="x", pady=(8, 0))

        # La marca y su texto se colocan con "relx" (0.0 a 1.0 del ancho)
        self.marker_tau = ctk.CTkFrame(bar_box, width=2, height=24, fg_color=TEXT, corner_radius=1)
        self.marker_tau.place(relx=0.59, y=0, anchor="n")
        self.lbl_tau_mark = ctk.CTkLabel(
            bar_box, text="Tau 0.59", text_color=MUTED, font=ctk.CTkFont(size=12)
        )
        self.lbl_tau_mark.place(relx=0.59, y=28, anchor="n")

        # Línea de información extra (errores, mensajes de captura, umbral usado)
        self.lbl_info = ctk.CTkLabel(
            card_res, text="", text_color=MUTED, wraplength=700,
            font=ctk.CTkFont(size=13)
        )
        self.lbl_info.pack(padx=24, pady=(4, 12))

    # ---------------------------------------------------------------
    # PASO 3: Lógica (igual que antes, solo cambia cómo se muestra)
    # ---------------------------------------------------------------
    def _load_devices(self):
        devices = list_input_devices()
        self.devices_map.clear()
        names = []
        for dev in devices:
            label = f"[{dev['id']}] {dev['name']}"
            self.devices_map[label] = dev["id"]
            names.append(label)

        if names:
            self.combo_mic.configure(values=names)
            self.combo_mic.set(names[0])
        else:
            self.combo_mic.configure(values=["Sin micrófonos detectados"])
            self.combo_mic.set("Sin micrófonos detectados")

    def _on_threshold_change(self, val):
        scaled_val = round(float(val) / 100.0, 2)
        # Actualiza el número, la marca de la barra y el texto "Tau"
        self.lbl_thresh.configure(text=f"{scaled_val:.2f}")
        self.lbl_tau_mark.configure(text=f"Tau {scaled_val:.2f}")
        self.marker_tau.place_configure(relx=scaled_val)
        self.lbl_tau_mark.place_configure(relx=scaled_val)
        if self.matcher is not None:
            self.matcher.threshold = scaled_val

    def _init_model(self):
        try:
            initial_tau = round(float(self.slider_thresh.get()) / 100.0, 2)
            self.matcher = SpeakerMatcher(threshold=initial_tau)
            self.after(0, self._on_model_ready)
        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda msg=err_msg: self._set_status(f"Error cargando modelo: {msg}", "error"))

    def _on_model_ready(self):
        self._set_status("Sistema listo · 30 hablantes", "ok")
        self.btn_record.configure(state="normal")

    def _start_recognition_thread(self):
        if self.is_recording or self.matcher is None:
            return
        self.is_recording = True
        self.btn_record.configure(state="disabled", text="Grabando... ¡Habla ahora!")

        # Dejar la tarjeta de resultados "limpia" para la nueva captura
        self._set_badge("Escuchando audio...", "listen")
        self.lbl_candidate.configure(text="—")
        self.lbl_score.configure(text="0.0000", text_color=ACCENT)
        self.lbl_info.configure(text="Capturando 4.0 segundos desde micrófono...")
        self.progress_score.configure(progress_color=ACCENT)
        self.progress_score.set(0.0)

        selected_label = self.combo_mic.get()
        device_id = self.devices_map.get(selected_label, None)

        threading.Thread(target=self._process_audio_task, args=(device_id,), daemon=True).start()

    def _process_audio_task(self, device_id):
        try:
            # Verificar que el modelo esté cargado
            if self.matcher is None:
                raise RuntimeError("El modelo no está cargado.")
            # Verificar que se eligió un micrófono válido
            if device_id is None:
                raise RuntimeError("No se ha seleccionado un micrófono válido.")

            wav_path = record_audio_to_wav(
                output_path="temp_mic_capture.wav",
                duration_sec=4.0,
                device_id=device_id,
            )
            self.after(0, lambda: self.btn_record.configure(text="Analizando huella vocal..."))

            ident, score, candidate, _ = self.matcher.identify_speaker(wav_path)
            self.after(0, lambda: self._display_result(ident, score, candidate))
        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda msg=err_msg: self._display_error(msg))

    def _display_result(self, ident, score, candidate):
        self.is_recording = False
        self.btn_record.configure(state="normal", text=BTN_TEXT)

        # Escalar score entre 0 y 1 para la barra
        clamped_score = max(0.0, min(1.0, score))
        identified = ident != "Desconocido"
        color = OK if identified else BAD

        self.progress_score.configure(progress_color=color)
        self.progress_score.set(clamped_score)
        self.lbl_candidate.configure(text=str(candidate))
        self.lbl_score.configure(text=f"{score:.4f}", text_color=color)

        if identified:
            self._set_badge(f"Identificado: {ident}", "ok")
        else:
            self._set_badge("Acceso denegado: desconocido", "bad")

        # Mostrar umbral sólo si el modelo está cargado
        threshold_str = f"{self.matcher.threshold:.2f}" if self.matcher is not None else "—"
        self.lbl_info.configure(text=f"Umbral usado: {threshold_str}")

    def _display_error(self, err_msg):
        self.is_recording = False
        self.btn_record.configure(state="normal", text=BTN_TEXT)
        self._set_badge("Error de captura", "bad")
        self.lbl_info.configure(text=err_msg)


if __name__ == "__main__":
    app = SpeakerRecognitionApp()
    app.mainloop()