# src/gui/app.py
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import threading
import customtkinter as ctk
from src.data.mic_recorder import list_input_devices, record_audio_to_wav
from src.models.speaker_matcher import SpeakerMatcher
from src.models.phrase_detector import PhraseDetector
from src.models.model_factory import MODEL_CONFIG, DEFAULT_MODEL, available_models
from src.gui.comparison_view import ComparisonPanel

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
        self.resizable(True, True)          # ahora se puede agrandar o achicar
        self.minsize(560, 420)              # tamaño mínimo para que no se deforme
        self.configure(fg_color=BG)
        self._fit_window(820, 830)          # tamaño deseado, ajustado a la pantalla

        # Contenedor con scroll: todo el contenido va dentro de este marco
        self.body = ctk.CTkScrollableFrame(self, fg_color="transparent", corner_radius=0)
        self.body.pack(fill="both", expand=True)

        # Modelo activo y caché de matchers (uno por modelo, se cargan bajo demanda)
        self.current_model = DEFAULT_MODEL
        self.matchers = {}
        self.matcher = None
        self.detector = None
        # Cada modelo recuerda su propio umbral (los espacios de similitud son distintos)
        self.model_thresholds = {n: MODEL_CONFIG[n]["threshold"] for n in available_models()}

        self.devices_map = {}
        self.is_recording = False
        self.is_loading = False

        self._build_ui()
        self._load_devices()
        # Cargar modelos en segundo plano al abrir la app
        threading.Thread(target=self._init_model, daemon=True).start()

    # ---------------------------------------------------------------
    # NUEVO: ajustar la ventana al tamaño de la pantalla
    # ---------------------------------------------------------------
    def _fit_window(self, want_w=820, want_h=830):
        """Abre la ventana del tamaño que quieras, pero sin pasarse del tamaño de la pantalla."""
        self.update_idletasks()
        # Escala de Windows (100%, 125%, 150%...). Si no se puede leer, se asume 1.0
        try:
            scale = self._get_window_scaling()
        except Exception:
            scale = 1.0
        screen_w = int(self.winfo_screenwidth() / scale)
        screen_h = int(self.winfo_screenheight() / scale)

        # Dejar espacio para la barra de tareas y la barra de título
        w = min(want_w, screen_w - 40)
        h = min(want_h, screen_h - 110)
        self.geometry(f"{w}x{h}")

        # Centrar la ventana en la pantalla
        self.update_idletasks()
        x = max((self.winfo_screenwidth() - self.winfo_width()) // 2, 0)
        y = max((self.winfo_screenheight() - self.winfo_height()) // 2 - 30, 0)
        self.wm_geometry(f"+{x}+{y}")

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
        """Tarjeta con esquinas redondeadas y borde fino (dentro del marco con scroll)."""
        return ctk.CTkFrame(
            self.body, fg_color=CARD, corner_radius=16,
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

    def _set_controls(self, enabled):
        """Habilita o deshabilita el selector de modelo y el botón de grabar."""
        state = "normal" if enabled else "disabled"
        self.seg_model.configure(state=state)
        self.btn_record.configure(state=state)

    def _ready_text(self):
        n = len(self.matcher.speaker_names) if self.matcher is not None else 0
        return f"Listo · {self.current_model} · {n} hablantes"

    # ---------------------------------------------------------------
    # PASO 2: Construcción de la interfaz
    # ---------------------------------------------------------------
    def _build_ui(self):
        # ----- Encabezado: logo + título a la izquierda, estado a la derecha
        header = ctk.CTkFrame(self.body, fg_color="transparent")
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
            titles, text="Biometría de hablante · ECAPA-TDNN / ERes2NetV2 / TitaNet-Large · 1:N",
            text_color=MUTED, font=ctk.CTkFont(size=13)
        ).pack(anchor="w")

        self.lbl_status = ctk.CTkLabel(
            header, text="", height=32, corner_radius=16,
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.lbl_status.pack(side="right")
        self._set_status("Cargando modelos y centroides...", "loading")

        # ----- Tarjeta de configuración (modelo + micrófono + umbral)
        card_cfg = self._make_card()
        card_cfg.pack(fill="x", padx=40, pady=(22, 0))

        # Selector de modelo
        self._section_label(card_cfg, "Modelo de voz").pack(anchor="w", padx=24, pady=(20, 6))
        self.seg_model = ctk.CTkSegmentedButton(
            card_cfg, values=available_models(), height=38, corner_radius=10,
            fg_color=BG, selected_color=ACCENT, selected_hover_color="#8FBBFF",
            unselected_color="#222737", unselected_hover_color="#2C3242",
            text_color=TEXT, font=ctk.CTkFont(size=13, weight="bold"),
            command=self._on_model_change
        )
        self.seg_model.set(self.current_model)
        self.seg_model.configure(state="disabled")   # se habilita al terminar de cargar
        self.seg_model.pack(fill="x", padx=24)

        self._section_label(card_cfg, "Micrófono").pack(anchor="w", padx=24, pady=(18, 6))

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
        initial_tau = self.model_thresholds[self.current_model]
        tau_row = ctk.CTkFrame(card_cfg, fg_color="transparent")
        tau_row.pack(fill="x", padx=24, pady=(22, 6))
        self._section_label(tau_row, "Umbral de decisión (Tau)").pack(side="left")
        self.lbl_thresh = ctk.CTkLabel(
            tau_row, text=f"{initial_tau:.2f}", height=26, corner_radius=8,
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
        # El slider trabaja con enteros: 59 = 0.59
        self.slider_thresh.set(round(initial_tau * 100))
        self.slider_thresh.pack(fill="x", padx=24)

        hint_row = ctk.CTkFrame(card_cfg, fg_color="transparent")
        hint_row.pack(fill="x", padx=24, pady=(4, 18))
        ctk.CTkLabel(hint_row, text="Más permisivo", text_color=MUTED,
                     font=ctk.CTkFont(size=12)).pack(side="left")
        ctk.CTkLabel(hint_row, text="Más estricto", text_color=MUTED,
                     font=ctk.CTkFont(size=12)).pack(side="right")

        # ----- Botón principal de grabación (grande y redondeado)
        self.btn_record = ctk.CTkButton(
            self.body, text=BTN_TEXT, height=64, width=360, corner_radius=32,
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
        self.marker_tau.place(relx=initial_tau, y=0, anchor="n")
        self.lbl_tau_mark = ctk.CTkLabel(
            bar_box, text=f"Tau {initial_tau:.2f}", text_color=MUTED, font=ctk.CTkFont(size=12)
        )
        self.lbl_tau_mark.place(relx=initial_tau, y=28, anchor="n")

        # Frase detectada por Whisper
        self.lbl_phrase = ctk.CTkLabel(
            card_res, text="", text_color=TEXT, wraplength=640,
            font=ctk.CTkFont(size=15, slant="italic")
        )
        self.lbl_phrase.pack(padx=24, pady=(6, 0))

        # Línea de información extra (errores, mensajes de captura, modelo y umbral usados)
        self.lbl_info = ctk.CTkLabel(
            card_res, text="", text_color=MUTED, wraplength=640,
            font=ctk.CTkFont(size=13)
        )
        self.lbl_info.pack(padx=24, pady=(4, 12))

        # ----- Panel de comparación de los 3 modelos (misma página, mismo scroll)
        self.comparison_panel = ComparisonPanel(
            self.body,
            get_microphone=self._get_microphone,
            can_start=self._can_start_comparison,
            on_busy_change=self._on_comparison_busy,
        )
        self.comparison_panel.pack(fill="x", padx=40, pady=(0, 28))

    # ---------------------------------------------------------------
    # Conexión con el panel de comparación
    # ---------------------------------------------------------------
    def _get_microphone(self):
        """Devuelve (id_del_dispositivo, etiqueta) del micrófono elegido arriba."""
        label = self.combo_mic.get()
        return self.devices_map.get(label), label

    def _can_start_comparison(self):
        """El panel solo puede grabar si la app ya cargó y no está grabando ni cambiando de modelo."""
        return (not self.is_recording and not self.is_loading
                and self.matcher is not None and self.detector is not None)

    def _on_comparison_busy(self, busy):
        """Mientras el panel graba, se bloquean los controles de arriba (y viceversa)."""
        self.is_recording = busy
        self._set_controls(not busy)

    # ---------------------------------------------------------------
    # PASO 3: Lógica
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

    def _update_threshold_ui(self, scaled_val):
        """Actualiza el número, la marca de la barra y el texto 'Tau'."""
        self.lbl_thresh.configure(text=f"{scaled_val:.2f}")
        self.lbl_tau_mark.configure(text=f"Tau {scaled_val:.2f}")
        self.marker_tau.place_configure(relx=scaled_val)
        self.lbl_tau_mark.place_configure(relx=scaled_val)

    def _on_threshold_change(self, val):
        scaled_val = round(float(val) / 100.0, 2)
        self._update_threshold_ui(scaled_val)
        # Guardar el umbral para el modelo activo
        self.model_thresholds[self.current_model] = scaled_val
        if self.matcher is not None:
            self.matcher.threshold = scaled_val

    # ----- Carga inicial -----
    def _init_model(self):
        try:
            self.detector = PhraseDetector(model_size="small")
            matcher = SpeakerMatcher(
                model_name=self.current_model,
                threshold=self.model_thresholds[self.current_model],
            )
            self.matchers[self.current_model] = matcher
            self.matcher = matcher
            self.after(0, self._on_model_ready)
        except Exception as e:
            traceback.print_exc()
            err_msg = str(e)
            self.after(0, lambda msg=err_msg: self._set_status(f"Error cargando modelo: {msg}", "error"))

    def _on_model_ready(self):
        self._set_status(self._ready_text(), "ok")
        self._set_controls(True)

    # ----- Cambio de modelo desde el selector -----
    def _on_model_change(self, new_model):
        if new_model == self.current_model:
            return
        if self.is_recording or self.is_loading:
            self.seg_model.set(self.current_model)   # revertir selección
            return

        self.is_loading = True
        self._set_controls(False)
        self._set_status(f"Cargando {new_model}...", "loading")
        threading.Thread(target=self._switch_model_task, args=(new_model,), daemon=True).start()

    def _switch_model_task(self, new_model):
        try:
            if new_model not in self.matchers:
                self.matchers[new_model] = SpeakerMatcher(
                    model_name=new_model,
                    threshold=self.model_thresholds[new_model],
                )
            self.after(0, lambda: self._on_model_switched(new_model))
        except Exception as e:
            traceback.print_exc()
            err_msg = str(e)
            self.after(0, lambda msg=err_msg: self._on_switch_failed(msg))

    def _on_model_switched(self, new_model):
        self.current_model = new_model
        self.matcher = self.matchers[new_model]

        # Mostrar el umbral que corresponde a este modelo
        tau = self.model_thresholds[new_model]
        self.slider_thresh.set(round(tau * 100))
        self._update_threshold_ui(tau)
        self.matcher.threshold = tau

        # Limpiar la tarjeta de resultados
        self._set_badge("En espera", "wait")
        self.lbl_candidate.configure(text="—")
        self.lbl_score.configure(text="0.0000", text_color=ACCENT)
        self.lbl_phrase.configure(text="")
        self.lbl_info.configure(text=f"Modelo activo: {new_model}")
        self.progress_score.configure(progress_color=ACCENT)
        self.progress_score.set(0.0)

        self.is_loading = False
        self._set_status(self._ready_text(), "ok")
        self._set_controls(True)

    def _on_switch_failed(self, err_msg):
        # Volver al modelo que sí funciona
        self.is_loading = False
        self.seg_model.set(self.current_model)
        self._set_status(f"Error cargando modelo: {err_msg}", "error")
        self.lbl_info.configure(text=err_msg)
        if self.matcher is not None:
            self._set_controls(True)

    # ----- Grabación e identificación -----
    def _start_recognition_thread(self):
        if self.is_recording or self.is_loading or self.matcher is None or self.detector is None:
            return
        self.is_recording = True
        self._set_controls(False)
        self.btn_record.configure(text="Grabando... ¡Habla ahora!")

        # Dejar la tarjeta de resultados "limpia" para la nueva captura
        self._set_badge("Escuchando audio...", "listen")
        self.lbl_candidate.configure(text="—")
        self.lbl_score.configure(text="0.0000", text_color=ACCENT)
        self.lbl_phrase.configure(text="")
        self.lbl_info.configure(text="Capturando 4.0 segundos desde micrófono... Di «Yo soy (tu nombre)»")
        self.progress_score.configure(progress_color=ACCENT)
        self.progress_score.set(0.0)

        selected_label = self.combo_mic.get()
        device_id = self.devices_map.get(selected_label, None)

        threading.Thread(target=self._process_audio_task, args=(device_id,), daemon=True).start()

    def _process_audio_task(self, device_id):
        try:
            matcher = self.matcher
            if matcher is None or self.detector is None:
                raise RuntimeError("El modelo no está cargado.")
            if device_id is None:
                raise RuntimeError("No se ha seleccionado un micrófono válido.")

            wav_path = record_audio_to_wav(
                output_path="temp_mic_capture.wav",
                duration_sec=4.0,
                device_id=device_id,
            )
            self.after(0, lambda: self.btn_record.configure(text="Analizando voz y frase..."))

            # 1) Quién es (huella vocal, con el modelo activo)
            ident, score, candidate, _ = matcher.identify_speaker(wav_path)

            # 2) Qué dijo (transcripción) y qué nombre declaró
            phrase = self.detector.transcribe(wav_path)
            has_claim, claimed = PhraseDetector.parse_claim(phrase, matcher.speaker_names)

            self.after(0, lambda: self._display_result(
                ident, score, candidate, phrase, has_claim, claimed))
        except Exception as e:
            traceback.print_exc()
            err_msg = str(e)
            self.after(0, lambda msg=err_msg: self._display_error(msg))

    def _display_result(self, ident, score, candidate, phrase, has_claim, claimed):
        self.is_recording = False
        self.btn_record.configure(text=BTN_TEXT)
        self._set_controls(True)

        clamped_score = max(0.0, min(1.0, score))
        self.lbl_candidate.configure(text=str(candidate))
        self.lbl_phrase.configure(text=f"Frase detectada: “{phrase or '(nada)'}”")

        threshold_str = f"{self.matcher.threshold:.2f}" if self.matcher is not None else "—"
        tail = f"Modelo: {self.current_model} · Umbral: {threshold_str}"
        voice_ok = ident != "Desconocido"

        # Caso 1: no dijo "yo soy"
        if not has_claim:
            color = BAD
            self._set_badge("No se detectó «yo soy»", "bad")
            self.lbl_info.configure(text=f"Debes decir «Yo soy (tu nombre)». {tail}")

        # Caso 2: dijo "yo soy" y la voz coincide con el nombre
        elif voice_ok and PhraseDetector.same_person(ident, claimed or ""):
            color = OK
            self._set_badge(f"Identidad verificada: {ident}", "ok")
            self.lbl_info.configure(text=f"Tu voz coincide con el nombre que dijiste. {tail}")

        # Caso 3: la voz es de otra persona registrada
        elif voice_ok:
            color = BAD
            self._set_badge(f"Tú no eres quien dices ser, eres {ident}", "bad")
            self.lbl_info.configure(
                text=f"Dijiste ser «{claimed}», pero tu voz corresponde a {ident}. {tail}")

        # Caso 4: la voz no coincide con nadie
        else:
            color = BAD
            self._set_badge("Tú no eres quien dices ser", "bad")
            self.lbl_info.configure(
                text=f"Dijiste ser «{claimed}», pero tu voz no coincide con ningún hablante registrado. {tail}")

        self.progress_score.configure(progress_color=color)
        self.progress_score.set(clamped_score)
        self.lbl_score.configure(text=f"{score:.4f}", text_color=color)

    def _display_error(self, err_msg):
        self.is_recording = False
        self.btn_record.configure(text=BTN_TEXT)
        self._set_controls(True)
        self._set_badge("Error de captura", "bad")
        self.lbl_phrase.configure(text="")
        self.lbl_info.configure(text=err_msg)


if __name__ == "__main__":
    app = SpeakerRecognitionApp()
    app.mainloop()