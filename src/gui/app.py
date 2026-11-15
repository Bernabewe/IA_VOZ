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


class SpeakerRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sistema Biométrico de Identificación de Hablante (1:N)")
        self.geometry("680x540")
        self.resizable(False, False)

        self.matcher = None
        self.devices_map = {}
        self.is_recording = False

        self._build_ui()
        self._load_devices()
        # Cargar el modelo ECAPA-TDNN en segundo plano al abrir la app
        threading.Thread(target=self._init_model, daemon=True).start()

    def _build_ui(self):
        # Título principal
        self.lbl_title = ctk.CTkLabel(
            self, text="Identificación Biométrica por Voz (ECAPA-TDNN)",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.lbl_title.pack(pady=(20, 5))

        self.lbl_status = ctk.CTkLabel(
            self, text="Cargando modelo neuronal y centroides...",
            text_color="#F59E0B", font=ctk.CTkFont(size=13)
        )
        self.lbl_status.pack(pady=(0, 15))

        # Marco de configuración de Hardware y Umbral
        frame_config = ctk.CTkFrame(self)
        frame_config.pack(fill="x", padx=30, pady=10)

        # Selector de Micrófono
        lbl_mic = ctk.CTkLabel(frame_config, text="Dispositivo de Entrada (Micrófono):", font=ctk.CTkFont(weight="bold"))
        lbl_mic.pack(anchor="w", padx=15, pady=(12, 2))

        mic_row = ctk.CTkFrame(frame_config, fg_color="transparent")
        mic_row.pack(fill="x", padx=15, pady=(0, 10))

        self.combo_mic = ctk.CTkComboBox(mic_row, width=480, state="readonly")
        self.combo_mic.pack(side="left", padx=(0, 10))

        self.btn_refresh_mic = ctk.CTkButton(
            mic_row, text="Actualizar", width=90, command=self._load_devices
        )
        self.btn_refresh_mic.pack(side="left")

        # Control de Umbral (Threshold)
        self.lbl_thresh = ctk.CTkLabel(
            frame_config, text="Umbral de Decisión (Tau): 0.59", font=ctk.CTkFont(weight="bold")
        )
        self.lbl_thresh.pack(anchor="w", padx=15, pady=(5, 2))

        self.slider_thresh = ctk.CTkSlider(
            frame_config, from_=30, to=85, number_of_steps=55,
            command=self._on_threshold_change
        )
        # Initialize with default threshold 0.59 (scaled to integer range)
        self.slider_thresh.set(59)
        self.slider_thresh.pack(fill="x", padx=15, pady=(0, 15))

        # Botón principal de grabación
        self.btn_record = ctk.CTkButton(
            self, text="Grabar e Identificar (4.0 seg)",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=48, width=280, state="disabled",
            command=self._start_recognition_thread
        )
        self.btn_record.pack(pady=15)

        # Panel de Resultados
        frame_res = ctk.CTkFrame(self)
        frame_res.pack(fill="both", expand=True, padx=30, pady=(5, 25))

        self.lbl_verdict = ctk.CTkLabel(
            frame_res, text="VEREDICTO: EN ESPERA",
            font=ctk.CTkFont(size=22, weight="bold"), text_color="#9CA3AF"
        )
        self.lbl_verdict.pack(pady=(20, 8))

        self.lbl_details = ctk.CTkLabel(
            frame_res, text="Candidato más cercano: --- | Similitud Coseno: 0.0000",
            font=ctk.CTkFont(size=14)
        )
        self.lbl_details.pack(pady=(0, 10))

        self.progress_score = ctk.CTkProgressBar(frame_res, width=450, height=14)
        self.progress_score.set(0.0)
        self.progress_score.pack(pady=(5, 15))

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
        self.lbl_thresh.configure(text=f"Umbral de Decisión (Tau): {scaled_val:.2f}")
        if self.matcher is not None:
            self.matcher.threshold = scaled_val

    def _init_model(self):
        try:
            initial_tau = round(float(self.slider_thresh.get()) / 100.0, 2)
            self.matcher = SpeakerMatcher(threshold=initial_tau)
            self.after(0, self._on_model_ready)
        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda msg=err_msg: self.lbl_status.configure(text=f"Error cargando modelo: {msg}", text_color="#EF4444"))

    def _on_model_ready(self):
        self.lbl_status.configure(text="Sistema Listo (28 Hablantes en Base de Datos)", text_color="#10B981")
        self.btn_record.configure(state="normal")

    def _start_recognition_thread(self):
        if self.is_recording or self.matcher is None:
            return
        self.is_recording = True
        self.btn_record.configure(state="disabled", text="Grabando... ¡Habla ahora!")
        self.lbl_verdict.configure(text="ESCUCHANDO AUDIO...", text_color="#3B82F6")
        self.lbl_details.configure(text="Capturando 4.0 segundos desde micrófono...")
        self.progress_score.set(0.0)

        selected_label = self.combo_mic.get()
        device_id = self.devices_map.get(selected_label, None)

        threading.Thread(target=self._process_audio_task, args=(device_id,), daemon=True).start()

    def _process_audio_task(self, device_id):
        try:
            # Verify the model is loaded
            if self.matcher is None:
                raise RuntimeError("El modelo no está cargado.")
            # Verify a valid microphone device was selected
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
        self.btn_record.configure(state="normal", text="Grabar e Identificar (4.0 seg)")

        # Escalar score entre 0 y 1 para la barra de progreso
        clamped_score = max(0.0, min(1.0, score))
        self.progress_score.set(clamped_score)

        if ident != "Desconocido":
            self.lbl_verdict.configure(text=f"IDENTIFICADO: {ident.upper()}", text_color="#10B981")
        else:
            self.lbl_verdict.configure(text="ACCESO DENEGADO: DESCONOCIDO", text_color="#EF4444")

        # Mostrar umbral sólo si el modelo está cargado
        threshold_str = f"{self.matcher.threshold:.2f}" if self.matcher is not None else "—"
        self.lbl_details.configure(
            text=f"Candidato más cercano: {candidate} | Similitud Coseno: {score:.4f} (Umbral: {threshold_str})"
        )

    def _display_error(self, err_msg):
        self.is_recording = False
        self.btn_record.configure(state="normal", text="Grabar e Identificar (4.0 seg)")
        self.lbl_verdict.configure(text="ERROR DE CAPTURA", text_color="#EF4444")
        self.lbl_details.configure(text=err_msg)


if __name__ == "__main__":
    app = SpeakerRecognitionApp()
    app.mainloop()