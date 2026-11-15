# src/data/mic_recorder.py
import os
import torch
import torchaudio
import sounddevice as sd
import soundfile as sf
import numpy as np


def list_input_devices():
    """
    Lista micrófonos disponibles priorizando APIs estables en Windows (WASAPI / DirectSound)
    para evitar errores WdmSyncIoctl (-9999) en auriculares USB.
    """
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    input_devices = []

    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            api_name = hostapis[dev["hostapi"]]["name"]
            if "WASAPI" in api_name or "DirectSound" in api_name:
                input_devices.append({
                    "id": idx,
                    "name": f"{dev['name']} ({api_name})",
                    "default_sr": int(dev["default_samplerate"])
                })

    # Respaldo si el sistema solo expone MME
    if not input_devices:
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                api_name = hostapis[dev["hostapi"]]["name"]
                input_devices.append({
                    "id": idx,
                    "name": f"{dev['name']} ({api_name})",
                    "default_sr": int(dev["default_samplerate"])
                })

    return input_devices


def record_audio_to_wav(output_path="temp_mic_capture.wav", duration_sec=4.0, target_sr=16000, device_id=None):
    """
    Captura audio desde el micrófono seleccionado respetando su frecuencia y canales nativos,
    convierte a mono y remuestrea a 16kHz para ECAPA-TDNN.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    dev_info = sd.query_devices(device_id, kind="input")
    capture_sr = int(dev_info["default_samplerate"])
    # Usar 1 canal si lo soporta, o los canales nativos máximos (máx. 2) para evitar PaErrorCode -9999
    channels = 1
    try:
        sd.check_input_settings(device=device_id, samplerate=capture_sr, channels=1)
    except Exception:
        channels = min(2, int(dev_info["max_input_channels"]))

    print(f"Micrófono activo: [{dev_info['name']}] @ {capture_sr} Hz ({channels} ch)")
    print(f"Grabando durante {duration_sec} segundos... ¡Habla ahora!")

    recording = sd.rec(
        int(duration_sec * capture_sr),
        samplerate=capture_sr,
        channels=channels,
        dtype="float32",
        device=device_id
    )
    sd.wait()
    print("Grabación finalizada.")

    # Convertir a tensor (channels, time) y reducir a mono si es estéreo
    tensor_audio = torch.from_numpy(recording.T)
    if tensor_audio.shape[0] > 1:
        tensor_audio = torch.mean(tensor_audio, dim=0, keepdim=True)

    # Remuestrear a 16000 Hz si la frecuencia nativa es distinta
    if capture_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=capture_sr, new_freq=target_sr)
        tensor_audio = resampler(tensor_audio)

    mono_16k = tensor_audio.squeeze(0).numpy()
    sf.write(output_path, mono_16k, target_sr)
    return output_path