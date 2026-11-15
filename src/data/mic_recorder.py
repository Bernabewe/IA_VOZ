# src/data/mic_recorder.py
import os
import torch
import torchaudio
import sounddevice as sd
import soundfile as sf


def list_input_devices():
    """
    Lista todos los micrófonos disponibles en el sistema (integrados y externos).
    Retorna una lista de diccionarios con id, nombre y frecuencia predeterminada.
    """
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    input_devices = []

    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            api_name = hostapis[dev["hostapi"]]["name"]
            # Filtrar principalmente MME o WASAPI para evitar duplicados excesivos en Windows
            if "MME" in api_name or "WASAPI" in api_name:
                input_devices.append({
                    "id": idx,
                    "name": f"{dev['name']} ({api_name})",
                    "default_sr": int(dev["default_samplerate"])
                })
    return input_devices


def record_audio_to_wav(output_path="temp_mic_capture.wav", duration_sec=4.0, target_sr=16000, device_id=None):
    """
    Graba audio desde el micrófono especificado (o el predeterminado si device_id=None).
    Si el hardware no soporta 16kHz nativos, captura a su frecuencia nativa y remuestrea a 16kHz mono.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Determinar frecuencia nativa del dispositivo seleccionado
    dev_info = sd.query_devices(device_id, kind="input")
    native_sr = int(dev_info["default_samplerate"])
    capture_sr = target_sr

    # Verificar si el micrófono acepta 16000 Hz directamente; si no, usar su frecuencia nativa
    try:
        sd.check_input_settings(device=device_id, samplerate=target_sr, channels=1)
    except Exception:
        capture_sr = native_sr

    print(f"Micrófono activo: [{dev_info['name']}] @ {capture_sr} Hz")
    print(f"Grabando durante {duration_sec} segundos... ¡Habla ahora!")

    recording = sd.rec(
        int(duration_sec * capture_sr),
        samplerate=capture_sr,
        channels=1,
        dtype="float32",
        device=device_id
    )
    sd.wait()
    print("Grabación finalizada.")

    # Si se capturó a una frecuencia distinta a 16000 Hz, remuestrear con PyTorch
    if capture_sr != target_sr:
        tensor_audio = torch.from_numpy(recording.T)  # (1, T)
        resampler = torchaudio.transforms.Resample(orig_freq=capture_sr, new_freq=target_sr)
        tensor_audio = resampler(tensor_audio)
        recording = tensor_audio.squeeze(0).numpy()

    sf.write(output_path, recording, target_sr)
    return output_path


if __name__ == "__main__":
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.models.speaker_matcher import SpeakerMatcher

    print("--- DISPOSITIVOS DE ENTRADA DETECTADOS ---")
    for d in list_input_devices():
        print(f"ID: {d['id']} | {d['name']} (Default SR: {d['default_sr']} Hz)")
    print("-" * 45)

    # Grabar usando el micrófono predeterminado (device_id=None)
    wav_file = record_audio_to_wav(duration_sec=4.0, device_id=None)

    matcher = SpeakerMatcher(threshold=0.5908)
    ident, score, candidate, _ = matcher.identify_speaker(wav_file)

    print("\n--- PRUEBA DE IDENTIFICACIÓN EN VIVO (MICRÓFONO) ---")
    print(f"Candidato más cercano: {candidate}")
    print(f"Similitud Coseno obtenida: {score:.4f}")
    print(f"Veredicto Final (Umbral = 0.5908): {ident}")