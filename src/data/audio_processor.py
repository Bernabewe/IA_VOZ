# src/data/audio_processor.py
import torch
import torchaudio
import soundfile as sf


def trim_silence_energy(waveform: torch.Tensor, sr: int = 16000, frame_ms: int = 25, top_db: float = 20.0) -> torch.Tensor:
    """
    Elimina tramas de silencio y valida que exista contraste dinámico real
    propio de la voz humana (descartando estática o ruido estacionario de AGC).
    """
    sig = waveform.squeeze(0) if waveform.dim() == 2 else waveform

    frame_len = int(sr * frame_ms / 1000)
    if sig.numel() <= frame_len:
        return waveform

    num_frames = sig.numel() // frame_len
    trimmed_len = num_frames * frame_len
    frames = sig[:trimmed_len].view(num_frames, frame_len)

    rms = torch.sqrt(torch.mean(frames ** 2, dim=1) + 1e-12)
    rms_db = 20.0 * torch.log10(rms + 1e-12)

    # Validar contraste dinámico entre tramas altas (P90) y tramas bajas (P20)
    p90_db = torch.quantile(rms_db, 0.90).item()
    p20_db = torch.quantile(rms_db, 0.20).item()
    dynamic_range = p90_db - p20_db
    max_db = torch.max(rms_db).item()

    print(f"[AudioProcessor] RMS Global: {torch.sqrt(torch.mean(sig**2)).item():.5f} | Rango Dinámico: {dynamic_range:.2f} dB")

    if dynamic_range < 8.0:
        raise ValueError(f"Ruido estacionario o silencio detectado (Contraste: {dynamic_range:.1f} dB). Habla más cerca del micrófono.")

    active_mask = rms_db >= (max_db - top_db)

    if torch.sum(active_mask) * frame_len < int(0.6 * sr):
        raise ValueError("Señal de voz insuficiente o muy corta. Habla más claro durante la captura.")

    active_audio = frames[active_mask].reshape(-1)
    return active_audio.unsqueeze(0)


def load_and_normalize_audio(file_path: str, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    """
    Carga un archivo WAV, valida energía y rango dinámico vocal, remuestrea a 16kHz,
    elimina silencios y normaliza amplitud para ECAPA-TDNN.
    """
    data, sr = sf.read(file_path, dtype="float32")
    waveform = torch.from_numpy(data)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    # 1. Remover DC offset
    waveform = waveform - torch.mean(waveform)

    # 2. Verificar energía global mínima real (evita silencio y ruido débil de AGC)
    global_rms = torch.sqrt(torch.mean(waveform ** 2)).item()
    if global_rms < 0.010:
        print(f"[AudioProcessor] Rechazado por RMS bajo: {global_rms:.5f}")
        raise ValueError("No se detectó voz (micrófono silenciado o volumen muy bajo).")

    # 3. Validar contraste dinámico vocal y recortar silencios
    waveform = trim_silence_energy(waveform, sr=sr, top_db=20.0)

    # 4. Normalización de amplitud pico segura (0.95)
    max_amp = torch.max(torch.abs(waveform))
    if max_amp > 1e-6:
        waveform = (waveform / max_amp) * 0.95

    return waveform, sr