# src/data/audio_processor.py
import torch
import soundfile as sf

def load_and_normalize_audio(file_path, target_sr=16000):
    """
    Carga un archivo audio WAV desde disco, convierte a mono si es necesario,
    garantiza la frecuencia de muestreo objetivo y devuelve un tensor PyTorch (1, T).
    """
    data, sr = sf.read(file_path)
    
    # 1. Convertir a PyTorch Tensor
    tensor = torch.from_numpy(data).float()
    
    # 2. Manejo de canales: Si es estereo (T, C), promediar a mono (T,)
    if tensor.ndim > 1:
        tensor = tensor.mean(dim=1)
        
    # 3. Asegurar dimensión de canal (1, T)
    tensor = tensor.unsqueeze(0)
    
    # 4. Validar frecuencia de muestreo
    if sr != target_sr:
        raise ValueError(f"Frecuencia de muestreo invalida ({sr} Hz). Se requiere {target_sr} Hz en '{file_path}'.")
        
    # 5. Normalización de amplitud pico
    max_val = torch.max(torch.abs(tensor))
    if max_val > 0:
        tensor = tensor / max_val
        
    return tensor, target_sr

if __name__ == "__main__":
    # Prueba unitaria rápida con el primer audio disponible del manifiesto
    import pandas as pd
    import os
    
    if os.path.exists("dataset_manifest.csv"):
        df = pd.read_csv("dataset_manifest.csv")
        sample_path = df.iloc[0]["file_path"]
        
        signal, sr = load_and_normalize_audio(sample_path)
        print("--- PRUEBA DE PROCESADOR DE AUDIO ---")
        print(f"Archivo probado: {sample_path}")
        print(f"Forma del tensor (Shape): {signal.shape}")
        print(f"Tipo de dato (Dtype): {signal.dtype}")
        print(f"Valor min/max de amplitud: [{signal.min():.2f}, {signal.max():.2f}]")
    else:
        print("No se encontró 'dataset_manifest.csv'. Ejecuta primero dataset_loader.py.")