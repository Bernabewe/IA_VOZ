# src/data/dataset_loader.py

# --- RAZÓN DE LAS IMPORTACIONES ---
# 1. os: Navegación del sistema de archivos local y manipulación de rutas sin problemas de S.O.
import os

# 2. pandas (pd): Estructuración de datos tabulares e inspección exportable en formato CSV.
import pandas as pd

# 3. soundfile (sf): Lectura directa de la cabecera de archivos WAV PCM a nivel de biblioteca C (libsndfile).
#    Se usa en lugar de torchaudio.info() para evitar fallos de bindings de C++ en Windows.
import soundfile as sf


def inspect_and_build_manifest(dataset_dir="dataset_voz", output_csv="dataset_manifest.csv"):
    """
    Escanea la carpeta del dataset, valida parámetros acústicos y genera
    un manifiesto estructurado en CSV.
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"No se encontró el directorio: '{dataset_dir}'")

    records = []
    print(f"Escaneando directorio '{dataset_dir}'...")

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)

                # Extraer metadatos técnicos de la cabecera WAV con soundfile
                info = sf.info(file_path)

                # Extraer metadata del nombre de archivo (ej. bernardo-Frase2-Dia1-MicA.wav)
                filename_no_ext = os.path.splitext(file)[0]
                parts = filename_no_ext.split("-")

                speaker = parts[0].lower() if len(parts) > 0 else "desconocido"
                frase = parts[1] if len(parts) > 1 else "N/A"
                dia = parts[2] if len(parts) > 2 else "N/A"
                mic = parts[3] if len(parts) > 3 else "N/A"

                records.append({
                    "speaker": speaker,
                    "phrase": frase,
                    "day": dia,
                    "mic": mic,
                    "file_path": file_path,
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                    "duration_sec": round(info.duration, 2)
                })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    print("\n--- RESUMEN TÉCNICO DEL DATASET ---")
    print(f"Archivos WAV procesados: {len(df)}")
    print(f"Hablantes únicos: {df['speaker'].nunique()}")
    print(f"Frecuencias de muestreo: {df['sample_rate'].unique().tolist()} Hz")
    print(f"Canales de audio: {df['channels'].unique().tolist()}")
    print(f"Duración promedio: {df['duration_sec'].mean():.2f} s")
    print(f"Manifiesto exportado a: '{output_csv}'")

    return df


if __name__ == "__main__":
    inspect_and_build_manifest()