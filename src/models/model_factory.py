# src/models/model_factory.py

# Un solo lugar para registrar los modelos disponibles.
# "threshold" es provisional: se ajusta en el Paso 7 con la evaluación.
MODEL_CONFIG = {
    "ECAPA-TDNN": {
        "centroids": "embeddings_db/centroids_ecapa.pt",
        "threshold": 0.59,
    },
    "ERes2NetV2": {
        "centroids": "embeddings_db/centroids_eres2netv2.pt",
        "threshold": 0.59,
    },
}

DEFAULT_MODEL = "ECAPA-TDNN"


def available_models():
    return list(MODEL_CONFIG.keys())


def create_extractor(model_name: str):
    """Devuelve el extractor del modelo pedido (import diferido para cargar solo lo necesario)."""
    if model_name == "ECAPA-TDNN":
        from src.models.feature_extractor import ECAPAFeatureExtractor
        return ECAPAFeatureExtractor()
    if model_name == "ERes2NetV2":
        from src.models.eres2net_extractor import ERes2NetV2FeatureExtractor
        return ERes2NetV2FeatureExtractor()
    raise ValueError(f"Modelo desconocido: '{model_name}'. Opciones: {available_models()}")