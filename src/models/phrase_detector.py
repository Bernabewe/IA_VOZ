# src/models/phrase_detector.py
import re
import difflib
import unicodedata
import soundfile as sf
from faster_whisper import WhisperModel


def _norm(text: str) -> str:
    """Minúsculas, sin acentos ni signos, para comparar texto de forma tolerante."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


class PhraseDetector:
    """Transcribe lo que dijo la persona y extrae el nombre que dice ser."""

    def __init__(self, model_size="small"):
        print("Cargando modelo Whisper...")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, wav_path: str) -> str:
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        segments, _ = self.model.transcribe(
            audio, language="es", beam_size=5, vad_filter=True
        )
        return " ".join(s.text.strip() for s in segments).strip()

    @staticmethod
    def parse_claim(text: str, known_speakers: list[str]):
        """
        Devuelve (hay_yo_soy, nombre_declarado).
        - hay_yo_soy: False si la frase no contiene "yo soy".
        - nombre_declarado: el hablante registrado que más se parece a lo que
          dijo después de "yo soy" (tolerante a errores de transcripción),
          o la palabra tal cual si no coincide con nadie.
        """
        norm_text = _norm(text)
        m = re.search(r"\byo\s*(?:soy|soi)\b\s*(.*)", norm_text)
        if not m:
            return False, None

        words = m.group(1).split()
        if not words:
            return True, ""

        norm_to_orig = {_norm(s): s for s in known_speakers}
        for w in words[:3]:  # revisa las primeras palabras tras "yo soy"
            hit = difflib.get_close_matches(w, list(norm_to_orig), n=1, cutoff=0.75)
            if hit:
                return True, norm_to_orig[hit[0]]
        return True, words[0]

    @staticmethod
    def same_person(a: str, b: str) -> bool:
        return _norm(a) == _norm(b)