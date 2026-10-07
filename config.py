"""Configuración, preferencias e internacionalización de Whispea."""
import os
import json

from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QIcon, QPainter, QPainterPath, QPixmap, QColor


PREFS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prefs.json")

OLLAMA_MODEL = "qwen2.5:3b"
PROMPT_TEMPLATE = (
    "Sos un asistente que resume textos. Devolvé ÚNICAMENTE el resumen en el idioma del texto original, "
    "en un párrafo conciso y claro.\n"
    "REGLAS ESTRICTAS:\n"
    "- NO escribas introducciones, saludos ni confirmaciones (nada de \"claro\", \"aquí está\", \"este es el resumen\", etc.).\n"
    "- NO repitas estas instrucciones.\n"
    "- NO uses Markdown ni títulos.\n"
    "- Empezá directamente con el contenido del resumen.\n\n"
    "Texto a resumir:\n{texto}\n\n"
    "Resumen:"
)
MAX_SUMMARY_INPUT_CHARS = 25000


def load_prefs():
    defaults = {
        "model_size": "distil-large-v3",
        "transcription_language": "auto",
        "ui_language": "es",
    }
    try:
        with open(PREFS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        defaults.update(data)
        if defaults.get("model_size") == "large":
            defaults["model_size"] = "distil-large-v3"
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return defaults


def save_prefs(prefs):
    try:
        with open(PREFS_PATH, 'w', encoding='utf-8') as f:
            json.dump(prefs, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def make_flag_icon(country, w=48, h=32):
    """Dibuja la bandera de España o USA en alta resolución (los emoji de bandera no renderizan en Windows)."""
    scale = 3
    pm = QPixmap(w * scale, h * scale)
    pm.setDevicePixelRatio(scale)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    try:
        p.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(0.5, 0.5, w - 1, h - 1, 5.0, 5.0)
        p.setClipPath(path)
        if country == "es":
            p.fillRect(QRectF(0, 0, w, h), QColor(170, 21, 27))
            p.fillRect(QRectF(0, h / 4, w, h / 2), QColor(241, 191, 0))
        else:
            stripe_h = h / 13.0
            p.fillRect(QRectF(0, 0, w, h), Qt.white)
            for i in range(13):
                if i % 2 == 0:
                    p.fillRect(QRectF(0, i * stripe_h, w, stripe_h + 0.5), QColor(179, 25, 66))
            canton_w = w * 0.42
            canton_h = stripe_h * 7
            p.fillRect(QRectF(0, 0, canton_w, canton_h), QColor(10, 49, 97))
            p.setPen(Qt.NoPen)
            p.setBrush(Qt.white)
            cols, rows = 6, 4
            for r in range(rows):
                for c_ in range(cols):
                    x = canton_w * (c_ + 0.75) / (cols + 0.5)
                    y = canton_h * (r + 0.75) / (rows + 0.5)
                    d = min(canton_w / (cols + 0.5), canton_h / (rows + 0.5)) * 0.32
                    p.drawEllipse(QRectF(x - d / 2, y - d / 2, d, d))
        p.setClipping(False)
        p.setBrush(Qt.NoBrush)
        p.setPen(QColor(0, 0, 0, 70))
        p.drawPath(path)
    finally:
        p.end()
    return QIcon(pm)


def load_translations():
    try:
        with open('translations.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback a traducciones básicas si el archivo no existe
        return {
            "es": {
                "app_title": "Whispea",
                "audio_transcription": "Transcripción de Audio",
                "whisper_model": "Modelo Whisper:",
                "select_audio": "Seleccionar archivo de audio",
                "drag_audio": "O arrastra un archivo de audio aquí",
                "ready": "Listo para transcribir",
                "transcription_history": "Historial de Transcripciones",
                "transcribed_text": "Texto Transcrito",
                "save_text": "Guardar texto",
                "copy_text": "Copiar texto",
                "clear": "Limpiar",
                "language": "Idioma:",
                "transcription_language": "Idioma de transcripción:",
                "auto_detect": "Detección automática",
                "loading_model": "Cargando modelo Whisper...",
                "transcribing": "Transcribiendo...",
                "saving_results": "Guardando resultados...",
                "transcription_complete": "Transcripción completada exitosamente",
                "error_saving": "Error al guardar en historial",
                "no_text_warning": "No hay texto para guardar.",
                "no_text_copy": "No hay texto para copiar.",
                "text_saved": "Texto guardado correctamente.",
                "text_copied": "Texto copiado al portapapeles.",
                "invalid_file": "Archivo no válido",
                "select_valid_audio": "Por favor, selecciona un archivo de audio válido.",
                "error": "Error",
                "success": "Éxito",
                "warning": "Advertencia",
                "starting_transcription": "Iniciando transcripción...",
                "unexpected_error": "Error inesperado",
                "load_error": "No se pudo cargar el texto",
                "summarize_ai": "Resumir con IA",
                "summarized_text": "Texto Resumido",
                "copy_summary": "Copiar resumen",
                "summarizing_ai": "Resumiendo con IA...",
                "summary_complete": "Resumen completado",
                "ollama_error": "Ollama no está disponible. Instalalo con 'winget install Ollama.Ollama' y asegurate de que esté corriendo.",
                "no_text_summary": "No hay texto para resumir.",
                "summary_error": "Error al generar el resumen"
            },
            "en": {
                "app_title": "Whispea",
                "audio_transcription": "Audio Transcription",
                "whisper_model": "Whisper Model:",
                "select_audio": "Select audio file",
                "drag_audio": "Or drag an audio file here",
                "ready": "Ready to transcribe",
                "transcription_history": "Transcription History",
                "transcribed_text": "Transcribed Text",
                "save_text": "Save text",
                "copy_text": "Copy text",
                "clear": "Clear",
                "language": "Language:",
                "transcription_language": "Transcription language:",
                "auto_detect": "Auto detect",
                "loading_model": "Loading Whisper model...",
                "transcribing": "Transcribing...",
                "saving_results": "Saving results...",
                "transcription_complete": "Transcription completed successfully",
                "error_saving": "Error saving to history",
                "no_text_warning": "No text to save.",
                "no_text_copy": "No text to copy.",
                "text_saved": "Text saved successfully.",
                "text_copied": "Text copied to clipboard.",
                "invalid_file": "Invalid file",
                "select_valid_audio": "Please select a valid audio file.",
                "error": "Error",
                "success": "Success",
                "warning": "Warning",
                "starting_transcription": "Starting transcription...",
                "unexpected_error": "Unexpected error",
                "load_error": "Could not load text",
                "summarize_ai": "Summarize with AI",
                "summarized_text": "Summarized Text",
                "copy_summary": "Copy summary",
                "summarizing_ai": "Summarizing with AI...",
                "summary_complete": "Summary completed",
                "ollama_error": "Ollama is not available. Install it with 'winget install Ollama.Ollama' and make sure it is running.",
                "no_text_summary": "There is no text to summarize.",
                "summary_error": "Error generating summary"
            }
        }


def load_whisper_languages():
    try:
        with open('whisper_languages.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback a idiomas básicos si el archivo no existe
        return {
            "auto": "Detección automática",
            "es": "Español",
            "en": "English",
            "fr": "Français",
            "de": "Deutsch",
            "it": "Italiano",
            "pt": "Português",
            "ja": "日本語",
            "zh": "中文",
            "ru": "Русский"
        }


TRANSLATIONS = load_translations()
WHISPER_LANGUAGES = load_whisper_languages()
