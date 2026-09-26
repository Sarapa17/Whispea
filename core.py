"""Hilos de trabajo (QThread) para transcripción, resumen y precarga de modelo."""
import os
import json
import shutil
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

from PyQt5.QtCore import QThread, pyqtSignal
from faster_whisper import WhisperModel, BatchedInferencePipeline

from config import (
    OLLAMA_MODEL,
    PROMPT_TEMPLATE,
    MAX_SUMMARY_INPUT_CHARS,
    TRANSLATIONS,
)


class TranscriptionThread(QThread):
    """Hilo para realizar la transcripción en segundo plano"""
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(str, str, bool)  # audio_path, text, success
    error_signal = pyqtSignal(str)

    def __init__(self, audio_path, model_size, history_dir, language, ui_language, model=None, batched_model=None):
        super().__init__()
        self.audio_path = audio_path
        self.model_size = model_size
        self.history_dir = history_dir
        self.language = language
        self.ui_language = ui_language
        self._model = model
        self._batched_model = batched_model

    def run(self):
        try:
            model = self._model
            batched_model = self._batched_model
            if model is None:
                self.progress_signal.emit(10, TRANSLATIONS[self.ui_language]["loading_model"])
                model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=8,
                )
                batched_model = BatchedInferencePipeline(model=model)

            self.progress_signal.emit(15, TRANSLATIONS[self.ui_language]["transcribing"])

            if self.language == "auto":
                segments, info = batched_model.transcribe(
                    self.audio_path,
                    batch_size=8,
                    beam_size=1,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
            else:
                segments, info = batched_model.transcribe(
                    self.audio_path,
                    language=self.language,
                    batch_size=8,
                    beam_size=1,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )

            text_parts = []
            duration = getattr(info, "duration", 0) or 0
            for segment in segments:
                text_parts.append(segment.text)
                if duration > 0:
                    progress = 15 + int(75 * segment.end / duration)
                    self.progress_signal.emit(min(progress, 90), TRANSLATIONS[self.ui_language]["transcribing"])

            text = "".join(text_parts).strip()

            self.progress_signal.emit(90, TRANSLATIONS[self.ui_language]["saving_results"])
            # Guardar en historial
            success = self.save_to_history(self.audio_path, text, self.language)

            self.finished_signal.emit(self.audio_path, text, success)

        except Exception as e:
            self.error_signal.emit(f"{TRANSLATIONS[self.ui_language]['unexpected_error']}: {str(e)}")

    def save_to_history(self, audio_path, text, language):
        try:
            # Crear directorio de historial si no existe
            os.makedirs(self.history_dir, exist_ok=True)

            # Nombre base para los archivos
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"audio_{timestamp}"

            # Copiar archivo de audio al historial
            audio_ext = Path(audio_path).suffix
            history_audio_path = os.path.join(self.history_dir, f"{base_name}{audio_ext}")
            shutil.copy2(audio_path, history_audio_path)

            # Guardar texto transcrito
            text_path = os.path.join(self.history_dir, f"{base_name}.txt")
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(f"Transcripción del audio: {Path(audio_path).name}\n")
                f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Modelo: {self.model_size}\n")
                f.write(f"Idioma: {language}\n")
                f.write("=" * 60 + "\n\n")
                f.write(text)

            # Guardar metadatos en JSON
            metadata = {
                "audio_file": f"{base_name}{audio_ext}",
                "text_file": f"{base_name}.txt",
                "original_name": Path(audio_path).name,
                "timestamp": datetime.now().isoformat(),
                "model": self.model_size,
                "language": language,
                "text_preview": text[:100] + "..." if len(text) > 100 else text
            }

            metadata_path = os.path.join(self.history_dir, f"{base_name}.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            return True

        except Exception:
            return False


class SummaryThread(QThread):
    """Hilo para generar el resumen con Ollama en segundo plano"""
    chunk_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str, str, bool)  # text_path, summary, success
    error_signal = pyqtSignal(str)

    def __init__(self, text, text_path):
        super().__init__()
        self.text = text[:MAX_SUMMARY_INPUT_CHARS]
        self.text_path = text_path

    def run(self):
        try:
            try:
                urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                self.error_signal.emit(TRANSLATIONS["es"]["ollama_error"])
                return

            try:
                body = json.dumps({
                    "model": OLLAMA_MODEL,
                    "prompt": PROMPT_TEMPLATE.format(texto=self.text),
                    "stream": True,
                    "options": {"num_ctx": 8192, "temperature": 0.2, "top_p": 0.9}
                }, ensure_ascii=False).encode("utf-8")
                req = urllib.request.Request(
                    "http://localhost:11434/api/generate",
                    data=body,
                    headers={"Content-Type": "application/json"}
                )

                full_summary = []
                with urllib.request.urlopen(req, timeout=300) as resp:
                    for raw_line in resp:
                        line = raw_line.decode("utf-8").strip()
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if "error" in chunk:
                            self.error_signal.emit(
                                f"{TRANSLATIONS['es']['summary_error']}: {chunk['error']}"
                            )
                            return

                        piece = chunk.get("response", "")
                        if piece:
                            full_summary.append(piece)
                            self.chunk_signal.emit(piece)

                        if chunk.get("done"):
                            break

            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                self.error_signal.emit(f"{TRANSLATIONS['es']['summary_error']}: {str(e)}")
                return

            resumen = "".join(full_summary)
            self.finished_signal.emit(self.text_path, resumen, True)

        except Exception as e:
            self.error_signal.emit(f"{TRANSLATIONS['es']['summary_error']}: {str(e)}")


class ModelPreloadThread(QThread):
    finished_signal = pyqtSignal(str, object)  # model_size, model
    error_signal = pyqtSignal(str)

    def __init__(self, model_size):
        super().__init__()
        self.model_size = model_size

    def run(self):
        try:
            model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=8,
            )
            self.finished_signal.emit(self.model_size, model)
        except Exception as e:
            self.error_signal.emit(str(e))
