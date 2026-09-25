import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
import json

from faster_whisper import WhisperModel, BatchedInferencePipeline

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QComboBox, QPushButton, 
                             QTextEdit, QListWidget, QFileDialog, QMessageBox,
                             QProgressBar, QSplitter, QListWidgetItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QDragEnterEvent, QDropEvent, QIcon, QPainter, QPixmap, QColor

import urllib.request
import urllib.error

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

PREFS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prefs.json")

def load_prefs():
    defaults = {
        "model_size": "large",
        "transcription_language": "auto",
        "ui_language": "es"
    }
    try:
        with open(PREFS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        defaults.update(data)
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


def make_flag_icon(country):
    """Dibuja la bandera de España o USA en un QPixmap (los emoji de bandera no renderizan en Windows)"""
    pm = QPixmap(24, 16)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    if country == "es":
        p.fillRect(0, 0, 24, 4, Qt.red)
        p.fillRect(0, 4, 24, 8, QColor(241, 191, 0))
        p.fillRect(0, 12, 24, 4, Qt.red)
    else:
        stripe_h = 16 // 13
        for i in range(13):
            if i % 2 == 0:
                p.fillRect(0, i * stripe_h, 24, stripe_h, Qt.red)
        p.fillRect(0, 0, 10, 8, QColor(60, 59, 110))
    p.end()
    return QIcon(pm)


# Cargar traducciones desde archivo externo
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

# Cargar idiomas de Whisper desde archivo externo
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

# Cargar las traducciones e idiomas
TRANSLATIONS = load_translations()
WHISPER_LANGUAGES = load_whisper_languages()

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

class AudioTranscriberApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # CAMBIO: guardar en carpeta del script
        self.history_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history")
        self.current_transcription = None
        self.current_text_path = None
        self.summary_thread = None
        self._prefs = load_prefs()
        self.current_language = self._prefs.get("ui_language", "es")
        self.model_cache = {}
        self.model_preload_thread = None
        self.initUI()
        self.load_history()
        self._start_model_preload()
        self._ensure_ollama_running()

    def initUI(self):
        self.setWindowTitle(TRANSLATIONS[self.current_language]["app_title"])
        self.setGeometry(100, 100, 1200, 800)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QHBoxLayout(central_widget)
        
        # Splitter para dividir la interfaz
        splitter = QSplitter(Qt.Horizontal)
        
        # Panel izquierdo - Controles e historial
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Título
        title_label = QLabel(TRANSLATIONS[self.current_language]["audio_transcription"])
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        left_layout.addWidget(title_label)
        
        # Selector de idioma con banderas (discreto, junto al título)
        lang_row = QHBoxLayout()
        self.btn_es = QPushButton()
        self.btn_es.setIcon(make_flag_icon("es"))
        self.btn_es.setCheckable(True)
        self.btn_es.setChecked(True)
        self.btn_es.setFixedSize(34, 26)
        self.btn_es.setToolTip("Español")
        self.btn_en = QPushButton()
        self.btn_en.setIcon(make_flag_icon("en"))
        self.btn_en.setCheckable(True)
        self.btn_en.setFixedSize(34, 26)
        self.btn_en.setToolTip("English")
        self.btn_es.clicked.connect(lambda: self.change_ui_language("es"))
        self.btn_en.clicked.connect(lambda: self.change_ui_language("en"))
        lang_row.addStretch(1)
        lang_row.addWidget(self.btn_es)
        lang_row.addWidget(self.btn_en)
        left_layout.addLayout(lang_row)
        
        # Selección de modelo
        model_layout = QHBoxLayout()
        model_label = QLabel(TRANSLATIONS[self.current_language]["whisper_model"])
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        saved_model = self._prefs.get("model_size", "large")
        if saved_model in ["tiny","base","small","medium","large"]:
            self.model_combo.setCurrentText(saved_model)
        else:
            self.model_combo.setCurrentText("large")
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        left_layout.addLayout(model_layout)
        
        # Selección de idioma de transcripción
        trans_lang_layout = QHBoxLayout()
        trans_lang_label = QLabel(TRANSLATIONS[self.current_language]["transcription_language"])
        self.trans_lang_combo = QComboBox()
        
        # Agregar idiomas soportados
        for code, name in WHISPER_LANGUAGES.items():
            self.trans_lang_combo.addItem(name, code)
            
        saved_lang = self._prefs.get("transcription_language", "auto")
        idx = self.trans_lang_combo.findData(saved_lang)
        if idx >= 0:
            self.trans_lang_combo.setCurrentIndex(idx)
        else:
            self.trans_lang_combo.setCurrentIndex(0)  # Auto-detect por defecto
        trans_lang_layout.addWidget(trans_lang_label)
        trans_lang_layout.addWidget(self.trans_lang_combo)
        left_layout.addLayout(trans_lang_layout)
        
        # Botón para seleccionar archivo
        self.select_btn = QPushButton(TRANSLATIONS[self.current_language]["select_audio"])
        self.select_btn.clicked.connect(self.select_audio_file)
        left_layout.addWidget(self.select_btn)
        
        # Área de drop
        drop_label = QLabel(TRANSLATIONS[self.current_language]["drag_audio"])
        drop_label.setAlignment(Qt.AlignCenter)
        drop_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                padding: 20px;
                background-color: #f0f0f0;
            }
        """)
        drop_label.setAcceptDrops(True)
        drop_label.dragEnterEvent = self.dragEnterEvent
        drop_label.dropEvent = self.dropEvent
        left_layout.addWidget(drop_label)
        
        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        # Etiqueta de estado
        self.status_label = QLabel(TRANSLATIONS[self.current_language]["ready"])
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        
        # Historial
        history_label = QLabel(TRANSLATIONS[self.current_language]["transcription_history"])
        history_label.setFont(QFont("Arial", 12, QFont.Bold))
        left_layout.addWidget(history_label)
        
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.load_history_item)
        left_layout.addWidget(self.history_list)
        
        # Panel derecho - Visualización de texto
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        right_title = QLabel(TRANSLATIONS[self.current_language]["transcribed_text"])
        right_title.setFont(QFont("Arial", 14, QFont.Bold))
        right_layout.addWidget(right_title)
        
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        right_layout.addWidget(self.text_edit, 2)
        
        # Botones de acción para el texto
        button_layout = QHBoxLayout()
        self.save_btn = QPushButton(TRANSLATIONS[self.current_language]["save_text"])
        self.save_btn.clicked.connect(self.save_text)
        self.copy_btn = QPushButton(TRANSLATIONS[self.current_language]["copy_text"])
        self.copy_btn.clicked.connect(self.copy_text)
        self.summarize_btn = QPushButton(TRANSLATIONS[self.current_language]["summarize_ai"])
        self.summarize_btn.clicked.connect(self.summarize_text)
        
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.copy_btn)
        button_layout.addWidget(self.summarize_btn)
        right_layout.addLayout(button_layout)
        
        # Bloque de resumen con IA
        self.summary_title = QLabel(TRANSLATIONS[self.current_language]["summarized_text"])
        self.summary_title.setFont(QFont("Arial", 14, QFont.Bold))
        right_layout.addWidget(self.summary_title)
        
        self.summary_edit = QTextEdit()
        self.summary_edit.setReadOnly(True)
        right_layout.addWidget(self.summary_edit, 1)
        
        summary_button_layout = QHBoxLayout()
        self.copy_summary_btn = QPushButton(TRANSLATIONS[self.current_language]["copy_summary"])
        self.copy_summary_btn.clicked.connect(self.copy_summary)
        summary_button_layout.addWidget(self.copy_summary_btn)
        right_layout.addLayout(summary_button_layout)
        
        # Añadir paneles al splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])
        
        main_layout.addWidget(splitter)
        
        # Aceptar drag and drop en toda la ventana
        self.setAcceptDrops(True)

    def _start_model_preload(self):
        model_size = self._prefs.get("model_size", "large")
        self._start_model_preload_for_size(model_size)

    def _start_model_preload_for_size(self, model_size):
        if model_size in self.model_cache:
            return
        self.status_label.setText(TRANSLATIONS[self.current_language].get("loading_model", "Cargando modelo..."))
        self.model_preload_thread = ModelPreloadThread(model_size)
        self.model_preload_thread.finished_signal.connect(self._on_model_preloaded)
        self.model_preload_thread.error_signal.connect(self._on_model_preload_error)
        self.model_preload_thread.start()

    def _on_model_preloaded(self, model_size, model):
        self.model_cache[model_size] = model
        if not self.progress_bar.isVisible():
            self.status_label.setText(f"Modelo {model_size} listo - {TRANSLATIONS[self.current_language]['ready']}")

    def _on_model_preload_error(self, error_msg):
        self.status_label.setText(f"Error cargando modelo: {error_msg}")

    def _ensure_ollama_running(self):
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1)
        except Exception:
            import subprocess
            try:
                subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

    def change_ui_language(self, lang_code):
        """Cambia el idioma de la interfaz (es/en)"""
        if lang_code in ("es", "en") and lang_code in TRANSLATIONS:
            self.current_language = lang_code
            self.btn_es.setChecked(lang_code == "es")
            self.btn_en.setChecked(lang_code == "en")
            self.retranslate_ui()
            self._prefs["ui_language"] = lang_code
            save_prefs(self._prefs)

    def retranslate_ui(self):
        """Actualiza todos los textos de la interfaz con el idioma seleccionado"""
        self.setWindowTitle(TRANSLATIONS[self.current_language]["app_title"])
        
        # Actualizar etiquetas: mapa inverso de cualquier texto conocido a su key (funciona en ambas direcciones)
        text_to_key = {}
        for key in TRANSLATIONS["es"].keys():
            for lang in TRANSLATIONS:
                val = TRANSLATIONS[lang].get(key)
                if val:
                    text_to_key[val] = key
        for widget in self.findChildren(QLabel):
            key = text_to_key.get(widget.text())
            if key:
                widget.setText(TRANSLATIONS[self.current_language][key])
        
        # Actualizar botones
        self.select_btn.setText(TRANSLATIONS[self.current_language]["select_audio"])
        self.save_btn.setText(TRANSLATIONS[self.current_language]["save_text"])
        self.copy_btn.setText(TRANSLATIONS[self.current_language]["copy_text"])
        self.summarize_btn.setText(TRANSLATIONS[self.current_language]["summarize_ai"])
        self.copy_summary_btn.setText(TRANSLATIONS[self.current_language]["copy_summary"])
        
        # Actualizar estado
        if not self.progress_bar.isVisible():
            self.status_label.setText(TRANSLATIONS[self.current_language]["ready"])

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def closeEvent(self, event):
        threads = []
        if self.model_preload_thread and self.model_preload_thread.isRunning():
            threads.append(self.model_preload_thread)
        if hasattr(self, 'transcription_thread') and self.transcription_thread and self.transcription_thread.isRunning():
            threads.append(self.transcription_thread)
        if self.summary_thread and self.summary_thread.isRunning():
            threads.append(self.summary_thread)
        for t in threads:
            t.quit()
            t.wait(3000)

        for model in self.model_cache.values():
            try:
                del model
            except Exception:
                pass
        self.model_cache.clear()
        super().closeEvent(event)

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if self.is_audio_file(file_path):
                self.process_audio_file(file_path)
            else:
                QMessageBox.warning(self, 
                                   TRANSLATIONS[self.current_language]["invalid_file"], 
                                   TRANSLATIONS[self.current_language]["select_valid_audio"])

    def is_audio_file(self, file_path):
        audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.wma', '.aac', '.opus']
        return os.path.isfile(file_path) and Path(file_path).suffix.lower() in audio_extensions

    def select_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, TRANSLATIONS[self.current_language]["select_audio"], "", 
            "Audio Files (*.mp3 *.wav *.ogg *.flac *.m4a *.wma *.aac *.opus)"
        )
        if file_path:
            self.process_audio_file(file_path)

    def process_audio_file(self, file_path):
        self.current_transcription = file_path
        self.summary_edit.clear()
        model_size = self.model_combo.currentText()
        language = self.trans_lang_combo.currentData()
        
        self._prefs["model_size"] = model_size
        self._prefs["transcription_language"] = language
        save_prefs(self._prefs)

        if model_size not in self.model_cache:
            self._start_model_preload_for_size(model_size)
        
        model = self.model_cache.get(model_size)
        batched_model = None
        if model is not None:
            batched_model = BatchedInferencePipeline(model=model)
        
        self.set_controls_enabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setText(TRANSLATIONS[self.current_language]["starting_transcription"])
        
        # Iniciar hilo de transcripción
        self.transcription_thread = TranscriptionThread(file_path, model_size, self.history_dir, language, self.current_language, model=model, batched_model=batched_model)
        self.transcription_thread.progress_signal.connect(self.update_progress)
        self.transcription_thread.finished_signal.connect(self.transcription_finished)
        self.transcription_thread.error_signal.connect(self.transcription_error)
        self.transcription_thread.start()

    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def transcription_finished(self, audio_path, text, success):
        self.progress_bar.setVisible(False)
        if not (self.summary_thread is not None and self.summary_thread.isRunning()):
            self.set_controls_enabled(True)
        
        if success:
            self.status_label.setText(TRANSLATIONS[self.current_language]["transcription_complete"])
            self.text_edit.setPlainText(text)
            self.load_history()  # Actualizar historial
            if self.history_list.count() > 0:
                metadata = self.history_list.item(0).data(Qt.UserRole)
                self.current_text_path = os.path.join(self.history_dir, metadata['text_file'])
        else:
            self.status_label.setText(TRANSLATIONS[self.current_language]["error_saving"])

    def transcription_error(self, error_message):
        self.progress_bar.setVisible(False)
        if not (self.summary_thread is not None and self.summary_thread.isRunning()):
            self.set_controls_enabled(True)
        self.status_label.setText(f"{TRANSLATIONS[self.current_language]['error']}: {error_message}")
        QMessageBox.critical(self, TRANSLATIONS[self.current_language]["error"], error_message)

    def set_controls_enabled(self, enabled):
        self.select_btn.setEnabled(enabled)
        self.model_combo.setEnabled(enabled)
        self.trans_lang_combo.setEnabled(enabled)
        self.summarize_btn.setEnabled(enabled)
        self.copy_summary_btn.setEnabled(enabled)

    def load_history(self):
        self.history_list.clear()
        
        if not os.path.exists(self.history_dir):
            return
            
        # Buscar archivos JSON en el directorio de historial
        json_files = [f for f in os.listdir(self.history_dir) if f.endswith('.json')]
        json_files.sort(reverse=True)  # Más recientes primero
        
        for json_file in json_files:
            try:
                with open(os.path.join(self.history_dir, json_file), 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Obtener el nombre del idioma
                lang_code = metadata.get('language', 'auto')
                lang_name = WHISPER_LANGUAGES.get(lang_code, lang_code)
                
                item = QListWidgetItem()
                item.setText(metadata['original_name'])
                item.setToolTip(f"{metadata['timestamp'][:10]} - {lang_name}")
                item.setData(Qt.UserRole, metadata)
                self.history_list.addItem(item)
                
            except Exception as e:
                print(f"Error loading history item: {e}")

    def load_history_item(self, item):
        metadata = item.data(Qt.UserRole)
        text_file = os.path.join(self.history_dir, metadata['text_file'])
        self.current_text_path = text_file
        
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()
            if "RESUMEN (IA):" in text:
                partes = text.split("RESUMEN (IA):", 1)
                self.text_edit.setPlainText(partes[0])
                self.summary_edit.setPlainText(partes[1] if len(partes) > 1 else "")
            else:
                self.text_edit.setPlainText(text)
                self.summary_edit.clear()
        except Exception as e:
            QMessageBox.critical(self, TRANSLATIONS[self.current_language]["error"], 
                               f"{TRANSLATIONS[self.current_language]['load_error']}: {str(e)}")

    def save_text(self):
        if not self.text_edit.toPlainText():
            QMessageBox.warning(self, TRANSLATIONS[self.current_language]["warning"], 
                              TRANSLATIONS[self.current_language]["no_text_warning"])
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, TRANSLATIONS[self.current_language]["save_text"], "", "Text Files (*.txt)"
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.text_edit.toPlainText())
                QMessageBox.information(self, TRANSLATIONS[self.current_language]["success"], 
                                      TRANSLATIONS[self.current_language]["text_saved"])
            except Exception as e:
                QMessageBox.critical(self, TRANSLATIONS[self.current_language]["error"], 
                                   f"{TRANSLATIONS[self.current_language]['error']}: {str(e)}")

    def copy_text(self):
        if self.text_edit.toPlainText():
            QApplication.clipboard().setText(self.text_edit.toPlainText())
            QMessageBox.information(self, TRANSLATIONS[self.current_language]["success"], 
                                  TRANSLATIONS[self.current_language]["text_copied"])
        else:
            QMessageBox.warning(self, TRANSLATIONS[self.current_language]["warning"], 
                              TRANSLATIONS[self.current_language]["no_text_copy"])

    def summarize_text(self):
        text = self.text_edit.toPlainText()
        if not text:
            QMessageBox.warning(self, TRANSLATIONS[self.current_language]["warning"],
                              TRANSLATIONS[self.current_language]["no_text_summary"])
            return

        if self.summary_thread is not None and self.summary_thread.isRunning():
            return

        self.set_controls_enabled(False)
        self.summary_edit.clear()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)
        self.status_label.setText(TRANSLATIONS[self.current_language]["summarizing_ai"])

        self.summary_thread = SummaryThread(text, self.current_text_path)
        self.summary_thread.chunk_signal.connect(self.append_summary_chunk)
        self.summary_thread.finished_signal.connect(self.summary_finished)
        self.summary_thread.error_signal.connect(self.summary_error)
        self.summary_thread.start()

    def append_summary_chunk(self, chunk):
        """Inserta un fragmento del resumen en el QTextEdit a medida que llega."""
        self.summary_edit.moveCursor(self.summary_edit.textCursor().End)
        self.summary_edit.insertPlainText(chunk)
        self.summary_edit.moveCursor(self.summary_edit.textCursor().End)

    def summary_finished(self, text_path, summary, success):
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.set_controls_enabled(True)
        self.status_label.setText(TRANSLATIONS[self.current_language]["summary_complete"])
        self.summary_edit.setPlainText(summary)
        self.save_summary_to_history(text_path, summary)
        self.load_history()

    def summary_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.set_controls_enabled(True)
        self.status_label.setText(f"{TRANSLATIONS[self.current_language]['error']}: {error_message}")
        QMessageBox.critical(self, TRANSLATIONS[self.current_language]["error"], error_message)

    def copy_summary(self):
        if self.summary_edit.toPlainText():
            QApplication.clipboard().setText(self.summary_edit.toPlainText())
            QMessageBox.information(self, TRANSLATIONS[self.current_language]["success"],
                                  TRANSLATIONS[self.current_language]["text_copied"])
        else:
            QMessageBox.warning(self, TRANSLATIONS[self.current_language]["warning"],
                              TRANSLATIONS[self.current_language]["no_text_copy"])

    def save_summary_to_history(self, text_path, summary):
        """Guarda el resumen en el historial: append al .txt y actualiza el metadata JSON"""
        if not text_path or not os.path.exists(text_path):
            return
        try:
            with open(text_path, 'a', encoding='utf-8') as f:
                f.write("\n\n" + "=" * 60 + "\nRESUMEN (IA):\n")
                f.write(summary)

            metadata_path = Path(text_path).with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                metadata["summary"] = True
                metadata["summary_model"] = OLLAMA_MODEL
                metadata["summary_preview"] = summary[:100]
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving summary to history: {e}")

def main():
    # CAMBIO: ahora en carpeta del script
    history_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history")
    os.makedirs(history_dir, exist_ok=True)
    
    # Iniciar aplicación
    app = QApplication(sys.argv)
    window = AudioTranscriberApp()
    window.history_dir = history_dir
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()