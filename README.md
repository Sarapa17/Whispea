# Whispea

¿Cansado de recibir audios en clase y no poder escucharlos? ¿De tener grabaciones de clases de 2 horas y querer saber si es la clase que necesitás ver ahora? ¿Siempre renegando porque las alternativas web no permiten cosas largas o te piden pagar?

Yo también. Por eso armé **Whispea**: una herramienta de escritorio que transcribe y resume audio de manera **100% local**, corriendo modelos en tu propia máquina — sin subir nada a ningún server, sin APIs pagas, sin límites de duración.

![Captura de la app](assets/screenshot.png)

## Características

- **Transcripción local con Whisper** — arrastrá o seleccioná un audio y obtené el texto, corriendo [OpenAI Whisper](https://github.com/openai/whisper) en tu PC
- **Resumen con IA local con Ollama** — un clic y un modelo local ([Ollama](https://ollama.com)) te resume la transcripción en un párrafo
- **Historial** — cada transcripción y su resumen quedan guardados y se pueden recargar con un clic
- **Multiidioma** — interfaz en español e inglés (selector con banderas), transcripción en 30 idiomas
- **Drag & drop** — arrastrá el archivo a la ventana y listo
- **Sin límites de duración** — tu máquina, tus reglas: audios de 2 horas incluidos

## Requisitos

- Python 3.10+
- [FFmpeg](https://ffmpeg.org) en el PATH (en Windows: `winget install ffmpeg`)
- Windows: [Microsoft VC++ Redistributable x64](https://aka.ms/vs/17/release/vc_redist.x64.exe) (necesario para PyTorch)
- [Ollama](https://ollama.com) instalado + un modelo para los resúmenes (ver Instalación)

## Instalación

```bash
git clone https://github.com/TU-USUARIO/whispea.git
cd whispea

# Crear y activar venv
python -m venv venv
.\venv\Scripts\Activate.ps1    # Windows PowerShell
# source venv/bin/activate     # Linux/Mac

# Dependencias
pip install -r requirements.txt

# Modelo para los resúmenes (~2GB)
ollama pull qwen2.5:3b
```

## Uso

```bash
python main.py
```

1. Seleccioná o arrastrá un archivo de audio (mp3, wav, ogg, flac, m4a, wma, aac, opus)
2. Esperá a que termine la transcripción
3. Clic en **"Resumir con IA"** para generar el resumen
4. **"Copiar resumen"** / **"Copiar texto"** / **"Guardar texto"** para lo que necesites

## Cómo funciona

- **Transcripción**: [OpenAI Whisper](https://github.com/openai/whisper) corre 100% local en CPU (el audio se convierte a WAV 16kHz con FFmpeg primero)
- **Resumen**: la app le pega a un server local de [Ollama](https://ollama.com) (`http://localhost:11434`) con el modelo configurado — el texto nunca sale de tu máquina
- **Historial**: todo queda en `ConvertYourFileHistory/` (fuera del repo, son tus datos)

## Configuración

En `main.py`, al inicio:

| Constante | Default | Qué hace |
|---|---|---|
| `OLLAMA_MODEL` | `"qwen2.5:3b"` | El modelo de Ollama que resume (cambialo por `qwen2.5:7b` si querés más calidad) |
| `PROMPT_TEMPLATE` | `"Resumí el siguiente texto..."` | El prompt que se le envía al modelo |
| `MAX_SUMMARY_INPUT_CHARS` | `25000` | Límite de caracteres que se envían al modelo |

## Notas

- El modelo elegido (`qwen2.5:3b`, ~2GB) corre rápido en CPU (~15-25 tok/s en un Ryzen 7) y es de lo mejor en español para resúmenes
- Si Ollama no está corriendo, la app te avisa con un mensaje claro (no crashea)
- Probado en Windows 11; debería funcionar en Linux/Mac ajustando la activación del venv
