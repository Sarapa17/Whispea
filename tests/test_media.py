from pathlib import Path
from types import SimpleNamespace

import core
from core import TranscriptionThread
from main import AudioTranscriberApp


class FakeBatchedModel:
    def __init__(self):
        self.transcribed_paths = []

    def transcribe(self, path, **kwargs):
        self.transcribed_paths.append(path)
        return iter(()), SimpleNamespace(duration=0)


def make_thread(video_path, history_dir, batched_model, ui_language="es"):
    return TranscriptionThread(
        audio_path=str(video_path),
        model_size="tiny",
        history_dir=str(history_dir),
        language="es",
        ui_language=ui_language,
        model=object(),
        batched_model=batched_model,
    )


def test_video_requires_ffmpeg_instead_of_transcribing_original(tmp_path, monkeypatch):
    video_path = tmp_path / "clase.mp4"
    video_path.write_bytes(b"video")
    batched_model = FakeBatchedModel()
    errors = []
    thread = make_thread(video_path, tmp_path / "history", batched_model)
    thread.error_signal.connect(errors.append)
    monkeypatch.setattr(core.shutil, "which", lambda _: None)

    thread.run()

    assert errors
    assert not batched_model.transcribed_paths


def test_video_conversion_error_uses_selected_ui_language(tmp_path, monkeypatch):
    video_path = tmp_path / "clase.mp4"
    video_path.write_bytes(b"video")
    batched_model = FakeBatchedModel()
    errors = []
    thread = make_thread(video_path, tmp_path / "history", batched_model, ui_language="en")
    thread.error_signal.connect(errors.append)
    monkeypatch.setattr(core.shutil, "which", lambda _: None)

    thread.run()

    assert errors
    assert "FFmpeg is required" in errors[0]


def test_video_conversion_disables_video_stream(tmp_path, monkeypatch):
    video_path = tmp_path / "clase.mp4"
    video_path.write_bytes(b"video")
    commands = []
    monkeypatch.setattr(core.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(core.subprocess, "run", lambda command, **kwargs: commands.append(command))

    converted_path = TranscriptionThread._convert_to_16k_mono(str(video_path))

    try:
        assert converted_path is not None
        assert "-vn" in commands[0]
    finally:
        if converted_path:
            core.os.remove(converted_path)


def test_video_without_audio_reports_error_and_removes_temporary_file(tmp_path, monkeypatch):
    video_path = tmp_path / "clase.mp4"
    video_path.write_bytes(b"video")
    commands = []
    batched_model = FakeBatchedModel()
    errors = []
    thread = make_thread(video_path, tmp_path / "history", batched_model)
    thread.error_signal.connect(errors.append)
    monkeypatch.setattr(core.shutil, "which", lambda _: "ffmpeg")

    def fail_conversion(command, **kwargs):
        commands.append(command)
        raise core.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(core.subprocess, "run", fail_conversion)

    thread.run()

    assert errors
    assert "pista de audio" in errors[0]
    assert not batched_model.transcribed_paths
    assert not Path(commands[0][-1]).exists()


def test_audio_conversion_can_still_fall_back_without_ffmpeg(tmp_path, monkeypatch):
    audio_path = tmp_path / "clase.mp3"
    audio_path.write_bytes(b"audio")
    monkeypatch.setattr(core.shutil, "which", lambda _: None)

    assert TranscriptionThread._convert_to_16k_mono(str(audio_path)) is None


def test_media_file_validation_accepts_video_and_audio_but_rejects_other_files(tmp_path):
    video_path = tmp_path / "clase.mp4"
    audio_path = tmp_path / "clase.mp3"
    text_path = tmp_path / "notas.txt"
    video_path.touch()
    audio_path.touch()
    text_path.touch()

    assert AudioTranscriberApp.is_media_file(str(video_path))
    assert AudioTranscriberApp.is_media_file(str(audio_path))
    assert not AudioTranscriberApp.is_media_file(str(text_path))


def test_file_picker_filter_includes_audio_and_video_formats(monkeypatch):
    captured = {}

    class FakeApp:
        current_language = "es"

        def process_audio_file(self, file_path):
            raise AssertionError("No file should be processed when the dialog is cancelled")

    def fake_get_open_file_name(parent, title, directory, file_filter):
        captured["filter"] = file_filter
        return "", ""

    monkeypatch.setattr("main.QFileDialog.getOpenFileName", fake_get_open_file_name)

    AudioTranscriberApp.select_audio_file(FakeApp())

    assert ".mp3" in captured["filter"]
    assert ".mp4" in captured["filter"]
