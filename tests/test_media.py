import json
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


def test_opus_archive_uses_mono_16khz_and_32kbps(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    commands = []
    monkeypatch.setattr(core.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(core.subprocess, "run", lambda command, **kwargs: commands.append(command))

    converted_path = TranscriptionThread._convert_to_opus(str(audio_path))

    try:
        assert converted_path is not None
        assert converted_path.endswith(".opus")
        assert commands[0][commands[0].index("-ar") + 1] == "16000"
        assert commands[0][commands[0].index("-ac") + 1] == "1"
        assert commands[0][commands[0].index("-c:a") + 1] == "libopus"
        assert commands[0][commands[0].index("-b:a") + 1] == "32k"
    finally:
        if converted_path:
            core.os.remove(converted_path)


def test_video_run_transcribes_wav_and_archives_only_opus(tmp_path, monkeypatch):
    video_path = tmp_path / "clase.mp4"
    wav_path = tmp_path / "transcription.wav"
    opus_path = tmp_path / "archive.opus"
    history_dir = tmp_path / "history"
    video_path.write_bytes(b"video")
    wav_path.write_bytes(b"uncompressed audio")
    opus_path.write_bytes(b"compressed audio")
    batched_model = FakeBatchedModel()
    finished = []
    thread = make_thread(video_path, history_dir, batched_model)
    thread.finished_signal.connect(lambda *result: finished.append(result))
    monkeypatch.setattr(
        TranscriptionThread,
        "_convert_to_16k_mono",
        staticmethod(lambda _: str(wav_path)),
    )
    monkeypatch.setattr(
        TranscriptionThread,
        "_convert_to_opus",
        staticmethod(lambda _: str(opus_path)),
    )

    thread.run()

    assert batched_model.transcribed_paths == [str(wav_path)]
    assert finished and finished[0][2] is True
    stored_audio = list(history_dir.glob("*.opus"))
    assert len(stored_audio) == 1
    assert not list(history_dir.glob("*.mp4"))
    metadata = json.loads(next(history_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert metadata["original_name"] == "clase.mp4"
    assert not wav_path.exists()
    assert not opus_path.exists()


def test_video_is_not_archived_when_opus_encoding_fails(tmp_path, monkeypatch):
    video_path = tmp_path / "clase.mp4"
    wav_path = tmp_path / "transcription.wav"
    history_dir = tmp_path / "history"
    video_path.write_bytes(b"video")
    wav_path.write_bytes(b"audio")
    batched_model = FakeBatchedModel()
    errors = []
    thread = make_thread(video_path, history_dir, batched_model, ui_language="en")
    thread.error_signal.connect(errors.append)
    monkeypatch.setattr(
        TranscriptionThread,
        "_convert_to_16k_mono",
        staticmethod(lambda _: str(wav_path)),
    )
    monkeypatch.setattr(TranscriptionThread, "_convert_to_opus", staticmethod(lambda _: None))

    thread.run()

    assert errors
    assert "Could not save compressed audio" in errors[0]
    assert not history_dir.exists()
    assert not wav_path.exists()


def test_audio_falls_back_to_original_when_opus_is_unavailable(tmp_path, monkeypatch):
    audio_path = tmp_path / "clase.mp3"
    history_dir = tmp_path / "history"
    audio_path.write_bytes(b"audio")
    batched_model = FakeBatchedModel()
    finished = []
    thread = make_thread(audio_path, history_dir, batched_model)
    thread.finished_signal.connect(lambda *result: finished.append(result))
    monkeypatch.setattr(TranscriptionThread, "_convert_to_16k_mono", staticmethod(lambda _: None))
    monkeypatch.setattr(TranscriptionThread, "_convert_to_opus", staticmethod(lambda _: None))

    thread.run()

    assert batched_model.transcribed_paths == [str(audio_path)]
    assert finished and finished[0][2] is True
    assert len(list(history_dir.glob("*.mp3"))) == 1


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
