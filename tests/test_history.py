import json

from core import TranscriptionThread


def test_save_to_history_creates_audio_text_and_metadata(tmp_path):
    audio_path = tmp_path / "clase.opus"
    history_dir = tmp_path / "history"
    audio_path.write_bytes(b"audio data")
    thread = TranscriptionThread(
        audio_path=str(audio_path),
        model_size="tiny",
        history_dir=str(history_dir),
        language="es",
        ui_language="es",
        model=object(),
        batched_model=object(),
    )

    result = thread.save_to_history(str(audio_path), "Texto transcripto", "es")

    assert result is True
    audio_files = list(history_dir.glob("*.opus"))
    text_files = list(history_dir.glob("*.txt"))
    metadata_files = list(history_dir.glob("*.json"))
    assert len(audio_files) == 1
    assert len(text_files) == 1
    assert len(metadata_files) == 1
    assert audio_files[0].read_bytes() == b"audio data"
    assert "Texto transcripto" in text_files[0].read_text(encoding="utf-8")

    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    assert metadata["original_name"] == "clase.opus"
    assert metadata["model"] == "tiny"
    assert metadata["language"] == "es"