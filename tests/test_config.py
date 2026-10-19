import json

import config


def test_load_prefs_returns_defaults_when_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PREFS_PATH", str(tmp_path / "prefs.json"))

    prefs = config.load_prefs()

    assert prefs == {
        "model_size": "large-v3-turbo",
        "transcription_language": "es",
        "ui_language": "es",
    }


def test_load_prefs_merges_saved_values_with_defaults(tmp_path, monkeypatch):
    prefs_path = tmp_path / "prefs.json"
    prefs_path.write_text(
        json.dumps({"model_size": "small", "ui_language": "en"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "PREFS_PATH", str(prefs_path))

    prefs = config.load_prefs()

    assert prefs == {
        "model_size": "small",
        "transcription_language": "es",
        "ui_language": "en",
    }


def test_save_prefs_writes_json_with_utf8_values(tmp_path, monkeypatch):
    prefs_path = tmp_path / "prefs.json"
    monkeypatch.setattr(config, "PREFS_PATH", str(prefs_path))
    prefs = {
        "model_size": "small",
        "transcription_language": "es",
        "ui_language": "es",
        "label": "Transcripción rápida",
    }

    config.save_prefs(prefs)

    assert json.loads(prefs_path.read_text(encoding="utf-8")) == prefs


def test_load_prefs_returns_defaults_when_file_is_invalid(tmp_path, monkeypatch):
    prefs_path = tmp_path / "prefs.json"
    prefs_path.write_text("{ preferencias inválidas", encoding="utf-8")
    monkeypatch.setattr(config, "PREFS_PATH", str(prefs_path))

    prefs = config.load_prefs()

    assert prefs == {
        "model_size": "large-v3-turbo",
        "transcription_language": "es",
        "ui_language": "es",
    }