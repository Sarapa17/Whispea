import config


def test_load_prefs_returns_defaults_when_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PREFS_PATH", str(tmp_path / "prefs.json"))

    prefs = config.load_prefs()

    assert prefs == {
        "model_size": "large-v3-turbo",
        "transcription_language": "es",
        "ui_language": "es",
    }