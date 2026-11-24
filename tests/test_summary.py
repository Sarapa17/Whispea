import io
import json
import urllib.error

import core
from core import SummaryThread, select_ollama_model


class FakeHTTPResponse:
    def __init__(self, body_bytes=b"", lines=None):
        self._body_bytes = body_bytes
        self._lines = lines or []

    def read(self):
        return self._body_bytes

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test_select_ollama_model_returns_preferred_when_installed():
    models = ["gemma4:12b", "qwen2.5:3b"]
    assert select_ollama_model(models, "qwen2.5:3b") == "qwen2.5:3b"


def test_select_ollama_model_prefers_general_model_over_coder_when_preferred_missing():
    models = ["qwen2.5-coder:14b", "gemma4:12b"]
    assert select_ollama_model(models, "qwen2.5:3b") == "gemma4:12b"


def test_select_ollama_model_matches_same_family_tag():
    models = ["gemma4:12b", "qwen2.5:7b"]
    assert select_ollama_model(models, "qwen2.5:3b") == "qwen2.5:7b"


def test_select_ollama_model_returns_none_when_empty():
    assert select_ollama_model([], "qwen2.5:3b") is None


def test_summary_thread_falls_back_to_installed_model(monkeypatch):
    captured_requests = []

    def fake_urlopen(req_or_url, timeout=0):
        url = req_or_url if isinstance(req_or_url, str) else req_or_url.full_url
        if url.endswith("/api/tags"):
            payload = json.dumps({
                "models": [
                    {"name": "qwen2.5-coder:14b"},
                    {"name": "gemma4:12b"},
                ]
            }).encode("utf-8")
            return FakeHTTPResponse(body_bytes=payload)
        captured_requests.append(json.loads(req_or_url.data.decode("utf-8")))
        lines = [
            json.dumps({"response": "Resumen ", "done": False}).encode("utf-8"),
            json.dumps({"response": "listo.", "done": True}).encode("utf-8"),
        ]
        return FakeHTTPResponse(lines=lines)

    monkeypatch.setattr(core.urllib.request, "urlopen", fake_urlopen)

    chunks = []
    finished = []
    errors = []
    thread = SummaryThread("Texto de prueba", "/tmp/test.txt", ui_language="es")
    thread.chunk_signal.connect(chunks.append)
    thread.finished_signal.connect(lambda *args: finished.append(args))
    thread.error_signal.connect(errors.append)

    thread.run()

    assert not errors
    assert captured_requests[0]["model"] == "gemma4:12b"
    assert thread.model_used == "gemma4:12b"
    assert chunks == ["Resumen ", "listo."]
    assert finished == [("/tmp/test.txt", "Resumen listo.", True)]


def test_summary_thread_reports_error_when_no_models_installed(monkeypatch):
    def fake_urlopen(req_or_url, timeout=0):
        payload = json.dumps({"models": []}).encode("utf-8")
        return FakeHTTPResponse(body_bytes=payload)

    monkeypatch.setattr(core.urllib.request, "urlopen", fake_urlopen)

    errors = []
    thread = SummaryThread("Texto de prueba", "/tmp/test.txt", ui_language="en")
    thread.error_signal.connect(errors.append)

    thread.run()

    assert len(errors) == 1
    assert "ollama pull" in errors[0]


def test_summary_thread_extracts_ollama_json_error_from_http_error(monkeypatch):
    def fake_urlopen(req_or_url, timeout=0):
        url = req_or_url if isinstance(req_or_url, str) else req_or_url.full_url
        if url.endswith("/api/tags"):
            payload = json.dumps({"models": [{"name": "gemma4:12b"}]}).encode("utf-8")
            return FakeHTTPResponse(body_bytes=payload)
        err_body = io.BytesIO(b'{"error":"model \'gemma4:12b\' not found"}')
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, err_body)

    monkeypatch.setattr(core.urllib.request, "urlopen", fake_urlopen)

    errors = []
    thread = SummaryThread("Texto de prueba", "/tmp/test.txt", ui_language="es")
    thread.error_signal.connect(errors.append)

    thread.run()

    assert len(errors) == 1
    assert "model 'gemma4:12b' not found" in errors[0]
