from vigia.core import models


def test_registry_entries_have_license_and_https():
    assert models.REGISTRY
    for name, spec in models.REGISTRY.items():
        assert spec.license, name
        assert spec.url is None or spec.url.startswith("https://"), name


def test_ensure_uses_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "MODELS_DIR", tmp_path)
    spec = models.REGISTRY["dfine-s"]
    (tmp_path / spec.filename).write_bytes(b"x")
    monkeypatch.setattr(models, "_download", lambda *a: (_ for _ in ()).throw(AssertionError("no network")))
    assert models.ensure("dfine-s") == tmp_path / spec.filename


def test_ensure_without_url_explains(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "MODELS_DIR", tmp_path)
    try:
        models.ensure("road-rfdetr-s")
    except FileNotFoundError as e:
        assert "export_road_model.py" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")
