import pytest

import train


def test_load_model_module_via_model_source(monkeypatch, tmp_path):
    models_dir = tmp_path / "models"
    base_dir = models_dir / "base"
    alias_dir = models_dir / "alias"
    base_dir.mkdir(parents=True)
    alias_dir.mkdir(parents=True)

    (base_dir / "model.py").write_text(
        "VALUE = 'base'\n"
        "def build_model(config, vocab_size):\n"
        "    return (config.get('name', 'none'), vocab_size)\n"
    )
    (alias_dir / "model.source").write_text("base\n")

    monkeypatch.setattr(train, "MODELS_DIR", models_dir)

    mod = train.load_model_module("alias")
    assert getattr(mod, "VALUE") == "base"
    assert mod.build_model({"name": "ok"}, 7) == ("ok", 7)


def test_load_model_module_model_source_chain(monkeypatch, tmp_path):
    models_dir = tmp_path / "models"
    a_dir = models_dir / "a"
    b_dir = models_dir / "b"
    c_dir = models_dir / "c"
    a_dir.mkdir(parents=True)
    b_dir.mkdir(parents=True)
    c_dir.mkdir(parents=True)

    (a_dir / "model.source").write_text("b\n")
    (b_dir / "model.source").write_text("c\n")
    (c_dir / "model.py").write_text("VALUE = 'c'\n")

    monkeypatch.setattr(train, "MODELS_DIR", models_dir)

    mod = train.load_model_module("a")
    assert getattr(mod, "VALUE") == "c"


def test_load_model_module_model_source_cycle(monkeypatch, tmp_path):
    models_dir = tmp_path / "models"
    a_dir = models_dir / "a"
    b_dir = models_dir / "b"
    a_dir.mkdir(parents=True)
    b_dir.mkdir(parents=True)

    (a_dir / "model.source").write_text("b\n")
    (b_dir / "model.source").write_text("a\n")

    monkeypatch.setattr(train, "MODELS_DIR", models_dir)

    with pytest.raises(ValueError, match=r"cycle detected"):
        train.load_model_module("a")


def test_load_model_module_model_source_rejects_path_traversal(monkeypatch, tmp_path):
    models_dir = tmp_path / "models"
    alias_dir = models_dir / "alias"
    alias_dir.mkdir(parents=True)
    (alias_dir / "model.source").write_text("../base\n")

    monkeypatch.setattr(train, "MODELS_DIR", models_dir)

    with pytest.raises(ValueError, match=r"Invalid model\.source"):
        train.load_model_module("alias")
