from pathlib import Path

from train.sft_lora import complete_config


def test_complete_config_with_a_local_base_dir_copies_configs_only(tmp_path: Path):
    base, out = tmp_path / "base", tmp_path / "out"
    base.mkdir()
    out.mkdir()
    for n in ("config.json", "chat_template.jinja", "model.safetensors", "model.safetensors.index.json"):
        (base / n).write_text("x")
    (out / "tokenizer.json").write_text("mine")
    complete_config(str(base), out)
    assert sorted(p.name for p in out.iterdir()) == ["chat_template.jinja", "config.json", "tokenizer.json"]


def test_adapter_kwargs_only_carries_the_variants_that_were_asked_for():
    from train.sft_lora import adapter_kwargs

    assert adapter_kwargs() == {}
    assert adapter_kwargs(rslora=True, dora=True, init="pissa") == {"use_rslora": True, "use_dora": True, "init_lora_weights": "pissa"}
