from ops.box import RemoteBox


def script(**kw):
    return RemoteBox.from_backend("l40s")._serve_script("/m/model", 1, **kw)


def test_quantization_flag_only_when_asked():
    assert "--quantization" not in script()
    assert "--quantization fp8" in script(quant="fp8")
