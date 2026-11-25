from ops.box import RemoteBox


def script(**kw):
    return RemoteBox.from_backend("l40s")._serve_script("/m/model", 1, **kw)


def test_quantization_flag_only_when_asked():
    assert "--quantization" not in script()
    assert "--quantization fp8" in script(quant="fp8")


def test_jobs_run_in_their_own_systemd_unit_so_back_to_back_runs_cannot_collide():
    session, cmd = RemoteBox.from_backend("l40s")._job("sh", 0, ["x", "echo", "hi"])
    assert "--unit=lawrag-job-$(date +%s%N)" in cmd and session.endswith("run-x")


def test_guard_ram_threshold_can_be_raised_for_the_tools_endpoint():
    box = RemoteBox.from_backend("l40s")
    assert '-ge 14 ]' in box._guard()
    assert '-ge 50 ]' in box._guard(50)
