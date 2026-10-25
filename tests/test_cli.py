"""CLI smoke tests: report (with plots), validate, list."""

import os

from llm_roofline.__main__ import _parse_batches, main


def test_parse_batches():
    assert _parse_batches("1..256") == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert _parse_batches("4..17") == [4, 8, 16]
    assert _parse_batches("1,8,64") == [1, 8, 64]


def test_report_runs_and_writes_plots(tmp_path, capsys):
    out = str(tmp_path / "results")
    rc = main(["report", "--model", "llama-3-70b", "--gpu", "h100", "--tp", "4",
               "--batch", "1..64", "--context", "8192", "--out", out])
    assert rc == 0
    text = capsys.readouterr().out
    assert "TPOT" in text and "Llama-3-70B" in text
    for name in ("roofline.png", "batch_curves.png", "tp_scaling.png"):
        assert os.path.getsize(os.path.join(out, name)) > 10_000


def test_validate_and_list(capsys):
    assert main(["validate"]) == 0
    assert "measured" in capsys.readouterr().out
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "deepseek-v3" in out and "h200" in out
