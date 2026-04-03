from pathlib import Path


def test_visual_compare_script_writes_report(tmp_path):
    import subprocess
    import sys

    out = tmp_path / "visual_compare.md"
    script = Path(__file__).resolve().parents[1] / "scripts" / "visual_compare.py"

    result = subprocess.run(
        [sys.executable, str(script), "--max-tokens", "8", "--head-dim", "32", "--bits", "2", "--outlier-bits", "3", "--output", str(out)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert out.exists()
    report = out.read_text(encoding="utf-8")
    assert "TurboQuant Reference Cache Report" in report
    assert "## Benchmark Results" in report
    assert "## Baseline vs TurboQuant KV Cache" in report
    assert "## Context Extension" in report
    assert "Baseline KV" in report
    assert "TurboQuant KV" in report
    assert "wrote" in result.stdout
