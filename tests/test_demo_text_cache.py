from pathlib import Path


def test_demo_text_cache_runs_and_prints_terminal_report():
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts" / "demo_text_cache.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--text",
            "TurboQuant demo text for cache testing.",
            "--max-tokens",
            "8",
            "--head-dim",
            "32",
            "--bits",
            "2",
            "--outlier-bits",
            "3",
            "--outlier-channels",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "TurboQuant Text Cache Demo" in result.stdout
    assert "Final cache sizes" in result.stdout
    assert "Visual trends" in result.stdout
    assert "Per-step snapshot" in result.stdout
