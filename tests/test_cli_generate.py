import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).parent.parent


def test_generate_help_mentions_prompt_option():
    result = subprocess.run(
        [sys.executable, "main.py", "generate", "--help"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, f"Help failed:\nSTDERR:\n{result.stderr}"
    help_text = result.stdout
    assert "--prompt" in help_text
    assert "Non-interactive" in help_text or "non-interactive" in help_text
