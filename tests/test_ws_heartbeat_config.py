"""Regression test for COMDEX_WS_HEARTBEAT_IDLE_SECONDS: proves the env var
is actually read into actionhandlerAPI.WS_HEARTBEAT_IDLE_SECONDS, with the
documented default when unset. No broker/API server needed - this imports
the module fresh in a subprocess so the env var is read at import time,
the same way it is when the real server starts.
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_heartbeat_value(env_override=None):
    env = dict(os.environ)
    if env_override is None:
        env.pop("COMDEX_WS_HEARTBEAT_IDLE_SECONDS", None)
    else:
        env["COMDEX_WS_HEARTBEAT_IDLE_SECONDS"] = env_override
    result = subprocess.run(
        [sys.executable, "-c", "import actionhandlerAPI as api; print(api.WS_HEARTBEAT_IDLE_SECONDS)"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"import failed: {result.stderr}"
    return float(result.stdout.strip())


class WsHeartbeatConfigTests(unittest.TestCase):
    def test_default_is_30_seconds_when_unset(self):
        self.assertEqual(30.0, read_heartbeat_value(env_override=None))

    def test_env_var_overrides_default(self):
        self.assertEqual(5.0, read_heartbeat_value(env_override="5"))

    def test_env_var_supports_fractional_seconds(self):
        self.assertEqual(2.5, read_heartbeat_value(env_override="2.5"))


if __name__ == "__main__":
    unittest.main()
