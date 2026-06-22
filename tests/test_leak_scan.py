"""leak-scan guards secrets/PII out of the public repo.

Intent (each test fails if the guard regresses, not merely if code is absent).
Several tests pin bugs an adversarial red-team reproduced against an earlier cut:
  - a BARE home path at end of line (/Users/<name>) must be caught, not just
    one with trailing segments;
  - the lowercase /users/ REST route must NOT be flagged (case-sensitive generic);
  - private RFC1918 LAN IPs must be caught, but loopback/bind-all must not;
  - modern sk-proj- keys must be caught;
  - a leak hidden in an added FILE PATH must be caught;
  - an added content line that itself begins with '+' must still be scanned.

The leak fixtures are assembled from fragments (HOME*, AWS, SK_PROJ, LAN_IP,
PATH_LEAK) so THIS source file contains no matching literal — otherwise
leak-scan, run as the repo's own pre-commit hook, would refuse to commit it.
"""

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCANNER = REPO / ".githooks" / "leak-scan"
PATTERNS = REPO / ".githooks" / "leak-patterns"

HOME = "/Users/" + "alice/secret/config.yaml"   # home path with trailing segments
HOME_BARE = "/Users/" + "bob"                    # bare home path, no trailing slash
HOME_SECRET = "/Users/" + "bob/secret"           # for the '+'-prefixed content test
PATH_LEAK = "docs/Users/" + "bob/notes.md"       # leak hidden in a new filename
AWS = "AKIA" + "IOSFODNN7EXAMPLE"
SK_PROJ = "sk-proj-" + "aB12cD34eF56gH78iJ90kLmNoPqRtUvWxYz12abcd"
LAN_IP = "192.168." + "1.50"


def run_scan(diff_text, tmp_path, local_patterns=""):
    """Invoke the scanner with the tracked generic patterns plus an isolated
    (test-controlled) local file — never the developer's real personal list."""
    local = tmp_path / "patterns.local"
    local.write_text(local_patterns)
    env = {
        **os.environ,
        "TW_LEAK_PATTERNS": str(PATTERNS),
        "TW_LEAK_PATTERNS_LOCAL": str(local),
    }
    return subprocess.run(
        [str(SCANNER)], input=diff_text, text=True, capture_output=True, env=env
    )


def _added(line):
    """A minimal unified diff that ADDS one line."""
    return f"--- a/x\n+++ b/x\n@@ -0,0 +1 @@\n+{line}\n"


def test_blocks_home_path_with_trailing_segments(tmp_path):
    # 'alice', not the author — proves the pattern is generic, not hardcoded.
    r = run_scan(_added(f"cfg = {HOME}"), tmp_path)
    assert r.returncode == 1, r.stderr
    assert "leak" in r.stderr.lower()


def test_blocks_bare_home_path_at_end_of_line(tmp_path):
    # Regression: a path with no trailing '/' used to slip through.
    r = run_scan(_added(f"export HOME={HOME_BARE}"), tmp_path)
    assert r.returncode == 1, r.stderr


def test_lowercase_users_route_passes(tmp_path):
    # Regression: case-insensitive matching wrongly blocked the REST route.
    r = run_scan(_added("self.client.get('/users/42/profile')"), tmp_path)
    assert r.returncode == 0, r.stderr


def test_blocks_private_lan_ip(tmp_path):
    r = run_scan(_added(f"upstream = http://{LAN_IP}:1234/v1"), tmp_path)
    assert r.returncode == 1, r.stderr


def test_loopback_and_bindall_pass(tmp_path):
    # The repo uses 127.0.0.1 / 0.0.0.0 pervasively; they must not be flagged.
    r = run_scan(_added("serve at http://127.0.0.1:8080 or 0.0.0.0:7000"), tmp_path)
    assert r.returncode == 0, r.stderr


def test_blocks_aws_access_key(tmp_path):
    r = run_scan(_added(f'aws_key = "{AWS}"'), tmp_path)
    assert r.returncode == 1, r.stderr


def test_blocks_sk_proj_key(tmp_path):
    # Regression: modern OpenAI project keys evaded the legacy sk- rule.
    r = run_scan(_added(f"OPENAI_API_KEY={SK_PROJ}"), tmp_path)
    assert r.returncode == 1, r.stderr


def test_blocks_leak_in_added_file_path(tmp_path):
    # The leak is only in the new filename; the file body is clean.
    diff = f"--- /dev/null\n+++ b/{PATH_LEAK}\n@@ -0,0 +1 @@\n+clean body\n"
    r = run_scan(diff, tmp_path)
    assert r.returncode == 1, r.stderr


def test_scans_added_line_that_begins_with_plus(tmp_path):
    # Regression: git renders such a content line as '+++ ...', which the old
    # header strip silently dropped. Here it carries a real leak and must block.
    diff = f"--- a/x\n+++ b/x\n@@ -1 +1 @@\n+++ {HOME_SECRET}\n"
    r = run_scan(diff, tmp_path)
    assert r.returncode == 1, r.stderr


def test_blocks_personal_literal_case_insensitive(tmp_path):
    # Personal literals match in any casing (employer name however written).
    r = run_scan(
        _added("commissioned by acme corporation"),
        tmp_path,
        local_patterns="Acme Corporation\n",
    )
    assert r.returncode == 1, r.stderr


def test_clean_diff_passes(tmp_path):
    r = run_scan(_added("def add(a, b):  return a + b"), tmp_path)
    assert r.returncode == 0, r.stderr


def test_removing_a_leak_is_not_blocked(tmp_path):
    diff = f"--- a/x\n+++ b/x\n@@ -1 +0,0 @@\n-old = {HOME}\n"
    r = run_scan(diff, tmp_path)
    assert r.returncode == 0, r.stderr


def test_context_line_is_not_an_addition(tmp_path):
    diff = f"--- a/x\n+++ b/x\n@@ -1,2 +1,2 @@\n {HOME}\n+clean line\n"
    r = run_scan(diff, tmp_path)
    assert r.returncode == 0, r.stderr
