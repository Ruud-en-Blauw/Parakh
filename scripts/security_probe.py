#!/usr/bin/env python3
"""Narrow offline tracked-secret scan; does not replace reviewed security testing."""
from __future__ import annotations

import argparse
import json
import hashlib
import re
import subprocess
from pathlib import Path

# High-specificity signatures only: never print matched content or a secret-bearing line.
PATTERNS = {
    "private-key-block": re.compile(
        rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github-token": re.compile(rb"\b(?:ghp_|gho_|ghs_)[A-Za-z0-9]{36,}\b"),
    "openai-token": re.compile(rb"\bsk-(?:proj-|or-v1-)[A-Za-z0-9_-]{24,}\b"),
}
KEY_HEADER = re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----")
# Reviewed fake credentials in redaction tests, pinned by (repo, file, signature, sha256 of the
# exact match). A new occurrence, an edited value or the same value in another file still fails.
REVIEWED_FIXTURES = {
    ("Parakh", "tests/test_observability_events.py", "openai-token",
     "8d9d47e9872e78bc06945d577b1e202a8fe5cd450dbfef4a23d4134a5fa49ed1"),
    ("Parakh", "tests/test_telemetry_import.py", "private-key-block",
     "9c026e94a2c607a2e427181318c6865ee0fa2a667ddb3e73c79612f12eb0a6e5"),
    ("Parakh", "tests/test_tui_console.py", "openai-token",
     "92cbf5c3be8f9c826361108d715e5909d3fa191e9acda39f9d0a9362cd7bfb4f"),
    # Fake "secret-test-key-never-leak" value in the OpenRouter key-redaction test.
    ("Parakh", "tests/test_openrouter_provider.py", "openai-token",
     "3a0b408bb4211c55c376b9b29d00609969f268f8ea5768fd1fbf08a093164cf8"),
}
MAX_BYTES = 2_000_000


def scan(repo: Path) -> list[dict[str, str]]:
    files = subprocess.run(["git", "-C", str(repo), "ls-files", "-z"],
                           capture_output=True, check=True).stdout.split(b"\0")
    issues: list[dict[str, str]] = []
    for raw in files:
        if not raw:
            continue
        relative = raw.decode("utf-8", errors="replace")
        file = repo / relative
        if not file.is_file() or file.stat().st_size > MAX_BYTES:
            continue
        data = file.read_bytes()
        if b"\0" in data:
            continue
        reviewed_spans: list[tuple[int, int]] = []
        for name, pattern in PATTERNS.items():
            for match in pattern.finditer(data):
                fingerprint = hashlib.sha256(match.group()).hexdigest()
                if (repo.name, relative, name, fingerprint) in REVIEWED_FIXTURES:
                    reviewed_spans.append(match.span())
                else:
                    issues.append({"repository": repo.name, "file": relative, "reason": name})
        # Any key header (including an unterminated or unusual block) must be a reviewed fixture.
        for header in KEY_HEADER.finditer(data):
            if not any(start <= header.start() < end for start, end in reviewed_spans):
                issues.append({"repository": repo.name, "file": relative, "reason": "private-key-header"})
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--karmi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parakh = Path(__file__).resolve().parents[1]
    findings = scan(parakh) + scan(args.karmi.resolve())
    (args.output / "security-findings.json").write_text(json.dumps({
        "mode": "offline", "scanned": "tracked text files <=2MB, three high-specificity signatures",
        "reviewed_fixture_fingerprints": len(REVIEWED_FIXTURES),
        "findings": findings, "dependency_vulnerability_database": "NOT_RUN",
        "authorization_scope": "existing native pytest tests required separately",
    }, indent=2) + "\n")
    print(f"Tracked-secret signature findings: {len(findings)} (no values emitted)")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
