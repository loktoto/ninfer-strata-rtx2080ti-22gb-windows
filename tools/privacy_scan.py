#!/usr/bin/env python3
"""Fail CI if tracked source contains common private data or credentials.

This scanner deliberately prints only category, path and line number,
never the matched secret. It is one layer of protection, not a guarantee.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
ALLOW_EMAIL_DOMAINS = {"example.com", "example.org", "example.net", "example.invalid"}
PATTERNS = {
    "Windows user directory": re.compile(r"\b[A-Z]:[\\/](?:Users|Documents and Settings)[\\/][^\\/\s\"'<>]+", re.I),
    "Unix user directory": re.compile(r"/(?:Users|home)/[^/\s\"'<>]+"),
    "Private network address": re.compile(r"\b(?:192\.168\.\d{1,3}|10\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3})\.\d{1,3}\b"),
    "Local machine name": re.compile(r"\b(?:DESKTOP|LAPTOP|WORKSTATION)-[A-Z0-9-]{5,}\b", re.I),
    "Private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "Credential token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{15,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|AIza[A-Za-z0-9_-]{30,})\b"),
}
SENSITIVE_PATHS = re.compile(
    r"(^|/)(?:\.env(?:\..*)?|\.npmrc|\.pypirc|\.git-credentials|"
    r"id_(?:rsa|ed25519)|credentials(?:\.[^/]*)?|service-account[^/]*\.json|"
    r"\.ssh|\.aws)(?:/|$)|\.(?:pem|p12|pfx|jks|sqlite3?|db)$",
    re.I,
)
CREDENTIAL_VALUE = re.compile(
    r"\b(?:password|passwd|api[_-]?key|client[_-]?secret|access[_-]?token|auth[_-]?token)"
    r"\s*[:=]\s*['\"]?([A-Za-z0-9_/-]{14,})",
    re.I,
)
PLACEHOLDER = re.compile(r"^(?:example|dummy|placeholder|changeme|your_|test_|not_a_|redacted)", re.I)


def find_issues(contents: str) -> list[tuple[int, str]]:
    findings = []
    for number, line in enumerate(contents.splitlines(), 1):
        for match in EMAIL.finditer(line):
            if match.group().rsplit("@", 1)[-1].lower() not in ALLOW_EMAIL_DOMAINS:
                findings.append((number, "Non-example email address"))
        for description, regex in PATTERNS.items():
            if regex.search(line):
                findings.append((number, description))
        match = CREDENTIAL_VALUE.search(line)
        if match and not PLACEHOLDER.match(match.group(1)):
            findings.append((number, "Possible embedded credential"))
    return findings


def main() -> int:
    names = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"]).split(b"\0")
    errors = 0
    checked = 0
    for raw in names:
        if not raw:
            continue
        path = raw.decode("utf-8", "surrogateescape")
        if SENSITIVE_PATHS.search(path):
            print(f"::error file={path}::Tracked potentially private file")
            errors += 1
            continue
        filename = ROOT / path
        if not filename.is_file():
            continue
        try:
            contents = filename.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # Non-text content; review binaries separately.
        checked += 1
        for line, kind in find_issues(contents):
            print(f"::error file={path},line={line}::{kind}; value intentionally withheld")
            errors += 1
    print(f"Privacy scan: checked {checked} tracked text files; findings: {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
