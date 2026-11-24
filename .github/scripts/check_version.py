#!/usr/bin/env python3
"""Checks that every version number of the SDK agrees, and that a release tag names the same version.

Usage (from the repository root):

    python .github/scripts/check_version.py [v0.2.0]

The version of the workspace (`rust/Cargo.toml`, `[workspace.package]`) is the reference. The Python package
(`rust/crates/quintrit-py/Cargo.toml`) and every `version = "..."` requirement on a sibling crate in a `path`
dependency must equal it. With a tag argument (`v` followed by the version) the tag must name the same version too.
Only the Python standard library is used.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CRATES = ROOT / "rust" / "crates"


def main() -> None:
    workspace = (ROOT / "rust" / "Cargo.toml").read_text(encoding="utf-8")
    section = workspace.split("[workspace.package]", 1)[1]
    match = re.search(r'^version\s*=\s*"([^"]+)"', section, re.MULTILINE)
    if not match:
        sys.exit("cannot read the version from rust/Cargo.toml")
    version = match.group(1)
    problems = []

    py = (CRATES / "quintrit-py" / "Cargo.toml").read_text(encoding="utf-8")
    py_version = re.search(r'^version\s*=\s*"([^"]+)"', py, re.MULTILINE)
    if not py_version or py_version.group(1) != version:
        problems.append(f"quintrit-py has version {py_version.group(1) if py_version else '?'}, expected {version}")

    for manifest in sorted(CRATES.glob("*/Cargo.toml")):
        for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            dep = re.match(r'^(quintrit-[a-z0-9-]+)\s*=\s*\{.*path\s*=.*version\s*=\s*"=?([^"]+)"', line)
            if dep and dep.group(2) != version:
                problems.append(f"{manifest.relative_to(ROOT)}:{number}: {dep.group(1)} requires {dep.group(2)}, expected {version}")

    if len(sys.argv) > 1:
        tag = sys.argv[1]
        if tag != f"v{version}":
            problems.append(f"the tag is {tag}, but the version in the manifests is {version}")

    if problems:
        sys.exit("version check failed:\n  " + "\n  ".join(problems))
    print(f"all versions agree: {version}")


if __name__ == "__main__":
    main()
