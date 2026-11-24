#!/usr/bin/env python3
"""Packs the command-line tool and the C library of one platform into release archives.

Usage (from the repository root, after `cargo build --release -p quintrit-runtime --bin quintrit` and
`cargo build --profile release-unwind -p quintrit-c`):

    python .github/scripts/package.py --platform linux-x86_64 --out dist [--licenses THIRD_PARTY_LICENSES.md]

It writes `quintrit-<platform>.<ext>` (the `quintrit` binary) and `quintrit-c-<platform>.<ext>` (headers, shared and
static library). `.zip` is used for Windows platforms, `.tar.gz` for the others. The names carry no version on purpose,
so that `https://github.com/<owner>/<repo>/releases/latest/download/quintrit-linux-x86_64.tar.gz` always names the newest
release; the version is written into the `VERSION` file inside each archive. Only the Python standard library is used.
"""

import argparse
import re
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "rust" / "target"


def workspace_version() -> str:
    text = (ROOT / "rust" / "Cargo.toml").read_text(encoding="utf-8")
    section = text.split("[workspace.package]", 1)[1]
    match = re.search(r'^version\s*=\s*"([^"]+)"', section, re.MULTILINE)
    if not match:
        sys.exit("cannot read the version from rust/Cargo.toml")
    return match.group(1)


def write_archive(out: Path, top: str, files: dict, windows: bool) -> Path:
    """Writes `files` (archive path -> source path) below the directory `top` and returns the archive path."""
    if windows:
        archive = out / f"{top}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for name, src in sorted(files.items()):
                z.write(src, f"{top}/{name}")
    else:
        archive = out / f"{top}.tar.gz"
        with tarfile.open(archive, "w:gz") as t:
            for name, src in sorted(files.items()):
                info = t.gettarinfo(str(src), f"{top}/{name}")
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with open(src, "rb") as f:
                    t.addfile(info, f)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", required=True, help="platform name used in the archive names, e.g. linux-x86_64")
    parser.add_argument("--out", required=True, type=Path, help="directory that receives the archives")
    parser.add_argument("--licenses", type=Path, help="third-party license list to include")
    parser.add_argument("--version", help="version to write into the archives (default: the workspace version)")
    args = parser.parse_args()

    windows = args.platform.startswith("windows")
    version = args.version or workspace_version()
    args.out.mkdir(parents=True, exist_ok=True)

    version_file = args.out / "_VERSION"
    version_file.write_bytes((version + "\n").encode("utf-8"))  # a LF line end on every platform
    common = {"LICENSE": ROOT / "LICENSE", "NOTICE": ROOT / "NOTICE", "VERSION": version_file}
    if args.licenses:
        common["THIRD_PARTY_LICENSES.md"] = args.licenses

    # Command-line tool.
    exe = "quintrit.exe" if windows else "quintrit"
    cli = TARGET / "release" / exe
    if not cli.is_file():
        sys.exit(f"missing {cli}")
    readme = args.out / "_README.txt"
    readme.write_text(
        "Quintrit command-line tool\n\n"
        "Run `quintrit --help`, or generate text with a GGUF model that has ternary weights:\n\n"
        f"    {exe} --model path/to/model.gguf --prompt \"What is ternary computing?\" --chat-template --temp 0.0\n\n"
        "No installation and no Rust toolchain are needed; the instruction set (AVX2, AVX-512, NEON) is chosen when the\n"
        "program starts. Documentation: https://github.com/velurex-labs/quintrit\n",
        encoding="utf-8",
    )
    files = dict(common)
    files[exe] = cli
    files["README.txt"] = readme
    cli_archive = write_archive(args.out, f"quintrit-{args.platform}", files, windows)
    readme.unlink()

    # C library: headers plus the shared and the static library of the profile `release-unwind` (panics unwind, so a
    # defect in the library becomes an error code instead of ending the host process).
    libdir = TARGET / "release-unwind"
    include = ROOT / "rust" / "crates" / "quintrit-c" / "include"
    if windows:
        libs = ["quintrit_c.dll", "quintrit_c.dll.lib", "quintrit_c.lib"]
    elif args.platform.startswith("macos"):
        libs = ["libquintrit_c.dylib", "libquintrit_c.a"]
    else:
        libs = ["libquintrit_c.so", "libquintrit_c.a"]
    files = dict(common)
    for header in ("quintrit.h", "quintrit.hpp"):
        files[f"include/{header}"] = include / header
    for lib in libs:
        if not (libdir / lib).is_file():
            sys.exit(f"missing {libdir / lib}")
        files[f"lib/{lib}"] = libdir / lib
    c_archive = write_archive(args.out, f"quintrit-c-{args.platform}", files, windows)

    version_file.unlink()
    print(cli_archive)
    print(c_archive)


if __name__ == "__main__":
    main()
