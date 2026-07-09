"""
Cross-platform checkpoint downloader for SAM 2 / SAM 2.1.

Parses the sibling ``download_ckpts.sh`` shell script, resolves the checkpoint
URLs it defines (expanding ``${VAR}`` references), and downloads each one using
the Python standard library. This avoids needing ``wget`` or ``curl``, so it
runs unchanged on Windows.

Usage:
    python download_ckpts.py                 # download all SAM 2.1 checkpoints
    python download_ckpts.py --script foo.sh # parse a different shell script
    python download_ckpts.py --dest ./out    # download into another directory
    python download_ckpts.py --list          # only print the resolved URLs
"""

import argparse
import re
import sys
import urllib.request
from pathlib import Path

# Matches:  NAME="value"  or  NAME=value  (value optionally quoted)
_ASSIGN_RE = re.compile(r'^\s*([A-Za-z_][A-Za-z0-9_]*)=(.+?)\s*$')
# Matches ${VAR} or $VAR references inside a value
_VAR_RE = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)')


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def parse_shell_vars(script_path: Path) -> dict:
    """Parse simple VAR=value assignments from a shell script.

    Skips comment lines (leading ``#``). Expands ``$VAR`` / ``${VAR}`` using
    variables already defined earlier in the file.
    """
    variables: dict[str, str] = {}
    for raw in script_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _ASSIGN_RE.match(line)
        if not m:
            continue
        name, raw_value = m.group(1), m.group(2)
        # Only keep straightforward assignments (no command substitution etc.)
        if "$(" in raw_value or "`" in raw_value:
            continue
        value = _strip_quotes(raw_value)

        def _expand(match: re.Match) -> str:
            key = match.group(1) or match.group(2)
            return variables.get(key, "")

        variables[name] = _VAR_RE.sub(_expand, value)
    return variables


def extract_urls(variables: dict) -> list:
    """Return the checkpoint URLs (variable values ending in .pt), de-duplicated."""
    urls: list[str] = []
    for value in variables.values():
        if value.startswith(("http://", "https://")) and value.endswith(".pt"):
            if value not in urls:
                urls.append(value)
    return urls


def download(url: str, dest_dir: Path) -> None:
    filename = url.rsplit("/", 1)[-1]
    out_path = dest_dir / filename
    print(f"Downloading {filename} ...")

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = done * 100 / total_size
        sys.stdout.write(f"\r  {pct:6.2f}%  ({done // (1024 * 1024)} / "
                         f"{total_size // (1024 * 1024)} MiB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, out_path, reporthook=_progress)
    sys.stdout.write("\n")


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--script", type=Path, default=here / "download_ckpts.sh",
                        help="Shell script to parse (default: download_ckpts.sh)")
    parser.add_argument("--dest", type=Path, default=here,
                        help="Directory to download into (default: script dir)")
    parser.add_argument("--list", action="store_true",
                        help="Only print resolved URLs, do not download")
    args = parser.parse_args()

    if not args.script.exists():
        print(f"Shell script not found: {args.script}", file=sys.stderr)
        return 1

    variables = parse_shell_vars(args.script)
    urls = extract_urls(variables)
    if not urls:
        print("No checkpoint URLs found in script.", file=sys.stderr)
        return 1

    if args.list:
        for url in urls:
            print(url)
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            download(url, args.dest)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to download checkpoint from {url}: {exc}", file=sys.stderr)
            return 1

    print("All checkpoints are downloaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())