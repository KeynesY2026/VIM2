"""Check installer-owned model/config trees, not dependency directories."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

MODEL = "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
MODEL_FILES = (
    "conv_frontend.onnx",
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "tokenizer/merges.txt",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
)
PACKAGED_CONFIG = (
    "config/settings.json",
    "config/hotkey.conf",
    "config/hotwords.template.txt",
)
PACKAGED_PATHS = tuple(
    f".models/{MODEL}/{name}" for name in MODEL_FILES
) + PACKAGED_CONFIG
FORBIDDEN = {
    ".squad",
    ".git",
    "runtime",
    "temp",
    "tests",
    "tools",
    "__pycache__",
    "torch",
    "qwen_asr",
    "transformers",
    "bitsandbytes",
}
_OWNED_CHILDREN = {
    ".models": {MODEL},
    f".models/{MODEL}": {
        "conv_frontend.onnx",
        "encoder.int8.onnx",
        "decoder.int8.onnx",
        "tokenizer",
    },
    f".models/{MODEL}/tokenizer": {
        "merges.txt",
        "tokenizer_config.json",
        "vocab.json",
    },
    "config": {"settings.json", "hotkey.conf", "hotwords.template.txt"},
}
_MODEL_DIRS = {".models", f".models/{MODEL}", f".models/{MODEL}/tokenizer"}


def hash_key(relative: Path) -> str:
    return "\\".join(relative.parts)


def packaged_source_files(repo: Path) -> dict[str, Path]:
    """Map packaged relative paths to the exact source files the spec copies."""
    selected: dict[str, Path] = {}
    for relative in PACKAGED_PATHS:
        source_relative = (
            Path("packaging/hotwords.template.txt")
            if relative == "config/hotwords.template.txt"
            else Path(relative)
        )
        selected[relative] = _exact_regular_file(repo, source_relative)
    return selected


def verify_source_payload(
    repo: Path, *, hashes: Mapping[str, str] | None = None
) -> None:
    """Validate installer inputs without treating the whole repository as a bundle."""
    if not repo.is_dir():
        raise FileNotFoundError(repo)
    selected = packaged_source_files(repo)
    expected = _hashes(hashes)
    for relative, source in selected.items():
        if not relative.startswith(".models/"):
            continue
        _check_hash(source, hash_key(Path(relative)), expected, relative)
    print(f"Source payload allowlist checks passed: {repo}")


def verify(root: Path, *, hashes: Mapping[str, str] | None = None) -> None:
    if not root.is_dir():
        raise FileNotFoundError(root)
    _reject_forbidden_and_developer_files(root)
    resource_roots = _project_resource_roots(root)
    expected = _hashes(hashes)
    for resource_root in resource_roots:
        _verify_owned_trees(resource_root, expected)
    _reject_escaping_project_links(root, resource_roots)
    print(f"Payload allowlist checks passed: {root}")


def _hashes(hashes: Mapping[str, str] | None) -> Mapping[str, str]:
    if hashes is not None:
        return hashes
    hash_file = Path(__file__).resolve().parents[1] / "release-files.sha256.json"
    return json.loads(hash_file.read_text(encoding="utf-8"))


def _reject_forbidden_and_developer_files(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        for name in dirnames:
            path = current / name
            if set(path.relative_to(root).parts) & FORBIDDEN:
                raise RuntimeError(f"Forbidden payload path: {path}")
        for name in filenames:
            path = current / name
            if set(path.relative_to(root).parts) & FORBIDDEN:
                raise RuntimeError(f"Forbidden payload path: {path}")
            if path.is_symlink():
                continue
            if path.suffix.lower() in {".h", ".hpp", ".md", ".py"}:
                raise RuntimeError(f"Developer file in payload: {path}")


def _project_resource_roots(bundle: Path) -> list[Path]:
    if (bundle / "Contents" / "Info.plist").is_file() or bundle.suffix == ".app":
        resources = bundle / "Contents" / "Resources"
        if _has_owned_trees(resources):
            return [resources]
        raise RuntimeError(f"No project resource root: {bundle}")
    internal = bundle / "_internal"
    if internal.is_dir() and _has_owned_trees(internal):
        return [internal]
    if _has_owned_trees(bundle):
        return [bundle]
    raise RuntimeError(f"No project resource root: {bundle}")


def _has_owned_trees(root: Path) -> bool:
    return any(_is_real_dir(root / name) for name in (".models", "config"))


def _is_real_dir(path: Path) -> bool:
    return path.is_dir() and not path.is_symlink()


def _verify_owned_trees(resource_root: Path, hashes: Mapping[str, str]) -> None:
    for name in (".models", "config"):
        _exact_child(resource_root, Path(name))
        _walk_owned(resource_root, Path(name), hashes)


def _walk_owned(
    resource_root: Path, relative: Path, hashes: Mapping[str, str]
) -> None:
    directory = _exact_child(resource_root, relative)
    if not _is_real_dir(directory):
        raise RuntimeError(f"Missing payload directory: {relative.as_posix()}")
    expected = _OWNED_CHILDREN[relative.as_posix()]
    seen: set[str] = set()
    for name in os.listdir(directory):
        child_relative = relative / name
        posix = child_relative.as_posix()
        child = directory / name
        if child.is_symlink():
            raise RuntimeError(f"Symlink not allowed: {posix}")
        if not _is_within(child, resource_root):
            raise RuntimeError(f"Path escapes resource root: {posix}")
        if posix == "config/hotwords.txt" or (
            relative.as_posix() == "config" and name.casefold() == "hotwords.txt"
        ):
            raise RuntimeError("Private hotwords must not be bundled")
        if name not in expected:
            if any(item.casefold() == name.casefold() for item in expected):
                raise RuntimeError(f"Case variant is not allowed: {posix}")
            raise RuntimeError(f"Extra model or config path: {posix}")
        seen.add(name)
        if posix in _MODEL_DIRS or (relative.as_posix() == ".models" and name == MODEL):
            _walk_owned(resource_root, child_relative, hashes)
        elif child.is_dir():
            _walk_owned(resource_root, child_relative, hashes)
        elif child.is_file():
            if posix.startswith(".models/"):
                _check_hash(child, hash_key(child_relative), hashes, posix)
        else:
            raise RuntimeError(f"Unsupported payload entry: {posix}")
    missing = expected - seen
    if missing:
        missing_name = sorted(missing)[0]
        folded = [
            entry
            for entry in os.listdir(directory)
            if entry.casefold() == missing_name.casefold()
        ]
        if folded:
            raise RuntimeError(
                f"Case variant is not allowed: {(relative / folded[0]).as_posix()}"
            )
        raise RuntimeError(
            f"Missing payload path: {(relative / missing_name).as_posix()}"
        )


def _exact_child(parent: Path, relative: Path) -> Path:
    current = parent if relative.parts and parent.name != relative.parts[-1] else parent
    if relative == Path(".models") or relative == Path("config"):
        current_parent = parent
        name = relative.name
    else:
        return parent / relative
    if current_parent.is_symlink():
        raise RuntimeError(f"Symlink not allowed: {relative.as_posix()}")
    for entry in os.listdir(current_parent):
        if entry == name:
            return current_parent / entry
    folded = [
        entry
        for entry in os.listdir(current_parent)
        if entry.casefold() == name.casefold()
    ]
    if folded:
        raise RuntimeError(f"Case variant is not allowed: {folded[0]}")
    raise RuntimeError(f"Missing payload path: {name}")


def _exact_regular_file(repo: Path, relative: Path) -> Path:
    current = repo
    walked: list[str] = []
    for part in relative.parts:
        if current.is_symlink():
            raise RuntimeError(f"Symlink not allowed: {relative.as_posix()}")
        walked.append(part)
        found = None
        try:
            entries = os.listdir(current)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Missing payload path: {relative.as_posix()}"
            ) from exc
        for entry in entries:
            if entry == part:
                found = current / entry
                break
        if found is None:
            folded = [entry for entry in entries if entry.casefold() == part.casefold()]
            if folded:
                raise RuntimeError(
                    f"Case variant is not allowed: {'/'.join([*walked[:-1], folded[0]])}"
                )
            raise RuntimeError(f"Missing payload path: {relative.as_posix()}")
        if found.is_symlink():
            raise RuntimeError(f"Symlink not allowed: {relative.as_posix()}")
        current = found
    if not current.is_file():
        raise RuntimeError(f"Missing payload path: {relative.as_posix()}")
    return current


def _check_hash(
    path: Path, key: str, hashes: Mapping[str, str], label: str
) -> None:
    try:
        expected = hashes[key]
    except KeyError as exc:
        raise RuntimeError(f"No SHA-256 pin for {label}") from exc
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected:
        raise RuntimeError(f"Bundled model SHA-256 mismatch: {label}")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _reject_escaping_project_links(bundle: Path, resource_roots: list[Path]) -> None:
    canonical = {
        name: {
            (resource_root / name).resolve()
            for resource_root in resource_roots
            if (resource_root / name).exists()
        }
        for name in (".models", "config")
    }
    for dirpath, dirnames, _filenames in os.walk(bundle, followlinks=False):
        current = Path(dirpath)
        for name in dirnames:
            if name not in canonical:
                continue
            path = current / name
            if not path.is_symlink():
                continue
            try:
                resolved = path.resolve()
            except OSError as exc:
                raise RuntimeError(
                    f"Project tree symlink escapes resource root: {path}"
                ) from exc
            if resolved not in canonical[name]:
                raise RuntimeError(
                    f"Project tree symlink escapes resource root: {path}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument(
        "--source",
        nargs="?",
        const=Path(__file__).resolve().parents[1],
        type=Path,
        help="validate installer inputs; defaults to the repository root",
    )
    args = parser.parse_args()
    if args.source is not None:
        verify_source_payload(args.source.resolve())
    elif args.root is not None:
        verify(args.root.resolve())
    else:
        parser.error("pass a built payload root or --source REPO")
