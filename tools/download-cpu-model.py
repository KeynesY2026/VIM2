"""Fetch only the six verified CPU runtime files from a fixed HF revision."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

REPO = 'csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25'
REVISION = '68818b2313fe77bd06f6a7c5068ff3ef59d02b8a'
DIRECTORY = 'sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25'
FILES = ('conv_frontend.onnx', 'encoder.int8.onnx', 'decoder.int8.onnx',
         'tokenizer/merges.txt', 'tokenizer/tokenizer_config.json', 'tokenizer/vocab.json')


def download(root: Path) -> None:
    hashes = json.loads((root / 'release-files.sha256.json').read_text(encoding='utf-8'))
    target_root = root / '.models' / DIRECTORY
    for name in FILES:
        expected = hashes['.models\\' + DIRECTORY + '\\' + name.replace('/', '\\')]
        target = target_root / name
        digest = hashlib.sha256()
        if target.is_file():
            with target.open('rb') as local:
                for chunk in iter(lambda: local.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() == expected:
                print(f'Verified: {name}')
                continue
            raise RuntimeError(f'Existing model file fails SHA-256: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + '.download')
        url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{quote(name, safe="/")}'
        digest = hashlib.sha256()
        try:
            with urlopen(url, timeout=120) as response, temp.open('wb') as output:
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    digest.update(chunk)
                    output.write(chunk)
            if digest.hexdigest() != expected:
                raise RuntimeError(f'Model SHA-256 mismatch: {name}')
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
        print(f'Downloaded and verified: {name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    download(parser.parse_args().root.resolve())
