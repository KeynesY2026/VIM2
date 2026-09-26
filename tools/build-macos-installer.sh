#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-$HOME/.venvs/vim2-mac/bin/python}
[[ "$(uname -m)" == 'arm64' ]] || { echo 'Native Apple Silicon required' >&2; exit 1; }
[[ "$($PYTHON -c 'import platform; print(platform.machine())')" == 'arm64' ]] || exit 1
[[ "$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == '3.11' ]] || exit 1
[[ "$($PYTHON -m PyInstaller --version)" == '6.16.0' ]] || { echo 'Install requirements-build.lock' >&2; exit 1; }
df -h .

user_data=""
stage=""
staged_dmg=""
mount_dir=""
mount_data=""
cleanup() {
  local status=$?
  if [[ -n "${mount_dir}" ]]; then
    hdiutil detach "${mount_dir}" >/dev/null 2>&1 || true
    rmdir "${mount_dir}" 2>/dev/null || true
  fi
  if [[ -n "${mount_data}" ]]; then rm -rf "${mount_data}" || true; fi
  if [[ -n "${user_data}" ]]; then rm -rf "${user_data}" || true; fi
  if [[ -n "${stage}" ]]; then rm -rf "${stage}" || true; fi
  if [[ -n "${staged_dmg}" ]]; then rm -f "${staged_dmg}" || true; fi
  exit "${status}"
}
trap cleanup EXIT

PYTHONDONTWRITEBYTECODE=1 "$PYTHON" tools/download-cpu-model.py
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" tools/verify-installer-layout.py --source "$PWD"
PYTHONPATH=app PYTHONDONTWRITEBYTECODE=1 QT_QPA_PLATFORM=offscreen "$PYTHON" -m pytest -q -p no:cacheprovider
PYTHONPATH=app PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m vim2 --root "$PWD" --check --skip-runtime-check
"$PYTHON" -m PyInstaller --noconfirm --clean --distpath dist/build --workpath build/pyinstaller packaging/vim2.spec
app="$PWD/dist/build/VIM2.app"
[[ -x "$app/Contents/MacOS/VIM2" ]] || exit 1
"$PYTHON" tools/verify-installer-layout.py "$app"
/usr/bin/lipo -archs "$app/Contents/MacOS/VIM2"
/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$app/Contents/Info.plist"
user_data="$(mktemp -d "${TMPDIR:-/tmp}/vim2-check.XXXXXX")"
QT_QPA_PLATFORM=offscreen "$app/Contents/MacOS/VIM2" --check --skip-runtime-check --data-root "$user_data"
[[ -f "$user_data/config/settings.json" && -d "$user_data/runtime" ]]
QT_QPA_PLATFORM=offscreen "$app/Contents/MacOS/VIM2" --import-smoke --data-root "$user_data"
# The onedir and PyInstaller work directory are intermediates. Keep the final DMG.
rm -rf build/pyinstaller
if [[ -d "$PWD/dist/build/VIM2" && -d "$app" ]]; then
  rm -rf "$PWD/dist/build/VIM2"
fi
df -h .
mkdir -p dist/installers build
stage="$(mktemp -d "$PWD/build/vim2-dmg.XXXXXX")"
ditto "$app" "$stage/VIM2.app"
ln -s /Applications "$stage/Applications"
"$PYTHON" packaging/stage_dmg_notes.py "$stage"
[[ -s "$stage/安装说明.txt" ]]
dmg="$PWD/dist/installers/VIM2-0.1.0-macos-arm64.dmg"
# hdiutil appends .dmg unless the destination already uses that suffix.
staged_dmg="$PWD/dist/installers/VIM2-0.1.0-macos-arm64.partial.dmg"
rm -f "$staged_dmg" "${dmg}.partial" "${dmg}.partial.dmg"
hdiutil create -format UDZO -fs HFS+ -volname 'VIM2 0.1.0' -srcfolder "$stage" "$staged_dmg"
[[ -f "$staged_dmg" ]] || { echo "hdiutil did not create $staged_dmg" >&2; exit 1; }
mount_dir="$(mktemp -d "${TMPDIR:-/tmp}/vim2-mount.XXXXXX")"
hdiutil attach -readonly -nobrowse -mountpoint "$mount_dir" "$staged_dmg"
[[ -x "$mount_dir/VIM2.app/Contents/MacOS/VIM2" ]]
[[ "$(readlink "$mount_dir/Applications")" == '/Applications' ]]
[[ -s "$mount_dir/安装说明.txt" ]]
cmp -s "$PWD/packaging/安装说明.txt" "$mount_dir/安装说明.txt"
"$PYTHON" tools/verify-installer-layout.py "$mount_dir/VIM2.app"
mount_data="$(mktemp -d "${TMPDIR:-/tmp}/vim2-mount-check.XXXXXX")"
QT_QPA_PLATFORM=offscreen "$mount_dir/VIM2.app/Contents/MacOS/VIM2" --check --skip-runtime-check --data-root "$mount_data"
QT_QPA_PLATFORM=offscreen "$mount_dir/VIM2.app/Contents/MacOS/VIM2" --import-smoke --data-root "$mount_data"
hdiutil detach "$mount_dir"
rmdir "$mount_dir"
mount_dir=""
mv -f "$staged_dmg" "$dmg"
staged_dmg=""
shasum -a 256 "$dmg" | sed "s|$dmg|$(basename "$dmg")|" > "$dmg.sha256"
rm -rf "$stage"
stage=""
(cd dist/installers && shasum -a 256 -c "$(basename "$dmg").sha256")
df -h .
echo "Mac installer: $dmg"
