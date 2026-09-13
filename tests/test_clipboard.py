import pytest

from vim2.clipboard import ClipboardPasteError, WindowsClipboardPaster


class FakeWindowsApi:
    def __init__(self, *, foreground_ok: bool = True, send_ok: bool = True):
        self.foreground_ok = foreground_ok
        self.send_ok = send_ok
        self.calls: list[tuple[str, object]] = []
        self.clipboard_text = ""

    def set_clipboard_text(self, text: str) -> None:
        self.calls.append(("clipboard", text))
        self.clipboard_text = text

    def set_foreground_window(self, handle: int) -> bool:
        self.calls.append(("foreground", handle))
        return self.foreground_ok

    def send_ctrl_v(self) -> bool:
        self.calls.append(("ctrl_v", None))
        return self.send_ok


def test_paste_waits_for_hotkey_release_and_targets_original_window() -> None:
    api = FakeWindowsApi()
    calls: list[str] = []
    paster = WindowsClipboardPaster(
        api,
        wait_until_hotkey_released=lambda: calls.append("released"),
    )

    paster.paste("Unicode 文本", target_window=123)

    assert calls == ["released"]
    assert api.calls == [
        ("clipboard", "Unicode 文本"),
        ("foreground", 123),
        ("ctrl_v", None),
    ]


def test_foreground_failure_preserves_recognized_text_in_clipboard() -> None:
    api = FakeWindowsApi(foreground_ok=False)
    paster = WindowsClipboardPaster(
        api, wait_until_hotkey_released=lambda: None
    )

    with pytest.raises(ClipboardPasteError, match="target window"):
        paster.paste("recoverable text", target_window=123)

    assert api.clipboard_text == "recoverable text"
    assert ("ctrl_v", None) not in api.calls


def test_send_input_failure_is_reported_without_clearing_clipboard() -> None:
    api = FakeWindowsApi(send_ok=False)
    paster = WindowsClipboardPaster(
        api, wait_until_hotkey_released=lambda: None
    )

    with pytest.raises(ClipboardPasteError, match="Ctrl\\+V"):
        paster.paste("recoverable text", target_window=123)

    assert api.clipboard_text == "recoverable text"
