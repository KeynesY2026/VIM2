import time

from vim2.hotkey import (
    HotkeyDispatcher,
    HotkeyMatcher,
    KeyEvent,
    key_name_from_virtual_key,
    parse_hotkey,
)


def test_generic_modifier_matches_either_side_and_allows_extra_keys() -> None:
    matcher = HotkeyMatcher(parse_hotkey("Ctrl+K"))

    assert not matcher.process(KeyEvent("LeftCtrl", True)).triggered
    assert not matcher.process(KeyEvent("Shift", True)).triggered
    decision = matcher.process(KeyEvent("K", True))

    assert decision.triggered
    assert decision.suppress


def test_specific_modifier_does_not_match_other_side() -> None:
    matcher = HotkeyMatcher(parse_hotkey("RightAlt"))

    decision = matcher.process(KeyEvent("LeftAlt", True))

    assert not decision.triggered
    assert not decision.suppress


def test_auto_repeat_triggers_only_once_until_hotkey_is_fully_released() -> None:
    matcher = HotkeyMatcher(parse_hotkey("RightAlt"))

    first = matcher.process(KeyEvent("RightAlt", True))
    repeated = matcher.process(KeyEvent("RightAlt", True))
    matcher.process(KeyEvent("RightAlt", False))
    second = matcher.process(KeyEvent("RightAlt", True))

    assert first.triggered
    assert not repeated.triggered
    assert second.triggered


def test_configured_keys_are_suppressed_on_press_and_release() -> None:
    matcher = HotkeyMatcher(parse_hotkey("LeftCtrl+RightAlt"))

    events = (
        KeyEvent("LeftCtrl", True),
        KeyEvent("RightAlt", True),
        KeyEvent("RightAlt", False),
        KeyEvent("LeftCtrl", False),
    )

    assert all(matcher.process(event).suppress for event in events)


def test_invalid_hotkey_is_rejected() -> None:
    for value in ("", "Ctrl++K", "NotAKey"):
        try:
            parse_hotkey(value)
        except ValueError:
            continue
        raise AssertionError(f"{value!r} should be invalid")


def test_semantically_duplicate_aliases_are_collapsed() -> None:
    assert parse_hotkey("Ctrl+Control+K").keys == ("Ctrl", "K")


def test_dispatcher_debounces_rapid_second_toggle(monkeypatch) -> None:
    now = 10.0
    monkeypatch.setattr(time, "monotonic", lambda: now)
    actions: list[str] = []
    dispatcher = HotkeyDispatcher(
        parse_hotkey("RightAlt"),
        on_toggle=lambda: actions.append("toggle"),
        on_cancel=lambda: None,
        is_cancellable=lambda: False,
    )

    dispatcher.process(
        KeyEvent("RightAlt", True),
        lower_integrity_injected=False,
        vim2_injected=False,
    )
    dispatcher.process(
        KeyEvent("RightAlt", False),
        lower_integrity_injected=False,
        vim2_injected=False,
    )
    now += 0.1
    dispatcher.process(
        KeyEvent("RightAlt", True),
        lower_integrity_injected=False,
        vim2_injected=False,
    )

    assert actions == ["toggle"]

    dispatcher.process(
        KeyEvent("RightAlt", False),
        lower_integrity_injected=False,
        vim2_injected=False,
    )
    now += 0.2
    dispatcher.process(
        KeyEvent("RightAlt", True),
        lower_integrity_injected=False,
        vim2_injected=False,
    )

    assert actions == ["toggle", "toggle"]


def test_dispatcher_allows_rdp_input_but_rejects_lower_integrity_injection() -> None:
    actions: list[str] = []
    dispatcher = HotkeyDispatcher(
        parse_hotkey("RightAlt"),
        on_toggle=lambda: actions.append("toggle"),
        on_cancel=lambda: actions.append("cancel"),
        is_cancellable=lambda: True,
    )

    assert dispatcher.process(
        KeyEvent("RightAlt", True),
        lower_integrity_injected=False,
        vim2_injected=False,
    ) is True
    assert dispatcher.process(
        KeyEvent("RightAlt", False),
        lower_integrity_injected=False,
        vim2_injected=False,
    ) is True
    assert dispatcher.process(
        KeyEvent("RightAlt", True),
        lower_integrity_injected=True,
        vim2_injected=False,
    ) is False
    assert dispatcher.process(
        KeyEvent("RightAlt", True),
        lower_integrity_injected=False,
        vim2_injected=True,
    ) is False
    assert dispatcher.process(
        KeyEvent("Esc", True),
        lower_integrity_injected=False,
        vim2_injected=False,
    ) is True

    assert actions == ["toggle", "cancel"]


def test_virtual_key_mapping_preserves_modifier_side() -> None:
    assert key_name_from_virtual_key(0xA4) == "LeftAlt"
    assert key_name_from_virtual_key(0xA5) == "RightAlt"
    assert key_name_from_virtual_key(ord("K")) == "K"
