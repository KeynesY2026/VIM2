from vim2.foreground import ForegroundWindowHistory


def test_history_keeps_last_application_and_ignores_vim2_and_taskbar() -> None:
    windows = {
        101: (10, "Notepad", "notepad.exe"),
        202: (20, "Shell_TrayWnd", "explorer.exe"),
        303: (30, "Qt5152QWindowIcon", "vim2.exe"),
        404: (40, "CabinetWClass", "explorer.exe"),
        505: (
            50,
            "Windows.UI.Core.CoreWindow",
            "ShellExperienceHost.exe",
        ),
        606: (60, "Windows.UI.Core.CoreWindow", "my_uwp_app.exe"),
    }
    history = ForegroundWindowHistory(
        process_id=30,
        describe_window=windows.get,
    )

    history.record(101)
    history.record(202)
    history.record(303)
    history.record(505)
    assert history.last_window == 101

    history.record(404)
    assert history.last_window == 404

    history.record(606)
    assert history.last_window == 606
