from vim2.application import SingleInstanceGuard


class FakeKernel32:
    def __init__(self, last_error: int) -> None:
        self.last_error = last_error
        self.closed: list[int] = []

    def CreateMutexW(self, security, initially_owned, name: str) -> int:
        return 99

    def get_last_error(self) -> int:
        return self.last_error

    def CloseHandle(self, handle: int) -> None:
        self.closed.append(handle)


def test_first_process_owns_single_instance_mutex() -> None:
    api = FakeKernel32(last_error=0)
    guard = SingleInstanceGuard(api=api)

    assert guard.acquire()
    guard.close()

    assert api.closed == [99]


def test_second_process_is_rejected_and_handle_is_closed() -> None:
    api = FakeKernel32(last_error=183)
    guard = SingleInstanceGuard(api=api)

    assert not guard.acquire()

    assert api.closed == [99]
