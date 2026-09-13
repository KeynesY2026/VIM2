import pytest

from vim2.state import AppState, StateMachine


def test_happy_path_transitions_return_to_ready() -> None:
    machine = StateMachine()

    for state in (
        AppState.MODEL_LOADING,
        AppState.READY,
        AppState.RECORDING,
        AppState.FINALIZING,
        AppState.READY,
    ):
        machine.transition_to(state)

    assert machine.current is AppState.READY


def test_recording_cannot_start_while_model_is_loading() -> None:
    machine = StateMachine()
    machine.transition_to(AppState.MODEL_LOADING)

    with pytest.raises(ValueError, match="model_loading.*recording"):
        machine.transition_to(AppState.RECORDING)


def test_retry_pending_blocks_model_switch_until_audio_is_resolved() -> None:
    machine = StateMachine()
    machine.transition_to(AppState.MODEL_LOADING)
    machine.transition_to(AppState.READY)
    machine.transition_to(AppState.RECORDING)
    machine.transition_to(AppState.FINALIZING)
    machine.transition_to(AppState.RETRY_PENDING)

    with pytest.raises(ValueError, match="retry_pending.*model_switching"):
        machine.transition_to(AppState.MODEL_SWITCHING)

    machine.transition_to(AppState.FINALIZING)
    machine.transition_to(AppState.READY)
    machine.transition_to(AppState.MODEL_SWITCHING)

    assert machine.current is AppState.MODEL_SWITCHING
