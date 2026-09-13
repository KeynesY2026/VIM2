from __future__ import annotations

from enum import StrEnum


class AppState(StrEnum):
    STARTING = "starting"
    MODEL_LOADING = "model_loading"
    READY = "ready"
    RECORDING = "recording"
    LIVE_TRANSCRIBING = "live_transcribing"
    FINALIZING = "finalizing"
    RETRY_PENDING = "retry_pending"
    MODEL_SWITCHING = "model_switching"
    ERROR = "error"
    EXITING = "exiting"


_TRANSITIONS: dict[AppState, frozenset[AppState]] = {
    AppState.STARTING: frozenset(
        {AppState.MODEL_LOADING, AppState.ERROR, AppState.EXITING}
    ),
    AppState.MODEL_LOADING: frozenset(
        {AppState.READY, AppState.ERROR, AppState.EXITING}
    ),
    AppState.READY: frozenset(
        {
            AppState.RECORDING,
            AppState.MODEL_SWITCHING,
            AppState.ERROR,
            AppState.EXITING,
        }
    ),
    AppState.RECORDING: frozenset(
        {
            AppState.LIVE_TRANSCRIBING,
            AppState.FINALIZING,
            AppState.READY,
            AppState.ERROR,
            AppState.EXITING,
        }
    ),
    AppState.LIVE_TRANSCRIBING: frozenset(
        {
            AppState.RECORDING,
            AppState.FINALIZING,
            AppState.READY,
            AppState.ERROR,
            AppState.EXITING,
        }
    ),
    AppState.FINALIZING: frozenset(
        {
            AppState.READY,
            AppState.RETRY_PENDING,
            AppState.ERROR,
            AppState.EXITING,
        }
    ),
    AppState.RETRY_PENDING: frozenset(
        {AppState.FINALIZING, AppState.READY, AppState.EXITING}
    ),
    AppState.MODEL_SWITCHING: frozenset(
        {AppState.READY, AppState.MODEL_LOADING, AppState.ERROR, AppState.EXITING}
    ),
    AppState.ERROR: frozenset(
        {
            AppState.MODEL_LOADING,
            AppState.READY,
            AppState.FINALIZING,
            AppState.EXITING,
        }
    ),
    AppState.EXITING: frozenset(),
}


class StateMachine:
    def __init__(self) -> None:
        self.current = AppState.STARTING

    def transition_to(self, next_state: AppState) -> None:
        if next_state not in _TRANSITIONS[self.current]:
            raise ValueError(
                f"Invalid state transition: {self.current.value} -> {next_state.value}"
            )
        self.current = next_state
