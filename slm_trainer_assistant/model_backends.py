"""Model backend interface for baseline eval collection.

Real providers can be added later behind this interface. The current stub backend
keeps the runner deterministic and testable before model/API integrations exist.
"""

from __future__ import annotations

from typing import Protocol

from slm_trainer_assistant.schemas import EvalExample


class ModelBackend(Protocol):
    @property
    def name(self) -> str:
        """Stable backend name stored in eval reports."""

    def generate(self, example: EvalExample) -> str:
        """Return a response for one eval example."""


class StubBackend:
    @property
    def name(self) -> str:
        return "stub"

    def generate(self, example: EvalExample) -> str:
        media_note = ""
        if example.media:
            media_paths = ", ".join(media.path for media in example.media)
            media_note = f" media={media_paths}"
        return (
            f"[stub:{example.id}] Baseline placeholder for "
            f"{example.category}/{example.difficulty}:{media_note} {example.question}"
        )


def get_backend(name: str) -> ModelBackend:
    normalized = name.strip().lower()
    if normalized == "stub":
        return StubBackend()
    raise ValueError(f"unknown backend '{name}'. Available backends: stub")
