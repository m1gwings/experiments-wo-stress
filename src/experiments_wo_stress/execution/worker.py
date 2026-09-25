"""A run's component lifecycle, protocol steps, and checkpoint boundaries."""

from __future__ import annotations

import logging
import signal
import time
import traceback
from pathlib import Path
from typing import Any

from ..components.loading import construct, create_instance
from ..storage.experiment import load_instance, save_instance
from ..storage.files import StorageError, atomic_json, read_json
from ..storage.run import Recorder, RunStore
from ..study.rng import make_rngs
from ..study.specs import RunSpec
from .logging import run_logging

_WORKER_STOP_EVENT: Any = None


def initialize_worker(event: Any) -> None:
    """Install the shared stop event in a spawned process."""
    global _WORKER_STOP_EVENT
    _WORKER_STOP_EVENT = event
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class RunSession:
    """Own fresh components, RNG streams, and persistence for one active run.

    Components are constructed in scientific order and restored together with their
    RNGs. A checkpoint is published only after a complete protocol step and after
    the corresponding observations have been flushed.

    ``run`` first checks for reusable completion, then restores or initializes the
    components, advances the protocol, and commits either a pause or completion.
    The session controls persistence timing; the protocol owns interaction order,
    the algorithm owns learning state, and the data generator owns environment state.
    """

    def __init__(
        self,
        spec: RunSpec,
        root: Path,
        store: RunStore,
        execution: dict[str, Any],
        recording: dict[str, Any],
        logger: logging.Logger,
        stop_event: Any,
    ) -> None:
        self.spec = spec
        self.root = root
        self.store = store
        self.execution = execution
        self.recording = recording
        self.logger = logger
        self.stop_event = stop_event
        self.components: dict[str, Any] = {}
        self.metadata = read_json(store.directory / "metadata.json")
        self.rngs = {}
        self.protocol: Any = None
        self.recorder: Recorder | None = None
        self.checkpoint_time = 0.0
        self.checkpoint_step = 0

    def run(self, max_steps: int | None) -> str:
        """Reuse, resume, or finish the requested prefix; return its execution status."""
        if self.store.completion(self.spec.budget_steps) is not None:
            self.logger.info("Reusing completed results")
            return "skipped"
        self.restore_or_initialize()
        started_step = self.protocol.step
        while not self.protocol.is_finished():
            if self.stop_event.is_set() or (
                max_steps is not None and self.protocol.step - started_step >= max_steps
            ):
                self.checkpoint("paused")
                self.logger.info("Paused at step %s", self.protocol.step)
                return "paused"
            self.advance()
            if self.checkpoint_due() and not self.protocol.is_finished():
                self.checkpoint()
        self.finalize()
        return "completed"

    def restore_or_initialize(self) -> None:
        """Construct components and restore a complete checkpoint, or initialize once."""
        self.rngs = make_rngs(self.spec)
        if self.metadata.get("instance_id"):
            instance = load_instance(self.root, self.metadata["instance_id"])
        else:
            instance = create_instance(self.spec.data, self.rngs["instance"])
            self.metadata["instance_id"] = save_instance(
                self.root, instance, compression=self.store.compression
            )
            atomic_json(self.store.directory / "metadata.json", self.metadata)
        # Preserve construction order and register each component immediately, so
        # cleanup also runs if construction of a later component fails.
        self.components["algorithm"] = construct(
            self.spec.algorithm, self.rngs["algorithm"], logger=self.logger.getChild("algorithm")
        )
        self.components["data"] = construct(
            self.spec.data,
            self.rngs["data"],
            instance=instance,
            logger=self.logger.getChild("data"),
        )
        self.components["protocol"] = self.protocol = construct(
            self.spec.protocol, self.rngs["protocol"], logger=self.logger.getChild("protocol")
        )
        if self.spec.budget_steps is not None:
            setter = getattr(self.protocol, "set_budget", None)
            if not callable(setter):
                raise ValueError("A budget requires protocol.set_budget(steps)")
            setter(self.spec.budget_steps)
        checkpoint, manifest, restored_step = self.store.restore()
        if checkpoint is None:
            self.protocol.initialize(self.components["algorithm"], self.components["data"])
        else:
            for name, component in self.components.items():
                component.load_state_dict(checkpoint[name])
            # Construction or state loading may consume randomness. Restore RNGs
            # afterward so the next step starts at the saved draw positions.
            for name, rng in self.rngs.items():
                rng.bit_generator.state = checkpoint["rngs"][name]
            if self.protocol.step != restored_step:
                raise StorageError("Checkpoint protocol step does not match result boundary")
        self.logger.info(
            "Executing from step %s to %s",
            self.protocol.step,
            self.spec.budget_steps or "completion",
        )
        self.recorder = Recorder(self.store, self.recording, manifest)
        self.checkpoint_time = time.monotonic()
        self.checkpoint_step = self.protocol.step

    def advance(self) -> None:
        """Advance exactly one protocol step and record its observations."""
        previous_step = self.protocol.step
        observations = self.protocol.advance(self.components["algorithm"], self.components["data"])
        if self.protocol.step != previous_step + 1:
            raise ValueError("Protocol.advance() must increment step by exactly one")
        # A request-specific final sample would break prefix invariance when extended.
        final = self.protocol.is_finished() and not self.metadata["extendable"]
        self.recorder.record(self.protocol.step, observations, final=final)

    def checkpoint_due(self) -> bool:
        """Check the configured wall-clock and complete-step checkpoint intervals."""
        seconds = self.execution.get("checkpoint_seconds", 120)
        steps = self.execution.get("checkpoint_steps")
        elapsed = seconds is not None and time.monotonic() - self.checkpoint_time >= seconds
        advanced = steps is not None and self.protocol.step - self.checkpoint_step >= steps
        return elapsed or advanced

    def checkpoint(self, status: str = "running") -> None:
        """Commit observations, component state, and RNG state at the same boundary."""
        # Flushing creates chunks; RunStore's progress publication commits them
        # together with the component and RNG snapshot at this exact step.
        self.recorder.flush()
        state = {name: component.state_dict() for name, component in self.components.items()}
        state["rngs"] = {name: rng.bit_generator.state for name, rng in self.rngs.items()}
        self.store.checkpoint(state, self.recorder.manifest, self.protocol.step, status=status)
        self.checkpoint_time, self.checkpoint_step = time.monotonic(), self.protocol.step
        self.logger.debug("Checkpoint committed at step %s", self.protocol.step)

    def finalize(self) -> None:
        """Commit the final step before publishing a reusable completion."""
        self.checkpoint()
        self.store.finish(self.recorder.manifest, self.protocol.step)
        self.logger.info("Completed at step %s", self.protocol.step)

    def close(self) -> None:
        """Release component resources even when execution or restoration failed."""
        for component in self.components.values():
            close = getattr(component, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    self.logger.exception("Component cleanup failed")


def execute_run(
    spec_dict: dict[str, Any],
    root: str,
    execution: dict[str, Any],
    recording: dict[str, Any],
    max_steps: int | None,
    storage_id: str,
    *,
    stop_event: Any = None,
) -> tuple[str, str, str | None]:
    """Spawn-safe process entry point; construct all run resources inside the worker."""
    spec = RunSpec.from_dict(spec_dict)
    directory = Path(root) / "runs" / storage_id
    store = None
    session = None
    with run_logging(directory, spec.run_id, execution) as logger:
        try:
            store = RunStore(
                directory,
                compression=execution.get("compression", False),
                keep_checkpoints=execution.get("keep_checkpoints", 2),
            )
            session = RunSession(
                spec,
                Path(root),
                store,
                execution,
                recording,
                logger,
                _WORKER_STOP_EVENT if stop_event is None else stop_event,
            )
            return spec.run_id, session.run(max_steps), None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("Run failed: %s", error)
            if store is not None:
                try:
                    store.fail(error, traceback.format_exc())
                except OSError:
                    pass
            return spec.run_id, "failed", error
        finally:
            if session is not None:
                session.close()
