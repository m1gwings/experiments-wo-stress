"""Own fixed GPU visibility and the lifetime of dedicated spawned workers.

The brief parent-environment change is protected against other EWS GPU launches
and validation. Unrelated application threads must not launch subprocesses or
initialize CUDA while an EWS GPU process is being started: Python's public spawn
API has no per-child environment argument.
"""

from __future__ import annotations

import os
import signal
import threading
import traceback
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..study.config import ExperimentConfig
    from ..study.specs import RunSpec
    from .provenance import PreparedRequest


_GPU_LAUNCH_LOCK = threading.Lock()


def validate_gpu_environment() -> None:
    """Reject inherited CUDA masks without mistaking another EWS launch for one."""
    with _GPU_LAUNCH_LOCK:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            raise ValueError(
                "execution.gpu_ids cannot be used when CUDA_VISIBLE_DEVICES is already set; "
                "unset CUDA_VISIBLE_DEVICES and configure the desired GPU IDs explicitly"
            )


@dataclass(eq=False)
class GPUWorker:
    """Own one spawned process and its command pipe for one fixed GPU.

    The coordinator sends at most one command at a time. The process retains
    its inherited GPU visibility until shutdown; completing a run releases its
    components, not its GPU assignment. ``gpu_workers`` owns final cleanup.
    """

    gpu_id: int
    process: Any
    connection: Connection

    @property
    def pid(self) -> int:
        """Return the spawned process ID for operational diagnostics."""
        return self.process.pid

    def prepare(
        self, config: ExperimentConfig
    ) -> tuple[list[RunSpec], dict[str, Any], PreparedRequest]:
        """Plan and inspect components inside an already GPU-bound process."""
        self.connection.send(("prepare", config))
        wait_gpu_workers([self])
        return self.result()

    def submit(self, arguments: tuple) -> None:
        """Submit one run; receive its result before submitting another."""
        self.connection.send(("run", arguments))

    def result(self) -> Any:
        """Receive a command result or describe a worker crash or remote error."""
        try:
            # A study-created descendant may inherit this pipe under fork. Its
            # open descriptor must not hide the assigned worker's abrupt death.
            if self.process.exitcode is not None and not self.connection.poll():
                raise EOFError
            succeeded, value = self.connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(
                f"GPU worker {self.pid} assigned GPU {self.gpu_id} exited without a result"
            ) from exc
        if not succeeded:
            raise RuntimeError(f"GPU worker {self.pid} assigned GPU {self.gpu_id} failed:\n{value}")
        return value


def wait_gpu_workers(workers: Sequence[GPUWorker]) -> list[GPUWorker]:
    """Wait for results or process exits, including abrupt exits without a reply."""
    if not workers:
        return []
    watched = [worker.connection for worker in workers] + [
        worker.process.sentinel for worker in workers
    ]
    while True:
        ready = wait(watched, timeout=0.1)
        # A study may fork descendants that retain the process sentinel's write
        # end as well as its command pipe. Poll the actual child exit status too,
        # so their open descriptors cannot hide the assigned worker's death.
        completed = [
            worker
            for worker in workers
            if worker.connection in ready
            or worker.process.sentinel in ready
            or worker.process.exitcode is not None
        ]
        if completed:
            return completed


def _worker_commands(connection: Connection, stop_event: Any, progress_queue: Any = None) -> None:
    """Serve plain commands after spawn has inherited its final CUDA visibility."""
    # The coordinator handles both signals and asks for safe checkpoints through
    # the event. Importing components, including custom planners, happens later.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    from .provenance import prepare_gpu_request
    from .worker import execute_run, initialize_worker

    initialize_worker(stop_event, progress_queue)
    try:
        while True:
            command, payload = connection.recv()
            if command == "close":
                return
            try:
                if command == "prepare":
                    result = prepare_gpu_request(payload)
                elif command == "run":
                    result = execute_run(*payload)
                else:
                    raise ValueError(f"Unknown GPU worker command: {command}")
            except Exception:
                # Return text rather than pickling a study-defined exception,
                # which could import scientific modules into the coordinator.
                connection.send((False, traceback.format_exc()))
            else:
                connection.send((True, result))
                # Preparation can describe the entire study; do not retain that
                # payload while executing the next scientific run in this worker.
                del result
    except (EOFError, BrokenPipeError):
        return
    finally:
        connection.close()


def _start_gpu_worker(
    gpu_id: int, context: Any, stop_event: Any, progress_queue: Any = None
) -> GPUWorker:
    """Launch with visibility established before spawn reimports caller modules."""
    parent_connection, child_connection = context.Pipe()
    process = context.Process(
        target=_worker_commands,
        args=(child_connection, stop_event, progress_queue),
        name=f"ews-gpu-{gpu_id}",
    )
    try:
        # An executor initializer is too late: spawn reimports __main__ first.
        # Process.start synchronously creates the OS process, so the child gets
        # this mask before any Python imports. Serialize EWS launches and restore
        # the parent immediately; never change visibility between worker tasks.
        with _GPU_LAUNCH_LOCK:
            previous = os.environ.get("CUDA_VISIBLE_DEVICES")
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                process.start()
            finally:
                if previous is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = previous
    except BaseException:
        parent_connection.close()
        if process.pid is not None:
            # A launch interrupted after OS creation must not orphan its child.
            process.kill()
            process.join()
        process.close()
        raise
    finally:
        # Only the child may keep this endpoint: otherwise its abrupt death
        # would leave the parent's receive waiting forever instead of seeing EOF.
        child_connection.close()
    return GPUWorker(gpu_id, process, parent_connection)


def _join_gpu_worker(worker: GPUWorker) -> None:
    """Drain abandoned replies so graceful shutdown cannot block on a full pipe."""
    watched = [worker.connection, worker.process.sentinel]
    try:
        while worker.process.exitcode is None:
            ready = wait(watched, timeout=0.1)
            # Descendants may retain the sentinel; only the assigned worker's
            # lifetime belongs to this resource owner.
            if worker.process.exitcode is not None:
                break
            if worker.connection not in ready:
                continue
            try:
                worker.connection.recv()
            except (EOFError, OSError):
                watched = [worker.process.sentinel]
        worker.process.join()
    finally:
        worker.connection.close()
        worker.process.close()


@contextmanager
def gpu_workers(
    gpu_ids: Sequence[int], context: Any, stop_event: Any, progress_queue: Any = None
) -> Iterator[list[GPUWorker]]:
    """Launch one worker per GPU and close every process and pipe on all exits.

    Explicit IDs refer to the unmasked CUDA device order. Reject an inherited
    mask instead of guessing whether IDs refer to physical or remapped devices.
    Graceful interruption uses the shared event; shutdown commands follow any
    active run so its checkpoint and component cleanup can finish first.
    """
    validate_gpu_environment()
    workers: list[GPUWorker] = []
    try:
        for gpu_id in gpu_ids:
            workers.append(_start_gpu_worker(gpu_id, context, stop_event, progress_queue))
        yield workers
    except BaseException:
        stop_event.set()
        raise
    finally:
        for worker in workers:
            try:
                worker.connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for worker in workers:
            _join_gpu_worker(worker)
