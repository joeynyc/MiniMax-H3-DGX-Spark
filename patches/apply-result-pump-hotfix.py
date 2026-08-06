#!/usr/bin/env python3
"""Harden the vLLM-Omni async diffusion result pump.

Upstream bug observed on 2026-08-05: a long denoise step (50 steps / 4 s, where
the tail steps ran 44-48 s each) outlived ``_ASYNC_OUTPUT_TIMEOUT`` (30 s), so
``DiffusionEngine.step_streaming`` cancelled the output future.  The worker
still finished the step and handed the result back, but ``_result_pump`` called
``fut.set_result()`` on the now-cancelled future.  That raises
``InvalidStateError``, and because the dispatch half of the pump loop has no
exception guard, the sole ``DiffusionResultPump`` thread died.

The server stayed superficially healthy afterwards (``/health`` 200,
``/v1/models`` 200) but every later request hung forever: the GPU produced the
video and nothing was left alive to hand it back.  Recovery required a restart.

Three defects, three fixes:

1. Settling a cancelled/already-done future must be a no-op, not a raise.
2. The pump thread must survive an unexpected per-message exception instead of
   unwinding and leaving the executor without a reader.
3. The 30 s D2H/SHM wait is far too tight for slow steps.  It only covers the
   background device-to-host copy (normally milliseconds), so raising it costs
   nothing in the common case and stops the cancellation being provoked.

Each substitution is anchored to exact upstream text and asserted afterwards,
so a base-image change breaks the build loudly rather than silently skipping a
fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

DIST = Path("/usr/local/lib/python3.12/dist-packages/vllm_omni")
EXECUTOR = DIST / "diffusion/executor/multiproc_executor.py"
ENGINE = DIST / "diffusion/diffusion_engine.py"

MARKER = "_h3_settle_result"


class PatchError(RuntimeError):
    pass


def substitute(text: str, old: str, new: str, *, count: int, what: str) -> str:
    """Replace `old` exactly `count` times or fail loudly."""
    found = text.count(old)
    if found != count:
        raise PatchError(f"{what}: expected {count} occurrence(s) of anchor, found {found}")
    return text.replace(old, new)


# --------------------------------------------------------------------------
# 1 + 2: multiproc_executor.py — safe settling and a pump thread that survives
# --------------------------------------------------------------------------

HELPERS = '''

# --- MiniMax-H3-DGX-Spark hotfix -------------------------------------------
# A future is cancelled when its caller timed out (see _ASYNC_OUTPUT_TIMEOUT).
# The worker result still arrives afterwards; settling it must not raise, or it
# takes the sole result-pump thread down with it.


def _h3_settle_result(fut, value, *, what: str = "result") -> bool:
    """Best-effort ``fut.set_result``. Returns True when the future took it."""
    if fut is None or fut.done():
        return False
    try:
        fut.set_result(value)
        return True
    except concurrent.futures.InvalidStateError:
        logger.warning("Dropping %s for a future that is no longer awaiting it", what)
        return False


def _h3_settle_exception(fut, exc, *, what: str = "error") -> bool:
    """Best-effort ``fut.set_exception``. Returns True when the future took it."""
    if fut is None or fut.done():
        return False
    try:
        fut.set_exception(exc)
        return True
    except concurrent.futures.InvalidStateError:
        logger.warning("Dropping %s for a future that is no longer awaiting it", what)
        return False


# --- end hotfix ------------------------------------------------------------
'''

PUMP_WRAPPER = '''    def _h3_pump_forever(self) -> None:
        """Run the result pump so no single bad message can kill the thread.

        ``_result_pump`` only guards its dequeue; an exception raised while
        dispatching a message unwinds the thread and leaves the executor with
        no reader, which strands every later request. Re-enter the loop instead.
        """
        while not self._pump_stop.is_set():
            try:
                self._result_pump()
                return  # clean exit: the loop observed _pump_stop / _is_failed
            except Exception:
                logger.exception("Result pump raised; restarting dispatch loop")
                # Back off on the stop event rather than spinning: a failure
                # that reproduces immediately would otherwise pin a core and
                # flood the log, and waiting here still observes shutdown at
                # once instead of sleeping through it.
                if self._pump_stop.wait(1.0):
                    return

    def _result_pump(self) -> None:'''


def patch_executor(text: str) -> str:
    if MARKER in text:
        raise PatchError("multiproc_executor.py already patched")

    # Helpers go after the logger is defined so they can use it.
    text = substitute(
        text,
        "logger = init_logger(__name__)\n",
        "logger = init_logger(__name__)\n" + HELPERS,
        count=1,
        what="executor: helper injection",
    )

    # Route the thread through the restarting wrapper.
    text = substitute(
        text,
        "target=self._result_pump, daemon=True, name=\"DiffusionResultPump\"",
        "target=self._h3_pump_forever, daemon=True, name=\"DiffusionResultPump\"",
        count=1,
        what="executor: pump thread target",
    )
    text = substitute(
        text,
        "    def _result_pump(self) -> None:",
        PUMP_WRAPPER,
        count=1,
        what="executor: pump wrapper",
    )

    # RPC_RESULT / COMPUTE_DONE branch.
    text = substitute(
        text,
        """                if fut is not None and not fut.done():
                    if msg.error:
                        fut.set_exception(RuntimeError(msg.error))
                    else:
                        fut.set_result(msg)""",
        """                if fut is not None and not fut.done():
                    if msg.error:
                        _h3_settle_exception(fut, RuntimeError(msg.error), what="rpc error")
                    else:
                        _h3_settle_result(fut, msg, what="rpc result")""",
        count=1,
        what="executor: rpc branch",
    )

    # Batch split branch: the freshly created future is private, so only the
    # shared `pending` future can race with a cancelling caller.
    text = substitute(
        text,
        """                            if pending is not None and not pending.done():
                                pending.set_result(per_req_result)""",
        """                            if pending is not None and not pending.done():
                                _h3_settle_result(pending, per_req_result, what="batch output")""",
        count=1,
        what="executor: batch branch",
    )

    # OUTPUT_READY branch — the exact site that killed the thread.
    text = substitute(
        text,
        """                    if fut is not None and not fut.done():
                        if msg.error:
                            fut.set_exception(RuntimeError(msg.error))
                        else:
                            try:
                                unpack_diffusion_output_shm(msg.output)
                            except Exception as e:
                                logger.exception("SHM unpack failed in result pump")
                                fut.set_exception(e)
                                continue
                            fut.set_result(msg.output)""",
        """                    if fut is not None and not fut.done():
                        if msg.error:
                            _h3_settle_exception(fut, RuntimeError(msg.error), what="output error")
                        else:
                            try:
                                unpack_diffusion_output_shm(msg.output)
                            except Exception as e:
                                logger.exception("SHM unpack failed in result pump")
                                _h3_settle_exception(fut, e, what="unpack error")
                                continue
                            _h3_settle_result(fut, msg.output, what="output")""",
        count=1,
        what="executor: output_ready branch",
    )

    if "concurrent.futures" not in text.split("logger = init_logger")[0]:
        raise PatchError("executor: concurrent.futures is not imported at module scope")

    return text


# --------------------------------------------------------------------------
# 3: diffusion_engine.py — make the D2H/SHM wait configurable and generous
# --------------------------------------------------------------------------

TIMEOUT_BLOCK = '''import os as _h3_os

# MiniMax-H3-DGX-Spark hotfix: this bounds only the background device-to-host /
# shared-memory copy of a finished step, which normally completes in
# milliseconds. The upstream 30 s was tight enough that a slow denoise step
# (48 s/step at 50 steps) tripped it, cancelling the output future mid-flight.
# Override with H3_ASYNC_OUTPUT_TIMEOUT (seconds).
_H3_DEFAULT_ASYNC_OUTPUT_TIMEOUT = 900.0


def _h3_async_output_timeout() -> float:
    """Read the override, falling back rather than failing to import.

    This runs at import time, so raising here would take the whole server
    down at startup over a typo in an environment variable — and the
    traceback would point at an import, not at the setting that caused it.
    """
    raw = _h3_os.environ.get("H3_ASYNC_OUTPUT_TIMEOUT")
    if raw is None or raw.strip() == "":
        return _H3_DEFAULT_ASYNC_OUTPUT_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Ignoring invalid H3_ASYNC_OUTPUT_TIMEOUT=%r; using %.1f s",
            raw, _H3_DEFAULT_ASYNC_OUTPUT_TIMEOUT)
        return _H3_DEFAULT_ASYNC_OUTPUT_TIMEOUT
    if value <= 0:
        logger.warning(
            "Ignoring non-positive H3_ASYNC_OUTPUT_TIMEOUT=%r; using %.1f s",
            raw, _H3_DEFAULT_ASYNC_OUTPUT_TIMEOUT)
        return _H3_DEFAULT_ASYNC_OUTPUT_TIMEOUT
    return value


_ASYNC_OUTPUT_TIMEOUT = _h3_async_output_timeout()'''


def patch_engine(text: str) -> str:
    if "H3_ASYNC_OUTPUT_TIMEOUT" in text:
        raise PatchError("diffusion_engine.py already patched")

    return substitute(
        text,
        "_ASYNC_OUTPUT_TIMEOUT = 30.0  # seconds",
        TIMEOUT_BLOCK,
        count=1,
        what="engine: async output timeout",
    )


# --------------------------------------------------------------------------


def verify(path: Path, required: list[str], counts: dict[str, int]) -> None:
    text = path.read_text(encoding="utf-8")
    for needle in required:
        if needle not in text:
            raise PatchError(f"{path.name}: post-patch check failed, missing {needle!r}")
    for needle, expected in counts.items():
        found = text.count(needle)
        if found != expected:
            raise PatchError(
                f"{path.name}: post-patch check failed, {needle!r} appears "
                f"{found} time(s), expected {expected}"
            )
    compile(text, str(path), "exec")  # syntax must still be valid


def main() -> int:
    try:
        for path, patcher in ((EXECUTOR, patch_executor), (ENGINE, patch_engine)):
            if not path.is_file():
                raise PatchError(f"target not found: {path}")
            path.write_text(patcher(path.read_text(encoding="utf-8")), encoding="utf-8")

        verify(
            EXECUTOR,
            required=[
                "_h3_settle_result",
                "_h3_settle_exception",
                "target=self._h3_pump_forever",
                "def _h3_pump_forever",
            ],
            counts={
                # Every settle on a caller-visible future now goes through the
                # helpers. The one raw set_result left is in the `elif batch_id`
                # cached branch, whose future is constructed on the spot and so
                # can never be cancelled by a timed-out caller.
                "fut.set_result(msg.output)": 1,
                "fut.set_result(msg)\n": 0,
                "pending.set_result(per_req_result)": 0,
                "_h3_settle_result(": 4,  # 1 definition + 3 call sites
                "_h3_settle_exception(": 4,  # 1 definition + 3 call sites
            },
        )
        verify(
            ENGINE,
            required=["H3_ASYNC_OUTPUT_TIMEOUT"],
            counts={"_ASYNC_OUTPUT_TIMEOUT = 30.0": 0},
        )
    except PatchError as exc:
        print(f"result-pump hotfix FAILED: {exc}", file=sys.stderr)
        return 1

    print("result-pump hotfix applied and verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
