"""``VoxtralClient`` and ``ServerGuard`` — speech requests that survive a bad answer.

The first version of the brick A job posted every request through a thread
pool and let the first non-200 answer raise out of ``pool.map``: one timeout
in thirty thousand clips ended a paid A100 run. And the vLLM child process was
started once and never looked at again — dead server, dead job, silently.

Here a request is retried, a clip that still fails is *skipped and counted*,
and the server is restarted (a bounded number of times) when it dies. The job
loses one clip at worst, never the run.
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

Wave = tuple[Any, int]
"""(samples, sample_rate)."""


class Response(Protocol):
    status_code: int
    content: bytes
    text: str


class Http(Protocol):
    def post(self, url: str, *, json: dict[str, Any]) -> Response: ...


class Process(Protocol):
    returncode: int | None

    def poll(self) -> int | None: ...


def decode_wav(content: bytes) -> Wave:
    import soundfile as sf

    samples, rate = sf.read(io.BytesIO(content), dtype="float32")
    return samples, int(rate)


@dataclass
class VoxtralClient:
    """Concurrent ``/audio/speech`` requests; ``None`` for a clip that never came back."""

    http: Http
    model: str
    voice_args: dict[str, Any]
    base_url: str = ""
    concurrency: int = 16
    retries: int = 3
    backoff_s: float = 2.0
    sleep: Callable[[float], None] = time.sleep
    decode: Callable[[bytes], Wave] = decode_wav
    failures: int = field(default=0, init=False)

    def one(self, text: str) -> Wave | None:
        reason = "aucun essai"
        for attempt in range(self.retries):
            try:
                payload = {"input": text, "model": self.model, "response_format": "wav", **self.voice_args}
                response = self.http.post(f"{self.base_url}/audio/speech", json=payload)
                if response.status_code == 200:
                    return self.decode(response.content)
                reason = f"{response.status_code}: {response.text[:120]}"
            except Exception as error:  # timeout, reset, decode: every one of them means "ask again"
                reason = f"{type(error).__name__}: {str(error)[:120]}"
            if attempt + 1 < self.retries:
                self.sleep(self.backoff_s * (attempt + 1))
        self.failures += 1
        logger.warning("clip abandonné après %d essais (%s) : %r", self.retries, reason, text[:60])
        return None

    def speak(self, texts: list[str]) -> tuple[list[Any | None], int]:
        """One entry per text, in order; the sample rate of whatever came back."""
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            results = list(pool.map(self.one, texts))
        rate = next((wave[1] for wave in results if wave is not None), 0)
        return [wave[0] if wave is not None else None for wave in results], rate


class ServerRestartsExhaustedError(RuntimeError):
    """The server died more often than a healthy run can explain."""


@dataclass
class ServerGuard:
    """Keeps the ``vllm serve`` child alive across a long run."""

    start: Callable[[], tuple[Process, str]]
    max_restarts: int = 3
    note: Callable[[str], None] = logger.warning
    process: Process | None = field(default=None, init=False)
    base_url: str = field(default="", init=False)
    restarts: int = field(default=0, init=False)

    def ensure_alive(self) -> str:
        """The base URL of a live server, restarting it if it died."""
        if self.process is None:
            self.process, self.base_url = self.start()
            return self.base_url
        if self.process.poll() is None:
            return self.base_url
        if self.restarts >= self.max_restarts:
            code = self.process.returncode
            raise ServerRestartsExhaustedError(
                f"serveur vLLM mort {self.restarts + 1} fois (dernier code {code}) — voir vllm_serve.log"
            )
        self.restarts += 1
        self.note(
            f"serveur vLLM mort (code {self.process.returncode}) — redémarrage {self.restarts}/{self.max_restarts}"
        )
        self.process, self.base_url = self.start()
        return self.base_url
