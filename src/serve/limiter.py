"""Admission control: at most `max_concurrent` requests run, up to `max_queue` wait, the rest get Overloaded (HTTP 429).
None = unlimited. Protects the LLM endpoint from a burst that would only grow every request's latency."""
import asyncio


class Overloaded(Exception):
    pass


class Limiter:
    def __init__(self, max_concurrent: int | None = None, max_queue: int | None = None):
        self.max_concurrent, self.max_queue = max_concurrent, max_queue
        self.running = self.waiting = 0
        self._sem = asyncio.Semaphore(max_concurrent) if max_concurrent else None

    async def __aenter__(self):
        if self._sem is None:
            self.running += 1
            return self
        if self._sem.locked():
            if self.max_queue is not None and self.waiting >= self.max_queue:
                raise Overloaded(f"{self.running} running, {self.waiting} waiting")
            self.waiting += 1
            try:
                await self._sem.acquire()
            finally:
                self.waiting -= 1
        else:
            await self._sem.acquire()
        self.running += 1
        return self

    async def __aexit__(self, *exc):
        self.running -= 1
        if self._sem is not None:
            self._sem.release()
