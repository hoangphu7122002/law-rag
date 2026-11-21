import asyncio

import pytest

from serve.limiter import Limiter, Overloaded


def test_requests_beyond_the_slots_wait_and_beyond_the_queue_are_rejected():
    async def run():
        lim = Limiter(max_concurrent=1, max_queue=1)
        gate = asyncio.Event()
        order = []

        async def job(name):
            async with lim:
                order.append(name)
                await gate.wait()

        a = asyncio.create_task(job("a"))
        await asyncio.sleep(0)
        b = asyncio.create_task(job("b"))            # waits in the queue
        await asyncio.sleep(0)
        assert lim.running == 1 and lim.waiting == 1
        with pytest.raises(Overloaded):
            async with lim:                           # queue is full
                pass
        gate.set()
        await asyncio.gather(a, b)
        assert order == ["a", "b"] and lim.running == 0 and lim.waiting == 0

    asyncio.run(run())


def test_no_limit_by_default():
    async def run():
        lim = Limiter()
        async with lim, lim:
            return lim.running

    assert asyncio.run(run()) == 2
