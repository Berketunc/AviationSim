import asyncio

from pl_control.setpoint_queue import consume_latest


def test_consumes_only_newest_queued_setpoint():
    queue = asyncio.Queue()
    queue.put_nowait("old")
    queue.put_nowait("newer")
    queue.put_nowait("latest")

    assert consume_latest(queue, "current") == "latest"
    assert queue.empty()


def test_preserves_current_setpoint_when_queue_is_empty():
    assert consume_latest(asyncio.Queue(), "current") == "current"
