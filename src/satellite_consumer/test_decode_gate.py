import asyncio
import threading
import time
import unittest

from satellite_consumer.consume import (
    _buffered_apply,
    _gated,
    _InOrderGate,
    _taken_by_writer,
)


def _run(items: list[int], slots: int, buffer_size: int, max_workers: int) -> tuple[list[int], int]:
    """Push ``items`` through the gated pipeline; return the results and the most items
    ever decoding or decoded and not yet taken by the writer."""
    lock = threading.Lock()
    held = 0
    most = 0

    def decode(item: int, wait_turn=None) -> int:
        nonlocal held, most
        time.sleep(0.01 * (item % 3))  # downloads finish out of order
        wait_turn()
        with lock:
            held += 1
            most = max(most, held)
        return item

    class CountingGate(_InOrderGate):
        def taken_one(self) -> None:
            nonlocal held
            with lock:
                held -= 1
            super().taken_one()

    gate = CountingGate(slots)

    async def consume() -> list[int]:
        out = []
        async for result in _taken_by_writer(
            _buffered_apply(
                enumerate(items),
                lambda job: _gated(decode, gate, job),
                buffer_size=buffer_size,
                max_workers=max_workers,
                executor="threads",
            ),
            gate,
        ):
            await asyncio.sleep(0.005)  # the write
            out.append(result)
        return out

    return asyncio.run(consume()), most


class TestDecodeGate(unittest.TestCase):
    def test_results_keep_input_order_and_bound_decoded_images(self) -> None:
        for slots in (1, 2):
            with self.subTest(slots=slots):
                results, most = _run(list(range(12)), slots=slots, buffer_size=4, max_workers=4)
                self.assertEqual(results, list(range(12)))
                self.assertLessEqual(most, slots)

    def test_more_slots_than_workers_does_not_deadlock(self) -> None:
        results, _ = _run(list(range(5)), slots=3, buffer_size=2, max_workers=1)
        self.assertEqual(results, list(range(5)))

    def test_taking_an_item_lets_the_next_one_decode(self) -> None:
        gate = _InOrderGate(1)

        async def first_only() -> int:
            async def two():
                yield 1
                yield 2

            async for item in _taken_by_writer(two(), gate):
                return item
            return -1

        self.assertEqual(asyncio.run(first_only()), 1)
        # Item 1 may decode once the writer took item 0; this would block otherwise.
        gate.wait_turn(1)


if __name__ == "__main__":
    unittest.main()
