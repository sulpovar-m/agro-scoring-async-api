import asyncio
import logging
from dataclasses import dataclass

from app.config import MAX_BATCH_SIZE, MAX_BATCH_WAIT_MS, QUEUE_MAX_SIZE
from app.errors import ServiceError
from app.schemas.prediction import FarmRequest

logger = logging.getLogger(__name__)


@dataclass
class QueueItem:
    data: FarmRequest
    request_id: str
    future: asyncio.Future


class DynamicBatcher:
    def __init__(self, scoring):
        self.scoring = scoring
        self.queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
        self.task = None

    def start(self):
        self.task = asyncio.create_task(self.run())

    async def stop(self):
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass
        while not self.queue.empty():
            item = self.queue.get_nowait()
            if not item.future.done():
                item.future.set_exception(ServiceError(503, "Worker stopped"))
            self.queue.task_done()

    async def submit(self, data, request_id):
        self.scoring.validate([data])
        if self.task is None or self.task.done():
            raise ServiceError(503, "Worker is unavailable")
        future = asyncio.get_running_loop().create_future()
        try:
            self.queue.put_nowait(QueueItem(data, request_id, future))
        except asyncio.QueueFull as error:
            raise ServiceError(429, "Prediction queue is full") from error
        return await future

    async def run(self):
        while True:
            batch = [await self.queue.get()]
            try:
                deadline = asyncio.get_running_loop().time() + MAX_BATCH_WAIT_MS / 1000
                while len(batch) < MAX_BATCH_SIZE:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        batch.append(
                            await asyncio.wait_for(self.queue.get(), remaining)
                        )
                    except TimeoutError:
                        break
                active = [item for item in batch if not item.future.cancelled()]
                if active:
                    logger.info(
                        "Dynamic batch size=%s request_ids=%s",
                        len(active),
                        [item.request_id for item in active],
                    )
                    results = await self.scoring.predict_batch(
                        [i.data for i in active],
                        [i.request_id for i in active],
                        "queued",
                    )
                    for item, result in zip(active, results, strict=True):
                        if not item.future.done():
                            item.future.set_result(result)
            except asyncio.CancelledError:
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(ServiceError(503, "Worker stopped"))
                raise
            except Exception as error:
                logger.exception(
                    "Dynamic batch failed request_ids=%s",
                    [item.request_id for item in batch],
                )
                failure = (
                    error
                    if isinstance(error, ServiceError)
                    else ServiceError(500, "Internal server error")
                )
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(failure)
            finally:
                for _ in batch:
                    self.queue.task_done()
