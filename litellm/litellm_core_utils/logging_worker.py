import asyncio
import contextlib
import contextvars
from typing import Coroutine, Optional

from typing_extensions import TypedDict

from litellm._logging import verbose_logger


class LoggingTask(TypedDict):
    """
    A logging task with its associated context to ensure logging is executed in
    the original task's context.
    """

    coroutine: Coroutine
    context: contextvars.Context


class LoggingWorker:
    """
    A simple, async logging worker that processes log coroutines in the background.
    Designed to be best-effort with bounded queues to prevent backpressure.

    This leads to a +200 RPS performance improvement when using LiteLLM Python SDK or Proxy Server.
    - Use this to queue coroutine tasks that are not critical to the main flow of the application. e.g Success/Error callbacks, logging, etc.
    """

    LOGGING_WORKER_MAX_QUEUE_SIZE = 50_000
    LOGGING_WORKER_MAX_TIME_PER_COROUTINE = 20.0

    MAX_ITERATIONS_TO_CLEAR_QUEUE = 200
    MAX_TIME_TO_CLEAR_QUEUE = 5.0

    def __init__(
        self,
        timeout: float = LOGGING_WORKER_MAX_TIME_PER_COROUTINE,
        max_queue_size: int = LOGGING_WORKER_MAX_QUEUE_SIZE,
    ):
        self.timeout = timeout
        self.max_queue_size = max_queue_size
        self._queue: Optional[asyncio.Queue[LoggingTask]] = None
        self._worker_task: Optional[asyncio.Task] = None

    def _ensure_queue(self) -> None:
        """Initialize the queue if it doesn't exist."""
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self.max_queue_size)

    def start(self) -> None:
        """Start the logging worker. Idempotent - safe to call multiple times."""
        self._ensure_queue()
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def _worker_loop(self) -> None:
        """Main worker loop that processes log coroutines sequentially."""
        try:
            _queue = self._queue
            if _queue is None:
                return

            # Cache frequently used locals to increase loop speed
            queue_get = _queue.get
            queue_task_done = _queue.task_done
            wait_for = asyncio.wait_for
            timeout = self.timeout

            while True:
                # Process one coroutine at a time to keep event loop load predictable
                task = await queue_get()
                try:
                    # Run the coroutine in its original context
                    await wait_for(
                        task["context"].run(asyncio.create_task, task["coroutine"]),
                        timeout=timeout,
                    )
                except Exception as e:
                    # Prevent expensive formatting if not logging this level
                    if verbose_logger.isEnabledFor(verbose_logger.level):
                        verbose_logger.exception(f"LoggingWorker error: {e}")
                    pass
                finally:
                    queue_task_done()

        except asyncio.CancelledError:
            # Minimize logging verbosity for high-performance shutdown
            if verbose_logger.isEnabledFor(verbose_logger.level):
                verbose_logger.debug("LoggingWorker cancelled during shutdown")
            # Attempt to clear remaining items to prevent "never awaited" warnings
            await self.clear_queue()

    def enqueue(self, coroutine: Coroutine) -> None:
        """
        Add a coroutine to the logging queue.
        Hot path: never blocks, drops logs if queue is full.
        """
        if self._queue is None:
            return

        try:
            # Capture the current context when enqueueing
            task = LoggingTask(coroutine=coroutine, context=contextvars.copy_context())
            self._queue.put_nowait(task)
        except asyncio.QueueFull as e:
            verbose_logger.exception(f"LoggingWorker queue is full: {e}")
            # Drop logs on overload to protect request throughput
            pass

    def ensure_initialized_and_enqueue(self, async_coroutine: Coroutine):
        """
        Ensure the logging worker is initialized and enqueue the coroutine.
        """
        self.start()
        self.enqueue(async_coroutine)

    async def stop(self) -> None:
        """Stop the logging worker and clean up resources."""
        if self._worker_task:
            self._worker_task.cancel()
            with contextlib.suppress(Exception):
                await self._worker_task
            self._worker_task = None

    async def flush(self) -> None:
        """Flush the logging queue."""
        if self._queue is None:
            return
        while not self._queue.empty():
            await self._queue.join()

    async def clear_queue(self):
        """
        Clear the queue with a maximum time limit.
        """
        _queue = self._queue
        if _queue is None:
            return

        loop_time = asyncio.get_event_loop().time
        start_time = loop_time()

        queue_get_nowait = _queue.get_nowait
        queue_task_done = _queue.task_done
        wait_for = asyncio.wait_for
        timeout = self.timeout
        max_iterations = self.MAX_ITERATIONS_TO_CLEAR_QUEUE
        max_time = self.MAX_TIME_TO_CLEAR_QUEUE

        for _ in range(max_iterations):
            # Check if we've exceeded the maximum time
            if (loop_time() - start_time) >= max_time:
                if verbose_logger.isEnabledFor(verbose_logger.level):
                    verbose_logger.warning(f"clear_queue exceeded max_time of {max_time}s, stopping early")
                break

            try:
                task = queue_get_nowait()
                # Await the coroutine to properly execute and avoid "never awaited" warnings
                # Await the coroutine to properly execute and avoid "never awaited" warnings
                try:
                    await wait_for(
                        task["context"].run(asyncio.create_task, task["coroutine"]),
                        timeout=timeout,
                    )
                except Exception:
                    # Suppress errors during cleanup
                    pass
                queue_task_done()  # If you're using join() elsewhere
            except asyncio.QueueEmpty:
                break


# Global instance for backward compatibility
GLOBAL_LOGGING_WORKER = LoggingWorker()
