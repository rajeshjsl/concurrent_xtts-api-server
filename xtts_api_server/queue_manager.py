from dataclasses import dataclass
from typing import Optional, Callable, Any, Dict
import asyncio
from queue import Queue
import threading
import time
from loguru import logger

@dataclass
class TTSQueueItem:
    text: str
    speaker: str
    language: str
    output_path: str
    future: asyncio.Future
    loop: asyncio.AbstractEventLoop
    callback_fn: Optional[Callable[[str], Any]] = None

class TTSQueueManager:
    def __init__(self, tts_wrapper):
        self.tts_wrapper = tts_wrapper
        self.queue = Queue()
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.running = True
        self.processing_thread.start()
        self.current_item: Optional[TTSQueueItem] = None
        self._lock = threading.Lock()
        
    def shutdown(self):
        """Gracefully shutdown the queue manager"""
        logger.info("Initiating queue manager shutdown...")
        self.running = False
        
        if self.processing_thread.is_alive():
            logger.info("Waiting for processing thread to complete...")
            self.processing_thread.join(timeout=30)
            
            if self.processing_thread.is_alive():
                logger.warning("Processing thread did not complete within timeout")
            else:
                logger.info("Processing thread completed successfully")
        
        while not self.queue.empty():
            try:
                item = self.queue.get_nowait()
                if item.future and not item.future.done():
                    item.future.set_exception(
                        Exception("Server shutdown before request could be processed")
                    )
                self.queue.task_done()
            except Exception as e:
                logger.error(f"Error clearing queue during shutdown: {str(e)}")
        
        logger.info("Queue manager shutdown complete")

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status in a thread-safe way"""
        with self._lock:
            current_text = None
            if self.current_item:
                current_text = (
                    self.current_item.text[:100] + "..." 
                    if len(self.current_item.text) > 100 
                    else self.current_item.text
                )
            
            return {
                "queue_size": self.queue.qsize(),
                "is_processing": self.current_item is not None,
                "current_text": current_text,
                "current_speaker": self.current_item.speaker if self.current_item else None,
                "current_language": self.current_item.language if self.current_item else None
            }

    def _set_future_result(self, future: asyncio.Future, loop: asyncio.AbstractEventLoop, result):
        if not future.done():
            loop.call_soon_threadsafe(future.set_result, result)

    def _set_future_exception(self, future: asyncio.Future, loop: asyncio.AbstractEventLoop, exc):
        if not future.done():
            loop.call_soon_threadsafe(future.set_exception, exc)
    
    def _process_queue(self):
        while self.running:
            try:
                if not self.queue.empty():
                    with self._lock:
                        self.current_item = self.queue.get()
                    try:
                        output_file = self.tts_wrapper.process_tts_to_file(
                            text=self.current_item.text,
                            speaker_name_or_path=self.current_item.speaker,
                            language=self.current_item.language,
                            file_name_or_path=self.current_item.output_path
                        )
                        self._set_future_result(
                            self.current_item.future,
                            self.current_item.loop,
                            output_file
                        )
                        
                        if self.current_item.callback_fn:
                            self.current_item.loop.call_soon_threadsafe(
                                self.current_item.callback_fn,
                                output_file
                            )
                            
                    except Exception as e:
                        logger.error(f"Error processing TTS request: {str(e)}")
                        self._set_future_exception(
                            self.current_item.future,
                            self.current_item.loop,
                            e
                        )
                    finally:
                        self.queue.task_done()
                        with self._lock:
                            self.current_item = None
                else:
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Queue processing error: {str(e)}")
                time.sleep(1)

    async def submit_request(self, text: str, speaker: str, language: str, output_path: str, callback_fn: Optional[Callable[[str], Any]] = None) -> str:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        item = TTSQueueItem(
            text=text,
            speaker=speaker,
            language=language,
            output_path=output_path,
            future=future,
            loop=loop,
            callback_fn=callback_fn
        )
        
        self.queue.put(item)
        
        if callback_fn:
            return "Request queued"
        else:
            try:
                result = await future
                return result
            except Exception as e:
                logger.error(f"Error in submit_request: {str(e)}")
                raise