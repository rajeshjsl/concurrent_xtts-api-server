from dataclasses import dataclass
from typing import Optional
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

class TTSQueueManager:
    def __init__(self, tts_wrapper):
        self.tts_wrapper = tts_wrapper
        self.queue = Queue()
        self.processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.running = True
        self.processing_thread.start()
        self.current_item: Optional[TTSQueueItem] = None
        
    def _set_future_result(self, future: asyncio.Future, loop: asyncio.AbstractEventLoop, result):
        """Safely set future result from another thread"""
        if not future.done():
            loop.call_soon_threadsafe(future.set_result, result)

    def _set_future_exception(self, future: asyncio.Future, loop: asyncio.AbstractEventLoop, exc):
        """Safely set future exception from another thread"""
        if not future.done():
            loop.call_soon_threadsafe(future.set_exception, exc)
    
    def _process_queue(self):
        """Process items in the queue sequentially"""
        while self.running:
            try:
                if not self.queue.empty():
                    self.current_item = self.queue.get()
                    try:
                        # Process TTS request
                        output_file = self.tts_wrapper.process_tts_to_file(
                            text=self.current_item.text,
                            speaker_name_or_path=self.current_item.speaker,
                            language=self.current_item.language,
                            file_name_or_path=self.current_item.output_path
                        )
                        # Set the result in the future
                        self._set_future_result(
                            self.current_item.future,
                            self.current_item.loop,
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
                        self.current_item = None
                else:
                    # Sleep briefly to prevent busy waiting
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Queue processing error: {str(e)}")
                time.sleep(1)  # Sleep on error to prevent rapid retries

    async def submit_request(self, text: str, speaker: str, language: str, output_path: str) -> str:
        """
        Submit a TTS request to the queue
        Returns: Path to the output file when processing is complete
        """
        # Create a future to track completion
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        # Create queue item
        item = TTSQueueItem(
            text=text,
            speaker=speaker,
            language=language,
            output_path=output_path,
            future=future,
            loop=loop
        )
        
        # Add to queue
        self.queue.put(item)
        
        # Wait for processing to complete
        try:
            result = await future
            return result
        except Exception as e:
            logger.error(f"Error in submit_request: {str(e)}")
            raise

    def get_queue_status(self):
        """Get current queue status"""
        return {
            "queue_size": self.queue.qsize(),
            "currently_processing": self.current_item.text if self.current_item else None
        }

    def shutdown(self):
        """Cleanup resources"""
        self.running = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)