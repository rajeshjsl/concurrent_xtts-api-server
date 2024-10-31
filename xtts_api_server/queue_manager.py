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
    callback_fn: Optional[Callable[[str], Any]] = None  # Optional callback function

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
                        
                        # If there's a callback function, schedule it
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
                        self.current_item = None
                else:
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Queue processing error: {str(e)}")
                time.sleep(1)


    async def submit_request(self, text: str, speaker: str, language: str, output_path: str, callback_fn: Optional[Callable[[str], Any]] = None) -> str:
        """
        Submit a TTS request to the queue
        Args:
            text: Text to convert to speech
            speaker: Speaker identifier
            language: Language code
            output_path: Path to save the output file
            callback_fn: Optional callback function to call with the output path
        Returns: Path to the output file when processing is complete (only if no callback provided)
        """
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
            # If there's a callback, don't wait for the result
            return "Request queued"
        else:
            # If no callback, wait for and return the result
            try:
                result = await future
                return result
            except Exception as e:
                logger.error(f"Error in submit_request: {str(e)}")
                raise

def shutdown(self):
        """Gracefully shutdown the queue manager"""
        logger.info("Initiating queue manager shutdown...")
        
        # Signal the processing thread to stop
        self.running = False
        
        # Wait for current processing to complete
        if self.processing_thread.is_alive():
            logger.info("Waiting for processing thread to complete...")
            self.processing_thread.join(timeout=30)  # Wait up to 30 seconds
            
            if self.processing_thread.is_alive():
                logger.warning("Processing thread did not complete within timeout")
            else:
                logger.info("Processing thread completed successfully")
        
        # Clear any remaining items in the queue
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
                # Get first 100 characters of text for preview
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
