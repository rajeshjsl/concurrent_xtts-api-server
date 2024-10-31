from TTS.api import TTS
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse,StreamingResponse
from xtts_api_server.queue_manager import TTSQueueManager
import uuid

from pydantic import BaseModel, AnyHttpUrl
import uvicorn

import aiohttp
import asyncio

import os
import time
from pathlib import Path
import shutil
from loguru import logger
from argparse import ArgumentParser
from pathlib import Path
from uuid import uuid4

from xtts_api_server.tts_funcs import TTSWrapper,supported_languages,InvalidSettingsError
from xtts_api_server.RealtimeTTS import TextToAudioStream, CoquiEngine
from xtts_api_server.modeldownloader import check_stream2sentence_version,install_deepspeed_based_on_python_version

# Default Folders , you can change them via API
DEVICE = os.getenv('DEVICE',"cuda")
OUTPUT_FOLDER = os.getenv('OUTPUT', 'output')
SPEAKER_FOLDER = os.getenv('SPEAKER', 'speakers')
MODEL_FOLDER = os.getenv('MODEL', 'models')
BASE_HOST = os.getenv('BASE_URL', '127.0.0.1:8020')
BASE_URL = os.getenv('BASE_URL', '127.0.0.1:8020')
MODEL_SOURCE = os.getenv("MODEL_SOURCE", "local")
MODEL_VERSION = os.getenv("MODEL_VERSION","v2.0.2")
LOWVRAM_MODE = os.getenv("LOWVRAM_MODE") == 'true'
DEEPSPEED = os.getenv("DEEPSPEED") == 'true'
USE_CACHE = os.getenv("USE_CACHE") == 'true'

# STREAMING VARS
STREAM_MODE = os.getenv("STREAM_MODE") == 'true'
STREAM_MODE_IMPROVE = os.getenv("STREAM_MODE_IMPROVE") == 'true'
STREAM_PLAY_SYNC = os.getenv("STREAM_PLAY_SYNC") == 'true'

if(DEEPSPEED):
  install_deepspeed_based_on_python_version()

# Create an instance of the TTSWrapper class and server
app = FastAPI()
XTTS = TTSWrapper(OUTPUT_FOLDER,SPEAKER_FOLDER,MODEL_FOLDER,LOWVRAM_MODE,MODEL_SOURCE,MODEL_VERSION,DEVICE,DEEPSPEED,USE_CACHE)
queue_manager = TTSQueueManager(XTTS)

# Check for old format model version
XTTS.model_version = XTTS.check_model_version_old_format(MODEL_VERSION)
MODEL_VERSION = XTTS.model_version

# Create version string
version_string = ""
if MODEL_SOURCE == "api" or MODEL_VERSION == "main":
    version_string = "lastest"
else:
    version_string = MODEL_VERSION

# Load model
if STREAM_MODE or STREAM_MODE_IMPROVE:
    # Load model for Streaming
    check_stream2sentence_version()

    logger.warning("'Streaming Mode' has certain limitations, you can read about them here https://github.com/daswer123/xtts-api-server#about-streaming-mode")

    if STREAM_MODE_IMPROVE:
        logger.info("You launched an improved version of streaming, this version features an improved tokenizer and more context when processing sentences, which can be good for complex languages like Chinese")
        
    model_path = XTTS.model_folder
    
    engine = CoquiEngine(specific_model=MODEL_VERSION,use_deepspeed=DEEPSPEED,local_models_path=str(model_path))
    stream = TextToAudioStream(engine)
else:
  logger.info(f"Model: '{version_string}' starts to load,wait until it loads")
  XTTS.load_model() 

if USE_CACHE:
    logger.info("You have enabled caching, this option enables caching of results, your results will be saved and if there is a repeat request, you will get a file instead of generation")

# Add CORS middleware 
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Help funcs
def play_stream(stream,language):
  if STREAM_MODE_IMPROVE:
    # Here we define common arguments in a dictionary for DRY principle
    play_args = {
        'minimum_sentence_length': 2,
        'minimum_first_fragment_length': 2,
        'tokenizer': "stanza",
        'language': language,
        'context_size': 2
    }
    if STREAM_PLAY_SYNC:
        # Play synchronously
        stream.play(**play_args)
    else:
        # Play asynchronously
        stream.play_async(**play_args)
  else:
    # If not improve mode just call the appropriate method based on sync_play flag.
    if STREAM_PLAY_SYNC:
      stream.play()
    else:
      stream.play_async()

async def send_callback(callback_url: str, file_path: str):
    """Send the generated audio file to the callback URL"""
    try:
        async with aiohttp.ClientSession() as session:
            with open(file_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('audio_file',
                             f,
                             filename='output.wav',
                             content_type='audio/wav')
                
                async with session.post(callback_url, data=data) as response:
                    if response.status != 200:
                        logger.error(f"Callback failed with status {response.status}: {await response.text()}")
                    else:
                        logger.info(f"Successfully sent callback to {callback_url}")
                        
                    if not XTTS.enable_cache_results:
                        os.unlink(file_path)
    except Exception as e:
        logger.error(f"Error sending callback: {str(e)}")

class OutputFolderRequest(BaseModel):
    output_folder: str

class SpeakerFolderRequest(BaseModel):
    speaker_folder: str

class ModelNameRequest(BaseModel):
    model_name: str

class TTSSettingsRequest(BaseModel):
    stream_chunk_size: int
    temperature: float
    speed: float
    length_penalty: float
    repetition_penalty: float
    top_p: float
    top_k: int
    enable_text_splitting: bool

# Update the request model to include optional callback_url
class SynthesisRequest(BaseModel):
    text: str
    speaker_wav: str 
    language: str
    callback_url: AnyHttpUrl | None = None  # Optional callback URL

class SynthesisFileRequest(BaseModel):
    text: str
    speaker_wav: str 
    language: str
    file_name_or_path: str  

@app.get("/speakers_list")
def get_speakers():
    speakers = XTTS.get_speakers()
    return speakers

@app.get("/speakers")
def get_speakers():
    speakers = XTTS.get_speakers_special()
    return speakers

@app.get("/languages")
def get_languages():
    languages = XTTS.list_languages()
    return {"languages": languages}

@app.get("/get_folders")
def get_folders():
    speaker_folder = XTTS.speaker_folder
    output_folder = XTTS.output_folder
    model_folder = XTTS.model_folder
    return {"speaker_folder": speaker_folder, "output_folder": output_folder,"model_folder":model_folder}

@app.get("/get_models_list")
def get_models_list():
    return XTTS.get_models_list()

@app.get("/get_tts_settings")
def get_tts_settings():
    settings = {**XTTS.tts_settings,"stream_chunk_size":XTTS.stream_chunk_size}
    return settings

@app.get("/sample/{file_name:path}")
def get_sample(file_name: str):
    # A fix for path traversal vulenerability. 
    # An attacker may summon this endpoint with ../../etc/passwd and recover the password file of your PC (in linux) or access any other file on the PC
    if ".." in file_name:
        raise HTTPException(status_code=404, detail=".. in the file name! Are you kidding me?") 
    file_path = os.path.join(XTTS.speaker_folder, file_name)
    if os.path.isfile(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    else:
        logger.error("File not found")
        raise HTTPException(status_code=404, detail="File not found")

@app.post("/set_output")
def set_output(output_req: OutputFolderRequest):
    try:
        XTTS.set_out_folder(output_req.output_folder)
        return {"message": f"Output folder set to {output_req.output_folder}"}
    except ValueError as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/set_speaker_folder")
def set_speaker_folder(speaker_req: SpeakerFolderRequest):
    try:
        XTTS.set_speaker_folder(speaker_req.speaker_folder)
        return {"message": f"Speaker folder set to {speaker_req.speaker_folder}"}
    except ValueError as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/switch_model")
def switch_model(modelReq: ModelNameRequest):
    try:
        XTTS.switch_model(modelReq.model_name)
        return {"message": f"Model switched to {modelReq.model_name}"}
    except InvalidSettingsError as e:  
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/set_tts_settings")
def set_tts_settings_endpoint(tts_settings_req: TTSSettingsRequest):
    try:
        XTTS.set_tts_settings(**tts_settings_req.dict())
        return {"message": "Settings successfully applied"}
    except InvalidSettingsError as e: 
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))

@app.get('/tts_stream')
async def tts_stream(request: Request, text: str = Query(), speaker_wav: str = Query(), language: str = Query()):
    # Validate local model source.
    if XTTS.model_source != "local":
        raise HTTPException(status_code=400,
                            detail="HTTP Streaming is only supported for local models.")
    # Validate language code against supported languages.
    if language.lower() not in supported_languages:
        raise HTTPException(status_code=400,
                            detail="Language code sent is either unsupported or misspelled.")
            
    async def generator():
        chunks = XTTS.process_tts_to_file(
            text=text,
            speaker_name_or_path=speaker_wav,
            language=language.lower(),
            stream=True,
        )
        # Write file header to the output stream.
        yield XTTS.get_wav_header()
        async for chunk in chunks:
            # Check if the client is still connected.
            disconnected = await request.is_disconnected()
            if disconnected:
                break
            yield chunk

    return StreamingResponse(generator(), media_type='audio/x-wav')

@app.post("/tts_to_audio/")
async def tts_to_audio(request: SynthesisRequest, background_tasks: BackgroundTasks):
    if STREAM_MODE or STREAM_MODE_IMPROVE:
        try:
            global stream
            if request.language.lower() not in supported_languages:
                raise HTTPException(status_code=400,
                                    detail="Language code sent is either unsupported or misspelled.")

            speaker_wav = XTTS.get_speaker_wav(request.speaker_wav)
            language = request.language[0:2]

            if stream.is_playing() and not STREAM_PLAY_SYNC:
                stream.stop()
                stream = TextToAudioStream(engine)

            engine.set_voice(speaker_wav)
            engine.language = request.language.lower()
           
            stream.feed(request.text)
            play_stream(stream,language)

            this_dir = Path(__file__).parent.resolve()
            output = this_dir / "RealtimeTTS" / "silence.wav"

            if request.callback_url:
                background_tasks.add_task(send_callback, str(request.callback_url), str(output))
                return {"message": "Audio generation started. Results will be sent to callback URL."}

            return FileResponse(
                path=output,
                media_type='audio/wav',
                filename="silence.wav",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    else:
        try:
            if XTTS.model_source == "local":
                logger.info(f"Processing TTS to audio with request: {request}")

            if request.language.lower() not in supported_languages:
                raise HTTPException(status_code=400,
                                  detail="Language code sent is either unsupported or misspelled.")

            output_file = f'{str(uuid4())}.wav'

            if request.callback_url:
                # Create callback handler
                callback_url = str(request.callback_url)
                
                def handle_completion(file_path: str):
                    # Create task for async callback
                    asyncio.create_task(send_callback(callback_url, file_path))
                
                # Submit with callback
                await queue_manager.submit_request(
                    text=request.text,
                    speaker=request.speaker_wav,
                    language=request.language.lower(),
                    output_path=output_file,
                    callback_fn=handle_completion
                )
                
                return {"message": "Audio generation started. Results will be sent to callback URL."}
            else:
                # No callback - wait for result and return directly
                result_path = await queue_manager.submit_request(
                    text=request.text,
                    speaker=request.speaker_wav,
                    language=request.language.lower(),
                    output_path=output_file
                )

                if not XTTS.enable_cache_results:
                    background_tasks.add_task(os.unlink, result_path)

                return FileResponse(
                    path=result_path,
                    media_type='audio/wav',
                    filename="output.wav"
                )

        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/queue_status")
async def get_queue_status():
    """Get current queue status"""
    try:
        status = queue_manager.get_queue_status()
        return status
    except Exception as e:
        logger.error(f"Error getting queue status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting queue status: {str(e)}")

@app.post("/tts_to_file")
async def tts_to_file(request: SynthesisFileRequest):
    try:
        if XTTS.model_source == "local":
            logger.info(f"Processing TTS to file with request: {request}")

        # Validate language code against supported languages
        if request.language.lower() not in supported_languages:
            raise HTTPException(status_code=400,
                              detail="Language code sent is either unsupported or misspelled.")

        # Submit to queue and wait for completion
        try:
            output_file = await queue_manager.submit_request(
                text=request.text,
                speaker=request.speaker_wav,
                language=request.language.lower(),
                output_path=request.file_name_or_path
            )
            return {"message": "The audio was successfully made and stored.", "output_path": output_file}
        except Exception as e:
            logger.error(f"Queue processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Queue processing error: {str(e)}")

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
# Add cleanup for queue manager in case you need to shutdown gracefully
# This could go at the end of the file or in an appropriate shutdown handler
@app.on_event("shutdown")
def shutdown_event():
    queue_manager.shutdown()

if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=8002)
