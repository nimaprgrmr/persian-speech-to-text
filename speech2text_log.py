import os
import logging
import asyncio
from timeit import default_timer as timer
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydub import AudioSegment
import speech_recognition as sr

# Initialize the FastAPI app
app = FastAPI(title="Convert Speech to Text")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory for temporary audio files
TEMP_AUDIO_DIR = "temp_audio"


def convert_to_wav(input_file):
    # Load the input audio file using pydub
    audio = AudioSegment.from_file(input_file)

    # Check if the file is not in WAV format
    if audio.channels != 1 or audio.frame_rate != 16000:
        # Convert the audio to WAV format
        audio = audio.set_channels(1).set_frame_rate(16000)

    # Define the output WAV file path
    output_file = os.path.join(TEMP_AUDIO_DIR, "temp_audio.wav")

    # Export the audio to WAV format
    audio.export(output_file, format="wav")

    return output_file


async def cleanup_temp_files(temp_audio_file: str, wav_file: str):
    # Clean up: Remove temporary files
    try:
        os.remove(temp_audio_file)
        os.remove(wav_file)
        logger.info("Temporary files removed: %s, %s", temp_audio_file, wav_file)
    except Exception as e:
        logger.error("Error cleaning up temporary files: %s", e)


@app.post("/upload")
async def audio_to_text(audio_file: UploadFile = File(...)):
    start_time = timer()
    try:
        # Create temporary directory if it doesn't exist
        os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

        # Save the uploaded audio file to a temporary location
        temp_audio_file = os.path.join(TEMP_AUDIO_DIR, audio_file.filename)
        with open(temp_audio_file, "wb") as temp_audio:
            temp_audio.write(audio_file.file.read())

        # Convert the audio to WAV format if needed
        wav_file = convert_to_wav(temp_audio_file)

        # Initialize the speech recognizer
        recognizer = sr.Recognizer()

        # Load the audio file
        with sr.AudioFile(wav_file) as source:
            audio_data = recognizer.record(source)

        # Use Google Web Speech API to transcribe the audio
        try:
            transcription = recognizer.recognize_google(audio_data, language="fa-IR")
            end_time = timer()
            return {"transcription": transcription, "time": str(round(end_time - start_time)) + " seconds"}
        except sr.UnknownValueError:
            raise HTTPException(status_code=400, detail="Speech Recognition could not understand the audio")
        except sr.RequestError:
            raise HTTPException(status_code=500, detail="Could not request results from Google Speech Recognition")
    finally:
        # Schedule cleanup of temporary files
        asyncio.ensure_future(cleanup_temp_files(temp_audio_file, wav_file))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
