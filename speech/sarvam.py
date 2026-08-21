import os
from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

client=SarvamAI(
    api_subscription_key=os.getenv("SARVAM_API_KEY")
)

def transcribe_audio(audio_path):
    with open(audio_path, "rb") as audio_file:

        response=client.speech_to_text.transcribe(
            file=audio_file,
            model="saaras:v3",
            mode="transcribe"
        )
    return response.transcript

def translate_audio(audio_path, target_language="en"):
    with open(audio_path, "rb") as audio_file:
        
        response = client.speech_to_text.transcribe(
            file=audio_file,
            model="saaras:v3",
            mode="translate"
        )
    return response.transcript