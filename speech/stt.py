from sarvam import transcribe_audio
from sarvam import translate_audio

audio_file="speech/audio.wav"
text=transcribe_audio(audio_file)
text2=translate_audio(audio_file, target_language="en")

print("Transcription:")
print(text)
print("Translation:")
print(text2)