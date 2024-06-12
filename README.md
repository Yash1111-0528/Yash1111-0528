import streamlit as st
import numpy as np
import pyaudio
import wave
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import soundfile as sf

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

def resample_audio(audio_input, orig_sample_rate, target_sample_rate):
    return torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=target_sample_rate)(audio_input)

def transcribe_audio(audio_input, sample_rate):
    input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

def record_audio(duration, sample_rate, output_file):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,rate=sample_rate, input=True,frames_per_buffer=CHUNK)

    frames = []

    st.write("Recording...")
    for i in range(0, int(sample_rate / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    st.write("Recording finished.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(output_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def main():
    st.title("Audio to Text Transcription")

    duration = 15 
    sample_rate = 16000  
    output_file = "temp_recording.wav"

    if st.button("Record audio"):
        record_audio(duration, sample_rate, output_file)

        audio_input, orig_sample_rate = torchaudio.load(output_file)
        audio_input_resampled = resample_audio(audio_input, orig_sample_rate, sample_rate)
        transcription = transcribe_audio(audio_input_resampled.squeeze().numpy(), sample_rate)
        st.write("Transcription:")
        st.write(transcription)

    uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3"])
    if uploaded_file:
        audio_input, orig_sample_rate = torchaudio.load(uploaded_file)
        audio_input_resampled = resample_audio(audio_input, orig_sample_rate, sample_rate)
        transcription = transcribe_audio(audio_input_resampled.squeeze().numpy(), sample_rate)
        st.write("Transcription:")
        st.write(transcription)

if __name__ == "__main__":
    main()
