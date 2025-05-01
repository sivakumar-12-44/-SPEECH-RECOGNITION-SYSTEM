import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa

def transcribe_audio_wav2vec(audio_path):
    print("🎧 Loading Wav2Vec2 model...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    print("🔊 Loading audio...")
    speech, sr = librosa.load(audio_path, sr=16000)
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values

    print("🤖 Transcribing...")
    with torch.no_grad():
