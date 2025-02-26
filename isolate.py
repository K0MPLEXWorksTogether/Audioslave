import os
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio

class AudioIsolater:
    def __init__(self, base_dir, input_file):
        self.base_dir = base_dir
        self.output_dir = os.path.join(base_dir, "separated_tracks")
        self.input_file = input_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = get_model(name="htdemucs").to(self.device)
    
    def load_audio(self):
        waveform, sr = torchaudio.load(self.input_file)
        self.waveform = waveform.to(self.device)
        self.sr = sr
    
    def apply_model(self):
        self.sources = apply_model(self.model, self.waveform[None], device=self.device, split=True)
    
    def save_separated_files(self):
        instruments = ["drums", "bass", "guitar", "vocals"]
        os.makedirs(self.output_dir, exist_ok=True)
        for i, instrument in enumerate(instruments):
            output_path = os.path.join(self.output_dir, f"{instrument}.wav")
            torchaudio.save(output_path, self.sources[0, i].cpu(), self.sr)
        print(f"✅ Separation complete! Files saved in {self.output_dir}")

# Usage
base_dir = r"E:\VNR hackathon\Audio-Isolater"
input_file = os.path.join(base_dir, "input1.wav")
audio_isolater = AudioIsolater(base_dir, input_file)
audio_isolater.load_audio()
audio_isolater.apply_model()
audio_isolater.save_separated_files()