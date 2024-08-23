import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
from datasets import load_dataset
import difflib
import random

# Load a small subset of the Mozilla Common Voice dataset
common_voice = load_dataset("mozilla-foundation/common_voice_17_0", "he", split="train[:10]", use_auth_token=True, trust_remote_code=True)

# Select a random sample from the dataset
sample = random.choice(common_voice)
audio_file = sample["path"]
reference_text = sample["sentence"]


class Wav2Vec2Sample:
    def __init__(self):
        try:
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
            self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")
            raise

    def get_transcription(self, audio_file):
        # Load the audio file
        waveform, sample_rate = torchaudio.load(audio_file)

        # Preprocess the audio file
        input_values = self.processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=sample_rate).input_values

        # Perform inference
        with torch.no_grad():
            logits = self.model(input_values).logits

        # Decode the output
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        print(f"Transcription: {transcription}")
        return transcription

    def score_transcription(self, transcription, reference_text):
        similarity_score = difflib.SequenceMatcher(None, transcription, reference_text).ratio()
        print(f"Similarity Score: {similarity_score}")
        return similarity_score


if __name__ == '__main__':
    w2v2 = Wav2Vec2Sample()
    transcription = w2v2.get_transcription(audio_file)
    similarity_score = w2v2.score_transcription(transcription, reference_text)
