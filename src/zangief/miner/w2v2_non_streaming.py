import torchaudio
from transformers import AutoProcessor, AutoModelForCTC, T5ForConditionalGeneration, T5Tokenizer, BertModel, BertTokenizer
import torch
from torch.nn.functional import cosine_similarity
from comet.models import load_from_checkpoint, download_model
from datasets import load_dataset
import random
import torchaudio.transforms as T
import time
import os
import requests
from Levenshtein import distance as l_distance


def buffer_dataset(dataset, buffer_size):
    buffer = []
    try:
        for item in dataset:
            if len(buffer) < buffer_size:
                buffer.append(item)
            else:
                break
    except StopIteration:
        pass
    return buffer


def ensure_file_downloaded(file_path, retries=5, delay=2):
    for i in range(retries):
        if os.path.exists(file_path):
            return True
        time.sleep(delay)
    return False


# Load a small subset of the Mozilla Common Voice dataset

start_time = time.time()
language = "hu"
hf_repo = "mozilla-foundation/common_voice_17_0"
common_voice = load_dataset(hf_repo, language, split="train",
                                 use_auth_token=True, trust_remote_code=True)

# Select a random sample from the dataset
sample = random.choice(common_voice)

audio_file = sample['path']


print(f"Audio File: {audio_file}")
reference_text = sample["sentence"]


class Wav2Vec2Sample:
    def __init__(self):
        try:
            model_name = "jz703/wav2vec2_lj_speech_phonemes_word_level"
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForCTC.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading model or tokenizer: {e}")
            raise

    def get_transcription(self, audio_file):
        # Load the audio file
        waveform, sample_rate = torchaudio.load(audio_file)
        print("Audio File Loaded.")

        # Resample the audio if needed
        if sample_rate != 16000:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Preprocess the audio file
        input_values = self.processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000).input_values

        # Perform inference
        with torch.no_grad():
            logits = self.model(input_values).logits

        # Decode the output with grouped tokens
        predicted_ids = torch.argmax(logits, dim=-1)
        phonemes = self.processor.batch_decode(predicted_ids, skip_special_tokens=True, group_tokens=True)[0]
        print(f"\n\n\nTranscription: {phonemes}")
        print(f"Reference: {reference_text}\n\n\n")
        return phonemes

    def score_transcription(self, transcription, reference_text):
        # similarity_score = difflib.SequenceMatcher(None, transcription, reference_text).ratio()
        # print(f'Reference Text: {reference_text}')
        # print(f"Similarity Score: {similarity_score}")
        similarity_score = self.compute_similarity(transcription, reference_text)
        return similarity_score

    def string_similarity(self, str1: str, str2: str) -> float:
        # Calculate Levenshtein distance
        lev_distance = l_distance(str1, str2)

        # Calculate the maximum length of the two strings
        max_len = max(len(str1), len(str2))

        # Prevent division by zero if both strings are empty
        if max_len == 0:
            return 1.0  # Both strings are identical in being empty

        # Calculate similarity score
        similarity = 1 - (lev_distance / max_len)
        return similarity

    def phoneme_score(self, str1, str2, steepness=1.5):
        words_one = len(str1.split(" "))
        words_two = len(str2.split(" "))

        if words_one == words_two:
            return 1.0
        elif words_one < words_two - (words_two / steepness) or words_one > words_two + (words_two / steepness):
            return 0.0
        else:
            high = words_two + (words_two / steepness)
            low = words_two - (words_two / steepness)

            # Calculate normalized score between 0 and 1 based on proximity to `words_two`
            score = 1 - abs(words_one - words_two) / (high - low)
            return score

    def character_score(self, str1, str2, steepness=3):
        chars_one = len([char for char in str1 if char.isalpha()])
        chars_two = len([char for char in str2 if char.isalpha()])

        if chars_one == chars_two:
            score = 1.0
        elif chars_one < chars_two - (chars_two / steepness) or chars_one > chars_two + (chars_two / steepness):
            score = 0.0
        else:
            high = chars_two + (chars_two / steepness)
            low = chars_two - (chars_two / steepness)
            score = 1 - abs(chars_one - chars_two) / (high - low)
        return score

    def compute_similarity(self, str1, str2):
        similarity_score = self.string_similarity(str1, str2)
        phoneme_score = self.phoneme_score(str1, str2)
        character_score = self.character_score(str1, str2)
        return {
            "Similarity Score": similarity_score,
            "Phoneme Score": phoneme_score,
            "Character Score": character_score
        }


if __name__ == '__main__':
    w2v2 = Wav2Vec2Sample()
    transcription = w2v2.get_transcription(audio_file)
    similarity_score = w2v2.score_transcription(transcription, reference_text)
    print(f"\n\n{similarity_score}")
