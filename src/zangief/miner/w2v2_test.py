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
language = "en"
buffer_size = 100
hf_repo = "mozilla-foundation/common_voice_17_0"
streaming_dataset = load_dataset(hf_repo, language, split="train",
                                 use_auth_token=True, trust_remote_code=True, streaming=True)
print("Dataset Loaded.")
dataset = streaming_dataset.shuffle(
    seed=1137, buffer_size=buffer_size
)
print("Dataset Shuffled.")
common_voice = buffer_dataset(dataset, buffer_size)

# Select a random sample from the dataset
sample = random.choice(common_voice)
end_time = time.time()
execution_time = end_time - start_time

sample_keys = [k for k in sample.keys()]
sample_audio_keys = []
if 'audio' in sample_keys:
    sample_audio_keys += [k for k in sample['audio'].keys()]

print(f"Sample Keys: {sample_keys}")
print(f"Sample Audio Keys: {sample_audio_keys}")

audio_file_url = sample["audio"]["path"]
print(f"Audio File URL: {audio_file_url}")

audio_file = "temp_audio_file.mp3"  # Define the local path
with requests.get(audio_file_url, stream=True) as r:
    r.raise_for_status()
    with open(audio_file, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

print(f"Execution Time: {execution_time} seconds")


print(f"Audio File: {audio_file}")
reference_text = sample["sentence"]


class Wav2Vec2Sample:
    def __init__(self):
        try:
            model_name = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForCTC.from_pretrained(model_name)
            self.comet_model_path = download_model("Unbabel/wmt20-comet-qe-da")
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-multilingual-uncased')
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
        # print(f"Transcription Before De-Phonemize: {phonemes}")
        # transcription = self.de_phoneme_ize(phonemes)
        # print(f"Transcription: {transcription}")
        print(f"Transcription: {phonemes}")
        return phonemes

    def de_phoneme_ize(self, phonemes, language):
        # Preprocess phonemes input for T5
        input_text = f"convert these phonemes into {language} words: {phonemes}"
        input_tokens = self.p2w_tokenizer(input_text, return_tensors="pt")

        # Generate output (e.g., converting phonemes to words)
        output_tokens = self.p2w_model.generate(**input_tokens)
        output_text = self.p2w_tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return output_text

    def score_transcription(self, transcription, reference_text):
        # similarity_score = difflib.SequenceMatcher(None, transcription, reference_text).ratio()
        # print(f'Reference Text: {reference_text}')
        # print(f"Similarity Score: {similarity_score}")
        similarity_score = self.compute_similarity(transcription, reference_text)
        return similarity_score

    def bert_similarity(self, str1, str2):
        model = self.bert_model
        tokenizer = self.bert_tokenizer

        # Tokenize and encode inputs
        inputs1 = tokenizer(str1, return_tensors='pt', padding=True, truncation=True)
        inputs2 = tokenizer(str2, return_tensors='pt', padding=True, truncation=True)

        # Get embeddings from the last hidden layer
        with torch.no_grad():
            outputs1 = model(**inputs1)
            outputs2 = model(**inputs2)

        # Mean pooling
        embeddings1 = outputs1.last_hidden_state.mean(dim=1)
        embeddings2 = outputs2.last_hidden_state.mean(dim=1)

        # Calculate cosine similarity
        similarity = cosine_similarity(embeddings1, embeddings2).item()
        return similarity

    # Function to get COMET score using "Unbabel/wmt20-comet-qe-da"
    def comet_score(self, str1, str2):
        comet_model = load_from_checkpoint(self.comet_model_path)

        data = [
            {
                "src": str1,  # Assign str1 to 'src'
                "ref": str2,  # Assign str2 to 'ref'
                'mt': str2
            }
        ]

        # Compute the quality estimation score
        predictions = comet_model.predict(data, batch_size=8, gpus=0)
        score = self.normalize_comet(predictions['scores'][0])
        return score

    def normalize_comet(self, value, min_value=-1.5, max_value=1.5):
        """
        Normalize a value from the range [min_value, max_value] to [0, 1].

        Parameters:
        value (float): The value to be normalized.
        min_value (float): The minimum value of the original range. Default is -1.5.
        max_value (float): The maximum value of the original range. Default is 1.5.

        Returns:
        float: The normalized value in the range [0, 1].
        """
        # Ensure the value is within the expected range
        if value < min_value:
            value = min_value

        if value > max_value:
            value = max_value

        # Normalize the value
        normalized_value = (value - min_value) / (max_value - min_value)
        return normalized_value

    # Main function to compute both BERT similarity and COMET score
    def compute_similarity(self, str1, str2):
        bert_sim = self.bert_similarity(str1, str2)
        comet_sim = self.comet_score(str1, str2)

        return {
            "BERT Similarity": bert_sim,
            "COMET Score": comet_sim
        }


if __name__ == '__main__':
    w2v2 = Wav2Vec2Sample()
    transcription = w2v2.get_transcription(audio_file)
    similarity_score = w2v2.score_transcription(transcription, reference_text)
    print(similarity_score)
    os.remove(audio_file)
