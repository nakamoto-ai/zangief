from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Define the model name
model_name = "facebook/wav2vec2-large-xlsr-53"

# Download the processor and the model
processor = Wav2Vec2Processor.from_pretrained(model_name)
# processor.save_pretrained("wav2vec2_model")
model = Wav2Vec2ForCTC.from_pretrained(model_name)
# model.save_pretrained("wav2vec2_model")

print("Model and processor downloaded and saved locally.")
