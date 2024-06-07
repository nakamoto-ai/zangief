from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer)
from base_miner import BaseMiner


class M2MMiner(BaseMiner):

    def __init__(self):
        super().__init__()
        config = self.get_config()
        self.model_name = config.get_value("model", "facebook/m2m100_1.2B")
        self.device = config.get_value("device", "cuda:0")
        self.max_length = config.get_value("max_length", 1024)
        self.do_sample = config.get_value("do_sample", "store_true")
        self.temperature = config.get_value("temperature", 1.0)
        self.top_k = config.get_value("top_k", 10)
        self.no_repeat_ngram_size = config.get_value("no_repeat_ngram_size", 3)
        self.num_beams = config.get_value("num_beams", 1)
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            self.model_name
        )
        self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)
        if self.device != "cpu":
            self.model.to(self.device)

    def generate_translation(self, prompt: str, source_language: str, target_language: str):
        self.tokenizer.src_lang = source_language
        source_tokenizer = self.tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        ).to(self.model.device)

        generated_tokens = self.model.generate(
            **source_tokenizer,
            do_sample=self.do_sample,
            forced_bos_token_id=self.tokenizer.get_lang_id(target_language),
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            num_beams=self.num_beams,
            temperature=self.temperature,
            top_k=self.top_k,
        )

        translation = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]

        return translation

if __name__ == "__main__":
    miner = M2MMiner()
    M2MMiner.start_miner_server(miner)