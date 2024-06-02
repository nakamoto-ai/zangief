import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from .base_miner import BaseMiner


class M2MMiner(BaseMiner):

    def __init__(self):
        config = self.get_config()
        super().__init__()
        self.model_name = config.get_value("model_name", "facebook/m2m100_1.2B")
        self.device = self.get_device()
        self.max_length = config.get_value("max_length", "2048")
        self.do_sample = config.get_value("do_sample", "True")
        self.temperature = config.get_value("temperature", "0.1")
        self.top_k = config.get_value("top_k", "4")
        self.no_repeat_ngram_size = config.get_value("no_repeat_ngram_size", "3")
        self.num_beams = config.get_value("num_beams", "2")
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            "facebook/m2m100_1.2B"
        )
        if isinstance(self.model, M2M100ForConditionalGeneration):
            self.model.to(self.device)
        self.tokenizer = M2M100Tokenizer.from_pretrained(str(self.model_name))

    def get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def generate_translation(
        self, prompt: str, source_language: str, target_language: str
    ):
        self.tokenizer.src_lang = source_language
        source_tokenizer = self.tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        ).to(self.device)
        if isinstance(self.model, M2M100ForConditionalGeneration):
            generated_tokens = self.model.generate(
                **source_tokenizer,
                do_sample=self.do_sample,
                forced_bos_token_id=self.tokenizer.get_lang_id(target_language),
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                num_beams=self.num_beams,
                temperature=self.temperature,
                top_k=self.top_k,
            )

        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[
            0
        ]


if __name__ == "__main__":
    miner = M2MMiner()
    M2MMiner.start_miner_server(miner)
