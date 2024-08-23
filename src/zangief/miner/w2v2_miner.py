from openai import OpenAI, APIError
from loguru import logger
from openai.types.chat.chat_completion import ChatCompletion
from base_miner import BaseMiner



class Wav2Vec2Miner(BaseMiner):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        # startup model and add inference params/configs

    def generate_transcription(
        self, audio
    ) -> str | None:
        user_prompt: str = (
            f"Translate the following text from {source_language} to {target_language}: {prompt}"
        )
        system_prompt: str = self.system_prompt
        completion: ChatCompletion = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        try:
            translation: str | None = completion.choices[0].message.content
        except APIError as e:
            logger.error(f"Error parsing OpenAI response: {e}")
            translation = None

        return translation