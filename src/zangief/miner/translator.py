import io
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import mmap
import numpy
import soundfile
import torchaudio
import torch

from collections import defaultdict
from IPython.display import Audio, display
from pathlib import Path
from pydub import AudioSegment
from pydub.playback import play

from loguru import logger
from typing import Union, Tuple, List, Dict

from model.seamless_communication.src.seamless_communication.inference import Translator
from model.seamless_communication.src.seamless_communication.streaming.dataloaders.s2tt import (
    SileroVADSilenceRemover,
)


class SeamlessTranslator:
    model_name: str
    vocoder_name: str
    translator: Translator
    target_languages: Dict[str, str]
    task_strings: Dict[str, str]

    def __init__(self):
        self.model_name = "seamlessM4T_v2_large"
        self.vocoder_name = (
            "vocoder_v2"
            if self.model_name == "seamlessM4T_v2_large"
            else "vocoder_36langs"
        )

        self.translator = Translator(
            self.model_name,
            self.vocoder_name,
            device=torch.device("cuda:0"),
            dtype=torch.float16,
        )
        self.target_languages = {
            "Afrikaans": "af",
            "Amharic": "am",
            "Arabic": "ar",
            "Asturian": "ast",
            "Azerbaijani": "az",
            "Bashkir": "ba",
            "Belarusian": "be",
            "Bulgarian": "bg",
            "Bengali": "bn",
            "Breton": "br",
            "Bosnian": "bs",
            "Catalan Valencian": "ca",
            "Cebuano": "ceb",
            "Czech": "cs",
            "Welsh": "cy",
            "Danish": "da",
            "German": "de",
            "Greeek": "el",
            "English": "en",
            "Spanish": "es",
            "Estonian": "et",
            "Persian": "fa",
            "Fulah": "ff",
            "Finnish": "fi",
            "French": "fr",
            "Western Frisian": "fy",
            "Irish": "ga",
            "Gaelic; Scottish Gaelic": "gd",
            "Galician": "gl",
            "Gujarati": "gu",
            "Hausa": "ha",
            "Hebrew": "he",
            "Hindi": "hi",
            "Croatian": "hr",
            "Haitian; Haitian Creole": "ht",
            "Hungarian": "hu",
            "Armenian": "hy",
            "Indonesian": "id",
            "Igbo": "ig",
            "Iloko": "ilo",
            "Icelandic": "is",
            "Italian": "it",
            "Japanese": "ja",
            "Javanese": "jv",
            "Georgian": "ka",
            "Kazakh": "kk",
            "Central Khmer": "km",
            "Kannada": "kn",
            "Korean": "ko",
            "Luxembourgish; Letzeburgesch": "lb",
            "Ganda": "lg",
            "Lingala": "ln",
            "Lao": "lo",
            "Lithuanian": "lt",
            "Latvian": "lv",
            "Malagasy": "mg",
            "Macedonian": "mk",
            "Malayalam": "ml",
            "Mongolian": "mn",
            "Marathi": "mr",
            "Malay": "ms",
            "Burmese": "my",
            "Nepali": "ne",
            "Dutch; Flemish": "nl",
            "Norwegian": "no",
            "Northern Sotho": "ns",
            "Occitan (post 1500)": "oc",
            "Oriya": "or",
            "Panjabi; Punjabi": "pa",
            "Polish": "pl",
            "Pushto; Pashto": "ps",
            "Portuguese": "pt",
            "Romanian; Moldavian; Moldovan": "ro",
            "Russian": "ru",
            "Sindhi": "sd",
            "Sinhala; Sinhalese": "si",
            "Slovak": "sk",
            "Slovenian": "sl",
            "Somali": "so",
            "Albanian": "sq",
            "Serbian": "sr",
            "Swati": "ss",
            "Sundanese": "su",
            "Swedish": "sv",
            "Swahili": "sw",
            "Tamil": "ta",
            "Thai": "th",
            "Tagalog": "tl",
            "Tswana": "tn",
            "Turkish": "tr",
            "Ukrainian": "uk",
            "Urdu": "ur",
            "Uzbek": "uz",
            "Vietnamese": "vi",
            "Wolof": "wo",
            "Xhosa": "xh",
            "Yiddish": "yi",
            "Yoruba": "yo",
            "Chinese": "zh",
            "Zulu": "zu",
        }
        self.task_strings = {
            "Speech-to-Text Translation": "s2tt",
            "Speech-to-Speech Translation": "s2st",
            "Automatic Speech Recognition": "asr",
            "Text-to-Speech Translation": "t2st",
            "Text-to-Text Translation": "t2tt",
        }

    def translation_inference(
        self,
        in_file: Union[
            str, Path
        ] = "model/seamless_communication/demo/dino_pretssel/audios/employee_eng_spa/ref/noisy_spk1_default_00240.wav",
        task_string: str = "s2st",
        target_languages: List[str] = ["eng"],
    ):
        logger.info("English audio:")

        if not Path(in_file).exists():
            logger.error(f"File {in_file} not found")
            raise FileNotFoundError(f"File {in_file} not found")

        input_file = Path(in_file)
        output_text = Path(f"model/output/{input_file.stem}.txt")
        output_audio = Path(f"model/output/{input_file.stem}.wav")

        task_str = self.task_strings[task_string]
        if not task_str:
            logger.error("Invalid task string")
            raise ValueError("Invalid task string")

        for tgt_lang in target_languages:

            tgt_lang = self.target_languages[tgt_lang]
            if not tgt_lang:
                logger.error("Invalid target language")
                raise ValueError("Invalid target language")

            text_output, speech_output = self.translator.predict(
                input=str(in_file),
                task_str=task_str,
                tgt_lang=tgt_lang,
            )
            logger.info(f"Translated text in {tgt_lang}: {text_output[0]}")

            if speech_output:
                torchaudio.save(
                    output_audio,
                    speech_output.audio_wavs[0][0].to(torch.float32).cpu(),
                    speech_output.sample_rate,
                )
            if text_output:
                output_text.write_text(str(text_output[0]), encoding="utf-8")

            logger.info("Translated target file")
