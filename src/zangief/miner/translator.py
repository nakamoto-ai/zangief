import torchaudio
import torch

from pathlib import Path

from loguru import logger
from typing import Union, Tuple, List, Dict

from seamless_communication.inference.translator import Translator


class SeamlessTranslator:
    """A class for performing translation tasks using a specified model and vocoder.

    Attributes:
        model_name (str): The name of the translation model.
        vocoder_name (str): The name of the vocoder.
        translator (Translator): The translator object.
        target_languages (Dict[str, str]): A dictionary mapping language names to language codes.
        task_strings (Dict[str, str]): A dictionary mapping task strings to abbreviations.

    Args:
        in_file (Union[str, Path]): The input file for translation.
        task_string (str): The type of translation task.
        target_languages (List[str]): The target languages for translation.

    Returns:
        Tuple[Path, Path] | None: A tuple containing the paths to the translated text and audio files, or None if translation fails.
    """

    model_name: str
    vocoder_name: str
    translator: Translator
    target_languages: Dict[str, str]
    task_strings: Dict[str, str]

    def __init__(self) -> None:
        """
        Initializes the SeamlessTranslator object with the specified model and vocoder names,
        and creates a translator object using the specified model and vocoder. The translator
        object is created with the device set to "cuda:0" and the data type set to "torch.float16".
        The target_languages dictionary maps language names to language codes, and the task_strings
        dictionary maps task strings to abbreviations.
        """
        self.model_name = "seamlessM4T_v2_large"
        self.vocoder_name = (
            "vocoder_v2"
            if self.model_name == "seamlessM4T_v2_large"
            else "vocoder_36langs"
        )

        self.translator = Translator(
            model_name_or_card=self.model_name,
            vocoder_name_or_card=self.vocoder_name,
            device=torch.device(device="cuda:0"),
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
        in_file: Union[str, Path],
        task_string: str = "s2st",
        target_languages: List[str] = ["eng"],
    ) -> Tuple[Path, Path] | None:
        """
        Perform translation inference on the given input file.

        Args:
            in_file (Union[str, Path]): The path to the input file.
            task_string (str, optional): The task string. Defaults to "s2st".
            target_languages (List[str], optional): The list of target languages. Defaults to ["eng"].

        Returns:
            Tuple[Path, Path] | None: A tuple containing the absolute paths to the output text file and output audio file, or None if the input file is not found.

        Raises:
            FileNotFoundError: If the input file is not found.
            ValueError: If the task string or target language is invalid.

        """

        if not Path(in_file).exists():
            logger.error(f"File {in_file} not found")
            raise FileNotFoundError(f"File {in_file} not found")

        input_file = Path(in_file)
        output_text = Path(f"model/output/{input_file.stem}.txt")
        output_audio = Path(f"model/output/{input_file.stem}.wav")

        task_str: str = self.task_strings[task_string]
        if not task_str:
            logger.error("Invalid task string")
            raise ValueError("Invalid task string")

        for tgt_lang in target_languages:

            tgt_lang: str = self.target_languages[tgt_lang]
            if not tgt_lang:
                logger.error("Invalid target language")
                raise ValueError("Invalid target language")

            text_output, speech_output = self.translator.predict(
                input=str(object=in_file),
                task_str=task_str,
                tgt_lang=tgt_lang,
            )
            logger.info(f"Translated text in {tgt_lang}: {text_output[0]}")

            if speech_output:
                torchaudio.save(
                    uri=output_audio,
                    src=speech_output.audio_wavs[0][0].to(torch.float32).cpu(),
                    sample_rate=speech_output.sample_rate,
                )
            if text_output:
                output_text.write_text(
                    data=str(object=text_output[0]), encoding="utf-8"
                )

            logger.info("Translated target file")

            return output_text.absolute(), output_audio.absolute()
