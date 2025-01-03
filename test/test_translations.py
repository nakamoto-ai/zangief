import pytest
import logging
from dataclasses import dataclass
from unittest.mock import MagicMock
from zangief.validator.reward import Reward, get_comet_model, get_bert_model
from zangief.validator.validator import TranslateValidator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

reward = Reward(comet_model=get_comet_model(), bert_model=get_bert_model())


LANGUAGES = [
    "ar", "bn", "cs", "de", "el", "en", "es", "fa", "fr", "he",
    "hi", "hu", "it", "ja", "jv", "ko", "my", "nl", "pa", "pl",
    "pt", "ro", "ru", "sv", "ta", "te", "th", "tr", "uk", "ur",
    "vi", "zh"
]


TRANSLATIONS = {
    "ar": "هذه جملة اختبار.",
    "bn": "এটি একটি পরীক্ষার বাক্য।",
    "cs": "Toto je testovací věta.",
    "de": "Dies ist ein Testsatz.",
    "el": "Αυτή είναι μια δοκιμαστική πρόταση.",
    "en": "This is a test sentence.",
    "es": "Esta es una frase de prueba.",
    "fa": "این یک جمله آزمایشی است.",
    "fr": "Ceci est une phrase de test.",
    "he": "זהו משפט בדיקה.",
    "hi": "यह एक परीक्षण वाक्य है।",
    "hu": "Ez egy teszt mondat.",
    "it": "Questa è una frase di prova.",
    "ja": "これはテスト文です。",
    "jv": "Iki ukara tes.",
    "ko": "이것은 테스트 문장입니다.",
    "my": "ဒါက စမ်းသပ်မှုစာကြောင်းပါ။",
    "nl": "Dit is een testzin.",
    "pa": "ਇਹ ਇੱਕ ਟੈਸਟ ਵਾਕ ਹੈ।",
    "pl": "To jest zdanie testowe.",
    "pt": "Esta é uma frase de teste.",
    "ro": "Aceasta este o propoziție de test.",
    "ru": "Это тестовое предложение.",
    "sv": "Det här är en testsats.",
    "ta": "இது ஒரு சோதனை வாக்கியம்.",
    "te": "ఇది ఒక పరీక్ష వాక్యం.",
    "th": "นี่คือประโยคทดสอบ.",
    "tr": "Bu bir test cümlesidir.",
    "uk": "Це тестове речення.",
    "ur": "یہ ایک ٹیسٹ جملہ ہے۔",
    "vi": "Đây là một câu thử nghiệm.",
    "zh": "这是一个测试句子。"
}


@dataclass
class TranslationTestCase:
    source_language: str
    target_language: str
    source_sentence: str
    expected_translation: str


test_cases = [
    TranslationTestCase(
        source_language=src,
        target_language=tgt,
        source_sentence=TRANSLATIONS[src],
        expected_translation=TRANSLATIONS[tgt]
    )
    for src in LANGUAGES for tgt in LANGUAGES if src != tgt
]


@pytest.mark.parametrize("test_case", test_cases, ids=[f"{tc.source_language}-{tc.target_language}" for tc in test_cases])
def test_translation_scores_greater_than_zero(test_case: TranslationTestCase):
    assert test_case.source_sentence, "Source sentence is empty"
    assert test_case.expected_translation, "Expected translation is empty"

    validator = TranslateValidator(
        key="dummy_key",
        netuid=1,
        client=MagicMock(selected_languages=LANGUAGES),
        module_client=MagicMock(),
        reward=reward,
        cc100=MagicMock()
    )

    miner_prompt = test_case.source_sentence
    miner_answers = [test_case.expected_translation]

    logger.info(f"Testing translation from '{test_case.source_language}' to '{test_case.target_language}'")
    logger.info(f"Prompt: {miner_prompt}")
    logger.info(f"Expected Translation: {miner_answers}")

    scores, scores_to_return = validator.reward.get_scores(
        miner_prompt,
        test_case.source_language,
        test_case.target_language,
        miner_answers
    )

    logger.info(f"Scores: {scores}")

    assert all(score > 0 for score in scores), (
        f"Score not greater than 0 for translation from '{test_case.source_language}' "
        f"to '{test_case.target_language}'. Scores: {scores}"
    )
