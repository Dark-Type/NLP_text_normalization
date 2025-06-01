import re
import logging
import pandas as pd
import numpy as np
import torch
import json
import os
import argparse
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from tqdm import tqdm

# Advanced imports for T5 training
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    GPT2Tokenizer
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./data/log_file.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Enhanced configuration for text normalization training."""
    model_name: str = "ai-forever/ruT5-base"
    max_source_length: int = 128
    max_target_length: int = 128

    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 1
    warmup_steps: int = 100
    weight_decay: float = 0.05
    gradient_clip_val: float = 0.5

    sample_size: int = 15000
    stratify_by_class: bool = True
    stratify_by_length: bool = True

    dataloader_num_workers: int = 4
    mixed_precision: bool = True
    compile_model: bool = False
    pin_memory: bool = True
    seed: int = 42

class TextCleaner:
    """ Text cleaner for dataset."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.russian_letters = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')
        self.valid_chars = (
            self.russian_letters |
            set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789') |
            set(' .,!?-()[]{}\";:\'\"/@#$%^&*+=<>~`|\\€$₽°')
        )

    def safe_load_csv(self, file_path: str) -> pd.DataFrame:
        """Safely load CSV with encoding issues handling."""
        encodings_to_try = ['utf-8', 'utf-8-sig', 'cp1251', 'iso-8859-1', 'latin1']

        for encoding in encodings_to_try:
            try:
                self.logger.info(f"Trying to load with encoding: {encoding}")
                df = pd.read_csv(file_path, encoding=encoding)
                self.logger.info(f"Successfully loaded with {encoding}")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self.logger.warning(f"Failed with {encoding}: {e}")
                continue

        try:
            self.logger.info("Loading with error handling (replacing bad characters)")
            df = pd.read_csv(file_path, encoding='utf-8', errors='replace')
            return df
        except Exception as e:
            self.logger.error(f"All encoding attempts failed: {e}")
            return None

    def clean_dataset(self, df: pd.DataFrame, aggressive_cleaning: bool = False) -> pd.DataFrame:
        """Clean Russian text normalization dataset with character handling."""
        self.logger.info(f"Starting robust cleaning of {len(df)} rows")
        original_size = len(df)

        df = self._handle_basic_issues(df)
        df = self._fix_character_encoding(df)
        df = self._clean_problematic_characters(df, gentle=True)

        if aggressive_cleaning:
            df = self._validate_normalizations(df)

        df = self._handle_clear_duplicates(df)
        df = self._minimal_final_cleanup(df)

        cleaned_size = len(df)
        removal_percentage = ((original_size - cleaned_size) / original_size) * 100
        self.logger.info(f"Cleaning complete: {original_size} -> {cleaned_size} rows "
                        f"({removal_percentage:.1f}% removed)")

        return df.reset_index(drop=True)

    def _handle_basic_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle basic data integrity issues."""
        required_cols = ['before', 'after']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df['before'] = df['before'].astype(str).replace('nan', '')
        df['after'] = df['after'].astype(str).replace('nan', '')

        initial_len = len(df)
        df = df[(df['before'].str.strip() != '') & (df['after'].str.strip() != '')]
        self.logger.info(f"Removed {initial_len - len(df)} clearly empty rows")

        return df

    def _fix_character_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix character encoding issues."""
        def fix_encoding(text):
            if pd.isna(text):
                return ""

            text = str(text)

            try:
                text = unicodedata.normalize('NFKC', text)
            except:
                pass

            encoding_fixes = {
                'Ã¡': 'а', 'Ã ': 'а', 'Ã«': 'е', 'Ã¬': 'и', 'Ã®': 'о', 'Ã³': 'у',
                'â€œ': '"', 'â€': '"', 'â€™': "'", 'â€"': '–', 'â€"': '—'
            }

            for wrong, correct in encoding_fixes.items():
                text = text.replace(wrong, correct)

            return text

        df['before'] = df['before'].apply(fix_encoding)
        df['after'] = df['after'].apply(fix_encoding)

        return df

    def _clean_problematic_characters(self, df: pd.DataFrame, gentle: bool = True) -> pd.DataFrame:
        """Remove problematic characters with gentle approach."""
        def clean_text(text, gentle_mode=True):
            if pd.isna(text):
                return ""

            text = str(text)

            if gentle_mode:
                problematic_chars = set(['', '', '', '﻿', '\x00', '\x01', '\x02', '\x03'])
                text = ''.join(char for char in text if char not in problematic_chars)
                text = re.sub(r'\s+', ' ', text).strip()
            else:
                text = ''.join(char for char in text if char in self.valid_chars or char.isspace())
                text = re.sub(r'\s+', ' ', text).strip()

            return text

        initial_len = len(df)

        df['before'] = df['before'].apply(lambda x: clean_text(x, gentle))
        df['after'] = df['after'].apply(lambda x: clean_text(x, gentle))

        df = df[(df['before'].str.strip() != '') & (df['after'].str.strip() != '')]

        self.logger.info(f"Character cleaning removed {initial_len - len(df)} rows")
        return df

    def _validate_normalizations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate normalizations"""
        initial_len = len(df)

        def is_valid_normalization(before, after):
            if len(after) > len(before) * 5:
                return False
            if len(after) < len(before) * 0.1 and len(before) > 10:
                return False
            return True

        mask = df.apply(lambda row: is_valid_normalization(row['before'], row['after']), axis=1)
        df = df[mask]

        self.logger.info(f"Normalization validation removed {initial_len - len(df)} rows")
        return df

    def _handle_clear_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle only clear duplicates."""
        initial_len = len(df)
        df = df.drop_duplicates(subset=['before', 'after'])
        self.logger.info(f"Duplicate removal: {initial_len - len(df)} exact duplicates removed")
        return df

    def _minimal_final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Minimal final cleanup."""
        initial_len = len(df)
        df = df[df['before'].str.len() <= 500]
        df = df[df['after'].str.len() <= 600]
        self.logger.info(f"Final cleanup removed {initial_len - len(df)} overly long texts")
        return df

class My_TextNormalization_Model:
    """
    Enhanced Russian text normalization model combining rule-based and neural approaches.
    """

    def __init__(self):
        """Initialize the enhanced text normalization model."""
        logger.info("Initializing Advanced Russian Text Normalization Model")

        # File paths
        self.dictionary_path = 'data/dictionary/model_dictionary.json'
        self.results_path = 'data/results.csv'
        self.train_path = 'data/ru_train.csv'
        self.test_path = 'data/ru_test_2.csv'
        self.result_path = 'data/result.csv'

        os.makedirs('data/dictionary', exist_ok=True)
        os.makedirs('data/results', exist_ok=True)
        os.makedirs('models_cache', exist_ok=True)

        self.cleaner = TextCleaner()

        self.numbers = {
            '0': 'ноль', '1': 'один', '2': 'два', '3': 'три', '4': 'четыре',
            '5': 'пять', '6': 'шесть', '7': 'семь', '8': 'восемь', '9': 'девять',
            '10': 'десять', '11': 'одиннадцать', '12': 'двенадцать',
            '13': 'тринадцать', '14': 'четырнадцать', '15': 'пятнадцать',
            '16': 'шестнадцать', '17': 'семнадцать', '18': 'восемнадцать',
            '19': 'девятнадцать', '20': 'двадцать', '30': 'тридцать',
            '40': 'сорок', '50': 'пятьдесят', '60': 'шестьдесят',
            '70': 'семьдесят', '80': 'восемьдесят', '90': 'девяносто'
        }

        self.hundreds = {
            '100': 'сто', '200': 'двести', '300': 'триста', '400': 'четыреста',
            '500': 'пятьсот', '600': 'шестьсот', '700': 'семьсот',
            '800': 'восемьсот', '900': 'девятьсот'
        }

        self.ordinals = {
            '1': 'первое', '2': 'второе', '3': 'третье', '4': 'четвертое',
            '5': 'пятое', '6': 'шестое', '7': 'седьмое', '8': 'восьмое',
            '9': 'девятое', '10': 'десятое', '11': 'одиннадцатое',
            '12': 'двенадцатое', '13': 'тринадцатое', '14': 'четырнадцатое',
            '15': 'пятнадцатое', '16': 'шестнадцатое', '17': 'семнадцатое',
            '18': 'восемнадцатое', '19': 'девятнадцатое', '20': 'двадцатое',
            '21': 'двадцать первое', '22': 'двадцать второе', '23': 'двадцать третье',
            '24': 'двадцать четвертое', '25': 'двадцать пятое', '26': 'двадцать шестое',
            '27': 'двадцать седьмое', '28': 'двадцать восьмое', '29': 'двадцать девятое',
            '30': 'тридцатое', '31': 'тридцать первое'
        }

        self.months = {
            '1': 'января', '2': 'февраля', '3': 'марта', '4': 'апреля',
            '5': 'мая', '6': 'июня', '7': 'июля', '8': 'августа',
            '9': 'сентября', '10': 'октября', '11': 'ноября', '12': 'декабря',
            '01': 'января', '02': 'февраля', '03': 'марта', '04': 'апреля',
            '05': 'мая', '06': 'июня', '07': 'июля', '08': 'августа',
            '09': 'сентября'
        }

        self.roman_numerals = {
            'I': 'первый', 'II': 'второй', 'III': 'третий', 'IV': 'четвертый',
            'V': 'пятый', 'VI': 'шестой', 'VII': 'седьмой', 'VIII': 'восьмой',
            'IX': 'девятый', 'X': 'десятый', 'XI': 'одиннадцатый',
            'XII': 'двенадцатый', 'XIII': 'тринадцатый', 'XIV': 'четырнадцатый',
            'XV': 'пятнадцатый', 'XVI': 'шестнадцатый', 'XVII': 'семнадцатый',
            'XVIII': 'восемнадцатый', 'XIX': 'девятнадцатый', 'XX': 'двадцатый'
        }

        logger.info("Advanced Russian Text Normalization Model initialized successfully")

    def train_enhanced_dict(self):
        """Create enhanced dictionary with robust data cleaning."""
        logger.info("Starting enhanced dictionary creation...")

        # Load and clean training data
        train_df = self.cleaner.safe_load_csv(self.train_path)
        if train_df is None:
            logger.error("Failed to load training data")
            return

        train_df = self.cleaner.clean_dataset(train_df, aggressive_cleaning=False)

        train_df['before'] = train_df['before'].str.lower()
        train_df['after'] = train_df['after'].str.lower()
        train_df['after_c'] = train_df['after'].map(lambda x: len(str(x).split()))
        train_df = train_df[~((train_df['class'] == 'LETTERS') & (train_df['after_c'] > 4))]

        train_df = train_df.groupby(['before', 'after'], as_index=False)['sentence_id'].count()
        train_df = train_df.sort_values(['sentence_id', 'before'], ascending=[False, True])
        train_df = train_df.drop_duplicates(['before'])

        dictionary = {key: value for (key, value) in train_df[['before', 'after']].values}
        logger.info(f"Dictionary created with {len(dictionary)} entries")

        try:
            with open(self.dictionary_path, 'w', encoding='utf-8') as f:
                json.dump(dictionary, f, indent=4, ensure_ascii=False)
            logger.info(f"Dictionary saved to {self.dictionary_path}")
        except Exception as e:
            logger.error(f"Failed to save dictionary: {e}")

        logger.info("Enhanced dictionary creation completed successfully")

    def normalize_with_rules(self, text: str) -> str:
        """Apply rule-based normalization to text."""
        if not text or not isinstance(text, str):
            return ""

        try:
            normalized_text = self._preprocess_text(text)
            normalized_text = self._normalize_dates(normalized_text)
            normalized_text = self._normalize_time(normalized_text)
            normalized_text = self._normalize_currency(normalized_text)
            normalized_text = self._normalize_measurements(normalized_text)
            normalized_text = self._normalize_percentages(normalized_text)
            normalized_text = self._normalize_phone_numbers(normalized_text)
            normalized_text = self._normalize_urls_emails(normalized_text)
            normalized_text = self._normalize_abbreviations(normalized_text)
            normalized_text = self._normalize_roman_numerals(normalized_text)
            normalized_text = self._normalize_numbers(normalized_text)
            normalized_text = self._normalize_punctuation(normalized_text)
            normalized_text = self._postprocess_text(normalized_text)

            return normalized_text
        except Exception as e:
            logger.error(f"Error during rule-based normalization: {str(e)}")
            return text

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text by cleaning and standardizing format."""
        text = re.sub(r'\s+', ' ', text.strip())
        text = text.replace('—', '-').replace('–', '-')
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        return text

    def _normalize_dates(self, text: str) -> str:
        """Normalize dates in various formats."""
        def convert_date(match):
            day, month, year = match.groups()
            day_word = self.ordinals.get(str(int(day)), f"{day}-е")
            month_word = self.months.get(str(int(month)), month)
            year_word = self._convert_year(year)
            return f"{day_word} {month_word} {year_word} года"

        text = re.sub(r'\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b', convert_date, text)

        def convert_short_date(match):
            day, month = match.groups()
            day_word = self.ordinals.get(str(int(day)), f"{day}-е")
            month_word = self.months.get(str(int(month)), month)
            return f"{day_word} {month_word}"

        text = re.sub(r'\b(\d{1,2})[./](\d{1,2})\b(?!/)', convert_short_date, text)
        return text

    def _normalize_time(self, text: str) -> str:
        """Normalize time expressions."""
        def convert_time(match):
            hours, minutes = match.groups()
            hour_int = int(hours)
            minute_int = int(minutes)

            hour_word = self._number_to_words(hour_int)

            if minute_int == 0:
                return f"{hour_word} часов"
            else:
                minute_word = self._number_to_words(minute_int)
                return f"{hour_word} часов {minute_word} минут"

        text = re.sub(r'\b(\d{1,2}):(\d{2})\b', convert_time, text)
        return text

    def _normalize_currency(self, text: str) -> str:
        """Normalize currency expressions."""
        def convert_currency(match):
            amount, currency = match.groups()
            amount_int = int(amount)
            amount_word = self._number_to_words(amount_int)

            if currency == '₽' or currency == 'руб':
                if amount_int == 1:
                    currency_word = 'рубль'
                elif 2 <= amount_int <= 4:
                    currency_word = 'рубля'
                else:
                    currency_word = 'рублей'
            elif currency == '$':
                if amount_int == 1:
                    currency_word = 'доллар'
                elif 2 <= amount_int <= 4:
                    currency_word = 'доллара'
                else:
                    currency_word = 'долларов'
            elif currency == '€':
                currency_word = 'евро'
            else:
                currency_word = currency

            return f"{amount_word} {currency_word}"

        text = re.sub(r'(\d+)\s*([₽$€]|руб\.?)', convert_currency, text)

        def convert_decimal_currency(match):
            amount, currency = match.groups()
            amount_word = self._convert_decimal_number(amount)
            currency_word = self._get_currency_word(currency)
            return f"{amount_word} {currency_word}"

        text = re.sub(r'(\d+[.,]\d+)\s*([₽$€]|руб\.?)', convert_decimal_currency, text)
        return text

    def _normalize_measurements(self, text: str) -> str:
        """Normalize measurement units."""
        def convert_measurement(match):
            amount, unit = match.groups()

            if '.' in amount or ',' in amount:
                amount_word = self._convert_decimal_number(amount)
            else:
                amount_word = self._number_to_words(int(amount))

            unit_mappings = {
                'кг': 'килограммов', 'г': 'граммов', 'км': 'километров',
                'м': 'метров', 'см': 'сантиметров', 'мм': 'миллиметров',
                'л': 'литров', 'мл': 'миллилитров', '°C': 'градусов цельсия',
                '°': 'градусов'
            }

            unit_word = unit_mappings.get(unit, unit)
            return f"{amount_word} {unit_word}"

        text = re.sub(r'(\d+(?:[.,]\d+)?)\s*(кг|г|км|м|см|мм|л|мл|°C|°)', convert_measurement, text)
        return text

    def _normalize_percentages(self, text: str) -> str:
        """Normalize percentage expressions."""
        def convert_percentage(match):
            amount = match.group(1)
            if '.' in amount or ',' in amount:
                amount_word = self._convert_decimal_number(amount)
            else:
                amount_word = self._number_to_words(int(amount))
            return f"{amount_word} процентов"

        text = re.sub(r'(\d+(?:[.,]\d+)?)%', convert_percentage, text)
        return text

    def _normalize_phone_numbers(self, text: str) -> str:
        """Normalize phone numbers."""
        def convert_phone(match):
            phone = match.group()
            digits = re.sub(r'[^\d]', '', phone)
            spoken_digits = []
            for digit in digits:
                spoken_digits.append(self.numbers.get(digit, digit))
            return ' '.join(spoken_digits)

        phone_patterns = [
            r'\+7\s*\(\d{3}\)\s*\d{3}-\d{2}-\d{2}',
            r'\+7\s*\d{10}',
            r'8\s*\(\d{3}\)\s*\d{3}-\d{2}-\d{2}',
            r'8\s*\d{10}'
        ]

        for pattern in phone_patterns:
            text = re.sub(pattern, convert_phone, text)

        return text

    def _normalize_urls_emails(self, text: str) -> str:
        """Normalize URLs and email addresses."""
        text = re.sub(r'https?://[^\s]+', 'ссылка', text)
        text = re.sub(r'www\.[^\s]+', 'веб сайт', text)
        text = re.sub(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', 'электронная почта', text)
        return text

    def _normalize_abbreviations(self, text: str) -> str:
        """Normalize common abbreviations."""
        abbreviations = {
            'т.е.': 'то есть', 'и т.д.': 'и так далее', 'и т.п.': 'и тому подобное',
            'т.к.': 'так как', 'т.н.': 'так называемый', 'см.': 'смотрите',
            'стр.': 'страница', 'гл.': 'глава', 'рис.': 'рисунок',
            'табл.': 'таблица', 'г.': 'год', 'гг.': 'годы', 'в.': 'век',
            'вв.': 'века', 'др.': 'другие', 'пр.': 'прочие',
            'напр.': 'например', 'англ.': 'английский', 'рус.': 'русский'
        }

        for abbr, expansion in abbreviations.items():
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, text, flags=re.IGNORECASE)

        return text

    def _normalize_roman_numerals(self, text: str) -> str:
        """Normalize Roman numerals."""
        def convert_roman(match):
            roman = match.group()
            return self.roman_numerals.get(roman, roman)

        text = re.sub(r'\b(I{1,3}|IV|V|VI{0,3}|IX|X{1,2}|XI{0,3}|XIV|XV|XVI{0,3}|XIX|XX)\b', convert_roman, text)
        return text

    def _normalize_numbers(self, text: str) -> str:
        """Normalize standalone numbers to words."""
        def convert_number(match):
            number = match.group()
            try:
                num = int(number)
                return self._number_to_words(num)
            except ValueError:
                return number

        text = re.sub(r'\b\d{1,4}\b', convert_number, text)
        return text

    def _normalize_punctuation(self, text: str) -> str:
        """Handle punctuation marks."""
        text = text.replace('&', ' и ')
        text = text.replace('+', ' плюс ')
        text = text.replace('=', ' равно ')
        text = text.replace('№', 'номер ')

        text = re.sub(r'[()[\]{}]', '', text)
        text = re.sub(r'[.!?]+', '.', text)
        text = re.sub(r'[,;:]+', ',', text)

        return text

    def _postprocess_text(self, text: str) -> str:
        """Final cleanup of normalized text."""
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[.!?,;:]+$', '', text)
        text = text.lower()
        return text

    def _number_to_words(self, num: int) -> str:
        """Convert numbers to Russian words."""
        if num == 0:
            return 'ноль'

        if 1 <= num <= 19:
            return self.numbers[str(num)]
        elif 20 <= num <= 99:
            tens = (num // 10) * 10
            units = num % 10
            if units == 0:
                return self.numbers[str(tens)]
            else:
                return f"{self.numbers[str(tens)]} {self.numbers[str(units)]}"
        elif 100 <= num <= 999:
            hundreds = (num // 100) * 100
            remainder = num % 100
            result = self.hundreds[str(hundreds)]
            if remainder > 0:
                result += f" {self._number_to_words(remainder)}"
            return result
        elif 1000 <= num <= 9999:
            thousands = num // 1000
            remainder = num % 1000

            if thousands == 1:
                result = "тысяча"
            elif 2 <= thousands <= 4:
                result = f"{self.numbers[str(thousands)]} тысячи"
            else:
                result = f"{self.numbers[str(thousands)]} тысяч"

            if remainder > 0:
                result += f" {self._number_to_words(remainder)}"
            return result
        else:
            return str(num)

    def _convert_decimal_number(self, number_str: str) -> str:
        """Convert decimal numbers to words."""
        number_str = number_str.replace(',', '.')

        try:
            if '.' in number_str:
                integer_part, decimal_part = number_str.split('.')
                integer_word = self._number_to_words(int(integer_part))
                decimal_digits = ' '.join([self.numbers.get(digit, digit) for digit in decimal_part])
                return f"{integer_word} целых {decimal_digits}"
            else:
                return self._number_to_words(int(number_str))
        except ValueError:
            return number_str

    def _convert_year(self, year: str) -> str:
        """Convert year to spoken form."""
        try:
            year_int = int(year)
            if 1000 <= year_int <= 2099:
                return self._number_to_words(year_int)
            else:
                return year
        except ValueError:
            return year

    def _get_currency_word(self, currency: str) -> str:
        """Get proper currency word."""
        currency_map = {
            '₽': 'рублей', 'руб': 'рублей', '$': 'долларов', '€': 'евро'
        }
        return currency_map.get(currency, currency)

    def normalize_with_dict(self):
        """Normalize text using dictionary approach."""
        logger.info("Starting dictionary-based normalization...")

        # Load dictionary
        try:
            with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                dictionary = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load dictionary: {e}")
            return None

        try:
            test = self.cleaner.safe_load_csv(self.test_path)
            if test is None:
                logger.error("Failed to load test data")
                return None
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            return None

        # Process test data
        test['id'] = test['sentence_id'].astype(str) + '_' + test['token_id'].astype(str)
        test['before_l'] = test['before'].str.lower()
        test['after'] = test['before_l'].map(lambda x: dictionary.get(x, x))

        def fix_case(original, lower, after):
            if lower == after:
                return original
            else:
                return after

        test['after'] = test.apply(lambda r: fix_case(r['before'], r['before_l'], r['after']), axis=1)

        return test

    def normalize_with_neural(self, test_mode=False):
        """
        Normalize text using neural T5 model.
        Args:
            test_mode (bool): If True, only process first 10 items for testing
        """
        logger.info("Starting neural normalization with T5 model...")

        try:
            # Load test data
            test = self.cleaner.safe_load_csv(self.test_path)
            if test is None:
                logger.error("Failed to load test data")
                return

            test['id'] = test['sentence_id'].astype(str) + '_' + test['token_id'].astype(str)

            if test_mode:
                logger.info("Running in test mode - processing only first 10 elements")
                test = test.head(10)

        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            return

        try:
            # Setup device and model
            MODEL_NAME = "saarus72/russian_text_normalizer"
            cache_dir = Path('models_cache')

            if torch.cuda.is_available():
                logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
                device = torch.device("cuda")
                torch.cuda.empty_cache()
            else:
                logger.warning("CUDA not available. Using CPU.")
                device = torch.device("cpu")

            # Load model and tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
            model = T5ForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                cache_dir=cache_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                use_cache=True
            )

            model = model.to(device)
            model.eval()

            logger.info(f"Model loaded and running on {device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return

        normalized_texts = []
        batch_size = 256 if torch.cuda.is_available() else 32
        if test_mode:
            batch_size = 2

        logger.info(f"Batch size: {batch_size}")

        try:
            with torch.inference_mode():
                for i in tqdm(range(0, len(test), batch_size)):
                    batch_texts = test['before'].iloc[i:i + batch_size].tolist()

                    formatted_texts = []
                    max_len = 0
                    for text in batch_texts:
                        if any(c.isdigit() or (c.isascii() and c.isalpha()) for c in text):
                            if text.isdigit():
                                text_rev = text[::-1]
                                groups = [text_rev[i:i+3][::-1] for i in range(0, len(text_rev), 3)]
                                text = ' '.join(groups[::-1])
                            formatted_text = f"<SC1>[{text}]<extra_id_0>"
                        else:
                            formatted_text = text
                        formatted_texts.append(formatted_text)
                        max_len = max(max_len, len(formatted_text))

                    inputs = tokenizer(
                        formatted_texts,
                        padding=True,
                        truncation=True,
                        max_length=min(max_len + 10, 128),
                        return_tensors="pt"
                    )
                    input_ids = inputs["input_ids"].to(device)
                    attention_mask = inputs["attention_mask"].to(device)

                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=min(max_len + 20, 128),
                        num_beams=2,
                        early_stopping=True,
                        do_sample=False,
                        use_cache=True,
                        eos_token_id=tokenizer.eos_token_id
                    )

                    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                    cleaned_outputs = []
                    for output, original_text in zip(decoded_outputs, batch_texts):
                        text = output.replace("<SC1>", "").replace("<extra_id_0>", "").strip()
                        text = text.strip('[]')
                        if not any(c.isdigit() or (c.isascii() and c.isalpha()) for c in original_text):
                            text = original_text
                        cleaned_outputs.append(text)

                    normalized_texts.extend(cleaned_outputs)


            results_df = pd.DataFrame({
                'id': test['id'],
                'after': normalized_texts
            })

            output_path = self.result_path.replace('.csv', '_test.csv') if test_mode else self.result_path
            logger.info(f"Saving results to {output_path}")
            results_df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info("Neural normalization completed successfully")

        except Exception as e:
            logger.error(f"Error during neural normalization: {e}")
            return

    def normalize_combined(self, test_mode=False):
        """
        Combined normalization approach: dictionary first, then neural for remaining tokens.
        Args:
            test_mode (bool): If True, only process first 10 items for testing
        """
        logger.info("Starting combined normalization (dictionary + neural)...")

        logger.info("Step 1: Applying dictionary normalization...")
        dict_results = self.normalize_with_dict()
        if dict_results is None:
            logger.error("Failed to perform dictionary normalization")
            return

        needs_neural = []
        for idx, row in dict_results.iterrows():
            if (row['before'] == row['after'] and
                any(c.isdigit() or (c.isascii() and c.isalpha()) for c in row['before'])):
                needs_neural.append(idx)

        logger.info(f"Found {len(needs_neural)} tokens requiring neural normalization")

        if not needs_neural:
            logger.info("All tokens successfully normalized with dictionary")
            output_path = self.result_path.replace('.csv', '_test.csv') if test_mode else self.result_path
            dict_results[['id', 'after']].to_csv(output_path, index=False, encoding='utf-8')
            return

        logger.info("Step 2: Applying neural normalization for remaining tokens...")

        try:
            MODEL_NAME = "saarus72/russian_text_normalizer"
            cache_dir = Path('models_cache')

            if torch.cuda.is_available():
                logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
                device = torch.device("cuda")
                torch.cuda.empty_cache()
            else:
                logger.warning("CUDA not available. Using CPU.")
                device = torch.device("cpu")

            tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
            model = T5ForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                cache_dir=cache_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                use_cache=True
            )

            model = model.to(device)
            model.eval()

            neural_texts = dict_results.iloc[needs_neural]
            batch_size = 256 if torch.cuda.is_available() else 32
            if test_mode:
                batch_size = 2

            logger.info(f"Batch size: {batch_size}")
            normalized_neural = []

            with torch.inference_mode():
                for i in tqdm(range(0, len(neural_texts), batch_size)):
                    batch_texts = neural_texts['before'].iloc[i:i + batch_size].tolist()

                    formatted_texts = []
                    max_len = 0
                    for text in batch_texts:
                        if text.isdigit():
                            text_rev = text[::-1]
                            groups = [text_rev[i:i+3][::-1] for i in range(0, len(text_rev), 3)]
                            text = ' '.join(groups[::-1])
                        formatted_text = f"<SC1>[{text}]<extra_id_0>"
                        formatted_texts.append(formatted_text)
                        max_len = max(max_len, len(formatted_text))

                    inputs = tokenizer(
                        formatted_texts,
                        padding=True,
                        truncation=True,
                        max_length=min(max_len + 10, 128),
                        return_tensors="pt"
                    )
                    input_ids = inputs["input_ids"].to(device)
                    attention_mask = inputs["attention_mask"].to(device)

                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=min(max_len + 20, 128),
                        num_beams=2,
                        early_stopping=True,
                        do_sample=False,
                        use_cache=True,
                        eos_token_id=tokenizer.eos_token_id
                    )

                    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                    cleaned_outputs = []
                    for output in decoded_outputs:
                        text = output.replace("<SC1>", "").replace("<extra_id_0>", "").strip()
                        text = text.strip('[]')
                        cleaned_outputs.append(text)

                    normalized_neural.extend(cleaned_outputs)

            for idx, neural_text in zip(needs_neural, normalized_neural):
                dict_results.at[idx, 'after'] = neural_text

            output_path = self.result_path.replace('.csv', '_test.csv') if test_mode else self.result_path
            logger.info(f"Saving results to {output_path}")
            dict_results[['id', 'after']].to_csv(output_path, index=False, encoding='utf-8')
            logger.info("Combined normalization completed successfully")

        except Exception as e:
            logger.error(f"Error during neural normalization: {e}")
            return

    def create_submission(self):
        """Create final submission file."""
        logger.info("Creating submission file...")

        try:
            if os.path.exists(self.result_path):
                df = pd.read_csv(self.result_path, encoding='utf-8')
                logger.info(f"Loaded results with {len(df)} rows")

                if 'id' in df.columns and 'after' in df.columns:
                    submission_df = df[['id', 'after']].copy()
                    submission_path = self.result_path.replace('.csv', '_submission.csv')
                    submission_df.to_csv(submission_path, index=False, encoding='utf-8')
                    logger.info(f"Submission file created: {submission_path}")
                    return submission_path
                else:
                    logger.error("Result file missing required columns 'id' and 'after'")
                    return None
            else:
                logger.error(f"Result file not found: {self.result_path}")
                return None

        except Exception as e:
            logger.error(f"Error creating submission file: {e}")
            return None


def main():
    """Main function to run the text normalization."""
    parser = argparse.ArgumentParser(description="Enhanced Russian Text Normalization")
    parser.add_argument("mode", choices=["train", "normalize"],
                       help="Operation mode: train or normalize")
    parser.add_argument("method", choices=["dictionary", "neural", "combined", "rules"],
                       help="Method: dictionary, neural, combined, or rules")
    parser.add_argument("--test", action="store_true",
                       help="Run in test mode (only 10 elements)")
    parser.add_argument("--create-submission", action="store_true",
                       help="Create submission file after normalization")

    args = parser.parse_args()

    normalizer = My_TextNormalization_Model()

    if args.mode == "train":
        if args.method == "dictionary":
            normalizer.train_enhanced_dict()
        else:
            logger.info(f"Training for {args.method} method not implemented yet")

    elif args.mode == "normalize":
        if args.method == "dictionary":
            result = normalizer.normalize_with_dict()
            if result is not None:
                output_path = normalizer.result_path.replace('.csv', '_test.csv') if args.test else normalizer.result_path
                result[['id', 'after']].to_csv(output_path, index=False, encoding='utf-8')
                logger.info(f"Dictionary normalization results saved to {output_path}")

        elif args.method == "neural":
            normalizer.normalize_with_neural(test_mode=args.test)

        elif args.method == "combined":
            normalizer.normalize_combined(test_mode=args.test)

        elif args.method == "rules":
            try:
                test = normalizer.cleaner.safe_load_csv(normalizer.test_path)
                if test is not None:
                    test['id'] = test['sentence_id'].astype(str) + '_' + test['token_id'].astype(str)
                    if args.test:
                        test = test.head(10)

                    test['after'] = test['before'].apply(normalizer.normalize_with_rules)

                    output_path = normalizer.result_path.replace('.csv', '_test.csv') if args.test else normalizer.result_path
                    test[['id', 'after']].to_csv(output_path, index=False, encoding='utf-8')
                    logger.info(f"Rule-based normalization results saved to {output_path}")
                else:
                    logger.error("Failed to load test data")
            except Exception as e:
                logger.error(f"Error in rule-based normalization: {e}")

        if args.create_submission:
            submission_path = normalizer.create_submission()
            if submission_path:
                logger.info(f"Submission file ready: {submission_path}")


if __name__ == '__main__':
    main()