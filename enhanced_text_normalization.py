import re
import logging
import pandas as pd
import numpy as np
import torch
import json
import os
import argparse
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
from pathlib import Path
import unicodedata
from collections import Counter, defaultdict
from tqdm import tqdm

from transformers import T5ForConditionalGeneration, GPT2Tokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./data/log_file.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TextCleaner:
    """Text cleaner for dataset with improved handling of encoding issues."""

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

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataset with improved handling."""
        if df is None:
            self.logger.error("DataFrame is None, cannot clean")
            return None

        self.logger.info(f"Starting dataset cleaning of {len(df)} rows")

        df['before'] = df['before'].astype(str).replace('nan', '')
        df['after'] = df['after'].astype(str).replace('nan', '')

        df['before'] = df['before'].apply(self._fix_encoding)
        df['after'] = df['after'].apply(self._fix_encoding)

        initial_len = len(df)
        df = df[(df['before'].str.strip() != '') & (df['after'].str.strip() != '')]
        self.logger.info(f"Removed {initial_len - len(df)} empty rows")

        df['before'] = df['before'].apply(self._clean_text)
        df['after'] = df['after'].apply(self._clean_text)

        initial_len = len(df)
        df = df.drop_duplicates(subset=['before', 'after'])
        self.logger.info(f"Removed {initial_len - len(df)} duplicates")

        return df.reset_index(drop=True)

    def _fix_encoding(self, text: str) -> str:
        """Fix common encoding issues in text."""
        if pd.isna(text):
            return ""

        text = str(text)

        try:
            text = unicodedata.normalize('NFKC', text)
        except:
            pass

        fixes = {
            'Ã¡': 'а', 'Ã ': 'а', 'Ã«': 'е', 'Ã¬': 'и', 'Ã®': 'о', 'Ã³': 'у',
            'â€œ': '"', 'â€': '"', 'â€™': "'", 'â€"': '–', 'â€"': '—',
            '\u200b': '', '\ufeff': '', '\xa0': ' '
        }

        for wrong, correct in fixes.items():
            text = text.replace(wrong, correct)

        return text

    def _clean_text(self, text: str) -> str:
        """Clean problematic characters from text."""
        if pd.isna(text):
            return ""

        text = str(text)

        problematic_chars = set(['', '', '', '﻿', '\x00', '\x01', '\x02', '\x03'])
        text = ''.join(char for char in text if char not in problematic_chars)

        text = re.sub(r'\s+', ' ', text).strip()

        return text


class My_TextNormalization_Model:
    """
    Enhanced Russian text normalization model with improved accuracy.
    """

    def __init__(self):
        """Initialize the enhanced text normalization model."""
        logger.info("Initializing Enhanced Russian Text Normalization Model")

        self.dictionary_path = 'data/dictionary/model_dictionary.json'
        self.class_dict_path = 'data/dictionary/class_dictionaries.json'
        self.special_dict_path = 'data/dictionary/special_cases.json'
        self.results_path = 'data/results.csv'
        self.train_path = 'data/ru_train.csv'
        self.test_path = 'data/ru_test_2.csv'
        self.result_path = 'data/result.csv'

        os.makedirs('data/dictionary', exist_ok=True)
        os.makedirs('data/results', exist_ok=True)
        os.makedirs('models_cache', exist_ok=True)

        self.cleaner = TextCleaner()

        self.general_dict = {}
        self.class_dictionaries = {}
        self.special_cases = {}

        self._init_rule_resources()

        self.model_name = "saarus72/russian_text_normalizer"
        self.tokenizer = None
        self.model = None
        self.device = None

    def _init_rule_resources(self):
        """Initialize resources for rule-based normalization."""
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
        self.centuries = {
            '19': 'девятнадцатый',
            '18': 'восемнадцатый',
            '17': 'семнадцатый',
            '20': 'двадцатый',
            '21': 'двадцать первый'
        }
        self.special_years = {
            '1812': 'тысяча восемьсот двенадцатый',
            '1917': 'тысяча девятьсот семнадцатый',
            '1941': 'тысяча девятьсот сорок первый',
            '1945': 'тысяча девятьсот сорок пятый',
            '2000': 'двухтысячный'
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

        self.abbreviations = {
            'т.е.': 'то есть', 'и т.д.': 'и так далее', 'и т.п.': 'и тому подобное',
            'т.к.': 'так как', 'т.н.': 'так называемый', 'см.': 'смотрите',
            'стр.': 'страница', 'гл.': 'глава', 'рис.': 'рисунок',
            'табл.': 'таблица', 'г.': 'год', 'гг.': 'годы', 'в.': 'век',
            'вв.': 'века', 'др.': 'другие', 'пр.': 'прочие',
            'напр.': 'например', 'англ.': 'английский', 'рус.': 'русский',
            'обл.': 'область', 'р-н': 'район', 'пос.': 'поселок', 'с.': 'село',
            'пр-т': 'проспект', 'ул.': 'улица', 'тыс.': 'тысяч', 'млн.': 'миллионов',
            'млрд.': 'миллиардов', 'руб.': 'рублей', 'коп.': 'копеек',
            'св.': 'святой', 'им.': 'имени', 'проф.': 'профессор',
            'акад.': 'академик', 'доц.': 'доцент', 'тел.': 'телефон'
        }

        self.currency_symbols = {
            '₽': 'рублей', 'руб': 'рублей', '$': 'долларов', '€': 'евро',
            'USD': 'долларов', 'EUR': 'евро', 'RUB': 'рублей'
        }

        self.gendered_words = {
            'один': {'masculine': 'один', 'feminine': 'одна', 'neuter': 'одно'},
            'два': {'masculine': 'два', 'feminine': 'две', 'neuter': 'два'}
        }

    def train_dict(self):
        """Create improved dictionary with better class awareness."""
        logger.info("Starting enhanced dictionary creation...")

        train_df = self.cleaner.safe_load_csv(self.train_path)
        if train_df is None:
            logger.error("Failed to load training data")
            return

        train_df = self.cleaner.clean_dataset(train_df)

        train_df['before'] = train_df['before'].str.lower()
        train_df['after'] = train_df['after'].str.lower()

        train_df['after_c'] = train_df['after'].map(lambda x: len(str(x).split()))
        train_df = train_df[~((train_df['class'] == 'LETTERS') & (train_df['after_c'] > 4))]

        self._create_general_dictionary(train_df)

        self._create_class_dictionaries(train_df)

        self._extract_special_cases(train_df)

        try:
            with open(self.dictionary_path, 'w', encoding='utf-8') as f:
                json.dump(self.general_dict, f, indent=4, ensure_ascii=False)

            with open(self.class_dict_path, 'w', encoding='utf-8') as f:
                json.dump(self.class_dictionaries, f, indent=4, ensure_ascii=False)

            with open(self.special_dict_path, 'w', encoding='utf-8') as f:
                json.dump(self.special_cases, f, indent=4, ensure_ascii=False)

            logger.info(f"Dictionaries saved successfully")
        except Exception as e:
            logger.error(f"Failed to save dictionaries: {e}")

        logger.info("Enhanced dictionary creation completed successfully")

    def _create_general_dictionary(self, df: pd.DataFrame):
        """Create general dictionary - original approach which works well."""
        train_df = df.groupby(['before', 'after'], as_index=False)['sentence_id'].count()
        train_df = train_df.sort_values(['sentence_id', 'before'], ascending=[False, True])
        train_df = train_df.drop_duplicates(['before'])
        self.general_dict = {key: value for (key, value) in train_df[['before', 'after']].values}
        logger.info(f"General dictionary created with {len(self.general_dict)} entries")

    def _create_class_dictionaries(self, df: pd.DataFrame):
        """Create specialized dictionaries for each token class."""
        self.class_dictionaries = {}

        for class_name, class_df in df.groupby('class'):
            if len(class_df) < 10:
                continue

            class_counts = class_df.groupby(['before', 'after']).size().reset_index(name='count')
            class_counts = class_counts.sort_values(['before', 'count'], ascending=[True, False])

            class_dict = {}
            for before, group in class_counts.groupby('before'):
                most_frequent = group.iloc[0]
                class_dict[before] = most_frequent['after']

            self.class_dictionaries[class_name] = class_dict
            logger.info(f"Class dictionary for '{class_name}' created with {len(class_dict)} entries")

    def _extract_special_cases(self, df: pd.DataFrame):
        """Extract special cases that need custom handling."""
        special_cases = {}

        ambiguous = df.groupby('before')['after'].nunique()
        ambiguous = ambiguous[ambiguous > 1].index.tolist()

        for token in ambiguous[:1000]:
            normalizations = df[df['before'] == token][['after', 'class']].drop_duplicates()
            if len(normalizations) <= 1:
                continue

            norm_dict = {}
            for _, row in normalizations.iterrows():
                cls = row['class'] if not pd.isna(row['class']) else 'UNKNOWN'
                norm_dict[cls] = row['after']

            special_cases[token] = norm_dict

        self.special_cases = special_cases
        logger.info(f"Extracted {len(special_cases)} special cases with context-dependent normalizations")

    def load_dictionaries(self):
        """Load all dictionaries from files."""
        try:
            with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                self.general_dict = json.load(f)
                logger.info(f"General dictionary loaded with {len(self.general_dict)} entries")
        except Exception as e:
            logger.warning(f"Failed to load general dictionary: {e}")
            self.general_dict = {}

        try:
            with open(self.class_dict_path, 'r', encoding='utf-8') as f:
                self.class_dictionaries = json.load(f)
                logger.info(f"Class dictionaries loaded: {list(self.class_dictionaries.keys())}")
        except Exception as e:
            logger.warning(f"Failed to load class dictionaries: {e}")
            self.class_dictionaries = {}

        try:
            with open(self.special_dict_path, 'r', encoding='utf-8') as f:
                self.special_cases = json.load(f)
                logger.info(f"Special cases loaded: {len(self.special_cases)} items")
        except Exception as e:
            logger.warning(f"Failed to load special cases: {e}")
            self.special_cases = {}

    def normalize_text_dict(self):
        """Normalize text using improved dictionary approach."""
        logger.info("Starting dictionary-based normalization...")

        if not self.general_dict:
            self.load_dictionaries()

        try:
            test = self.cleaner.safe_load_csv(self.test_path)
            if test is None:
                logger.error("Failed to load test data")
                return None
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            return None

        test['id'] = test['sentence_id'].astype(str) + '_' + test['token_id'].astype(str)
        test['before_l'] = test['before'].str.lower()

        def normalize_token(row):
            token = row['before_l']
            token_class = row.get('class', '')

            if pd.isna(token) or token is None:
                return ''

            if token_class and token_class in self.class_dictionaries and token in self.class_dictionaries[token_class]:
                return self.class_dictionaries[token_class][token]

            if token in self.special_cases and token_class in self.special_cases[token]:
                return self.special_cases[token][token_class]

            if token in self.general_dict:
                return self.general_dict[token]

            if any(c.isdigit() or (c.isascii() and c.isalpha()) for c in token):
                rule_result = self.normalize_with_rules(token)
                if rule_result != token:
                    return rule_result

            return token

        test['after'] = test.apply(normalize_token, axis=1)

        def fix_case(original, lower, after):
            if pd.isna(original) or pd.isna(lower) or pd.isna(after):
                return "" if pd.isna(original) else original

            if lower == after:
                return original
            else:
                return after

        test['after'] = test.apply(lambda r: fix_case(r['before'], r['before_l'], r['after']), axis=1)

        logger.info("Dictionary-based normalization completed")
        return test

    def normalize_with_rules(self, text: str) -> str:
        """Apply rule-based normalization with improved handling."""
        if not text or not isinstance(text, str) or pd.isna(text):
            return ""

        try:
            if re.match(r'\d{1,2}[./]\d{1,2}[./]\d{4}', text):
                return self._normalize_dates(text)

            if re.match(r'\d{1,2}:\d{2}', text):
                return self._normalize_time(text)

            if re.match(r'\d+\s*(?:[₽$€]|руб\.?)', text):
                return self._normalize_currency(text)

            if re.match(r'\d+(?:[.,]\d+)?\s*%', text):
                return self._normalize_percentages(text)

            if re.match(r'\d+(?:[.,]\d+)?\s*(?:кг|г|км|м|см|мм|л|мл|°C|°)', text):
                return self._normalize_measurements(text)

            if re.match(r'(?:\+7|8)\s*(?:\d{10}|\(\d{3}\)\s*\d{3}-\d{2}-\d{2})', text):
                return self._normalize_phone_numbers(text)

            if re.match(r'\b(I{1,3}|IV|V|VI{0,3}|IX|X{1,2}|XI{0,3}|XIV|XV|XVI{0,3}|XIX|XX)\b', text):
                return self._normalize_roman_numerals(text)

            if re.match(r'\b\d{1,6}\b', text):
                return self._normalize_numbers(text)

            normalized_text = self._preprocess_text(text)
            normalized_text = self._normalize_abbreviations(normalized_text)
            normalized_text = self._normalize_urls_emails(normalized_text)
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
        date_match = re.match(r'(\d{1,2})[./](\d{1,2})[./](\d{4})', text)
        if date_match:
            day, month, year = date_match.groups()
            day_word = self.ordinals.get(str(int(day)), f"{day}-е")
            month_word = self.months.get(str(int(month)), month)
            year_word = self._convert_year(year)
            return f"{day_word} {month_word} {year_word} года"

        short_date_match = re.match(r'(\d{1,2})[./](\d{1,2})(?!/)', text)
        if short_date_match:
            day, month = short_date_match.groups()
            day_word = self.ordinals.get(str(int(day)), f"{day}-е")
            month_word = self.months.get(str(int(month)), month)
            return f"{day_word} {month_word}"

        return text

    def _normalize_time(self, text: str) -> str:
        """Normalize time expressions with correct grammatical forms."""
        time_match = re.match(r'(\d{1,2}):(\d{2})', text)
        if time_match:
            hours, minutes = time_match.groups()
            hour_int = int(hours)
            minute_int = int(minutes)

            if hour_int == 1:
                hour_word = "один час"
            elif 2 <= hour_int <= 4:
                hour_word = f"{self._number_to_words(hour_int)} часа"
            else:
                hour_word = f"{self._number_to_words(hour_int)} часов"

            if minute_int == 0:
                return hour_word
            else:
                if minute_int == 1:
                    minute_word = "одна минута"
                elif 2 <= minute_int <= 4:
                    minute_word = f"{self._number_to_words(minute_int)} минуты"
                else:
                    minute_word = f"{self._number_to_words(minute_int)} минут"
                return f"{hour_word} {minute_word}"

        return text

    def _normalize_currency(self, text: str) -> str:
        """Normalize currency expressions with correct grammatical forms."""
        currency_match = re.match(r'(\d+)\s*([₽$€]|руб\.?)', text)
        if currency_match:
            amount, currency = currency_match.groups()
            amount_int = int(amount)
            amount_word = self._number_to_words(amount_int)

            if currency == '₽' or currency == 'руб':
                if amount_int % 10 == 1 and amount_int % 100 != 11:
                    currency_word = 'рубль'
                elif 2 <= amount_int % 10 <= 4 and (amount_int % 100 < 10 or amount_int % 100 >= 20):
                    currency_word = 'рубля'
                else:
                    currency_word = 'рублей'
            elif currency == '$':
                if amount_int % 10 == 1 and amount_int % 100 != 11:
                    currency_word = 'доллар'
                elif 2 <= amount_int % 10 <= 4 and (amount_int % 100 < 10 or amount_int % 100 >= 20):
                    currency_word = 'доллара'
                else:
                    currency_word = 'долларов'
            elif currency == '€':
                currency_word = 'евро'
            else:
                currency_word = self.currency_symbols.get(currency, currency)

            return f"{amount_word} {currency_word}"

        decimal_currency_match = re.match(r'(\d+[.,]\d+)\s*([₽$€]|руб\.?)', text)
        if decimal_currency_match:
            amount, currency = decimal_currency_match.groups()
            amount_word = self._convert_decimal_number(amount)
            currency_word = self.currency_symbols.get(currency, currency)
            return f"{amount_word} {currency_word}"

        return text

    def _normalize_measurements(self, text: str) -> str:
        """Normalize measurement units with correct grammatical forms."""
        measurement_match = re.match(r'(\d+(?:[.,]\d+)?)\s*(кг|г|км|м|см|мм|л|мл|°C|°)', text)
        if measurement_match:
            amount, unit = measurement_match.groups()

            if '.' in amount or ',' in amount:
                amount_word = self._convert_decimal_number(amount)
                unit_mappings = {
                    'кг': 'килограммов', 'г': 'граммов', 'км': 'километров',
                    'м': 'метров', 'см': 'сантиметров', 'мм': 'миллиметров',
                    'л': 'литров', 'мл': 'миллилитров', '°C': 'градусов цельсия',
                    '°': 'градусов'
                }
                unit_word = unit_mappings.get(unit, unit)
            else:
                amount_int = int(amount)
                amount_word = self._number_to_words(amount_int)

                if unit in ['кг', 'г', 'км', 'м', 'см', 'мм', 'л', 'мл']:
                    if amount_int % 10 == 1 and amount_int % 100 != 11:
                        unit_mappings = {
                            'кг': 'килограмм', 'г': 'грамм', 'км': 'километр',
                            'м': 'метр', 'см': 'сантиметр', 'мм': 'миллиметр',
                            'л': 'литр', 'мл': 'миллилитр'
                        }
                    elif 2 <= amount_int % 10 <= 4 and (amount_int % 100 < 10 or amount_int % 100 >= 20):
                        unit_mappings = {
                            'кг': 'килограмма', 'г': 'грамма', 'км': 'километра',
                            'м': 'метра', 'см': 'сантиметра', 'мм': 'миллиметра',
                            'л': 'литра', 'мл': 'миллилитра'
                        }
                    else:
                        unit_mappings = {
                            'кг': 'килограммов', 'г': 'граммов', 'км': 'километров',
                            'м': 'метров', 'см': 'сантиметров', 'мм': 'миллиметров',
                            'л': 'литров', 'мл': 'миллилитров'
                        }
                else:
                    unit_mappings = {
                        '°C': 'градусов цельсия',
                        '°': 'градусов' if amount_int != 1 else 'градус'
                    }

                unit_word = unit_mappings.get(unit, unit)

            return f"{amount_word} {unit_word}"

        return text

    def _normalize_percentages(self, text: str) -> str:
        """Normalize percentage expressions with correct grammatical forms."""
        percentage_match = re.match(r'(\d+(?:[.,]\d+)?)%', text)
        if percentage_match:
            amount = percentage_match.group(1)

            if '.' in amount or ',' in amount:
                amount_word = self._convert_decimal_number(amount)
                return f"{amount_word} процентов"
            else:
                amount_int = int(amount)
                amount_word = self._number_to_words(amount_int)

                if amount_int % 10 == 1 and amount_int % 100 != 11:
                    return f"{amount_word} процент"
                elif 2 <= amount_int % 10 <= 4 and (amount_int % 100 < 10 or amount_int % 100 >= 20):
                    return f"{amount_word} процента"
                else:
                    return f"{amount_word} процентов"

        return text

    def _normalize_phone_numbers(self, text: str) -> str:
        """Normalize phone numbers with better formatting."""
        phone_match = re.match(r'(?:\+7|8)\s*(?:\d{10}|\(\d{3}\)\s*\d{3}-\d{2}-\d{2})', text)
        if phone_match:
            digits = re.sub(r'[^\d]', '', text)

            formatted_digits = []

            if digits.startswith('7') or digits.startswith('8'):
                formatted_digits.append(self.numbers.get(digits[0], digits[0]))
                digits = digits[1:]

            i = 0
            while i < len(digits):
                if i + 2 <= len(digits):
                    pair = digits[i:i + 2]
                    if pair[0] != '0' and int(pair) <= 19:
                        formatted_digits.append(self._number_to_words(int(pair)))
                    else:
                        formatted_digits.append(self.numbers.get(pair[0], pair[0]))
                        formatted_digits.append(self.numbers.get(pair[1], pair[1]))
                else:
                    formatted_digits.append(self.numbers.get(digits[i], digits[i]))
                i += 2

            return ' '.join(formatted_digits)

        return text

    def _normalize_urls_emails(self, text: str) -> str:
        """Normalize URLs and email addresses."""
        if re.match(r'https?://[^\s]+', text):
            return 'ссылка'

        if re.match(r'www\.[^\s]+', text):
            return 'веб сайт'

        if re.match(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', text):
            return 'электронная почта'

        return text

    def _normalize_abbreviations(self, text: str) -> str:
        """Normalize common abbreviations."""
        for abbr, expansion in sorted(self.abbreviations.items(), key=lambda x: len(x[0]), reverse=True):
            if text.lower() == abbr.lower():
                return expansion

        for abbr, expansion in sorted(self.abbreviations.items(), key=lambda x: len(x[0]), reverse=True):
            if re.search(r'\b' + re.escape(abbr) + r'\b', text, re.IGNORECASE):
                text = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, text, flags=re.IGNORECASE)

        return text

    def _normalize_roman_numerals(self, text: str) -> str:
        """Normalize Roman numerals with contextual awareness."""
        roman_match = re.match(r'\b(I{1,3}|IV|V|VI{0,3}|IX|X{1,2}|XI{0,3}|XIV|XV|XVI{0,3}|XIX|XX)\b', text)
        if roman_match:
            roman = roman_match.group(1)
            return self.roman_numerals.get(roman, roman)

        return text

    def _normalize_numbers(self, text: str) -> str:
        """Normalize standalone numbers to words."""
        number_match = re.match(r'\b\d{1,6}\b', text)
        if number_match:
            try:
                num = int(text)
                return self._number_to_words(num)
            except ValueError:
                pass

        return text

    def _normalize_punctuation(self, text: str) -> str:
        """Handle punctuation marks."""
        text = text.replace('&', ' и ')
        text = text.replace('+', ' плюс ')
        text = text.replace('=', ' равно ')
        text = text.replace('№', 'номер ')
        text = text.replace('*', ' звездочка ')

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
        """Convert numbers to Russian words with grammatical correctness."""
        if num == 0:
            return 'ноль'

        if num < 0:
            return f"минус {self._number_to_words(abs(num))}"

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

        elif 1000 <= num <= 999999:
            thousands = num // 1000
            remainder = num % 1000

            if thousands == 1:
                result = "одна тысяча"
            elif thousands == 2:
                result = "две тысячи"
            elif 3 <= thousands <= 4:
                result = f"{self._number_to_words(thousands)} тысячи"
            else:
                result = f"{self._number_to_words(thousands)} тысяч"

            if remainder > 0:
                result += f" {self._number_to_words(remainder)}"
            return result

        else:
            return str(num)

    def _convert_decimal_number(self, number_str: str) -> str:
        """Convert decimal numbers to words with proper Russian forms."""
        number_str = number_str.replace(',', '.')

        try:
            if '.' in number_str:
                integer_part, decimal_part = number_str.split('.')

                if integer_part == '0' or integer_part == '':
                    integer_word = 'ноль'
                else:
                    integer_word = self._number_to_words(int(integer_part))

                if decimal_part == '0' or decimal_part == '':
                    return integer_word

                decimal_length = len(decimal_part)
                if decimal_length == 1:
                    denominator = "десятых"
                elif decimal_length == 2:
                    denominator = "сотых"
                elif decimal_length == 3:
                    denominator = "тысячных"
                else:
                    denominator = f"десяти в минус {decimal_length} степени"

                decimal_int = int(decimal_part)
                numerator = self._number_to_words(decimal_int)

                return f"{integer_word} целых {numerator} {denominator}"
            else:
                return self._number_to_words(int(number_str))
        except ValueError:
            return number_str

    def _convert_year(self, year: str) -> str:
        """Convert year to spoken form with special cases."""
        try:
            year_int = int(year)

            if 1000 <= year_int <= 1999:
                century = year_int // 100
                remainder = year_int % 100

                if remainder == 0:
                    century_key = str(century * 100 % 1000)
                    if century_key in self.hundreds:
                        return f"{self.hundreds[century_key]}й"
                    else:
                        return self._number_to_words(year_int)
                else:
                    century_key = str(century * 100 % 1000)
                    if century_key in self.hundreds:
                        return f"{self.hundreds[century_key]} {self._number_to_words(remainder)}"
                    else:
                        return self._number_to_words(year_int)
            elif 2000 <= year_int <= 2099:
                remainder = year_int % 100
                if remainder == 0:
                    return "двухтысячный"
                else:
                    return f"две тысячи {self._number_to_words(remainder)}"
            else:
                return self._number_to_words(year_int)
        except ValueError:
            return year
        except Exception as e:
            logger.error(f"Error converting year {year}: {e}")
            return year

    def normalize_with_neural(self, test_mode=False):
        """
        Normalize text using neural T5 model with improved preprocessing.
        Args:
            test_mode (bool): If True, only process first 10 items for testing
        """
        logger.info("Starting neural normalization with T5 model...")

        try:
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
                    batch_classes = test['class'].iloc[i:i + batch_size].tolist() if 'class' in test.columns else None

                    formatted_texts = []
                    max_len = 0
                    for j, text in enumerate(batch_texts):
                        if any(c.isdigit() or (c.isascii() and c.isalpha()) for c in text):
                            class_tag = f"[{batch_classes[j]}]" if batch_classes and batch_classes[j] else ""

                            if text.isdigit():
                                text_rev = text[::-1]
                                groups = [text_rev[i:i + 3][::-1] for i in range(0, len(text_rev), 3)]
                                text = ' '.join(groups[::-1])

                            formatted_text = f"<SC1>{class_tag}[{text}]<extra_id_0>"
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
                        num_beams=3,
                        early_stopping=True,
                        do_sample=False,
                        use_cache=True,
                        eos_token_id=tokenizer.eos_token_id
                    )

                    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                    cleaned_outputs = []
                    for output, original_text in zip(decoded_outputs, batch_texts):
                        if any(c.isdigit() or (c.isascii() and c.isalpha()) for c in original_text):
                            text = output.replace("<SC1>", "").replace("<extra_id_0>", "").strip()
                            text = text.strip('[]')
                            text = re.sub(r'\[([A-Z_]+)\]', '', text).strip()
                        else:
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
        Combined normalization approach with enhanced dictionary and targeted neural application.
        Args:
            test_mode (bool): If True, only process first 10 items for testing
        """
        logger.info("Starting enhanced combined normalization...")

        logger.info("Step 1: Applying dictionary normalization...")
        dict_results = self.normalize_text_dict()
        if dict_results is None:
            logger.error("Failed to perform dictionary normalization")
            return

        needs_neural = []
        for idx, row in dict_results.iterrows():
            if (row['before'] == row['after'] and
                    any(c.isdigit() or (c.isascii() and c.isalpha()) for c in row['before']) and
                    len(row['before']) > 1):
                needs_neural.append(idx)

        logger.info(f"Found {len(needs_neural)} tokens requiring neural normalization")

        if not needs_neural:
            logger.info("All tokens successfully normalized with dictionary")
            output_path = self.result_path.replace('.csv', '_test.csv') if test_mode else self.result_path
            dict_results[['id', 'after']].to_csv(output_path, index=False, encoding='utf-8')
            return

        logger.info("Step 3: Applying neural normalization for selected tokens...")

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
                    batch_classes = neural_texts['class'].iloc[
                                    i:i + batch_size].tolist() if 'class' in neural_texts.columns else None

                    formatted_texts = []
                    max_len = 0
                    for j, text in enumerate(batch_texts):
                        class_tag = f"[{batch_classes[j]}]" if batch_classes and j < len(batch_classes) and \
                                                               batch_classes[j] else ""

                        if text.isdigit():
                            text_rev = text[::-1]
                            groups = [text_rev[i:i + 3][::-1] for i in range(0, len(text_rev), 3)]
                            text = ' '.join(groups[::-1])

                        formatted_text = f"<SC1>{class_tag}[{text}]<extra_id_0>"
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
                        num_beams=3,
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
                        text = re.sub(r'\[([A-Z_]+)\]', '', text).strip()
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

    def normalize_text(self, text: str) -> str:
        """
        Normalize a single text input using optimized combined approach.
        This is the main API method for external applications.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        if not text or not isinstance(text, str) or pd.isna(text):
            return ""

        try:
            if not self.general_dict:
                self.load_dictionaries()

            text_lower = text.lower()

            if text_lower in self.general_dict:
                return self.general_dict[text_lower]

            rule_result = self.normalize_with_rules(text)
            if rule_result != text:
                return rule_result

            if any(c.isdigit() or (c.isascii() and c.isalpha()) for c in text):
                try:
                    if not hasattr(self, 'model') or self.model is None:
                        MODEL_NAME = "saarus72/russian_text_normalizer"
                        cache_dir = Path('models_cache')

                        if torch.cuda.is_available():
                            self.device = torch.device("cuda")
                        else:
                            self.device = torch.device("cpu")

                        self.tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
                        self.model = T5ForConditionalGeneration.from_pretrained(
                            MODEL_NAME,
                            cache_dir=cache_dir,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            low_cpu_mem_usage=True
                        )

                        self.model = self.model.to(self.device)
                        self.model.eval()

                    if text.isdigit():
                        text_rev = text[::-1]
                        groups = [text_rev[i:i + 3][::-1] for i in range(0, len(text_rev), 3)]
                        formatted_text = ' '.join(groups[::-1])
                    else:
                        formatted_text = text

                    formatted_input = f"<SC1>[{formatted_text}]<extra_id_0>"

                    with torch.inference_mode():
                        inputs = self.tokenizer(
                            formatted_input,
                            return_tensors="pt"
                        )
                        input_ids = inputs["input_ids"].to(self.device)
                        attention_mask = inputs["attention_mask"].to(self.device)

                        outputs = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_length=128,
                            num_beams=3,
                            early_stopping=True,
                            do_sample=False,
                            use_cache=True,
                            eos_token_id=self.tokenizer.eos_token_id
                        )

                        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        normalized = decoded_output.replace("<SC1>", "").replace("<extra_id_0>", "").strip()
                        normalized = normalized.strip('[]')

                        return normalized

                except Exception as e:
                    logger.error(f"Error in neural normalization: {e}")
                    return text

            return text

        except Exception as e:
            logger.error(f"Error in text normalization: {e}")
            return text

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
            normalizer.train_dict()
        else:
            logger.info(f"Training for {args.method} method not implemented yet")

    elif args.mode == "normalize":
        if args.method == "dictionary":
            result = normalizer.normalize_text_dict()
            if result is not None:
                output_path = normalizer.result_path.replace('.csv',
                                                             '_test.csv') if args.test else normalizer.result_path
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

                    output_path = normalizer.result_path.replace('.csv',
                                                                 '_test.csv') if args.test else normalizer.result_path
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
