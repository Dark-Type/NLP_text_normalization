import json
import re
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Python model for rule-based text normalization."""

    def initialize(self, args):
        """Initialize the model with rule resources."""
        self.model_config = model_config = json.loads(args['model_config'])

        self._init_rule_resources()

        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(model_config, "NORMALIZED")['data_type'])

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

        self.months = {
            '1': 'января', '2': 'февраля', '3': 'марта', '4': 'апреля',
            '5': 'мая', '6': 'июня', '7': 'июля', '8': 'августа',
            '9': 'сентября', '10': 'октября', '11': 'ноября', '12': 'декабря',
            '01': 'января', '02': 'февраля', '03': 'марта', '04': 'апреля',
            '05': 'мая', '06': 'июня', '07': 'июля', '08': 'августа',
            '09': 'сентября'
        }

        self.special_years = {
            '1700': 'тысяча семьсот', '1800': 'тысяча восемьсот',
            '1900': 'тысяча девятьсот', '2000': 'две тысячи'
        }

    def execute(self, requests):
        """Process a batch of requests."""
        responses = []

        for request in requests:
            in_text = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()[0]
            text = in_text.decode('utf-8')

            normalized, changed = self._normalize_with_rules(text)

            out_tensor = pb_utils.Tensor("NORMALIZED",
                                         np.array([normalized.encode('utf-8')], dtype=np.object_))
            changed_tensor = pb_utils.Tensor("CHANGED",
                                             np.array([changed], dtype=np.bool_))

            response = pb_utils.InferenceResponse(output_tensors=[out_tensor, changed_tensor])
            responses.append(response)

        return responses

    def _normalize_with_rules(self, text: str) -> tuple:
        """Apply rule-based normalization with improved handling."""
        if not text or not isinstance(text, str):
            return "", False

        try:
            original_text = text

            if re.match(r'\d{1,2}[./]\d{1,2}[./]\d{4}', text):
                text = self._normalize_dates(text)
                return text, text != original_text

            if re.match(r'\d{1,2}:\d{2}', text):
                text = self._normalize_time(text)
                return text, text != original_text

            if re.match(r'\d+\s*(?:[₽$€]|руб\.?)', text):
                text = self._normalize_currency(text)
                return text, text != original_text

            if re.match(r'\d+(?:[.,]\d+)?\s*%', text):
                text = self._normalize_percentages(text)
                return text, text != original_text

            if re.match(r'\d+(?:[.,]\d+)?\s*(?:кг|г|км|м|см|мм|л|мл|°C|°)', text):
                text = self._normalize_measurements(text)
                return text, text != original_text

            if re.match(r'\b\d{4}\b', text):
                if text in self.special_years:
                    return self.special_years[text], True

                try:
                    year_int = int(text)
                    if 1000 <= year_int <= 2099:
                        normalized = self._convert_year(text)
                        return normalized, normalized != text
                except:
                    pass

            if re.match(r'\b\d{1,6}\b', text):
                try:
                    num = int(text)
                    normalized = self._number_to_words(num)
                    return normalized, normalized != text
                except:
                    pass

            return text, False

        except Exception as e:
            print(f"Error during rule-based normalization: {str(e)}")
            return text, False

    def _normalize_dates(self, text):
        date_match = re.match(r'(\d{1,2})[./](\d{1,2})[./](\d{4})', text)
        if date_match:
            day, month, year = date_match.groups()
            day_word = self.ordinals.get(str(int(day)), f"{day}-е")
            month_word = self.months.get(str(int(month)), month)
            year_word = self._convert_year(year)
            return f"{day_word} {month_word} {year_word} года"
        return text

    def _normalize_time(self, text):
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

    def _normalize_currency(self, text):
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
                currency_word = 'рублей'

            return f"{amount_word} {currency_word}"
        return text

    def _normalize_percentages(self, text):
        percentage_match = re.match(r'(\d+(?:[.,]\d+)?)%', text)
        if percentage_match:
            amount = percentage_match.group(1)

            if '.' in amount or ',' in amount:
                amount = amount.replace(',', '.')
                amount_parts = amount.split('.')
                integer_part = int(amount_parts[0])
                decimal_part = amount_parts[1]

                integer_word = self._number_to_words(integer_part)

                if decimal_part == '0':
                    amount_word = integer_word
                else:
                    decimal_int = int(decimal_part)
                    decimal_word = self._number_to_words(decimal_int)

                    if len(decimal_part) == 1:
                        decimal_denominator = "десятых"
                    elif len(decimal_part) == 2:
                        decimal_denominator = "сотых"
                    else:
                        decimal_denominator = "тысячных"

                    amount_word = f"{integer_word} целых {decimal_word} {decimal_denominator}"
            else:
                amount_int = int(amount)
                amount_word = self._number_to_words(amount_int)

            if amount == '1' or (amount.isdigit() and int(amount) % 10 == 1 and int(amount) % 100 != 11):
                return f"{amount_word} процент"
            elif amount.isdigit() and 2 <= int(amount) % 10 <= 4 and (
                    int(amount) % 100 < 10 or int(amount) % 100 >= 20):
                return f"{amount_word} процента"
            else:
                return f"{amount_word} процентов"
        return text

    def _normalize_measurements(self, text):
        measurement_match = re.match(r'(\d+(?:[.,]\d+)?)\s*(кг|г|км|м|см|мм|л|мл|°C|°)', text)
        if measurement_match:
            amount, unit = measurement_match.groups()

            if '.' in amount or ',' in amount:
                amount = amount.replace(',', '.')
                amount_parts = amount.split('.')
                integer_part = int(amount_parts[0])
                decimal_part = amount_parts[1]

                integer_word = self._number_to_words(integer_part)

                if decimal_part == '0':
                    amount_word = integer_word
                else:
                    decimal_int = int(decimal_part)
                    decimal_word = self._number_to_words(decimal_int)

                    if len(decimal_part) == 1:
                        decimal_denominator = "десятых"
                    elif len(decimal_part) == 2:
                        decimal_denominator = "сотых"
                    else:
                        decimal_denominator = "тысячных"

                    amount_word = f"{integer_word} целых {decimal_word} {decimal_denominator}"

                unit_mapping = {
                    'кг': 'килограммов', 'г': 'граммов', 'км': 'километров',
                    'м': 'метров', 'см': 'сантиметров', 'мм': 'миллиметров',
                    'л': 'литров', 'мл': 'миллилитров', '°C': 'градусов цельсия',
                    '°': 'градусов'
                }
                unit_word = unit_mapping.get(unit, unit)
            else:
                amount_int = int(amount)
                amount_word = self._number_to_words(amount_int)

                if amount_int % 10 == 1 and amount_int % 100 != 11:
                    unit_mapping = {
                        'кг': 'килограмм', 'г': 'грамм', 'км': 'километр',
                        'м': 'метр', 'см': 'сантиметр', 'мм': 'миллиметр',
                        'л': 'литр', 'мл': 'миллилитр', '°C': 'градус цельсия',
                        '°': 'градус'
                    }
                elif 2 <= amount_int % 10 <= 4 and (amount_int % 100 < 10 or amount_int % 100 >= 20):
                    unit_mapping = {
                        'кг': 'килограмма', 'г': 'грамма', 'км': 'километра',
                        'м': 'метра', 'см': 'сантиметра', 'мм': 'миллиметра',
                        'л': 'литра', 'мл': 'миллилитра', '°C': 'градуса цельсия',
                        '°': 'градуса'
                    }
                else:
                    unit_mapping = {
                        'кг': 'килограммов', 'г': 'граммов', 'км': 'километров',
                        'м': 'метров', 'см': 'сантиметров', 'мм': 'миллиметров',
                        'л': 'литров', 'мл': 'миллилитров', '°C': 'градусов цельсия',
                        '°': 'градусов'
                    }
                unit_word = unit_mapping.get(unit, unit)

            return f"{amount_word} {unit_word}"
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

        elif 1000 <= num <= 9999:
            thousands = num // 1000
            remainder = num % 1000

            if thousands == 1:
                result = "одна тысяча"
            elif thousands == 2:
                result = "две тысячи"
            elif 3 <= thousands <= 4:
                result = f"{self.numbers[str(thousands)]} тысячи"
            else:
                result = f"{self.numbers[str(thousands)]} тысяч"

            if remainder > 0:
                result += f" {self._number_to_words(remainder)}"
            return result

        else:
            return str(num)

    def _convert_year(self, year: str) -> str:
        """Convert year to spoken form with special cases."""
        if year in self.special_years:
            return self.special_years[year]

        try:
            year_int = int(year)

            if 1000 <= year_int <= 1999:
                century = year_int // 100
                remainder = year_int % 100

                if remainder == 0:
                    return f"тысяча {self.hundreds[str((century - 10) * 100)]}"
                else:
                    return f"тысяча {self.hundreds[str((century - 10) * 100)]} {self._number_to_words(remainder)}"
            elif 2000 <= year_int <= 2099:
                remainder = year_int % 100
                if remainder == 0:
                    return "две тысячи"
                else:
                    return f"две тысячи {self._number_to_words(remainder)}"
            else:
                return self._number_to_words(year_int)
        except ValueError:
            return year