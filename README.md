# Russian Text Normalization for TTS

**Student:** Skazhutin Nikita  
**Group:** 972301 

## Description

This project implements a Russian text normalization model for Text-to-Speech (TTS) systems. It converts written text into appropriate spoken forms, handling numbers, dates, currencies, abbreviations, and other special cases.




## How To Use

### Installation

```bash
pip install pandas numpy torch transformers tqdm
```
or just install using poetry (lots of unnecessary dependencies there, though)
### Basic Usage

```python
from enhanced_text_normalization import My_TextNormalization_Model

# Initialize the model
normalizer = My_TextNormalization_Model()

# Normalize a single text
result = normalizer.normalize_text("У меня есть 1000 рублей")
print(result)  # "у меня есть одна тысяча рублей"

# Examples of normalization
print(normalizer.normalize_text("25.12.2023"))     # "двадцать пятое декабря две тысячи двадцать три года"
print(normalizer.normalize_text("15:30"))          # "пятнадцать часов тридцать минут"  
print(normalizer.normalize_text("500₽"))           # "пятьсот рублей"
print(normalizer.normalize_text("25%"))            # "двадцать пять процентов"
print(normalizer.normalize_text("т.е."))           # "то есть"
```

### Command Line Usage

#### 1. Train Dictionary (Required first step)
```bash
python enhanced_text_normalization.py train dictionary
```

#### 2. Run Normalization

**Dictionary-based normalization:**
```bash
python enhanced_text_normalization.py normalize dictionary
```

**Rule-based normalization:**
```bash
python enhanced_text_normalization.py normalize rules
```

**Neural normalization:**
```bash
python enhanced_text_normalization.py normalize neural
```

**Combined approach (recommended):**
```bash
python enhanced_text_normalization.py normalize combined
```

#### 3. Test Mode (process only 10 samples)
```bash
python enhanced_text_normalization.py normalize combined --test
```

#### 4. Create Submission File
```bash
python enhanced_text_normalization.py normalize combined --create-submission
```

### Data Structure Requirements

Your `data/` directory should contain:
```
data/
├── ru_train.csv          # Training data
├── ru_test_2.csv         # Test data
├── dictionary/           # Generated dictionaries (auto-created)
└── log_file.log         # Logs (auto-created)
```

The CSV files should have columns: `sentence_id`, `token_id`, `before`, `after`, `class`

## Methods Explained

### 1. Dictionary Method
- Fastest approach
- Uses pre-built dictionaries from training data
- Good for common cases


### 2. Rule-based Method  
- Uses linguistic rules for Russian language
- Handles numbers, dates, currencies, etc.
- No ML model required


### 3. Neural Method
- Uses T5 transformer model (`saarus72/russian_text_normalizer`)
- Best for complex/rare cases
- Requires GPU for good performance, which I do not have :(

### 4. Combined Method (Recommended)
- Dictionary first, then neural for unresolved cases
- Best balance of speed and accuracy



## Logging

All operations are logged to `./data/log_file.log` with detailed information about processing steps and any errors encountered.

## Output

The normalization process creates:
- `data/result.csv` - Main results file
- `data/result_submission.csv` - Kaggle submission format (with --create-submission)
- `data/dictionary/` - Generated dictionaries for future use
