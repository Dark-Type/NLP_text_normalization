import os
import logging
from pathlib import Path
import torch
from transformers import T5ForConditionalGeneration, GPT2Tokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_model_to_onnx():
    """Convert the T5 model to ONNX format using optimum library."""
    try:
        os.makedirs("../../text_normalizer_onnx/1", exist_ok=True)

        MODEL_NAME = "saarus72/russian_text_normalizer"
        cache_dir = Path('../../models_cache')
        cache_dir.mkdir(exist_ok=True)

        logger.info(f"Loading model {MODEL_NAME}")
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)

        tokenizer.save_pretrained("model_repository/text_normalizer/tokenizer_files")

        logger.info("Converting model to ONNX with optimum...")

        ort_model = ORTModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            from_transformers=True,
            cache_dir=cache_dir
        )

        onnx_path = Path("../../text_normalizer_onnx/1")
        ort_model.save_pretrained(onnx_path)

        logger.info(f"Model successfully converted to ONNX and saved to {onnx_path}")

        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            model_path = onnx_path / "model.onnx"
            decoder_path = onnx_path / "decoder_model.onnx"

            if model_path.exists():
                logger.info("Quantizing encoder model...")
                quantized_model_path = onnx_path / "model_quantized.onnx"
                quantize_dynamic(str(model_path), str(quantized_model_path), weight_type=QuantType.QUInt8)
                model_path.unlink()
                quantized_model_path.rename(model_path)

            if decoder_path.exists():
                logger.info("Quantizing decoder model...")
                quantized_decoder_path = onnx_path / "decoder_model_quantized.onnx"
                quantize_dynamic(str(decoder_path), str(quantized_decoder_path), weight_type=QuantType.QUInt8)
                decoder_path.unlink()
                quantized_decoder_path.rename(decoder_path)

            logger.info("Model quantization completed successfully")
        except Exception as e:
            logger.warning(f"Quantization step skipped: {e}")

        logger.info("ONNX conversion complete!")

    except Exception as e:
        logger.error(f"Error converting model to ONNX: {e}")
        raise


if __name__ == "__main__":
    convert_model_to_onnx()