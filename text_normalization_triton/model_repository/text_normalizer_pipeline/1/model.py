import json
import logging
import os
import sys
import numpy as np
import triton_python_backend_utils as pb_utils

sys.path.insert(0, '/app')

print(f"=== MODEL.PY STARTING ===", file=sys.stderr)
print(f"Python path: {sys.path}", file=sys.stderr)
print(f"Current working directory: {os.getcwd()}", file=sys.stderr)

try:
    from enhanced_text_normalization import My_TextNormalization_Model

    MODEL_AVAILABLE = True
    print("Successfully imported My_TextNormalization_Model", file=sys.stderr)
except ImportError as e:
    print(f"Error importing enhanced_text_normalization: {e}", file=sys.stderr)
    MODEL_AVAILABLE = False


class TritonPythonModel:
    """Simple text normalization pipeline (rules + dictionary only)."""

    def initialize(self, args):
        print("=== PIPELINE MODEL INITIALIZATION ===", file=sys.stderr)
        self.logger = pb_utils.Logger
        self.logger.log_info("Initializing Text Normalization Pipeline...")

        self.model_config = json.loads(args['model_config'])

        try:
            if MODEL_AVAILABLE:
                self.normalizer = My_TextNormalization_Model()
                self.logger.log_info("My_TextNormalization_Model created successfully")

                try:
                    if os.path.exists('/app/data/dictionary/model_dictionary.json'):
                        self.normalizer.load_dictionaries()
                        self.logger.log_info("Dictionaries loaded successfully")
                    else:
                        self.logger.log_warn("Dictionary files not found - using rules only")
                        print("Dictionary files not found at /app/data/dictionary/", file=sys.stderr)
                except Exception as dict_error:
                    self.logger.log_warn(f"Failed to load dictionaries: {dict_error}")
                    print(f"Dictionary loading error: {dict_error}", file=sys.stderr)

                self.logger.log_info("Enhanced normalization model initialized")
            else:
                self.normalizer = None
                self.logger.log_error("Enhanced normalization model not available")
        except Exception as e:
            self.logger.log_error(f"Error initializing normalizer: {str(e)}")
            print(f"Normalizer initialization error: {e}", file=sys.stderr)
            self.normalizer = None

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                input_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
                if input_tensor is None:
                    raise ValueError("Input tensor 'TEXT' not found")

                input_texts = input_tensor.as_numpy()

                if input_texts.ndim == 0:
                    input_texts = np.array([input_texts])
                elif input_texts.ndim > 1:
                    input_texts = input_texts.flatten()

                normalized_texts = []

                for text_item in input_texts:
                    try:
                        if isinstance(text_item, bytes):
                            text = text_item.decode('utf-8')
                        elif isinstance(text_item, np.bytes_):
                            text = text_item.decode('utf-8')
                        else:
                            text = str(text_item)

                        print(f"Processing text: '{text}'", file=sys.stderr)

                        normalized = self._normalize_text(text)

                        normalized_texts.append(normalized)
                        print(f"Result: '{text}' -> '{normalized}'", file=sys.stderr)

                    except Exception as e:
                        self.logger.log_error(f"Error processing text item: {str(e)}")
                        print(f"Text processing error: {e}", file=sys.stderr)
                        normalized_texts.append(str(text_item))

                output_array = np.array(normalized_texts, dtype=object)
                output_tensor = pb_utils.Tensor("NORMALIZED_TEXT", output_array)

                response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                responses.append(response)

            except Exception as e:
                self.logger.log_error(f"Error processing request: {str(e)}")
                print(f"Request processing error: {e}", file=sys.stderr)
                error_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"Error processing request: {str(e)}")
                )
                responses.append(error_response)

        return responses

    def _normalize_text(self, text):
        """Apply text normalization using available methods."""
        if self.normalizer is None:
            print(f"Using fallback normalization for: '{text}'", file=sys.stderr)
            return self._simple_normalize(text)

        try:
            if hasattr(self.normalizer, 'normalize_text'):
                result = self.normalizer.normalize_text(text)
                print(f"Used normalize_text method: '{text}' -> '{result}'", file=sys.stderr)
                return result

            if hasattr(self.normalizer, 'general_dict') and self.normalizer.general_dict:
                text_lower = text.lower()
                if text_lower in self.normalizer.general_dict:
                    result = self.normalizer.general_dict[text_lower]
                    print(f"Used dictionary: '{text}' -> '{result}'", file=sys.stderr)
                    return result

            if hasattr(self.normalizer, 'normalize_with_rules'):
                result = self.normalizer.normalize_with_rules(text)
                print(f"Used rules: '{text}' -> '{result}'", file=sys.stderr)
                return result

        except Exception as e:
            self.logger.log_error(f"Error in normalization: {e}")
            print(f"Normalization error: {e}", file=sys.stderr)

        return self._simple_normalize(text)

    def _simple_normalize(self, text):
        """Simple fallback normalization."""
        import re

        print(f"Using simple normalization for: '{text}'", file=sys.stderr)

        if text.isdigit():
            return f"число {text}"
        if re.match(r'\d+\s*руб', text):
            return re.sub(r'(\d+)\s*руб', r'\1 рублей', text)
        if re.match(r'\d+%', text):
            return re.sub(r'(\d+)%', r'\1 процентов', text)
        return text

    def finalize(self):
        self.logger.log_info("Finalizing Text Normalization Pipeline...")
        print("=== MODEL FINALIZATION ===", file=sys.stderr)