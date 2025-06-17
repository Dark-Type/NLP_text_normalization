import json
import numpy as np
import sys

try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    print("Warning: triton_python_backend_utils not available - running in test mode")
    pb_utils = None


class TritonPythonModel:
    """Python model for dictionary-based normalization."""

    def initialize(self, args):
        """Initialize model with hard-coded dictionaries."""
        if pb_utils:
            self.model_config = json.loads(args['model_config'])

        self.dictionary = {
            "1900": "тысяча девятьсот",
            "1945": "тысяча девятьсот сорок пять",
            "2000": "две тысячи",
            "км": "километр",
            "руб": "рублей",
            "т.е.": "то есть",
            "напр.": "например",
        }

        self.special_cases = {
            "xix": {"century": "девятнадцатый век"},
            "xx": {"century": "двадцатый век"},
            "xxi": {"century": "двадцать первый век"},
        }

        print(f"Dictionary normalizer initialized with {len(self.dictionary)} entries")

    def execute(self, requests):
        """Process a batch of requests."""
        if not pb_utils:
            return None

        responses = []

        for request in requests:
            in_text = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()[0]
            text = in_text.decode('utf-8').lower()
            original_text = in_text.decode('utf-8')

            normalized = text
            found = False

            if text in self.special_cases:
                for cls, value in self.special_cases[text].items():
                    normalized = value
                    found = True
                    break

            if not found and text in self.dictionary:
                normalized = self.dictionary[text]
                found = True

            if not found:
                normalized = original_text

            out_tensor = pb_utils.Tensor("NORMALIZED",
                                         np.array([normalized.encode('utf-8')], dtype=np.object_))
            found_tensor = pb_utils.Tensor("FOUND",
                                           np.array([found], dtype=np.bool_))

            response = pb_utils.InferenceResponse(output_tensors=[out_tensor, found_tensor])
            responses.append(response)

        return responses