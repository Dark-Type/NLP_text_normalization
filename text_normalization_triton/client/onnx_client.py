import argparse
import numpy as np
import torch
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
from transformers import GPT2Tokenizer


class ONNXClient:
    """Client for T5 ONNX model in Triton."""

    def __init__(self, url="localhost:8000", model_name="text_normalizer"):
        self.url = url
        self.model_name = model_name
        self.client = httpclient.InferenceServerClient(url=url)

        self.tokenizer = GPT2Tokenizer.from_pretrained("saarus72/russian_text_normalizer")

    def preprocess(self, text):
        """Preprocess text for the model."""
        if any(c.isdigit() or (c.isascii() and c.isalpha()) for c in text):
            if text.isdigit():
                text_rev = text[::-1]
                groups = [text_rev[i:i + 3][::-1] for i in range(0, len(text_rev), 3)]
                formatted_text = ' '.join(groups[::-1])
            else:
                formatted_text = text

            formatted_input = f"<SC1>[{formatted_text}]<extra_id_0>"
        else:
            return text, None, None

        inputs = self.tokenizer(
            formatted_input,
            return_tensors="np",
            padding=True
        )

        return text, inputs["input_ids"], inputs["attention_mask"]

    def postprocess(self, output, original_text):
        """Postprocess model output."""
        if output is None:
            return original_text

        token_ids = output.argmax(axis=-1)

        token_ids_tensor = torch.tensor(token_ids)

        decoded_text = self.tokenizer.decode(token_ids_tensor[0], skip_special_tokens=True)

        cleaned_text = decoded_text.replace("<SC1>", "").replace("<extra_id_0>", "").strip()
        cleaned_text = cleaned_text.strip('[]')

        return cleaned_text

    def normalize(self, text):
        """Normalize text using the T5 ONNX model."""
        original_text, input_ids, attention_mask = self.preprocess(text)

        if input_ids is None:
            return original_text

        inputs = []
        inputs.append(httpclient.InferInput("input_ids", input_ids.shape, np_to_triton_dtype(input_ids.dtype)))
        inputs.append(
            httpclient.InferInput("attention_mask", attention_mask.shape, np_to_triton_dtype(attention_mask.dtype)))

        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)

        response = self.client.infer(self.model_name, inputs)

        output = response.as_numpy("logits")

        normalized_text = self.postprocess(output, original_text)

        return normalized_text


def main():
    parser = argparse.ArgumentParser(description="ONNX model client")
    parser.add_argument("--url", default="localhost:8000", help="Triton server URL")
    parser.add_argument("--model", default="text_normalizer", help="Model name")

    args = parser.parse_args()

    client = ONNXClient(url=args.url, model_name=args.model)

    print("ONNX Model Text Normalization Client")
    print("Enter text to normalize (or 'exit' to quit):")

    while True:
        text = input("> ")
        if text.lower() == 'exit':
            break

        normalized = client.normalize(text)
        print(f"Normalized: {normalized}")


if __name__ == "__main__":
    main()