import argparse
import numpy as np
import requests
import json


class TextNormalizerClient:
    def __init__(self, url="localhost:8000"):
        self.url = f"http://{url}/v2/models/text_normalization_ensemble/infer"

    def normalize(self, text):
        """Normalize text using Triton server."""
        try:
            # Create request payload
            payload = {
                "inputs": [
                    {
                        "name": "TEXT",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": [text]
                    }
                ]
            }

            # Send request
            headers = {"Content-Type": "application/json"}
            response = requests.post(self.url, headers=headers, data=json.dumps(payload))

            if response.status_code == 200:
                result = response.json()
                normalized_text = result["outputs"][0]["data"][0]
                return normalized_text
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return text

        except Exception as e:
            print(f"Error normalizing text: {e}")
            return text


def main():
    parser = argparse.ArgumentParser(description="Text normalization client")
    parser.add_argument("--server", default="localhost:8000", help="Triton server URL")
    args = parser.parse_args()

    client = TextNormalizerClient(args.server)

    print("Text Normalization Client")
    print("Enter text to normalize (or 'exit' to quit):")

    while True:
        text = input("> ")
        if text.lower() == 'exit':
            break

        normalized = client.normalize(text)
        print(f"Normalized: {normalized}")


if __name__ == "__main__":
    main()