import argparse
import numpy as np
import requests
import json
import time
import sys


class TextNormalizerClient:
    def __init__(self, url="localhost:8000"):
        self.url = f"http://{url}/v2/models/text_normalizer/infer"
        print(f"Connecting to: {self.url}")

    def normalize(self, text):
        """Normalize text using Triton server."""
        try:
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

    def test_connection(self):
        """Test if the server is responding."""
        try:
            health_url = self.url.replace("/v2/models/text_normalizer/infer", "/v2/health/ready")
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except:
            return False


def main():
    parser = argparse.ArgumentParser(description="Text normalization client")
    parser.add_argument("--server", default="localhost:8000", help="Triton server URL")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    args = parser.parse_args()

    client = TextNormalizerClient(args.server)

    print("Testing connection to Triton server...")
    if not client.test_connection():
        print("Failed to connect to Triton server!")
        sys.exit(1)
    print("Connection successful!")

    if args.test:
        test_cases = [
            "1900",
            "10.02.2023",
            "123 руб",
            "30%",
            "привет",
            "км",
            "100 км",
            "20:30",
            "1234567890"
        ]

        print("\nRunning test cases:")
        print("=" * 50)

        for i, text in enumerate(test_cases, 1):
            normalized = client.normalize(text)
            print(f"{i:2d}. '{text}' -> '{normalized}'")

        return

    print("\nText Normalization Client (Interactive Mode)")
    print("Enter text to normalize (or 'exit' to quit):")

    while True:
        try:
            text = input("> ")
            if text.lower() in ['exit', 'quit']:
                break

            normalized = client.normalize(text)
            print(f"Normalized: {normalized}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
            break


if __name__ == "__main__":
    main()