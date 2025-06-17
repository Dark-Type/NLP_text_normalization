import argparse
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype


def test_model_performance(url, model_name, num_requests=100):
    """Test model performance with multiple requests."""
    client = httpclient.InferenceServerClient(url=url)

    test_texts = [
        "1900", "10.02.2023", "123 руб", "30%", "привет",
        "км", "100 км", "20:30", "1234567890"
    ]

    for text in test_texts:
        text_data = np.array([text.encode('utf-8')], dtype=np.object_)
        input_tensor = httpclient.InferInput("TEXT", text_data.shape, np_to_triton_dtype(text_data.dtype))
        input_tensor.set_data_from_numpy(text_data)
        client.infer(model_name=model_name, inputs=[input_tensor])

    latencies = []

    for _ in tqdm(range(num_requests)):
        text = np.random.choice(test_texts)

        text_data = np.array([text.encode('utf-8')], dtype=np.object_)
        input_tensor = httpclient.InferInput("TEXT", text_data.shape, np_to_triton_dtype(text_data.dtype))
        input_tensor.set_data_from_numpy(text_data)

        start_time = time.time()
        response = client.infer(model_name=model_name, inputs=[input_tensor])
        end_time = time.time()

        latencies.append((end_time - start_time) * 1000)  # ms

    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)

    print(f"\nPerformance Results for {model_name} ({num_requests} requests):")
    print(f"Mean latency: {mean_latency:.2f} ms")
    print(f"P50 latency: {p50_latency:.2f} ms")
    print(f"P95 latency: {p95_latency:.2f} ms")
    print(f"P99 latency: {p99_latency:.2f} ms")
    print(f"Min latency: {min_latency:.2f} ms")
    print(f"Max latency: {max_latency:.2f} ms")
    print(f"Throughput: {1000 / mean_latency:.2f} requests/second")


def main():
    parser = argparse.ArgumentParser(description="Test Triton model performance")
    parser.add_argument("--server", default="localhost:8000", help="Triton server URL")
    parser.add_argument("--model", default="text_normalization_ensemble",
                        help="Model name to test (dictionary_normalizer, rule_normalizer, text_normalization_ensemble)")
    parser.add_argument("--requests", type=int, default=100, help="Number of test requests")

    args = parser.parse_args()

    try:
        test_model_performance(args.server, args.model, args.requests)
    except Exception as e:
        print(f"Error during performance test: {e}")


if __name__ == "__main__":
    main()