#!/bin/bash
set -e

mkdir -p perf_results

if [ ! -f "perf_data/input_data_list.txt" ]; then
    echo "Generating test data..."
    python perf_data/generate_test_data.py
fi

echo "Starting performance tests..."

echo "Running throughput test for dictionary_normalizer..."
docker compose run perf_analyzer \
    --model-name dictionary_normalizer \
    --input-data perf_data/input_data_list.txt \
    --batch-size 1 \
    --concurrency-range 1:16:2 \
    --measurement-interval 10000 \
    --output-shared-memory-size 102400 \
    --measurement-mode count_windows \
    --measurement-request-count 1000 \
    --url host.docker.internal:8000 \
    --input-format text \
    > perf_results/dictionary_throughput.txt

echo "Running latency test for dictionary_normalizer..."
docker compose run perf_analyzer \
    --model-name dictionary_normalizer \
    --input-data perf_data/input_data_list.txt \
    --batch-size 1 \
    --concurrency-range 1 \
    --measurement-interval 10000 \
    --output-shared-memory-size 102400 \
    --measurement-mode count_windows \
    --measurement-request-count 1000 \
    --url host.docker.internal:8000 \
    --input-format text \
    > perf_results/dictionary_latency.txt

echo "Running throughput test for rule_normalizer..."
docker compose run perf_analyzer \
    --model-name rule_normalizer \
    --input-data perf_data/input_data_list.txt \
    --batch-size 1 \
    --concurrency-range 1:16:2 \
    --measurement-interval 10000 \
    --output-shared-memory-size 102400 \
    --measurement-mode count_windows \
    --measurement-request-count 1000 \
    --url host.docker.internal:8000 \
    --input-format text \
    > perf_results/rule_throughput.txt

echo "Running test for ensemble model..."
docker compose run perf_analyzer \
    --model-name text_normalization_ensemble \
    --input-data perf_data/input_data_list.txt \
    --batch-size 1 \
    --concurrency-range 1:8:1 \
    --measurement-interval 10000 \
    --output-shared-memory-size 102400 \
    --measurement-mode count_windows \
    --measurement-request-count 1000 \
    --url host.docker.internal:8000 \
    --input-format text \
    > perf_results/ensemble.txt

if docker compose exec triton ls /models/text_normalizer/1/encoder_model.onnx > /dev/null 2>&1; then
    echo "Running test for ONNX model..."
    

    
    docker compose run perf_analyzer \
        --model-name text_normalizer \
        --shape input_ids:1,16 \
        --shape attention_mask:1,16 \
        --batch-size 1 \
        --concurrency-range 1:4:1 \
        --measurement-interval 10000 \
        --measurement-mode count_windows \
        --measurement-request-count 100 \
        --url host.docker.internal:8000 \
        > perf_results/text_normalizer.txt
fi

echo "Performance tests completed! Results are in the perf_results directory."