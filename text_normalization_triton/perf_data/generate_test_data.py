import os
import json
import numpy as np

os.makedirs("perf_data", exist_ok=True)

dict_data = [
    "1900",
    "10.02.2023",
    "123 руб",
    "30%",
    "привет",
    "км",
    "100 км",
    "20:30",
    "1234567890",
    "С.-Петербург",
    "XIX",
    "25 €",
    "07.05.1945",
    "35°C",
    "т.е.",
    "50-60",
    "д. 5, корп. 2",
    "www.example.com",
    "hello@example.com",
    "5 тыс.",
]

for i, text in enumerate(dict_data):
    with open(f"perf_data/input_{i}.txt", "w", encoding="utf-8") as f:
        f.write(text)

with open("perf_data/input_data_list.txt", "w", encoding="utf-8") as f:
    for i in range(len(dict_data)):
        f.write(f"perf_data/input_{i}.txt\n")

print("Performance test data generated!")