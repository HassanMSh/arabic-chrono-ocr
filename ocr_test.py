import os, csv, re
import datetime
from pdf2image import convert_from_path
from PIL import Image
from kraken import binarization, blla, rpred
from kraken.lib import models
from dateutil import parser
import pandas as pd

# Create base dirs if they don't exist
os.makedirs("images", exist_ok=True)
os.makedirs("ocr", exist_ok=True)

# Create timestamped run dirs
timestamp = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
img_run_dir = os.path.join("images", timestamp)
ocr_run_dir = os.path.join("ocr", timestamp)
os.makedirs(img_run_dir, exist_ok=True)
os.makedirs(ocr_run_dir, exist_ok=True)

print(f"[INFO] Saving images to {img_run_dir}")
print(f"[INFO] Saving OCR text to {ocr_run_dir}")

# CSV output file
csv_path = os.path.join(ocr_run_dir, "ocr_output.csv")

# Load OCR model
model = models.load_any("models/arabic_best.mlmodel")

# Convert page 11 from PDF to images
pages = convert_from_path("books/attacks.pdf", dpi=300, first_page=11, last_page=11)

import json
# ...existing code...
json_path = os.path.join(ocr_run_dir, "ocr_output.json")
ocr_results = []

for i, page in enumerate(pages, start=11):
    w, h = page.size
    halves = {
        "right": page.crop((w // 2, 0, w, h)),  # RTL order: right first
        "left": page.crop((0, 0, w // 2, h)),
    }

    for side, img in halves.items():
        img_path = os.path.join(img_run_dir, f"page_{i}_{side}.png")
        img.save(img_path)

        # OCR
        bin_img = binarization.nlbin(img)
        seg = blla.segment(bin_img)
        pred = rpred.rpred(model, bin_img, seg)
        text = "\n".join([line.prediction for line in pred])

        # Collect results for JSON
        ocr_results.append({
            "page": i,
            "side": side,
            "text": text
        })

# Write to JSON file
with open(json_path, "w", encoding="utf-8") as jsonfile:
    json.dump(ocr_results, jsonfile, ensure_ascii=False, indent=2)