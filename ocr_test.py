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

# Convert Arabic-Indic digits → ASCII digits
def normalize_digits(s: str) -> str:
    trans = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    return s.translate(trans)

# Extract dates from text
def extract_dates(text: str):
    text_norm = normalize_digits(text)
    date_pattern = re.compile(r'(\d{2,4})[/-](\d{1,3})[/-](\d{1,4})')  # allow OCR glitches
    matches = date_pattern.findall(text_norm)

    raw_dates = []
    clean_dates = []

    for y, m, d in matches:
        raw_dates.append(f"{y}/{m}/{d}")  # save raw fragment

        try:
            y = int(y)
            m = int(m)
            d = int(d)

            # Fix 2-digit years (assume 1900s for this corpus)
            if y < 100:
                y = 1900 + y

            # Validate ranges
            if not (1 <= m <= 12):
                continue
            if not (1 <= d <= 31):
                continue

            clean_dates.append(f"{y:04d}-{m:02d}-{d:02d}")
        except Exception:
            continue

    # Return semicolon-separated
    return ";".join(raw_dates), ";".join(sorted(set(clean_dates)))

with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["page", "side", "text", "dates_raw", "dates_normalized"])  # header

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

            # Dates
            dates_raw, dates_norm = extract_dates(text)

            # Write to CSV
            writer.writerow([i, side, text, dates_raw, dates_norm])

print(f"[INFO] OCR results written to {csv_path}")

df = pd.read_csv(csv_path)

df_expanded = df.assign(
    normalized_date=df['dates_normalized'].str.split(';')
).explode('normalized_date')