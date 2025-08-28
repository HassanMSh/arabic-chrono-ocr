import os
import tempfile
import datetime
from pdf2image import convert_from_path
from PIL import Image
from kraken import binarization, pageseg, rpred
from kraken.lib.models import load_any

# Create images/ if it doesn't exist
os.makedirs("images", exist_ok=True)

# Create a temp run dir under images/
run_dir = os.path.join("images", datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S"))
os.makedirs(run_dir, exist_ok=True)

print(f"[INFO] Saving output images to {run_dir}")

# Convert page 11 from PDF to images
pages = convert_from_path("books/attacks.pdf", dpi=300, first_page=11, last_page=11)

for i, page in enumerate(pages, start=11):
    w, h = page.size
    left = page.crop((0, 0, w//2, h))
    right = page.crop((w//2, 0, w, h))

    # Save halves into run_dir
    left_path = os.path.join(run_dir, f"page_{i}_left.png")
    right_path = os.path.join(run_dir, f"page_{i}_right.png")
    left.save(left_path)
    right.save(right_path)

    # OCR example: left side only
    bin_img = binarization.nlbin(left)
    lines = pageseg.segment(bin_img)

    # Load pretrained Arabic model
    model = load_any("10.5281/zenodo.7050296")

    pred = rpred.rpred(model, bin_img, lines)
    text = "\n".join([seg.prediction for seg in pred])

    print(f"\n--- OCR Output for Page {i} (Left Half) ---\n")
    print(text)
