"""Download CLMET 3.1 (plain-text version, Parquet) from the Hugging Face Hub into data/clmet/raw/plain.

Source: biglam/clmet_3_1 (CC BY-SA 4.0), a mirror of De Smet, Diller & Tyrkkö (2015),
The Corpus of Late Modern English Texts, version 3.1.
"""
import os
from huggingface_hub import snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

path = snapshot_download("biglam/clmet_3_1", repo_type="dataset", revision="refs/convert/parquet",
                         allow_patterns=["plain/*"], local_dir=os.path.join(ROOT, "data", "clmet", "raw"))
print("Downloaded to", path)
