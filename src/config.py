import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Define directories
PARENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PARENT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"

# Create directories if they don't exist
for directory in [
    DATA_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)