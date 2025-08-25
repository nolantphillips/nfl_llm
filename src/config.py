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

# Define indices and descriptions
INDEX_DESCRIPTIONS = {
    "profiles": "Player biographies, colleges, draft info, positions, and general background.",
    "weekly_stats": "Player weekly performance stats (yards, touchdowns, completions, etc.).",
    "injuries": "Injury reports, player status, and health updates.",
    "schedules": "Game schedules, opponents, outcomes, and dates.",
    "play_by_play": "Detailed play-by-play data including drives, quarters, and play outcomes."
}

# Create directories if they don't exist
for directory in [
    DATA_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)