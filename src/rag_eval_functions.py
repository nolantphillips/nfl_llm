import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import re
import random
import pandas as pd
from typing import Any, Dict
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pinecone import Pinecone

from src.config import INDEX_DESCRIPTIONS
from src.rag_functions import answer_question

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI(api_key=OPENAI_API_KEY)

indices = {
    "weekly_stats": pc.Index("weekly-stats"),
    "play_by_play": pc.Index("pbp"),
    "profiles": pc.Index("player-profiles"),
    "injuries": pc.Index("injuries"),
    "schedules": pc.Index("schedules")
}

def evaluate_numeric_answer(pred: str, truth: float, tolerance: int = 5) -> Dict[str, Any]:
    try:
        pred_num = int(re.findall(r'\d+', pred)[0])
    except:
        return {"exact": 0, "within_tol": 0}
    
    exact = int(pred_num == truth)
    within_tol = int(abs(pred_num - truth) <= tolerance)

    return {"exact": exact, "within_tol": within_tol}

def evaluate_numeric_answer_kg(pred: str, truth: float, tolerance: float = 5.0) -> Dict[str, Any]:
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", pred)
    if not numbers:
        return {"exact": 0, "within_tol": 0}
    
    try:
        numbers = [float(n) for n in numbers]
    except ValueError:
        return {"exact": 0, "within_tol": 0}
    pred_num = min(numbers, key=lambda x: abs(x-truth))
    exact = int(pred_num == truth)
    within_tol = int(abs(pred_num - truth) <= tolerance)

    return {"exact": exact, "within_tol": within_tol}

def evaluate_numeric_from_prediction(prediction_dict: Dict[str, Any], truth: float, tolerance: float = 5.0) -> Dict[str, Any]:
    llm_answer = prediction_dict.get("llm_answer", "")
    return evaluate_numeric_answer_kg(llm_answer, truth, tolerance)

def evaluate_label_answer(pred: str, truth: str) -> Dict[str, Any]:
    pred_str = str(pred).lower()
    truth_str = str(truth).lower()

    statuses = ["out", "questionable", "doubtful"]

    found_status = None
    for s in statuses:
        if s in pred_str:
            found_status = s
            break
    
    exact = int(found_status == truth_str)
    return {"exact": exact}

def evaluate_label_from_prediction(prediction_dict: dict, truth: str) -> dict:
    llm_answer = prediction_dict.get("llm_answer", "")
    return evaluate_label_answer(llm_answer, truth)

def evaluate_text_answer(pred: str, truth: str) -> Dict[str, Any]:
    norm_pred, norm_truth = str(pred).lower(), str(truth).lower()
    exact = int(norm_pred == norm_truth)
    fuzzy = int(norm_truth in norm_pred or norm_pred in norm_truth)
    return {"exact": exact, "fuzzy": fuzzy}

def evaluate_text_from_prediction(prediction_dict: dict, truth: str) -> dict:
    llm_answer = prediction_dict.get("llm_answer", "")
    return evaluate_text_answer(llm_answer, truth)

def generate_weekly_stats_queries(df, n_samples=100):
    stat_templates = {
        "completions": "How many passes did {player} complete in Week {week}, {season}?",
        "attempts": "How many passes did {player} attempt in Week {week}, {season}?",
        "passing_yards": "How many passing yards did {player} have in Week {week}, {season}?",
        "passing_tds": "How many passing touchdowns did {player} throw in Week {week}, {season}?",
        "interceptions": "How many interceptions did {player} throw in Week {week}, {season}?",
        "sacks": "How many times was {player} sacked in Week {week}, {season}?",
        "carries": "How many carries did {player} have in Week {week}, {season}?",
        "rushing_yards": "How many rushing yards did {player} have in Week {week}, {season}?",
        "rushing_tds": "How many rushing touchdowns did {player} score in Week {week}, {season}?",
        "receptions": "How many catches did {player} have in Week {week}, {season}?",
        "targets": "How many targets did {player} have in Week {week}, {season}?",
        "receiving_yards": "How many receiving yards did {player} have in Week {week}, {season}?",
        "receiving_tds": "How many receiving touchdowns did {player} score in Week {week}, {season}?",
        "fantasy_points": "How many fantasy points did {player} score in Week {week}, {season}?",
        "fantasy_points_ppr": "How many PPR fantasy points did {player} score in Week {week}, {season}?"
    }

    queries = []
    for _ in range(n_samples):
        row = df.sample(1).iloc[0]
        player = row["player_display_name"]
        week = int(row["week"])
        season = int(row["season"])

        col = random.choice([c for c in stat_templates.keys() if pd.notnull(row[c])])
        query_text = stat_templates[col].format(player=player, week=week, season=season)

        queries.append({
            "query": query_text,
            "truth_lookup": lambda df, r=row, c=col: r[c],
            "eval_fn": evaluate_numeric_from_prediction,
            "index": "weekly-stats"
        })

    return queries


def generate_profile_queries(df, n_samples=100):
    queries = []

    profile_fields = [
        ("college", "Where did {player} play college football?"),
        ("draft_team", "Which team drafted {player}?"),
        ("draft_year", "What year was {player} drafted?"),
        ("position", "What position does {player} play?"),
        ("latest_team", "Which team does {player} currently play for?"),
        ("rookie_season", "What year did {player} enter the NFL?"),
        ("height", "How tall is {player}?"),
        ("weight", "What is the weight of {player}?"),
        ("draft_round", "What round was {player} drafted?"),
        ("draft_pick", "What pick was {player} in the draft?"),
        ("allpro", "How many times has {player} been named an All-Pro?"),
        ("jersey_number", "What number does {player} wear?")
    ]

    for _ in range(n_samples):
        row = df.sample(1).iloc[0]
        player = row["name"]
        field, template = random.choice([f for f in profile_fields if pd.notnull(row[f[0]])])
        query_text = template.format(player=player)
        truth_value = row[field]

        queries.append({
            "query": query_text,
            "truth_lookup": lambda df, r=row, f=field: r[f],
            "eval_fn": evaluate_text_from_prediction,
            "index": "profiles"
        })

    return queries

def generate_injury_queries(df, n_samples=100):
    """
    Generates queries for the injuries index.
    """
    queries = []

    for _ in range(n_samples):
        row = df.sample(1).iloc[0]
        player = row["full_name"]
        week = int(row["week"])
        season = int(row["season"])
        injury_status = row["report_status"]

        # Only generate if status exists
        if pd.notnull(injury_status):
            query_text = f"What was {player}'s injury status in Week {week}, {season}?"
            queries.append({
                "query": query_text,
                "truth_lookup": lambda df, r=row: r["report_status"],
                "eval_fn": evaluate_label_from_prediction,
                "index": "injuries"
            })

    return queries

def run_rag(query: str, index: str) -> str:

    index_descriptions = INDEX_DESCRIPTIONS
    answer, results, source = answer_question(query=query,
                                              embedder=embedder,
                                              indices=indices,
                                              index_descriptions=index_descriptions,
                                              client=client,
                                              top_k=5,
                                              research_mode=False
                                            )
    return answer