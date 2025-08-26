import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from tavily import TavilyClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional, Tuple
import time
import json
import re
from src.config import INDEX_DESCRIPTIONS

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

pc = Pinecone(api_key=PINECONE_API_KEY)

INDICES = {
    "weekly_stats": pc.Index("weekly-stats"),
    "play_by_play": pc.Index("pbp"),
    "profiles": pc.Index("player-profiles"),
    "injuries": pc.Index("injuries"),
    "schedules": pc.Index("schedules")
}

INDEX_DESCRIPTIONS = INDEX_DESCRIPTIONS

EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

def tavily_search_context(query: str, max_results: int=5) -> str:
    resp = tavily.search(query, max_results=max_results)

    if "results" not in resp:
        return "No web results found."
    
    snippets = []
    for r in resp["results"]:
        title = r.get("title", "Untitled")
        snippet = r.get("snippet", "")
        url = r.get("url", "")
        snippets.append(f"- {title}: {snippet} (Source: {url})")
    return "\n".join(snippets)

def choose_index(query, client, index_descriptions):
    system_prompt = "You are a router that decides which knowledge base to use for a query. Only respond with the index name from the list below:\n\n" + "\n".join([f"{k}: {v}" for k, v in index_descriptions.items()])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}"}
        ],
        temperature=0
    )

    choice = response.choices[0].message.content.strip()

    if choice not in index_descriptions:
        choice = "weekly-stats"
    return choice

def retrieve_context(query, embedder, indices, index_descriptions, client, top_k=5, research_mode=False):
    if research_mode:
        query_embedding = embedder.encode(query).tolist()
        results = []
        for name, idx in indices.items():
            res = idx.query(vector=query_embedding, top_k=top_k, include_metadata=True)
            for match in res['matches']:
                match['source'] = name
                results.append(match)
        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
    
    else:
        index_name = choose_index(query, client, index_descriptions)
        query_embedding = embedder.encode(query).tolist()
        res = indices[index_name].query(vector=query_embedding, top_k=top_k, include_metadata=True)

        results = []
        for match in res['matches']:
            match['source'] = index_name
            results.append(match)
        return results
    

def build_context(results):
    context = "\n\n".join([
        f"Source: {r['source']}\nText: {r['metadata']['text']}" for r in results
    ])
    return context

def answer_question(query, embedder, indices, index_descriptions, client, top_k=5, research_mode=False, score_threshold=0.5):
    results = retrieve_context(query, embedder, indices, index_descriptions, client, top_k=top_k)
    if not results or all(r["score"] < score_threshold for r in results):
        context = tavily_search_context(query)
        source = "web (Tavily)"
    else:
        context = build_context(results)
        source = f"index ({results[0]["source"]})"

    if research_mode:
        prompt = "You are an NFL research assistant. Retrieve and summarize the most relevant information. Always cite snippets from context."
    else:
        prompt = "You are an NFL Q&A assistant. Answer clearly and concisely using the provided context."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content":prompt},
            {"role": "user", "content":f"Question: {query}\n\nContext:\n{context}"}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content, results, source

# KG RAG Functions and Classes

class Neo4jClient:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def run_cypher(self, cypher: str, params: dict = None, timeout: int = 30) -> List[dict]:
        params = params or {}
        with self.driver.session() as session:
            result = session.run(cypher, **params)
            rows = [dict(record) for record in result]
        return rows
    
    def close(self):
        self.driver.close()

neo4j_client = Neo4jClient(uri, username, password)

class VectorRetriever:
    def __init__(self):
        pc = Pinecone(api_key=PINECONE_API_KEY)

        self.indices = {
            "weekly_stats": pc.Index("weekly-stats"),
            "play_by_play": pc.Index("pbp"),
            "profiles": pc.Index("player-profiles"),
            "injuries": pc.Index("injuries"),
            "schedules": pc.Index("schedules")
        }

    def search(self, query_embedding, top_k=5, index_name="weekly_stats", namespace=None):
        index = self.indices[index_name]
        res = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace or ""
        )
        return [
            {
                "id": match["id"],
                "score": match["score"],
                "text": match["metadata"].get("text"),
                "meta": match["metadata"],
            }
            for match in res["matches"]
        ]

vector_retriever = VectorRetriever()

def embed_text(text: str) -> List[float]:
    text_embedding = EMBEDDER.encode(text).tolist()
    return text_embedding

def choose_index_kg(query, client, INDEX_DESCRIPTIONS):
    system_prompt = "You are a router that decides which knowledge base to use for a query. Only respond with the index *key* from the list below:\n\n" + "\n".join([f"{k}: {v}" for k, v in INDEX_DESCRIPTIONS.items()])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}"}
        ],
        temperature=0
    )

    choice = response.choices[0].message.content.strip()

    if choice not in INDEX_DESCRIPTIONS:
        choice = "weekly-stats"
    return choice

client = OpenAI(api_key=OPENAI_API_KEY)
async def cypher_via_llm(question: str) -> Tuple[str, dict]:
    schema_hint = """
    Nodes:
    - "Player" : ["name", "gsis_id", "birth_date", "draft_year", "years_of_experience", "pass_ints", "pfr_id", "pick", "games", "season", "esb_id", "jersey_number", "def_ints", "height", "rookie_season", "otc_id", "weight", "pass_tds", "rec_tds", "pass_attempts", "def_sacks", "dr_av", "position_group", "pass_completions", "common_first_name", "position", "rush_tds", "status", "college_name", "allpro", "seasons_started", "rec_yards", "probowls", "rush_atts", "pff_position", "w_av", "def_solo_tackles", "rush_yards", "last_season", "receptions", "headshot", "first_name", "pass_yards", "smart_id", "pff_id", "draft_pick", "last_name", "draft_round", "latest_team", "round", "to", "espn_id", "age", "college", "nfl_id", "ngs_status", "football_name", "draft_team", "short_name", "pff_status", "hof", "ngs_status_short_description", "side", "pfr_player_id", "team", "cfb_player_id", "college_conference", "category", "ngs_position", "ngs_position_group", "suffix"]
    - "Team" : ["name"]
    - "WeeklyStat" : ["gsis_id", "season", "receptions", "week", "rushing_first_downs", "passing_yards_after_catch", "receiving_epa", "rushing_yards", "passing_epa", "fantasy_points", "receiving_2pt_conversions", "targets", "rushing_tds", "season_type", "pacr", "rushing_fumbles_lost", "fantasy_points_ppr", "passing_air_yards", "receiving_tds", "receiving_air_yards", "rushing_2pt_conversions", "receiving_yards", "sack_yards", "receiving_fumbles_lost", "attempts", "rushing_fumbles", "target_share", "special_teams_tds", "dakota", "passing_2pt_conversions", "receiving_fumbles", "receiving_first_downs", "completions", "racr", "passing_first_downs", "air_yards_share", "wopr", "interceptions", "passing_yards", "sack_fumbles_lost", "carries", "passing_tds", "rushing_epa", "sack_fumbles", "sacks", "receiving_yards_after_catch"]
    - "Injury" : ["gsis_id", "season", "week", "game_type", "practice_primary_injury", "practice_status", "date_modified", "report_primary_injury", "report_status", "report_secondary_injury", "practice_secondary_injury"]

    Relationships:
    - ["Injury"]-[:"AFFECTED_STATS]->["WeeklyStat"]
    - ["Player"]-[:"HAS_STATS"]->["WeeklyStat"]
    - ["Player"]-[:"INJURED_DURING"]->["Injury"]
    - ["WeeklyStat"]-[:"NEXT_WEEK_STATS"]->["WeeklyStat"]
    - ["Player"]-[:"TEAMMATE_WITH"]->["Player"]
    """

    prompt = f"""
    You are an assitant that generates Cypher queries for Neo4j. Only use the provided schema. Do not invent property names.
    Schema:
    {schema_hint}

    User questions: {question}

    Return only JSON with keys 'cypher' and 'params'.
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful assistant that translates natural language questions into Cypher queries."},
            {"role": "user", "content":prompt},
        ],
        temperature=0
    )
    content = resp.choices[0].message.content.strip()

    try:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            content = match.group(0)

        parsed = json.loads(content)
        cypher = parsed.get("cypher", "")
        params = parsed.get("params", {})
        return cypher, params
    except Exception as e:
        print("Cypher parse error:", e)
        return "", {}
    
def merge_results(vector_hits: List[dict], graph_rows: List[dict], merge_strategy: str = "facts-first") -> str:
    """
    Build a context string for the LLM from vector and graph results.
    Strategies:
     - 'facts-first': put KG facts first (structured), then vector snippets.
     - 'concatenate': just concatenate both sources with short headers.
    """
    parts = []
    if merge_strategy == "facts-first":
        if graph_rows:
            parts.append("=== Knowledge Graph Facts ===")
            for r in graph_rows:
                fact_str = ", ".join(f"{k}={v}" for k, v in r.items())
                parts.append(f"- {fact_str}")
        if vector_hits:
            parts.append("\n=== Supporting Passages (vector DB) ===")
            for v in vector_hits:
                score = v.get("score")
                text = v.get("text", "")
                parts.append(f"- (score {score:.3f}) {text}" if score is not None else f"- {text}")
    else:
        parts.append("=== Combined Evidence ===")
        for r in graph_rows:
            fact_str = ", ".join(f"{k}={v}" for k, v in r.items())
            parts.append(f"[KG FACT] {fact_str}")
        for v in vector_hits:
            text = v.get("text", "")
            parts.append(f"[VECTOR] {text}")
    
    ctx = "\n".join(parts)
    return ctx

async def hybrid_retrieve_and_answer(question: str, merge_strategy: str = "facts-first") -> dict:
    start = time.time()

    emb = embed_text(question)
    index_name = choose_index_kg(question, client, INDEX_DESCRIPTIONS)
    
    try:
        vector_hits = vector_retriever.search(query_embedding=emb, top_k=8, index_name=index_name)
    except Exception as e:
        print("Vector search error:", e)
        vector_hits = []

    cypher, params = await cypher_via_llm(question)

    graph_rows = []
    if cypher and cypher.strip():
        try:
            graph_rows = neo4j_client.run_cypher(cypher, params)
        except Exception as e:
            print("Cypher exec error:", e)
            graph_rows = []

    context = merge_results(vector_hits, graph_rows, merge_strategy)

    final_prompt = f"""
    You are an expert assitant for National Football League knowledge. Use the facts and passages below to answer the user's question. If facts are present in the KG, prefer them and cite them.
    Answer concisely and include the source type (KG or Vector) for each factual claim. If you are unsure state so.
    User question: {question}
    Context: {context}
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a concise, factual assitant that cites the source type for each claim."},
            {"role": "user", "content":final_prompt},
        ],
        temperature=0,
        max_tokens=400
    )
    answer = resp.choices[0].message.content

    elapsed = time.time() - start

    return {
        "question": question,
        "cypher_used": bool(cypher),
        "cypher": cypher,
        "cypher_params": params,
        "vector_hits": vector_hits,
        "graph_rows": graph_rows,
        "final_prompt": final_prompt,
        "llm_answer": answer,
        "elapsed_seconds": elapsed
    }