import os
import sys

from tavily import TavilyClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

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