# 🏈 NFL Knowledge Assistant (RAG Application)

An interactive **Retrieval-Augmented Generation (RAG)** app that answers questions about NFL player stats, injuries, and game performance.  
Built with **Streamlit**, deployed via **Streamlit Cloud**, and powered by **Neo4j, Pinecone, and OpenAI**.  

🔗 **Live App**: [https://nfl-rag.streamlit.app/]   

---

## 🚀 Features
- **Hybrid Retrieval**: Combines
  - **Neo4j Knowledge Graph** for structured data (player stats, injuries, teams, etc.)
  - **Pinecone Vector Search** for semantic retrieval from unstructured text  
- **LLM-Powered Q&A**: Uses OpenAI to generate context-rich answers  
- **Interactive Frontend**: Built in Streamlit with:
  - Query input & response display  
  - Expandable details (Cypher query, vector hits, raw results)  
  - Query log and drift/degradation monitoring tab  
- **Secure Deployment**: Credentials managed via GitHub Secrets & Streamlit Secrets

---

## 🛠️ Tech Stack
- **Frontend**: Streamlit  
- **LLM**: OpenAI GPT models  
- **Vector DB**: Pinecone  
- **Graph DB**: Neo4j AuraDB  
- **Additional**: Tavily API for supplementary retrieval  

---

## 📂 Repository Structure
nfl_llm/
│
├── app/ # Streamlit app files
│ └── app.py # Main entry point
│
├── src/ # Core RAG pipeline logic
│ ├── rag_functions.py # Functions for RAG pipelines
│ └── rag_eval_functions.py # Functions used to evaluate accuracy of answers from RAG pipelines
│ └── config.py # Defines directories to access data and defines vector DB index descriptions
|
├── requirements.txt # Python dependencies
├── requirements_with_versions.txt # Python dependencies with versions
├── README.md # Project documentation

1. **Clone the repo**
   ```bash
   git clone https://github.com/nolantphillips/nfl-llm.git
   cd nfl-llm
   ```

2. Create & activate environment
   ```bash
   conda create -n nfl_llm python=3.12
   conda activate nfl_llm
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Add API keys
   ```bash
   OPENAI_API_KEY = "your_openai_key"
   PINECONE_API_KEY = "your_pinecone_key"
   TAVILY_API_KEY = "your_tavily_key"
  
   NEO4J_URI = "your_neo4j_uri"
   NEO4J_USERNAME = "neo4j"
   NEO4J_PASSWORD = "your_password"
   NEO4J_DATABASE = "neo4j"
   ```

5. Run locally
   ```bash
   streamlit run app/app.py
   ```

6. Example queries
   - What were Josh Allen’s passing yards in Week 15 of 2024?
   - Show me players with a rush EPA over 5 in week 8, 2023.
