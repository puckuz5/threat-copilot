"""
RAG_ENGINE.PY — Threat Copilot Query Engine
──────────────────────────────────────────────
Takes a security analyst's question in plain English,
searches the 2,788-incident knowledge base, and returns
a grounded answer using Groq (LLaMA 3).

Includes a HALLUCINATION GUARD — this is the detail
that separates this project from a typical RAG demo.
Most people skip this. You won't.

Usage:
    from rag_engine import ThreatCopilot
    copilot = ThreatCopilot()
    answer, sources, confidence = copilot.ask("Show me Chinese APT attacks on energy sector")
"""

import os
import re
import faiss
import joblib
import numpy as np

from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

KB_DIR = "knowledge_base"


class ThreatCopilot:

    def __init__(self):
        print("Loading Threat Copilot engine...")

        # Load the embedding model — same one used in ingest.py
        # MUST be identical or vector search breaks
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load the pre-built knowledge base
        self.index           = faiss.read_index(f"{KB_DIR}/incidents.index")
        self.incident_texts  = joblib.load(f"{KB_DIR}/incident_texts.pkl")
        self.metadata        = joblib.load(f"{KB_DIR}/incident_metadata.pkl")
        self.stats           = joblib.load(f"{KB_DIR}/dashboard_stats.pkl")

        # Groq client for LLM generation
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        print(f"Loaded {self.index.ntotal} incidents. Ready.")

    # ──────────────────────────────────────────────
    # RETRIEVAL — find the most relevant incidents
    # for a given question using vector similarity
    # ──────────────────────────────────────────────
    def retrieve(self, question, top_k=5):
        question_vec = self.embedding_model.encode([question])
        question_vec = np.array(question_vec).astype("float32")

        distances, positions = self.index.search(question_vec, top_k)

        results = []
        for i, pos in enumerate(positions[0]):
            if pos != -1:
                results.append({
                    "text": self.incident_texts[pos],
                    "metadata": self.metadata[pos],
                    "distance": float(distances[0][i]),
                    "index": int(pos)
                })
        return results

    # ──────────────────────────────────────────────
    # HALLUCINATION GUARD
    # After the LLM generates an answer, we check
    # whether the specific claims it made (names,
    # countries, dates) actually appear in the
    # retrieved source incidents.
    #
    # If the LLM mentions a threat actor or country
    # that does NOT appear anywhere in the sources,
    # we flag it as a potential hallucination and
    # warn the user instead of presenting it as fact.
    #
    # This is a simple but real implementation —
    # not a black box. You can explain every line
    # of this in an interview.
    # ──────────────────────────────────────────────
    def check_hallucination(self, answer, sources):
        # Build a single blob of all source text to check against
        source_blob = " ".join([s["text"].lower() for s in sources])

        # Extract proper nouns / capitalized words from the answer
        # (a simple heuristic: words starting with capital letters
        # that are likely names, countries, or org names)
        answer_words = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', answer)

        # Remove common words that are capitalized but not entities
        common_words = {
            "The", "This", "That", "These", "Those", "There",
            "Based", "According", "Source", "Sources", "Answer",
            "Question", "Incident", "Attack", "Attacks", "Threat",
            "MITRE", "Initial", "Access", "Impact", "Unknown"
        }
        candidate_entities = [w for w in answer_words if w not in common_words]

        # Check what fraction of named entities in the answer
        # actually appear somewhere in the source text
        if not candidate_entities:
            return {"verified": True, "confidence": 100, "unverified_terms": []}

        unverified = []
        for entity in set(candidate_entities):
            if entity.lower() not in source_blob:
                unverified.append(entity)

        verified_ratio = 1 - (len(unverified) / len(set(candidate_entities)))
        confidence = round(verified_ratio * 100, 1)

        return {
            "verified": len(unverified) == 0,
            "confidence": confidence,
            "unverified_terms": unverified
        }

    # ──────────────────────────────────────────────
    # MAIN QUERY FUNCTION
    # Retrieves sources → builds grounded prompt →
    # calls Groq → runs hallucination check →
    # returns answer + sources + confidence score
    # ──────────────────────────────────────────────
    def ask(self, question, top_k=5):
        # Step 1: Retrieve relevant incidents
        sources = self.retrieve(question, top_k=top_k)

        if not sources:
            return (
                "No relevant incidents found in the knowledge base for this query.",
                [],
                {"verified": True, "confidence": 100, "unverified_terms": []}
            )

        # Step 2: Build context from retrieved incidents
        context = "\n\n---\n\n".join([s["text"] for s in sources])

        # Step 3: Build the grounded prompt
        system_message = """You are Threat Copilot, a cybersecurity threat intelligence assistant.

You will be given real incident records from a curated database of 2,788 documented cyberattacks (2021-2025).

RULES — follow these strictly:
1. Answer ONLY using information present in the provided incident records below.
2. Never invent threat actor names, countries, dates, or technical details not present in the records.
3. If asked about something not covered in the records, say so clearly.
4. When citing an incident, refer to it by its name as given.
5. Be precise and analytical — write like a security analyst briefing, not a casual chatbot.
6. If multiple incidents are relevant, synthesize patterns across them."""

        user_message = f"""INCIDENT RECORDS:
{context}

ANALYST QUESTION:
{question}

ANALYSIS:"""

        # Step 4: Call Groq
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user",   "content": user_message}
            ],
            max_tokens=1024,
            temperature=0.1   # low temperature = factual, not creative
        )

        answer = response.choices[0].message.content

        # Step 5: Run the hallucination guard
        guard_result = self.check_hallucination(answer, sources)

        return answer, sources, guard_result


# ──────────────────────────────────────────────
# QUICK TEST — run this file directly
# python rag_engine.py
# ──────────────────────────────────────────────
if __name__ == "__main__":
    copilot = ThreatCopilot()

    print("\n=== Threat Copilot — Quick Test ===")
    print("Type 'quit' to exit\n")

    while True:
        question = input("Ask a question: ").strip()
        if question.lower() == "quit":
            break
        if not question:
            continue

        print("\nSearching knowledge base and analyzing...\n")
        answer, sources, guard = copilot.ask(question)

        print("ANALYSIS:")
        print(answer)

        print(f"\n[Hallucination Guard] Confidence: {guard['confidence']}%")
        if guard['unverified_terms']:
            print(f"[Warning] Unverified terms found: {guard['unverified_terms']}")
        else:
            print("[Verified] All named entities confirmed in source records")

        print(f"\nBased on {len(sources)} retrieved incidents:")
        for s in sources:
            print(f"  - {s['metadata'].get('name', 'Unknown')} (distance: {s['distance']:.2f})")

        print("\n" + "=" * 60 + "\n")
