"""
INGEST.PY — Threat Copilot Data Pipeline
──────────────────────────────────────────────
This script takes your raw 2,788-incident cybersecurity
Excel file and turns it into:
  1. A cleaned, readable text summary for each incident
  2. A FAISS vector index for semantic search (RAG)
  3. A saved chunks list so the app can retrieve full text

Run this ONCE to build the knowledge base.
The Streamlit app then just loads the saved files —
it never re-processes the Excel file at runtime.

Usage: python ingest.py
"""

import pandas as pd
import numpy as np
import faiss
import joblib
import os

from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────
# CONFIG — change this if your filename differs
# ──────────────────────────────────────────────
DATA_PATH  = "data/data_2021-2025.xlsx"
OUTPUT_DIR = "knowledge_base"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# STEP 1: Load the raw Excel file
# ──────────────────────────────────────────────
print("Loading incident dataset...")
df = pd.read_excel(DATA_PATH)
print(f"Loaded {len(df)} incidents with {len(df.columns)} columns")


# ──────────────────────────────────────────────
# STEP 2: Select only the columns that matter
# for a security analyst asking questions.
# The raw file has 84 columns — most are metadata
# noise (legal response codes, currency fields etc).
# We pick the ones that actually answer questions like
# "who attacked whom, how, and what happened".
# ──────────────────────────────────────────────
KEEP_COLUMNS = [
    "ID", "name", "description", "start_date", "end_date",
    "incident_type", "receiver_name", "receiver_country",
    "receiver_region", "receiver_category", "initiator_name",
    "initiator_country", "initiator_category",
    "attributed_initiator", "attributed_initiator_country",
    "attribution_basis", "attribution_it_company",
    "MITRE_initial_access", "MITRE_impact",
    "data_theft", "disruption", "hijacking", "zero_days",
    "weighted_cyber_intensity", "economic_impact_exact_value",
    "sources_url"
]

# Only keep columns that actually exist (in case dataset varies)
available_cols = [c for c in KEEP_COLUMNS if c in df.columns]
df = df[available_cols].copy()

print(f"Kept {len(available_cols)} relevant columns")


# ──────────────────────────────────────────────
# STEP 3: Clean the data
# Fill missing values with readable placeholders
# instead of NaN, which breaks text generation
# ──────────────────────────────────────────────
print("Cleaning data...")

def clean_value(val):
    """Replace NaN / 'Not available' / empty with consistent label"""
    if pd.isna(val):
        return "Unknown"
    val_str = str(val).strip()
    if val_str.lower() in ["not available", "nan", ""]:
        return "Unknown"
    return val_str

for col in df.columns:
    if col != "ID":
        df[col] = df[col].apply(clean_value)

# Drop rows with no description — nothing to search on
df = df[df["description"] != "Unknown"]
df = df.reset_index(drop=True)

print(f"After cleaning: {len(df)} usable incidents")


# ──────────────────────────────────────────────
# STEP 4: Build a readable "incident card" for
# each row. This is what gets embedded and what
# the LLM reads later. We write it like a security
# analyst's case note — not a CSV dump.
# ──────────────────────────────────────────────
print("Building incident summaries...")

def build_incident_text(row):
    text = f"""INCIDENT: {row.get('name', 'Unknown')}
DATE: {row.get('start_date', 'Unknown')} to {row.get('end_date', 'Unknown')}
TYPE: {row.get('incident_type', 'Unknown')}

ATTACKER: {row.get('initiator_name', 'Unknown')} ({row.get('initiator_country', 'Unknown')})
ATTACKER CATEGORY: {row.get('initiator_category', 'Unknown')}
ATTRIBUTED TO: {row.get('attributed_initiator', 'Unknown')} ({row.get('attributed_initiator_country', 'Unknown')})
ATTRIBUTION CONFIDENCE: {row.get('attribution_basis', 'Unknown')}
ATTRIBUTING ORGANIZATION: {row.get('attribution_it_company', 'Unknown')}

TARGET: {row.get('receiver_name', 'Unknown')}
TARGET COUNTRY: {row.get('receiver_country', 'Unknown')}
TARGET REGION: {row.get('receiver_region', 'Unknown')}
TARGET SECTOR: {row.get('receiver_category', 'Unknown')}

MITRE INITIAL ACCESS: {row.get('MITRE_initial_access', 'Unknown')}
MITRE IMPACT: {row.get('MITRE_impact', 'Unknown')}
DATA THEFT INVOLVED: {row.get('data_theft', 'Unknown')}
DISRUPTION CAUSED: {row.get('disruption', 'Unknown')}
ZERO-DAY USED: {row.get('zero_days', 'Unknown')}
SEVERITY SCORE: {row.get('weighted_cyber_intensity', 'Unknown')}

DESCRIPTION:
{row.get('description', 'No description available')}

SOURCE: {row.get('sources_url', 'Unknown')}"""
    return text

df["incident_text"] = df.apply(build_incident_text, axis=1)

print("\nSample incident card:")
print("=" * 60)
print(df["incident_text"].iloc[0][:600] + "...")
print("=" * 60)


# ──────────────────────────────────────────────
# STEP 5: Load embedding model
# Same free local model used in your other projects
# ──────────────────────────────────────────────
print("\nLoading embedding model... (cached after first run)")
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


# ──────────────────────────────────────────────
# STEP 6: Build FAISS index
# Each incident card becomes one searchable vector.
# Unlike PDF chunking, here each incident is already
# a natural unit — no need to split further.
# ──────────────────────────────────────────────
print("Embedding all incidents... (this takes a few minutes)")

incident_texts = df["incident_text"].tolist()
embeddings = EMBEDDING_MODEL.encode(
    incident_texts,
    show_progress_bar=True,
    batch_size=32
)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"FAISS index built with {index.ntotal} incidents")


# ──────────────────────────────────────────────
# STEP 7: Save everything the app will need
# ──────────────────────────────────────────────
print("\nSaving knowledge base files...")

faiss.write_index(index, f"{OUTPUT_DIR}/incidents.index")
joblib.dump(incident_texts, f"{OUTPUT_DIR}/incident_texts.pkl")
joblib.dump(df.to_dict('records'), f"{OUTPUT_DIR}/incident_metadata.pkl")

# Save quick stats for the dashboard
stats = {
    "total_incidents": len(df),
    "date_range": f"{df['start_date'].min()} to {df['start_date'].max()}",
    "top_attacker_countries": df['initiator_country'].value_counts().head(10).to_dict(),
    "top_target_sectors": df['receiver_category'].value_counts().head(10).to_dict(),
    "top_target_countries": df['receiver_country'].value_counts().head(10).to_dict(),
    "data_theft_count": int((df['data_theft'] == 'True').sum()) if 'data_theft' in df.columns else 0,
    "zero_day_count": int((df['zero_days'] == 'True').sum()) if 'zero_days' in df.columns else 0,
}
joblib.dump(stats, f"{OUTPUT_DIR}/dashboard_stats.pkl")

print(f"""
Saved to {OUTPUT_DIR}/:
  incidents.index          ← FAISS vector index
  incident_texts.pkl       ← readable incident cards
  incident_metadata.pkl    ← structured data for filtering
  dashboard_stats.pkl      ← pre-computed stats for dashboard

Knowledge base ready! {len(df)} real-world cyber incidents indexed.
Next step: build the RAG query engine.
""")
