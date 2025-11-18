#!/usr/bin/env python3
"""
graph_script.py
- Reads research_report.md and vocabulary.md (same folder)
- Computes explicit frequency counts (exact token matches) for each vocab term
- Computes implicit counts by measuring semantic similarity between term and sentences using sentence-transformers
- Outputs graph_data.json with nodes (vocab terms + report sections/sentences) and edges with weights
Usage:
  pip install -r requirements.txt
  python graph_script.py
Outputs:
  - graph_data.json
  - counts_summary.json
"""

import re
import json
from pathlib import Path
from collections import Counter, defaultdict

try:
    from sentence_transformers import SentenceTransformer, util
except Exception as e:
    print("Missing sentence-transformers. Install with: pip install sentence-transformers")
    raise

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast model for semantic similarity
SIM_THRESHOLD = 0.55  # tune for implicit detection

def read_file(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    return p.read_text(encoding="utf-8")

def split_sentences(text):
    # naive split on punctuation; for better results use nltk or spacy sentence tokenizer
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]  

def normalize_term(t):
    return t.lower()

def explicit_counts(sentences, vocab):
    counts = Counter()
    word_re_cache = {}
    for term in vocab:
        pat = re.compile(r'\b' + re.escape(term) + r'\b', flags=re.IGNORECASE)
        word_re_cache[term] = pat
    for s in sentences:
        for term, pat in word_re_cache.items():
            if pat.search(s):
                counts[term] += 1
    return counts

def semantic_match(sentences, vocab, model, threshold=SIM_THRESHOLD):
    sent_embeddings = model.encode(sentences, convert_to_tensor=True)
    term_embeddings = model.encode(vocab, convert_to_tensor=True)
    sim_scores = util.cos_sim(term_embeddings, sent_embeddings)  # shape (len(vocab), len(sentences))
    implicit_counts = Counter()
    edges = []
    for i, term in enumerate(vocab):
        for j, sent in enumerate(sentences):
            score = float(sim_scores[i][j])
            if score >= threshold:
                implicit_counts[term] += 1
                edges.append({
                    "term": term,
                    "sentence_index": j,
                    "sentence": sentences[j],
                    "score": score
                })
    return implicit_counts, edges

def main():
    repo_dir = Path(".")
    report = read_file(repo_dir / "research_report.md")
    vocab_text = read_file(repo_dir / "vocabulary.md")
    # Extract vocab terms (we expect numbered lines starting with "1." etc.)
    vocab_terms = []
    for line in vocab_text.splitlines():
        m = re.match(r'^\s*\d+\.\s*([A-Za-z0-9\- ]+)', line)
        if m:
            vocab_terms.append(m.group(1).strip().lower())

    sentences = split_sentences(report)
    model = SentenceTransformer(MODEL_NAME)

    exp_counts = explicit_counts(sentences, vocab_terms)
    imp_counts, edges = semantic_match(sentences, vocab_terms, model, SIM_THRESHOLD)

    # Build nodes and edges for visualization
    nodes = []
    for term in vocab_terms:
        nodes.append({"id": term, "type": "term", "explicit": int(exp_counts.get(term, 0)), "implicit": int(imp_counts.get(term, 0))})
    # add sentence nodes for sentences that had matches (limited)
    sentence_nodes = {}
    for e in edges:
        idx = e["sentence_index"]
        if idx not in sentence_nodes:
            sentence_nodes[idx] = {"id": f"sent_{idx}", "type": "sentence", "text": e["sentence"]}
    nodes.extend(list(sentence_nodes.values()))

    graph_edges = []
    for e in edges:
        graph_edges.append({
            "source": e["term"],
            "target": f"sent_{e['sentence_index']}",
            "weight": e["score"]
        })

    out = {
        "nodes": nodes,
        "edges": graph_edges,
        "meta": {
            "model": MODEL_NAME,
            "sim_threshold": SIM_THRESHOLD
        }
    }

    Path("graph_data.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    summary = {
        "explicit_counts": dict(exp_counts),
        "implicit_counts": dict(imp_counts),
        "total_sentences": len(sentences),
        "vocab_terms": vocab_terms
    }
    Path("counts_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote graph_data.json and counts_summary.json")

if __name__ == "__main__":
    main()
