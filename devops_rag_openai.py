from __future__ import annotations
import os
import re
import json
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Iterable, Set

import numpy as np
import pandas as pd

# Optional libs
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

# OpenAI (>= 1.0 SDK)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Lexical retriever
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
except Exception:
    TfidfVectorizer = None
    def cos_sim(a, b): raise RuntimeError("scikit-learn not available for TF-IDF fallback")


# --------------------------
# Utilities
# --------------------------

def _norm(s: str) -> str:
    if not isinstance(s, str): s = str(s)
    return re.sub(r"\s+", " ", s.strip().lower())

_SYNONYMS = {
    "ci/cd": "continuous integration and delivery",
    "ci cd": "continuous integration and delivery",
    "ci": "continuous integration",
    "cd": "continuous delivery",
    "k8s": "kubernetes",
    "docker hub": "docker",
    "gh actions": "github actions",
}

def _norm_term(s: str) -> str:
    s = _norm(s)
    return _SYNONYMS.get(s, s)

def _unique(seq: Iterable[Any]) -> List[Any]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

@dataclass(frozen=True)
class NodeKey:
    t: str
    k: str
    def id(self) -> str:
        return f"{self.t}::{self.k}"


# --------------------------
# Index builder
# --------------------------

class DevOpsRAGIndex:
    def __init__(self, df_edges: pd.DataFrame):
        needed = {"file", "source_type", "source_keyword", "target_type", "target_keyword"}
        if not needed.issubset(df_edges.columns):
            raise ValueError(f"Missing columns: {needed - set(df_edges.columns)}")

        # Normalize/ensure str
        self.edges = df_edges.copy()
        for c in ["file", "source_type", "source_keyword", "target_type", "target_keyword"]:
            self.edges[c] = self.edges[c].astype(str)

        self.edges["source_type_n"] = self.edges["source_type"].apply(_norm)
        self.edges["source_keyword_n"] = self.edges["source_keyword"].apply(_norm_term)
        self.edges["target_type_n"] = self.edges["target_type"].apply(_norm)
        self.edges["target_keyword_n"] = self.edges["target_keyword"].apply(_norm_term)

        # Build adjacency and nodes
        self.nodes: Dict[str, Dict[str, Any]] = {}            # node_id -> payload
        self.neighbors: Dict[str, Dict[str, Set[str]]] = {}    # node_id -> type -> set(keywords)
        self.citations: Dict[str, Set[str]] = {}               # node_id -> set(file ids)

        for _, r in self.edges.iterrows():
            s_key = NodeKey(r["source_type_n"], r["source_keyword_n"]).id()
            t_key = NodeKey(r["target_type_n"], r["target_keyword_n"]).id()

            # register nodes
            for nid, (t, k) in [(s_key, (r["source_type_n"], r["source_keyword_n"])),
                                (t_key, (r["target_type_n"], r["target_keyword_n"]))]:
                if nid not in self.nodes:
                    self.nodes[nid] = {"type": t, "keyword": k}
                    self.neighbors[nid] = {}
                    self.citations[nid] = set()

            # undirected neighborhood with type separation
            self.neighbors[s_key].setdefault(self.nodes[t_key]["type"], set()).add(self.nodes[t_key]["keyword"])
            self.neighbors[t_key].setdefault(self.nodes[s_key]["type"], set()).add(self.nodes[s_key]["keyword"])

            # citations
            f = r["file"]
            self.citations[s_key].add(f)
            self.citations[t_key].add(f)

        # Build documents per node
        self.docs: Dict[str, str] = {}
        for nid, meta in self.nodes.items():
            t = meta["type"]; k = meta["keyword"]
            parts = [f"type: {t}", f"keyword: {k}"]

            # include neighbors by type for context
            neigh = self.neighbors.get(nid, {})
            for nt, kws in sorted(neigh.items()):
                if len(kws) == 0: continue
                sample = ", ".join(sorted(list(kws))[:50])
                parts.append(f"connected_{nt}s: {sample}")

            # attach a compact citation list
            cites = sorted(list(self.citations.get(nid, set())))[:50]
            if cites:
                parts.append("papers: " + ", ".join(cites))

            self.docs[nid] = "\n".join(parts)

        # Vector / lexical placeholders
        self.embeddings: Optional[np.ndarray] = None  # shape [N, D]
        self.ids: List[str] = list(self.docs.keys())
        self.id_to_ix = {nid: i for i, nid in enumerate(self.ids)}

        # Lexical TF-IDF (helps before vectors are built)
        self._tfidf = None
        self._tfidf_matrix = None
        if TfidfVectorizer is not None:
            self._tfidf = TfidfVectorizer(max_features=50000)
            self._tfidf_matrix = self._tfidf.fit_transform([self.docs[i] for i in self.ids])

    # ------------- Persistence -------------

    def save(self, path: str):
        path = os.path.abspath(path)
        os.makedirs(path, exist_ok=True)
        # dump nodes/docs/metadata
        meta = {
            "ids": self.ids,
            "nodes": self.nodes,
            "neighbors": {k:{t:list(v) for t,v in d.items()} for k,d in self.neighbors.items()},
            "citations": {k:list(v) for k,v in self.citations.items()},
            "docs": self.docs,
        }
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # embeddings
        if self.embeddings is not None:
            np.save(os.path.join(path, "embeddings.npy"), self.embeddings)

        # TF-IDF cache (optional)
        if self._tfidf is not None and self._tfidf_matrix is not None:
            try:
                from scipy import sparse
                sparse.save_npz(os.path.join(path, "tfidf.npz"), self._tfidf_matrix)
                with open(os.path.join(path, "tfidf_vocab.json"), "w", encoding="utf-8") as f:
                    json.dump(self._tfidf.vocabulary_, f)
            except Exception:
                # SciPy not available or filesystem issue; skip persisting TF-IDF
                pass

    @classmethod
    def load(cls, path: str) -> "DevOpsRAGIndex":
        path = os.path.abspath(path)
        with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)

        inst = object.__new__(cls)
        inst.nodes = meta["nodes"]
        inst.neighbors = {k:{t:set(v) for t,v in d.items()} for k,d in meta["neighbors"].items()}
        inst.citations = {k:set(v) for k,v in meta["citations"].items()}
        inst.docs = meta["docs"]
        inst.ids = meta["ids"]
        inst.id_to_ix = {nid: i for i, nid in enumerate(inst.ids)}
        inst.edges = pd.DataFrame([])  # not restored
        inst.embeddings = None
        inst._tfidf = None
        inst._tfidf_matrix = None

        # load embeddings if present
        emb_p = os.path.join(path, "embeddings.npy")
        if os.path.exists(emb_p):
            inst.embeddings = np.load(emb_p)

        # load tfidf if present (skip on any error)
        tfidf_p = os.path.join(path, "tfidf.npz")
        try:
            if os.path.exists(tfidf_p):
                from scipy import sparse
                inst._tfidf_matrix = sparse.load_npz(tfidf_p)
                with open(os.path.join(path, "tfidf_vocab.json"), "r", encoding="utf-8") as f:
                    vocab = json.load(f)
                if TfidfVectorizer is not None:
                    inst._tfidf = TfidfVectorizer(vocabulary=vocab)
        except Exception:
            # corrupted or partial TF-IDF cache — disable lexical index, vectors still work
            inst._tfidf = None
            inst._tfidf_matrix = None

        return inst

    # ------------- Embedding -------------

    def build_openai_embeddings(self, model: str = "text-embedding-3-large", batch_size: int = 64):
        if OpenAI is None:
            raise RuntimeError("openai package not available. Install `openai` >= 1.0.0")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        docs = [self.docs[i] for i in self.ids]
        embs = []
        for i in tqdm(range(0, len(docs), batch_size), desc="Embedding docs with OpenAI"):
            batch = docs[i:i+batch_size]
            resp = client.embeddings.create(model=model, input=batch)
            embs.extend([d.embedding for d in resp.data])
        self.embeddings = np.array(embs, dtype=np.float32)
        return self.embeddings

    # ------------- Retrieval -------------

    def _search_tfidf(self, query: str, k: int = 50) -> List[Tuple[str, float]]:
        if self._tfidf is None or self._tfidf_matrix is None:
            return []
        qv = self._tfidf.transform([query])
        sims = (qv @ self._tfidf_matrix.T).toarray()[0]
        best = np.argsort(-sims)[:k].tolist()
        return [(self.ids[i], float(sims[i])) for i in best]

    def _search_vectors(self, query: str, client: Optional[OpenAI], emb_model: str, k: int = 50) -> List[Tuple[str, float]]:
        if self.embeddings is None:
            return []
        if client is None:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.embeddings.create(model=emb_model, input=[query])
        q = np.array(resp.data[0].embedding, dtype=np.float32)
        # cosine similarity
        A = self.embeddings
        qn = q / (np.linalg.norm(q) + 1e-9)
        An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        sims = (An @ qn).astype(float)
        best = np.argsort(-sims)[:k].tolist()
        return [(self.ids[i], float(sims[i])) for i in best]

    def search(self, query: str, k: int = 50, use_openai: bool = True,
               emb_model: str = "text-embedding-3-large", alpha: float = 0.5) -> List[Tuple[str, float]]:
        """
        Hybrid search: mix TF-IDF and vector scores with weight alpha.
        """
        query_n = _norm(query)
        tfidf_hits = self._search_tfidf(query_n, k=k)
        tfidf_scores = {nid: s for nid, s in tfidf_hits}

        vec_hits = []
        if use_openai:
            vec_hits = self._search_vectors(query_n, client=None, emb_model=emb_model, k=k)
        vec_scores = {nid: s for nid, s in vec_hits}

        # combine
        ids = _unique([nid for nid,_ in tfidf_hits] + [nid for nid,_ in vec_hits])
        out = []
        for nid in ids:
            s = alpha * vec_scores.get(nid, 0.0) + (1 - alpha) * tfidf_scores.get(nid, 0.0)
            out.append((nid, s))
        out.sort(key=lambda x: -x[1])
        return out[:k]

    # ------------- Chain enforcement -------------

    def _neighbors_of_type(self, nid: str, t: str) -> List[str]:
        t = _norm(t)
        return sorted(list(self.neighbors.get(nid, {}).get(t, set())))

    def chain_recommend(self, query: str, top_k: int = 10, use_openai: bool = True,
                        emb_model: str = "text-embedding-3-large") -> Dict[str, List[Dict[str, Any]]]:
        """
        Enforce dual/tri-chains:
          (1) Outcome -> Capability -> Practice -> Tool
          (2) Outcome -> Capability -> Maturity Model
          (3) Outcome -> Capability -> Practice -> Maturity Model

        Returns dict with keys: outcomes, capabilities, practices, tools, maturity_models
        Each list item: {"name": str, "score": float, "papers": [...], "why": str}
        """
        hits = self.search(query, k=80, use_openai=use_openai, emb_model=emb_model, alpha=0.6)

        # Step 1: pick outcome-like nodes from hits
        outcome_nodes = [nid for nid,_ in hits if self.nodes[nid]["type"] in ("outcome",)]
        # if none found, expand by exploring neighbors of top hits for outcome type
        if not outcome_nodes and hits:
            for nid,_ in hits[:10]:
                for o in self.neighbors.get(nid, {}).get("outcome", []):
                    outcome_nodes.append(NodeKey("outcome", o).id())
            outcome_nodes = _unique(outcome_nodes)

        # Rank outcomes by combined score + degree centrality (proxy for support)
        out_scored = []
        vec = {nid: s for nid, s in hits}
        for nid in outcome_nodes:
            deg = sum(len(v) for v in self.neighbors.get(nid, {}).values())
            s = 0.7 * vec.get(nid, 0) + 0.3 * math.log1p(deg)
            out_scored.append((nid, s))
        out_scored.sort(key=lambda x: -x[1])
        outcomes = out_scored[:max(top_k, 5)]

        # Step 2: expand to capabilities
        cap_counts = {}
        for nid,_ in outcomes[:5]:
            for c in self._neighbors_of_type(nid, "capability"):
                cap_counts[c] = cap_counts.get(c, 0) + 1
        cap_scored = [(NodeKey("capability", c).id(), cnt) for c, cnt in cap_counts.items()]
        cap_scored.sort(key=lambda x: -x[1])
        capabilities = cap_scored[:max(top_k, 8)]

        # Step 3: expand to practices (from best capabilities)
        prac_counts = {}
        for nid,_ in capabilities[:8]:
            for p in self._neighbors_of_type(nid, "practice"):
                prac_counts[p] = prac_counts.get(p, 0) + 1
        prac_scored = [(NodeKey("practice", p).id(), cnt) for p, cnt in prac_counts.items()]
        prac_scored.sort(key=lambda x: -x[1])
        practices = prac_scored[:max(top_k, 10)]

        # Step 4: expand to maturity models (from capabilities and practices)
        mm_counts = {}
        # from capabilities
        for nid,_ in capabilities[:8]:
            for m in self._neighbors_of_type(nid, "maturity model"):
                mm_counts[m] = mm_counts.get(m, 0) + 1
        # from practices
        for nid,_ in practices[:10]:
            for m in self._neighbors_of_type(nid, "maturity model"):
                mm_counts[m] = mm_counts.get(m, 0) + 1
        maturity_scored = [(NodeKey("maturity model", m).id(), cnt) for m, cnt in mm_counts.items()]
        maturity_scored.sort(key=lambda x: -x[1])
        maturity_models = maturity_scored[:top_k]

        # Step 5: expand to tools (from best practices)
        tool_counts = {}
        for nid,_ in practices[:10]:
            for t in self._neighbors_of_type(nid, "tool"):
                tool_counts[t] = tool_counts.get(t, 0) + 1
        tools_scored = [(NodeKey("tool", t).id(), cnt) for t, cnt in tool_counts.items()]
        tools_scored.sort(key=lambda x: -x[1])
        tools = tools_scored[:top_k]

        def _pack(items: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
            out = []
            for nid, s in items:
                meta = self.nodes.get(nid, {})
                names = meta.get("keyword", "")
                papers = sorted(list(self.citations.get(nid, set())))[:20]
                why = self.docs.get(nid, "")[:800]
                out.append({"name": names, "score": float(s), "papers": papers, "why": why})
            return out

        return {
            "outcomes": _pack(outcomes),
            "capabilities": _pack(capabilities),
            "practices": _pack(practices),
            "tools": _pack(tools),
            "maturity_models": _pack(maturity_models),
        }


# --------------------------
# Generator
# --------------------------

SYSTEM_PROMPT = """You are a DevOps recommendation assistant.
Use the provided retrieved snippets ONLY. Do not invent facts.
For each bullet, include bracketed paper IDs like [references: 329, 540].
Enforce the chains:
  (1) Outcome -> Capability -> Practice -> Tool
  (2) Outcome -> Capability -> Maturity Model
  (3) Outcome -> Capability -> Practice -> Maturity Model
Deduplicate overlapping tools; prefer tools that satisfy multiple practices.
"""

def generate_answer_openai(query: str, retrieved: Dict[str, List[Dict[str, Any]]],
                           model: str = "gpt-4o-mini") -> str:
    """
    Compose a grounded answer using OpenAI responses.
    """
    if OpenAI is None:
        raise RuntimeError("openai package not available. Install `openai` >= 1.0.0")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Build context
    def _mk_section(title, items):
        lines = [f"{title}:"]
        for it in items:
            cites = ", ".join(it["papers"][:5])
            lines.append(f"- {it['name']} (score={it['score']:.3f}) [references: {cites}]")
            lines.append(f"  Why: {it['why']}")
        return "\n".join(lines)

    ctx = "\n\n".join([
        _mk_section("Outcomes", retrieved.get("outcomes", [])),
        _mk_section("Capabilities", retrieved.get("capabilities", [])),
        _mk_section("Practices", retrieved.get("practices", [])),
        _mk_section("Tools", retrieved.get("tools", [])),
        _mk_section("Maturity Models", retrieved.get("maturity_models", [])),
    ])

    user_prompt = (
        f"Query: {query}\n\nUse the retrieved context below to produce a concise, "
        f"structured recommendation with citations.\n\n{ctx}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# --------------------------
# Convenience API
# --------------------------

class DevOpsRAG:
    def __init__(self, df_edges: pd.DataFrame, index_dir: Optional[str] = None, use_openai: bool = True):
        """
        If index_dir/meta.json exists, try to load; otherwise build a fresh in-memory index and save (if index_dir set).
        Robust to corrupted TF-IDF cache.
        """
        self.use_openai = use_openai
        self.index_dir = index_dir
        try:
            if index_dir and os.path.exists(os.path.join(index_dir, "meta.json")):
                self.index = DevOpsRAGIndex.load(index_dir)
            else:
                self.index = DevOpsRAGIndex(df_edges)
                if index_dir:
                    os.makedirs(index_dir, exist_ok=True)
                    self.index.save(index_dir)
        except Exception:
            # fallback: rebuild fresh index in memory
            self.index = DevOpsRAGIndex(df_edges)
            if index_dir:
                try:
                    os.makedirs(index_dir, exist_ok=True)
                    self.index.save(index_dir)
                except Exception:
                    pass

    def build(self, index_dir: Optional[str] = None, embed_model: str = "text-embedding-3-large"):
        if self.use_openai:
            self.index.build_openai_embeddings(model=embed_model)
        if index_dir or self.index_dir:
            self.index.save(index_dir or self.index_dir)

    def recommend(self, query: str, top_k: int = 10, embed_model: str = "text-embedding-3-large") -> Dict[str, List[Dict[str, Any]]]:
        return self.index.chain_recommend(query, top_k=top_k, use_openai=self.use_openai, emb_model=embed_model)

    def answer(self, query: str, top_k: int = 10, gen_model: str = "gpt-4o-mini",
               embed_model: str = "text-embedding-3-large") -> str:
        recs = self.recommend(query, top_k=top_k, embed_model=embed_model)
        if not self.use_openai:
            # fallback: return a plain-text summary without generation
            lines = [f"# Recommendation for: {query}"]
            for sec in ["outcomes","capabilities","practices","tools","maturity_models"]:
                lines.append(f"\n## {sec.capitalize().replace('_',' ')}")
                for it in recs.get(sec, []):
                    lines.append(f"- {it['name']} (score={it['score']:.3f}) [references: {', '.join(it['papers'][:5])}]")
            return "\n".join(lines)
        return generate_answer_openai(query, recs, model=gen_model)


# --------------------------
# CLI usage example
# --------------------------

def _example_cli():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to CooccurrenceMapping.csv")
    p.add_argument("--index_dir", default="rag_index", help="Where to persist the index")
    p.add_argument("--build", action="store_true", help="Build embeddings with OpenAI")
    p.add_argument("--query", default="predictable faster releases with secure kubernetes deployments")
    p.add_argument("--top_k", type=int, default=12)
    p.add_argument("--use_openai", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    rag = DevOpsRAG(df, index_dir=args.index_dir, use_openai=args.use_openai)
    if args.build:
        rag.build(index_dir=args.index_dir)

    print(">>> Query:", args.query)
    txt = rag.answer(args.query, top_k=args.top_k)
    print(txt)


if __name__ == "__main__":
    # For quick testing without CLI
    import os
    csv_path = os.environ.get("COOCCURRENCE_CSV", "CooccurrenceMapping.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, low_memory=False)
        rag = DevOpsRAG(df, index_dir="rag_index", use_openai=bool(os.getenv("OPENAI_API_KEY")))
        print("Initialized RAG index. Set OPENAI_API_KEY and run rag.build() to compute embeddings.")
    else:
        print("Set COOCCURRENCE_CSV env var or pass --csv via CLI")