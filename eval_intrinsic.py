#!/usr/bin/env python3
"""
word_embedding_eval_harness.py

This module runs an evaluation harness on a given set of GloVe model
vector embeddings. The datasets to be evaluated are given via the
command line args.

Usage:

python3 eval_intrinsic.py --wordsim353-path eval/wordsim353/combined.csv \
                          --MEN-path eval/MEN/MEN_dataset_natural_form_full \
                          --SCSW-path eval/SCWS/ratings.txt \
                          --SIMLEX-path eval/SimLex-999/SimLex-999.txt \
                          --MC-folder  eval/Miller_and_Charles/ \
                          --BATS-folder eval/BATS_3_0 \
                          --vectors-path vectors.txt \
                          --skip-hypernym

"""
from __future__ import annotations
import csv
import argparse
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def read_SIMLEX(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=0)
    return df


def evaluate_SIMLEX(df: pd.DataFrame, embeddings: Dict[str, List[float]]) -> dict[str, Any]:
    err_cnt = 0
    actuals, preds = [], []
    for row in df.itertuples(name=None):
        w1, w2, metric = row[1], row[2], row[4]
        u = embeddings.get(w1.lower())
        v = embeddings.get(w2.lower())
        if metric is None or u is None or v is None:
            err_cnt += 1
            continue
        else:
            denom = np.linalg.norm(u) * np.linalg.norm(v)
            sim = float(np.dot(u, v) / denom)
        actuals.append(float(metric))
        preds.append(float(sim))
    rho, p = spearmanr(actuals, preds)
    return {"rho": rho.item(0), "pvalue": p.item(0), "n": len(actuals), "err": err_cnt}


def read_with_pandas(path: str) -> pd.DataFrame:
    """Read file into a pandas DataFrame and normalize column names."""
    if pd is None:
        raise RuntimeError("pandas is not installed. Install with: pip install pandas")

    df = pd.read_csv(path)
    # canonicalize column names: trim whitespace
    df.columns = [c.strip() for c in df.columns]

    # find columns (be flexible if header varies slightly)
    human_col = next((c for c in df.columns if "human" in c.lower()), df.columns[-1])
    w1_col = next((c for c in df.columns if "word" in c.lower() and "1" in c), df.columns[0])
    w2_col = next((c for c in df.columns if "word" in c.lower() and "2" in c), df.columns[1] if len(df.columns) > 1 else df.columns[0])

    df = df.rename(columns={w1_col: "word1", w2_col: "word2", human_col: "Human_mean"})
    df["word1"] = df["word1"].astype(str).str.strip()
    df["word2"] = df["word2"].astype(str).str.strip()
    df["Human_mean"] = pd.to_numeric(df["Human_mean"], errors="coerce")
    return df[["word1", "word2", "Human_mean"]]


def read_with_csv(path: str) -> List[Dict[str, Optional[float]]]:
    """Read file using the standard csv module and return list of dicts."""
    out: List[Dict[str, Optional[float]]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out
        keys = [k.strip() for k in reader.fieldnames]

        human_key = next((k for k in keys if "human" in k.lower()), keys[-1])
        w1_key = next((k for k in keys if "word" in k.lower() and "1" in k), keys[0])
        w2_key = next((k for k in keys if "word" in k.lower() and "2" in k), keys[1] if len(keys) > 1 else keys[0])

        for row in reader:
            score_raw = (row.get(human_key) or "").strip()
            try:
                score = float(score_raw) if score_raw != "" else None
            except ValueError:
                score = None
            out.append({
                "word1": (row.get(w1_key) or "").strip(),
                "word2": (row.get(w2_key) or "").strip(),
                "Human_mean": score
            })
    return out


def read_pair_scores(path: str, use_pandas: bool = True, colnames=('word1', 'word2', 'score')):
    """
    Read whitespace-separated pair+score files like MEN_dataset_natural_form_full:
      sun sunlight 50.000000
      automobile car 50.000000
    Returns a pandas.DataFrame if use_pandas=True (requires pandas), otherwise a list of dicts.
    """
    if use_pandas:
        out = pd.read_csv(path, sep=r'\s+', header=None, names=colnames, engine='python').assign(
            **{colnames[2]: lambda d: pd.to_numeric(d[colnames[2]], errors='coerce')}
            )
    else:
        out = []
        with open(path, encoding='utf-8') as f:
            for ln in f:
                s = ln.strip()
                if not s:
                    continue
                parts = s.split()
                if len(parts) < 3:
                    continue
                w1, w2, score_raw = parts[0], parts[1], parts[-1]
                try:
                    score = float(score_raw)
                except ValueError:
                    score = None
                out.append({colnames[0]: w1, colnames[1]: w2, colnames[2]: score})
    return out


def concat_csvs(folder: str | Path,
                pattern: str = "*.csv",
                sep: str = ";",
                encoding: str = "utf-8",
                add_source: bool = True,
                dedupe: bool = False,
                recursive: bool = False) -> pd.DataFrame:
    """
    Read Miller & Charles-style files (semicolon-separated).
    Concatenate all CSVs in `folder` matching `pattern` into one DataFrame.

    - folder: directory containing the CSVs
    - pattern: glob pattern (e.g. '*.csv' or '**/*.csv' if recursive=True)
    - sep: field separator used in files (Miller & Charles use ';' by default)
    - add_source: attach a `_source` column with the filename
    - dedupe: drop duplicate rows after concat
    - recursive: if True, will search subdirectories (uses rglob)

    Returns an empty DataFrame if no files found.
    """
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(p)

    files = sorted(p.rglob(pattern) if recursive else p.glob(pattern))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, sep=sep, header=0)
        except Exception as exc:
            # skip files that fail to parse but print a brief warning
            print(f"warning: failed to read {f}: {exc}")
            continue
        if add_source:
            df["_source"] = f.name
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True, sort=False)
    if dedupe:
        out = out.drop_duplicates(ignore_index=True)
    out.columns = [c.strip() for c in df.columns]
    return out


def spearman_vs_hj(df: pd.DataFrame, model_scores: Iterable[float] | str = "model",
                  human_col: str = "HJ") -> Tuple[float, float]:
    """
    Compute Spearman correlation between human ratings (HJ) and model scores.
    - df: DataFrame returned by concat_csvs
    - model_scores: either an iterable of scores aligned with df rows, or the name
      of a column in df containing model scores (e.g. "cosine")
    Returns (rho, pvalue).
    """
    if isinstance(model_scores, str):
        model = df[model_scores].astype(float)
    else:
        model = pd.Series(list(model_scores), index=df.index, dtype=float)
    human = df[human_col].astype(float)
    rho, p = spearmanr(human, model, nan_policy="omit")
    return float(rho), float(p)


_token_re = re.compile(r"[A-Za-z0-9']+")
_b_re = re.compile(r"</?b>", re.IGNORECASE)

def parse_scsw_line(line: str) -> Optional[Dict]:
    """
    Parse one line of SCSW ratings.txt (tab-delimited).
    Returns dict with keys: id, w1, pos1, w2, pos2, ctx1, ctx2, avg, ratings(list).
    """
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 8:
        return None
    try:
        _id = int(parts[0])
    except Exception:
        _id = parts[0]
    w1, pos1, w2, pos2 = parts[1], parts[2], parts[3], parts[4]
    ctx1, ctx2 = parts[5], parts[6]
    try:
        avg = float(parts[7])
    except Exception:
        avg = None
    ratings = []
    for x in parts[8:]:
        try:
            ratings.append(float(x))
        except Exception:
            pass
    return {"id": _id, "w1": w1, "pos1": pos1, "w2": w2, "pos2": pos2,
            "ctx1": ctx1, "ctx2": ctx2, "avg": avg, "ratings": ratings}


def read_scsw(path: str) -> List[Dict]:
    """Read whole SCSW file and return list of parsed rows (skips unparsable lines)."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(p)
    out = []
    with p.open(encoding="utf-8", errors="ignore") as fh:
        for ln in fh:
            row = parse_scsw_line(ln)
            if row:
                out.append(row)
    return out


def _tokens_from_context(ctx: str) -> List[str]:
    """Remove <b> tags and tokenize lowercased words/numbers/apostrophes."""
    s = _b_re.sub("", ctx)
    return _token_re.findall(s.lower())


def _avg_vector(tokens: List[str], embeddings: Dict[str, List[float]]):
    """Return average vector (list or numpy array) for tokens, or None if no tokens found in embeddings."""
    vecs = []
    for t in tokens:
        v = embeddings.get(t) or embeddings.get(t.lower()) or embeddings.get(t.capitalize())
        if v is not None:
            vecs.append(v)
    if not vecs:
        return None
    return np.mean([np.asarray(v, dtype=float) for v in vecs], axis=0)




def _rankdata(a: List[float]) -> List[float]:
    """Simple rankdata that assigns average ranks for ties."""
    sorted_idx = sorted(range(len(a)), key=lambda i: (a[i], i))
    ranks = [0.0] * len(a)
    i = 0
    while i < len(sorted_idx):
        j = i
        while j + 1 < len(sorted_idx) and a[sorted_idx[j + 1]] == a[sorted_idx[i]]:
            j += 1
        # average rank for positions i..j (1-based)
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    return ranks


def _spearmanr(a: List[float], b: List[float]):
    """Compute Spearman rho (pure-python). Returns (rho, pvalue=None)."""
    if len(a) != len(b) or not a:
        return None, None
    r, p = spearmanr(a, b, nan_policy="omit")
    return float(r), float(p)


def pair_cosine(embeddings: dict, w1: str, w2: str):
    """Return cosine similarity for w1,w2 using embeddings dict or None if missing."""

    u = embeddings.get(w1) or embeddings.get(w1.lower()) or embeddings.get(w1.capitalize())
    v = embeddings.get(w2) or embeddings.get(w2.lower()) or embeddings.get(w2.capitalize())
    if u is None or v is None:
        return None

    denom = np.linalg.norm(u) * np.linalg.norm(v)
    return float(np.dot(u, v) / denom) if denom else None


def evaluate_scws(embeddings: Dict[str, List[float]],
                  rows: List[Dict]) -> Dict:
    """
    Evaluate SCSW rows.
    method: "word"   -> cosine between target word embeddings (w1 vs w2)
    Returns dict: {'rho':..., 'pvalue':..., 'n':..., 'pairs': list of (human_avg, pred) used}
    """
    humans = []
    preds = []
    err_cnt = 0
    for r in rows:
        human = r.get("avg")
        u = embeddings.get(r["w1"].lower())
        v = embeddings.get(r["w2"].lower())

        if human is None or u is None or v is None:
            err_cnt += 1
            continue
        else:
            denom = np.linalg.norm(u) * np.linalg.norm(v)
            sim = float(np.dot(u, v) / denom)
        humans.append(float(human))
        preds.append(float(sim))
    rho, p = _spearmanr(humans, preds)
    return {"rho": rho, "pvalue": p, "n": len(humans), "err": err_cnt}



def read_embeddings(path: str, use_numpy: bool = True, expected_dim: int | None = None):
    """
    Read a word-embedding file where each row is: <token> <v1> <v2> ... <vN>
    Returns a dict: token -> vector (numpy array if use_numpy and numpy is available,
    otherwise a Python list of floats).

    This reader is forgiving about:
    - trailing commas after the token (e.g. "word, 0.1 0.2 ...")
    - occasional stray commas between numbers
    - small formatting corruption where parts of a float may be split (it will try
      to join adjacent tokens if a token doesn't parse as a float)

    Parameters:
    - path: path to embedding file
    - use_numpy: convert vectors to numpy arrays if numpy is available
    - expected_dim: if provided, used to validate (and warn) when parsed vector length
      doesn't match expected_dim

    Example:
      embs = read_embeddings("embeds.txt", use_numpy=True, expected_dim=300)
      vec = embs.get("computer")
    """
    import re
    try:
        import numpy as _np  # optional
    except Exception:
        _np = None

    def _is_float_token(s: str) -> bool:
        # allow optional sign, decimal, optional exponent
        return re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", s) is not None

    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            # split on whitespace first
            parts = line.split()
            if len(parts) < 2:
                # nothing to parse
                continue
            # token might end with a comma (e.g. "2,") or be like "word,"
            token = parts[0].rstrip(",")
            num_parts = parts[1:]

            # normalize: remove any pure-comma tokens
            num_parts = [p for p in num_parts if p != ","]

            nums: list[float] = []
            i = 0
            while i < len(num_parts):
                p = num_parts[i].replace(",", "")
                if _is_float_token(p):
                    nums.append(float(p))
                    i += 1
                    continue

                # if this token doesn't parse, try to join with one or more following tokens
                # to recover numbers that were split by stray spaces e.g. "0.397      389"
                joined = p
                j = i + 1
                parsed = False
                while j < len(num_parts) and j <= i + 3:  # don't join more than a few tokens
                    joined += num_parts[j].replace(",", "")
                    if _is_float_token(joined):
                        nums.append(float(joined))
                        i = j + 1
                        parsed = True
                        break
                    j += 1
                if parsed:
                    continue

                # fallback: try to extract a float substring from the current token
                m = re.search(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", num_parts[i])
                if m:
                    nums.append(float(m.group(0)))
                # move on
                i += 1

            if expected_dim is not None and len(nums) != expected_dim:
                # warn but still store the vector
                import warnings
                warnings.warn(
                    f"Line {lineno}: token={token!r} parsed dim={len(nums)} != expected_dim={expected_dim}"
                )

            if use_numpy and _np is not None:
                embeddings[token] = _np.array(nums, dtype=_np.float32)
            else:
                embeddings[token] = nums

    return embeddings


def add_cosine_to_pairs(embeddings: dict, pairs, w1_col="word1", w2_col="word2", out_col="cosine"):
    """
    Given embeddings and pairs (pandas.DataFrame or list[dict]), return pairs with cosine scores.
    - For DataFrame: returns a copy with a new column `out_col`.
    - For list of dicts: mutates dicts (adds out_col) and returns the list.
    """
    try:
        import pandas as pd
    except Exception:
        pd = None

    if pd is not None and hasattr(pairs, "iterrows"):
        df = pairs.copy()
        df[out_col] = df.apply(lambda r: pair_cosine(embeddings, r[w1_col], r[w2_col]), axis=1)
        return df

    # assume list of dicts
    for r in pairs:
        r[out_col] = pair_cosine(embeddings, r.get(w1_col), r.get(w2_col))
    return pairs


## analogies helper functions
def glob_bats_paths(root: Union[str, Path],
                    pattern: str = "*.txt",
                    recursive: bool = True,
                    include_metadata: bool = False) -> List[Path]:
    """
    Return a sorted list of file paths under `root` matching `pattern`.
    - root: path to eval/BATS_3_0
    - pattern: glob pattern (default '*.txt')
    - recursive: use rglob when True, glob when False
    - include_metadata: also include metadata.json if present
    """
    root = Path(root)
    if recursive:
        files = [p for p in root.rglob(pattern) if p.is_file()]
    else:
        files = [p for p in root.glob(pattern) if p.is_file()]

    if include_metadata:
        meta = root / "metadata.json"
        if meta.exists():
            files.append(meta)

    return sorted(files)


_nonword_edge = re.compile(r"^[^\w]+|[^\w]+$")

def parse_bats_line(line: str) -> List[Tuple[str, str]]:
    """
    Parse one BATS line and return list of (word1, word2) pairs.

    Handles two cases:
      - "w1<TAB>w2"                     -> [(w1, w2)]
      - "w1<TAB>w2a/w2b/w2c"            -> [(w1, w2a), (w1, w2b), (w1, w2c)]

    Skips blank lines and comment lines starting with '#'. Trims surrounding
    non-word characters from tokens.
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return []
    parts = s.split()
    if len(parts) < 2:
        return []
    w1 = _nonword_edge.sub("", parts[0])
    w2_field = parts[1]
    # split second field on '/', filter empties, clean edges
    items = [ _nonword_edge.sub("", it) for it in w2_field.split("/") if it.strip() ]
    return [(w1, it) for it in items]


def read_bats_file(path: Union[str, Path]) -> List[Tuple[str, str]]:
    """
    Read a BATS-format file from a POSIX path and return a list of (word1, word2) pairs.
    Requires `parse_bats_line(line)` to be defined (it should return a list of pairs for a line).
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(p)
    pairs: List[Tuple[str, str]] = []
    with p.open(encoding="utf-8", errors="ignore") as fh:
        for ln in fh:
            pairs.extend(parse_bats_line(ln))
    return pairs


def normalize_embeddings(emb: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {w: v / (np.linalg.norm(v) + 1e-12) for w, v in emb.items()}


def topk_scores(query: np.ndarray, vocab_mat: np.ndarray, k: int):
    # vocab_mat rows are normalized vectors; query should be normalized too
    sims = vocab_mat @ query
    idx = np.argpartition(-sims, k-1)[:k]
    return idx[np.argsort(-sims[idx])], sims


def solve_analogy_add(a, a_pos, b, emb_norm: Dict[str, np.ndarray],
                      vocab_list: List[str], vocab_mat: np.ndarray) -> Tuple[str, float]:
    q = emb_norm[a_pos] - emb_norm[a] + emb_norm[b]
    q = q / (np.linalg.norm(q) + 1e-12)
    sims = vocab_mat @ q
    # exclude source words
    exclude = {a, a_pos, b}
    for w in exclude:
        try:
            i = vocab_list.index(w)
            sims[i] = -np.inf
        except ValueError:
            pass
    i = int(np.argmax(sims))
    return vocab_list[i], float(sims[i])


def solve_analogy_3cosmul(a, a_pos, b, emb_norm, vocab_list, vocab_mat, eps=1e-8):
    # score(c) = cos(c, a_pos) * cos(c, b) / (cos(c, a) + eps)
    cos_a = vocab_mat @ emb_norm[a]
    cos_ap = vocab_mat @ emb_norm[a_pos]
    cos_b = vocab_mat @ emb_norm[b]
    score = (cos_ap * cos_b) / (cos_a + eps)
    for w in {a, a_pos, b}:
        try:
            score[vocab_list.index(w)] = -np.inf
        except ValueError:
            pass
    i = int(np.argmax(score))
    return vocab_list[i], float(score[i])


def evaluate_analogies(quads: List[Tuple[str,str,str,str]],
                       emb_norm: Dict[str, np.ndarray],
                       method: str = "add",
                       k_list=(1,5,10)):
    # prepare normalized emb and fast structures
    vocab_list = sorted(emb_norm.keys())
    vocab_mat = np.vstack([emb_norm[w] for w in vocab_list])
    total = 0
    hits = {k:0 for k in k_list}
    mrr_sum = 0.0
    for a,a_pos,b,b_pos in quads:
        if any(w not in emb_norm for w in (a,a_pos,b,b_pos)):
            continue
        total += 1
        if method == "add":
            q = emb_norm[a_pos] - emb_norm[a] + emb_norm[b]
            q /= np.linalg.norm(q) + 1e-12
            sims = vocab_mat @ q
        else:  # 3cosmul
            cos_a = vocab_mat @ emb_norm[a]
            cos_ap = vocab_mat @ emb_norm[a_pos]
            cos_b = vocab_mat @ emb_norm[b]
            sims = (cos_ap * cos_b) / (cos_a + 1e-8)

        # mask exclude words
        for w in (a, a_pos, b):
            if w in emb_norm:
                sims[vocab_list.index(w)] = -np.inf

        # rank
        order = np.argsort(-sims)
        ranks = {vocab_list[i]: rank for rank,i,rank in zip(vocab_list, order, range(len(order)))}
        if b_pos in ranks:
            rank = ranks[b_pos]
            mrr_sum += 1.0 / (rank + 1)
            for kk in k_list:
                if rank < kk:
                    hits[kk] += 1

    accs = {f"hit@{k}": (hits[k]/total if total else 0.0) for k in k_list}
    mrr = mrr_sum / total if total else 0.0
    return {"quads-evaluated": total, **accs, "MRR": mrr, "quad-total": len(quads)}


## now given the folder we process the bats data...
def all_analogies(tl_dir: Union[str, Path], emb: dict, skip:bool=True):
    """
    `skip` skips hypernyms which take a long time so this is helpful,
     for debugging, testing, adding new datasets, etc.
    """
    files = glob_bats_paths(tl_dir)
    data = []
    emb_norm = normalize_embeddings(emb)
    for file in files:
        print(f"starting {file.name}")
        if skip and file.name.startswith('L'):
            print(f"skipping {file.name}")
        else:
            all_pairs = read_bats_file(file)
            # now generate quads
            quads = []
            for two_pair in combinations(all_pairs, 2):
                (a,b),(c,d) = two_pair[0], two_pair[1]
                quads.append((a,b,c,d))
            print(f"attempting evaluation of {len(quads)} quads")
            result = evaluate_analogies(quads, emb_norm=emb_norm, method="add", k_list=(1,5))
            result["category"] = file.parent.name
            result["sub-category"] = file.name
            data.append(result)
    df = pd.DataFrame(data)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Read word-pair CSV")
    # single file datasets
    p.add_argument("--wordsim353-path", help="path to wordsim353 CSV file")
    p.add_argument("--MEN-path", help="path to MEN file")
    p.add_argument("--SCSW-path", help="path to SCSW file")
    p.add_argument("--SIMLEX-path", help="path to SIMLEX data.")
    # now multi-file folders
    p.add_argument("--MC-folder", help="path to Miller Charles benchmark data folder. DO NOT pass a full path to a file.")
    p.add_argument("--BATS-folder", help="path to BATS benchmark data folder. DO NOT pass a full path to a file.")
    p.add_argument("--vectors-path", help="path to word vectors file")
    # config flags
    p.add_argument("--skip-hypernym", action='store_true', help="including skip_hypernym will skip the `L` family from BATS which takes a long time to run.")
    args = p.parse_args()

    ## TODO: convert `print()` to logstreams and refactor common from copy-pasta
    embs = read_embeddings(args.vectors_path, use_numpy=True)
    if args.wordsim353_path is not None:
        df = read_with_pandas(args.wordsim353_path)
        vec = embs.get("computer")
        print(f"Processing WordSim353 data...")
        print(f"dimension of vectors read from file: {len(vec)!r}  ")
        pair_cosine = {"cosine_of_pair": []}
        for row in df.itertuples(name=None):
            try:
                r1, r2 = embs.get(row[1]), embs.get(row[2])
                denom = np.linalg.norm(r1) * np.linalg.norm(r2)
                cosine = float(np.dot(r1, r2) / denom) if denom else None
                pair_cosine["cosine_of_pair"].append(cosine)
            except TypeError as e:
                pair_cosine["cosine_of_pair"].append(None)
        # confirm matching length
        print(f"pair_cosine length: {len(pair_cosine['cosine_of_pair'])}  ...")
        print(f"df.shape: {df.shape}")
        df["pair_cosine"] = pair_cosine['cosine_of_pair']
        # now take the spearman correlation of the columns "pair_cosine" and "Human_mean"
        correlation_value = df["pair_cosine"].corr(df["Human_mean"], method="spearman")
        corr_coefficient, p_value = spearmanr(df["pair_cosine"].to_numpy(), df["Human_mean"].to_numpy(), nan_policy="omit")
        # NOTE: the nan-policy omit yields values that are essentially the same...
        print(correlation_value)x
        print(corr_coefficient, p_value)
        # report the nan count too though
        nan_count = np.sum(np.isnan(df["pair_cosine"].to_numpy()))
        print(f"nan-count: {nan_count}")

    if args.MEN_path is not None:
        print(f"Processing MEN data...")
        df = read_pair_scores(args.MEN_path)
        # NOTE: This is copy-pasta from the wordsim353 above. The only difference is the name of the column
        # in the dataset with ratings is called "score" whereas in wordsim353 it is called "Human_mean"
        # TODO: abstract this out...
        pair_cosine = {"cosine_of_pair": []}
        for row in df.itertuples(name=None):
            try:
                r1, r2 = embs.get(row[1]), embs.get(row[2])
                denom = np.linalg.norm(r1) * np.linalg.norm(r2)
                cosine = float(np.dot(r1, r2) / denom) if denom else None
                pair_cosine["cosine_of_pair"].append(cosine)
            except TypeError as e:
                pair_cosine["cosine_of_pair"].append(None)
        # confirm matching length
        print(f"pair_cosine length: {len(pair_cosine['cosine_of_pair'])}  ...")
        print(f"df.shape: {df.shape}")
        df["pair_cosine"] = pair_cosine['cosine_of_pair']
        # now take the spearman correlation of the columns "pair_cosine" and "Human_mean"
        correlation_value = df["pair_cosine"].corr(df["score"], method="spearman")
        corr_coefficient, p_value = spearmanr(df["pair_cosine"].to_numpy(), df["score"].to_numpy(), nan_policy="omit")
        # NOTE: the nan-policy omit yields values that are essentially the same...
        print(correlation_value)
        print(corr_coefficient, p_value)
        # report the nan count too though
        nan_count = np.sum(np.isnan(df["pair_cosine"].to_numpy()))
        print(f"nan-count: {nan_count}")

    if args.SCSW_path is not None:
        print(f"Processing SCSW data...")
        # NOTE: this read_scws() returns a list of dicts, hence 'lod'
        lod = read_scsw(args.SCSW_path)
        scws_results = evaluate_scws(embeddings=embs, rows=lod)
        print(scws_results)

    if args.SIMLEX_path is not None:
        print(f"Processing SIMLEX data...")
        simlex = read_SIMLEX(args.SIMLEX_path)
        print(f"simlex.shape: {simlex.shape}")
        simlex_results = evaluate_SIMLEX(simlex, embeddings=embs)
        print(simlex_results)

    if args.MC_folder is not None:
        # Now Miller-Charles data: The human similarity ratings you should use are in the HJ column
        # HJ ="human-judgment", the others are system/model scores or auxiliary stats (r1, r2, omega_k, etc.)
        print(f"Processing MC data...")
        # NOTE: Although we do not do so here, the `_source` column can be leveraged for a per file breakdown
        df = concat_csvs(args.MC_folder)
        # now calculate scores similar to other datasets
        pair_cosine = {"cosine_of_pair": []}
        for row in df.itertuples(name=None):
            try:
                r1, r2 = embs.get(row[1]), embs.get(row[2])
                denom = np.linalg.norm(r1) * np.linalg.norm(r2)
                cosine = float(np.dot(r1, r2) / denom) if denom else None
                pair_cosine["cosine_of_pair"].append(cosine)
            except TypeError as e:
                pair_cosine["cosine_of_pair"].append(None)
        # confirm matching length
        print(f"pair_cosine length: {len(pair_cosine['cosine_of_pair'])}  ...")
        print(f"df.shape: {df.shape}")
        df["pair_cosine"] = pair_cosine['cosine_of_pair']
        correlation_value = df["pair_cosine"].corr(df["HJ"], method="spearman")
        corr_coefficient, p_value = spearman_vs_hj(df, model_scores="pair_cosine", human_col="HJ")
        print(correlation_value)
        print(corr_coefficient, p_value)
        # report the nan count too though
        nan_count = np.sum(np.isnan(df["pair_cosine"].to_numpy()))
        print(f"nan-count: {nan_count}")

    if args.BATS_folder is not None:
        ## NOw do BATS-3.0 all analogies, this one takes awhile....
        print(f"Processing BATS analogies...")
        anal = all_analogies(args.BATS_folder, emb=embs, skip=args.skip_hypernym)
        print("BATS analogies breakdown:\n")
        print(anal)

if __name__ == "__main__":
    main()
