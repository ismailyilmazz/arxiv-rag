import json
import os
from pathlib import Path

import numpy as np

from arxiv_rag.storage.db import COLUMNS, connect, upsert_rows

STATE_FILE = "deltas_applied.json"


def write_delta(rows: list[dict], vectors: np.ndarray, folder: Path) -> dict:
    import pandas as pd

    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(COLUMNS)).to_parquet(folder / "papers.parquet", index=False)
    np.save(folder / "vectors.npy", vectors.astype(np.float16))
    np.save(folder / "ids.npy", np.array([r["id"] for r in rows], dtype=str))
    return {name: (folder / name).stat().st_size for name in ("papers.parquet", "vectors.npy", "ids.npy")}


def add_delta(manifest: dict, entry: dict) -> dict:
    deltas = [d for d in manifest.get("deltas", []) if d["date"] != entry["date"]]
    return {**manifest, "deltas": sorted(deltas + [entry], key=lambda d: d["date"])}


def _save(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def _load_delta(data_dir: Path, entry: dict, width: int):
    import pandas as pd

    folder = data_dir / entry["path"]
    for name, size in entry["files"].items():
        if not (folder / name).exists() or (folder / name).stat().st_size != size:
            raise ValueError(f"{name} eksik ya da boyutu farklı")
    frame = pd.read_parquet(folder / "papers.parquet")
    vectors = np.load(folder / "vectors.npy")
    ids = [str(i) for i in np.load(folder / "ids.npy")]
    if len(frame) != len(vectors) or list(frame["id"]) != ids or vectors.shape[1] != width:
        raise ValueError("satır, kimlik ya da vektör sayısı tutarsız")
    return frame, vectors, ids


def apply_deltas(data_dir: Path, manifest: dict) -> dict:
    data_dir = Path(data_dir)
    summary = {"applied": [], "errors": [], "added": 0, "updated": 0}
    deltas = sorted(manifest.get("deltas") or [], key=lambda d: d["date"])
    if deltas:
        vec_dir = data_dir / "vectors"
        vectors, pks = np.load(vec_dir / "vectors.npy"), np.load(vec_dir / "pks.npy")
        position = np.full(int(pks.max()) + 1 if len(pks) else 1, -1, dtype=np.int64)
        position[pks] = np.arange(len(pks))
        appended, extra_vectors, extra_pks = {}, [], []
        conn = connect(data_dir / "papers.db")
        try:
            for entry in deltas:
                try:
                    frame, delta_vectors, ids = _load_delta(data_dir, entry, vectors.shape[1])
                    upsert_rows(conn, frame[list(COLUMNS)].itertuples(index=False, name=None))
                    found = {}
                    for i in range(0, len(ids), 900):
                        chunk = ids[i : i + 900]
                        query = f"SELECT id, pk FROM papers WHERE id IN ({','.join('?' * len(chunk))})"
                        found.update({r[0]: int(r[1]) for r in conn.execute(query, chunk)})
                    missing = [pid for pid in ids if pid not in found]
                    if missing:
                        raise ValueError(f"{len(missing)} kimlik veritabanına yazılamadı")
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    summary["errors"].append(f"{entry.get('date')}: {type(e).__name__}: {e}"[:200])
                    continue
                for row, pid in enumerate(ids):
                    pk = found[pid]
                    if pk < len(position) and position[pk] >= 0:
                        vectors[position[pk]] = delta_vectors[row]
                        summary["updated"] += 1
                    elif pk in appended:
                        extra_vectors[appended[pk]] = delta_vectors[row]
                        summary["updated"] += 1
                    else:
                        appended[pk] = len(extra_pks)
                        extra_pks.append(pk)
                        extra_vectors.append(delta_vectors[row])
                        summary["added"] += 1
                summary["applied"].append(entry["date"])
        finally:
            conn.close()
        if summary["applied"]:
            if extra_vectors:
                vectors = np.vstack([vectors, np.asarray(extra_vectors, dtype=vectors.dtype)])
                pks = np.concatenate([pks, np.asarray(extra_pks, dtype=pks.dtype)])
            _save(vec_dir / "vectors.npy", vectors)
            _save(vec_dir / "pks.npy", pks)
    (data_dir / STATE_FILE).write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    return summary


def read_summary(data_dir: Path) -> dict:
    try:
        return json.loads((Path(data_dir) / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
