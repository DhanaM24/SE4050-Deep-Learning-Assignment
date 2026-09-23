"""Label loading, near-duplicate detection and leakage-safe splitting.

The damage task uses every train/Czech and train/India image that carries at
least one of the four label-map classes (D00, D10, D20, D40).  Splitting is
done on near-duplicate *clusters* (perceptual dHash) so identical/similar
patches can never leak between train, validation and test.

The resulting split manifest (results/metrics/split_manifest.csv) is the single
source of truth for every model in the group, guaranteeing fair comparison.
"""

import collections
import hashlib

import numpy as np
import pandas as pd
from PIL import Image

from src.config import Config
from src.utils import save_json
from src.validate_dataset import parse_damage_labels

MANIFEST_PATH = Config.results_dir / "metrics" / "split_manifest.csv"
NEAR_DUP_MAX_HAMMING = 4  # 64-bit dHash: <=4 means visually near-identical


def build_label_table() -> pd.DataFrame:
    """Multi-label table of images that have at least one main-class label."""
    df = parse_damage_labels()
    n_main = df[Config.class_names].sum(axis=1)
    df = df[n_main > 0].reset_index(drop=True)
    df["primary"] = df[Config.class_names].idxmax(axis=1)
    return df


def dhash(path, size: int = 8) -> int:
    """64-bit difference hash of an image (resize+grey, compare neighbours)."""
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        pixels = list(g.getdata())
    bits = 0
    for row in range(size):
        for col in range(size):
            left = pixels[row * (size + 1) + col]
            right = pixels[row * (size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def _hamming_matrix(hashes: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Pairwise Hamming distances between 64-bit hashes (chunked, low memory)."""
    b = np.unpackbits(hashes.view(np.uint8).reshape(hashes.size, 8), axis=1)
    n = b.shape[0]
    dist = np.empty((n, n), dtype=np.int16)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        diff = b[start:end, None, :] != b[None, :, :]
        dist[start:end] = diff.sum(axis=2)
    return dist


def find_near_duplicate_clusters(paths, max_hamming: int = NEAR_DUP_MAX_HAMMING):
    """Union-Find over perceptually near-identical images."""
    hashes = np.array([dhash(p) for p in paths], dtype=np.uint64)
    dist = _hamming_matrix(hashes)
    n = len(paths)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    iu = np.triu_indices(n, k=1)
    close = np.where(dist[iu] <= max_hamming)[0]
    for idx in close:
        union(int(iu[0][idx]), int(iu[1][idx]))

    groups = collections.defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    clusters = sorted((sorted(v) for v in groups.values()), key=lambda c: -len(c))
    return clusters, hashes


def _greedy_cluster_split(df: pd.DataFrame, clusters) -> pd.Series:
    """Assign whole clusters to train/val/test, hitting per-class targets."""
    rng = np.random.default_rng(Config.seed)
    n = len(df)
    test_target = Config.test_size
    val_target = Config.val_size

    primaries = df["primary"].to_numpy()
    split = np.full(n, "", dtype=object)
    used = collections.Counter()

    # cluster -> majority primary label
    cluster_primary = []
    for cl in clusters:
        votes = collections.Counter(primaries[cl])
        cluster_primary.append(votes.most_common(1)[0][0])

    order = np.arange(len(clusters))
    rng.shuffle(order)

    class_totals = collections.Counter(primaries)
    progress = collections.defaultdict(lambda: {"test": 0.0, "val": 0.0})

    for ci in order:
        cl = clusters[ci]
        label = cluster_primary[ci]
        size = len(cl)
        # choose the split that is currently furthest below its target share
        deficit = {}
        for sp, target in (("test", test_target), ("val", val_target)):
            want = class_totals[label] * target
            deficit[sp] = want - progress[label][sp]
        if deficit["test"] >= deficit["val"] and deficit["test"] > 0:
            choice = "test"
        elif deficit["val"] > 0:
            choice = "val"
        else:
            choice = "train"
        for i in cl:
            split[i] = choice
        progress[label][choice] = progress[label].get(choice, 0.0) + size
        used[choice] += size

    # guarantee non-empty splits even for tiny classes
    for sp in ("train", "val", "test"):
        if used[sp] == 0:
            raise RuntimeError(f"empty split: {sp}")
    return pd.Series(split, index=df.index, name="split")


def create_or_load_split(force: bool = False) -> pd.DataFrame:
    """Return the label table annotated with a split column.

    A saved manifest is reused so every group model sees the identical split.
    """
    df = build_label_table()
    if MANIFEST_PATH.exists() and not force:
        manifest = pd.read_csv(MANIFEST_PATH)
        # verify manifest matches current label table
        if list(manifest["path"]) == list(df["path"]):
            df["split"] = manifest["split"]
            df["cluster_id"] = manifest["cluster_id"]
            print(f"Loaded existing split manifest: {MANIFEST_PATH}")
            return df
        print("Manifest does not match current data - rebuilding split.")

    print("Computing perceptual hashes / near-duplicate clusters ...")
    clusters, hashes = find_near_duplicate_clusters(list(df["path"]))
    multi = [c for c in clusters if len(c) > 1]
    print(f"{len(df)} images -> {len(clusters)} clusters ({len(multi)} with duplicates)")

    print("Assigning leakage-safe splits ...")
    df["split"] = _greedy_cluster_split(df, clusters)

    df["dhash"] = [format(int(h), "016x") for h in hashes]
    df["cluster_id"] = -1
    for cid, cl in enumerate(clusters):
        df.loc[df.index[cl], "cluster_id"] = cid
    df.drop(columns="dhash").to_csv(MANIFEST_PATH, index=False)

    report = {
        "n_images": int(len(df)),
        "n_clusters": int(len(clusters)),
        "n_clusters_with_duplicates": int(len(multi)),
        "largest_cluster": int(max(len(c) for c in clusters)),
        "split_counts": df["split"].value_counts().to_dict(),
        "per_class_split": (
            df.groupby("split")[Config.class_names].sum().astype(int).to_dict()
        ),
        "per_source_split": pd.crosstab(df["split"], df["source"]).to_dict(),
    }
    save_json(report, Config.results_dir / "metrics" / "split_report.json")
    print("Saved split manifest ->", MANIFEST_PATH)
    return df


def load_xy(df: pd.DataFrame, split: str):
    """Paths and binary label matrix for one split."""
    part = df[df["split"] == split]
    y = part[Config.class_names].to_numpy(dtype=np.float32)
    return list(part["path"]), y


def leakage_report(df: pd.DataFrame) -> dict:
    """Verify no cluster and no path crosses split boundaries."""
    cluster_splits = df.groupby("cluster_id")["split"].nunique()
    cross = cluster_splits[cluster_splits > 1]
    dup_paths = df["path"].duplicated().sum()
    return {
        "clusters_spanning_multiple_splits": int(len(cross)),
        "duplicate_paths": int(dup_paths),
        "split_counts": df["split"].value_counts().to_dict(),
        "passed": bool(len(cross) == 0 and dup_paths == 0),
    }


if __name__ == "__main__":
    table = create_or_load_split(force=True)
    print(table.groupby(["split", "source"]).size())
    print(table.groupby("split")[Config.class_names].sum())
    print(leakage_report(table))
