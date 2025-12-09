from typing import List, Tuple

import numpy as np


def resolve_dataset(
    task: str,
    dataset: str,
    n_samples: int,
    seed: int = 42,
    split: str = "validation",
    stratified: bool = False,
) -> Tuple[List[str], List[int]]:
    """
    Dataset loader used for ILM layer selection.

    Mirrors the logic used in the main branch but kept standalone
    to avoid pulling the full refactor.
    """
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    task = (task or "").lower()
    name = (dataset or "").lower()
    texts: List[str] = []
    labels: List[int] = []

    def take_subset(ds, text_key: str, label_key: str):
        nonlocal texts, labels
        n = min(n_samples, len(ds))
        if stratified:
            per_class = {}
            for i in range(len(ds)):
                item = ds[int(i)]
                c = int(item[label_key])
                per_class.setdefault(c, []).append(i)
            if len(per_class) == 0:
                return
            k = len(per_class)
            per_cls_n = max(1, n // k)
            for c, arr in per_class.items():
                take = min(per_cls_n, len(arr))
                idx = rng.choice(arr, size=take, replace=False)
                for j in idx:
                    item = ds[int(j)]
                    texts.append(str(item[text_key]))
                    labels.append(int(item[label_key]))
        else:
            idx = rng.choice(len(ds), size=n, replace=False)
            for i in idx:
                item = ds[int(i)]
                texts.append(str(item[text_key]))
                labels.append(int(item[label_key]))

    if name == "sst2":
        try:
            ds = load_dataset("SetFit/sst2")
            chosen = ds.get(split if split in ds else "validation")
            take_subset(chosen, "text", "label")
        except Exception:
            ds = load_dataset("glue", "sst2")
            mapping = {"train": "train", "validation": "validation", "test": "test"}
            chosen = ds[mapping.get(split, "validation")]
            take_subset(chosen, "sentence", "label")
    elif name == "imdb":
        ds = load_dataset("imdb")
        take_subset(ds["test"], "text", "label")
    elif name == "cola":
        ds = load_dataset("nyu-mll/glue", "cola")
        take_subset(ds["validation"], "sentence", "label")
    elif name == "tweet_offensive":
        ds = load_dataset("cardiffnlp/tweet_eval", "offensive")
        take_subset(ds["validation"], "text", "label")
    elif name == "tweet_sentiment_binary":
        ds = load_dataset("tweet_eval", name="sentiment")
        val = ds["validation"]
        filtered = [ex for ex in val if int(ex["label"]) != 1]
        n = min(n_samples, len(filtered))
        idx = rng.choice(len(filtered), size=n, replace=False)
        for i in idx:
            item = filtered[int(i)]
            lab = 1 if int(item["label"]) == 2 else 0
            texts.append(str(item["text"]))
            labels.append(lab)
    elif name == "mnli":
        ds = load_dataset("glue", "mnli")
        split_map = {
            "train": "train",
            "validation": "validation_matched",
            "test": "validation_matched",
        }
        val = ds[split_map.get(split, "validation_matched")]
        n = min(n_samples, len(val))
        idx = rng.choice(len(val), size=n, replace=False)
        for i in idx:
            item = val[int(i)]
            texts.append(f"{item['premise']} [SEP] {item['hypothesis']}")
            labels.append(int(item["label"]))
    elif name == "snli":
        ds = load_dataset("snli")
        val = ds["validation"]
        n = min(n_samples, len(val))
        idx = rng.choice(len(val), size=n, replace=False)
        for i in idx:
            item = val[int(i)]
            if int(item["label"]) == -1:
                continue
            texts.append(f"{item['premise']} [SEP] {item['hypothesis']}")
            labels.append(int(item["label"]))

    return texts, labels
