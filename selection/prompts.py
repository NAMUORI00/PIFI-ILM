from typing import List, Tuple
import numpy as np


NEUTRAL_LABEL_WORDS = [
    "bar",
    "foo",
    "baz",
    "qux",
    "quux",
    "corge",
    "grault",
]


def _map_label_to_token(label: int, num_classes: int) -> str:
    """Map class index to a neutral label word."""
    if num_classes <= len(NEUTRAL_LABEL_WORDS):
        return NEUTRAL_LABEL_WORDS[label % len(NEUTRAL_LABEL_WORDS)]
    # Fallback to numeric string if classes exceed neutral list
    return str(label)


def build_label_prompts(
    texts: List[str],
    labels: List[int],
    k_shot: int = 1,
) -> Tuple[List[str], List[List[str]], List[int]]:
    """
    Build k-shot ICL-style prompts with neutral label words.

    Each prompt packs up to k examples:
        input: <text_i>\noutput: <label_word_i>\n ... (k times)
    The last example in the block is used for scoring; label_token_strs aligns per prompt
    with the last label word in that prompt.
    """
    labels_arr = np.asarray(labels, dtype=int)
    num_classes = int(labels_arr.max()) + 1 if labels_arr.size > 0 else 2
    prompts: List[str] = []
    label_token_strs: List[List[str]] = []
    prompt_labels: List[int] = []

    if k_shot <= 1:
        k_shot = 1

    for i in range(0, len(texts), k_shot):
        block_texts = texts[i : i + k_shot]
        block_labels = labels_arr[i : i + k_shot].tolist()
        lines = []
        for t, lab in zip(block_texts, block_labels):
            lw = _map_label_to_token(lab, num_classes)
            lines.append(f"input: {t}\noutput: {lw}")
        prompt = "\n".join(lines)
        prompts.append(prompt)
        if block_labels:
            lw_last = _map_label_to_token(block_labels[-1], num_classes)
            label_token_strs.append([lw_last])
            prompt_labels.append(int(block_labels[-1]))
        else:
            label_token_strs.append([])
            prompt_labels.append(0)

    return prompts, label_token_strs, prompt_labels
