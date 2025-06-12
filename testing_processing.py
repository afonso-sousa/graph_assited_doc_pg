# %%
example = {
    "source": [
        "He stopped dead immediately to say then followed slavishly."
    ],
    "target": [
        "He immediately obeyed."
    ],
    "semantic_graph": [
        {
            "nodes": [
                {"index": [1], "word": ["stopped"]},
                {"index": [2], "word": ["dead"]},
                {"index": [3], "word": ["immediately"]},
                {"index": [5], "word": ["say"]},
                {"index": [4], "word": ["to"]},
                {"index": [6], "word": ["then"]},
                {"index": [7], "word": ["followed"]},
                {"index": [8], "word": ["slavishly"]},
            ],
            "edges": [
                [3, 7],  # "immediately" <-> "followed"
                [7, 8],  # "followed" <-> "slavishly"
            ]
        }
    ]
}

# %%

from functools import partial
from types import SimpleNamespace

from transformers import AutoTokenizer

from train import preprocess_function

args = SimpleNamespace(max_seq_length=32, with_graph=True)
tokenizer = AutoTokenizer.from_pretrained("google/long-t5-local-base")

preprocess_partial = partial(preprocess_function, tokenizer=tokenizer, args=args)
processed = preprocess_partial(example)
print("Graph edges (token indices):", processed["graph_edges"][0])
print("Input tokens:", tokenizer.convert_ids_to_tokens(processed["input_ids"][0]))
# %%
