# %%
example = [
    {
        "source": ["He stopped dead immediately to say then followed slavishly."],
        "target": ["He immediately obeyed."],
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
                    {
                        "from": 3,
                        "to": 7,
                        "relation": "",
                    },  # "immediately" <-> "followed"
                    {"from": 7, "to": 8, "relation": ""},  # "followed" <-> "slavishly"
                ],
            }
        ],
    },
    [
        (3, 7),
        (8, 9),
        (9, 8),
        (10, 9),
        (9, 11),
        (11, 8),
        (7, 10),
        (9, 7),
        (8, 11),
        (9, 10),
        (11, 7),
        (10, 11),
        (11, 10),
        (10, 8),
        (7, 3),
        (7, 9),
        (8, 7),
        (8, 10),
        (10, 7),
        (11, 9),
        (7, 11),
        (7, 8),
    ],
]
# ['▁He', '▁stopped', '▁dead', '▁immediately', '▁to', '▁say', '▁then', '▁followed', '▁', 'slav', 'ish', 'ly', '▁', '.', '</s>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']


example_single_subwords = [
    {
        "source": ["Birds fly quickly."],
        "target": ["Birds move fast."],
        "semantic_graph": [
            {
                "nodes": [
                    {"index": [0], "word": ["Birds"]},
                    {"index": [1], "word": ["fly"]},
                    {"index": [2], "word": ["quickly"]},
                ],
                "edges": [
                    {"from": 1, "to": 2, "relation": ""},  # "fly" <-> "quickly"
                ],
            }
        ],
    },
    [(1, 2), (2, 1)],
]
# ['▁Birds', '▁fly', '▁quickly', '▁', '.', '</s>', ...]

example_multi_subwords = [
    {
        "source": ["The internationalization process progressed gradually."],
        "target": ["The process advanced slowly."],
        "semantic_graph": [
            {
                "nodes": [
                    {"index": [1], "word": ["internationalization"]},
                    {"index": [3], "word": ["progressed"]},
                    {"index": [4], "word": ["gradually"]},
                ],
                "edges": [
                    {
                        "from": 1,
                        "to": 3,
                        "relation": "",
                    },  # "internationalization" <-> "progressed"
                    # {"from": 3, "to": 4, "relation": ""},  # "progressed" <-> "gradually"
                ],
            }
        ],
    },
    [(2, 4), (1, 2), (2, 1), (4, 2), (1, 4), (4, 1)],
]  # ['▁The', '▁international', 'ization', '▁process', '▁progressed', '▁gradually', '▁', '.', '</s>', ...]

# %%

from functools import partial
from types import SimpleNamespace

from transformers import AutoTokenizer

from train import preprocess_function

args = SimpleNamespace(max_seq_length=32, with_graph=True)
tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")

preprocess_partial = partial(preprocess_function, tokenizer=tokenizer, args=args)
for data, expected in [example, example_single_subwords, example_multi_subwords]:
    processed = preprocess_partial(data)
    print(tokenizer.convert_ids_to_tokens(processed["input_ids"][0]))
    print(processed["graph_edges"][0])
    assert (
        processed["graph_edges"][0] == expected
    ), f"Expected {expected}, but got {processed['graph_edges'][0]}"

# %%
