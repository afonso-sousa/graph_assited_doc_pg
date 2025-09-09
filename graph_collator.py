from transformers import DataCollatorForSeq2Seq


class GraphDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        # Separate out graph_edges
        graph_edges = [f.pop("graph_edges") for f in features]

        # Use default collator for the rest
        batch = super().__call__(features)

        # Add graph_edges back manually
        batch["graph_edges"] = graph_edges
        return batch
