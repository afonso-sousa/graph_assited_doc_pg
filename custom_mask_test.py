# %%
import torch
from transformers import BigBirdPegasusConfig, BigBirdPegasusForConditionalGeneration

from graph_bigbird import GraphBigBirdPegasusForConditionalGeneration

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Configure and initialize the model
config = BigBirdPegasusConfig(
    vocab_size=32,
    d_model=64,
    encoder_layers=1,
    decoder_layers=1,
    encoder_attention_heads=1,
    attention_type="block_sparse",  # standard BigBird attention type
    block_size=2,  # Small block size for visualization
    num_random_blocks=2,
    use_bias=False,
)
# Custom model with graph-based sparse attention
my_model = GraphBigBirdPegasusForConditionalGeneration(config).to(device)
my_model.eval()

# Standard BigBird model
std_model = BigBirdPegasusForConditionalGeneration(config).to(device)
std_model.eval()

# %%
batch_size = 3
seq_len = 32

input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
attention_mask = torch.ones((batch_size, seq_len), device=device)

# Step 3: Define graph edges
# Edges: (0 -> 1), (1 -> 2), (2 -> 3), etc.
graph_edges = [(i, i + 1) for i in range(seq_len - 1)]
graph_edges_batch = [graph_edges for _ in range(batch_size)]

# %%
with torch.no_grad():
    outputs_std = std_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=torch.zeros((batch_size, 1), dtype=torch.long, device=device),
        output_attentions=True,
        return_dict=True,
    )
print(
    "Standard BigBird forward pass completed. Logits shape:", outputs_std.logits.shape
)

if outputs_std.encoder_attentions:
    print("Standard Attention Shape:", outputs_std.encoder_attentions[0].shape)

# %%
with torch.no_grad():
    outputs_graph = my_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=torch.zeros((batch_size, 1), dtype=torch.long, device=device),
        graph_edges=graph_edges_batch,
        output_attentions=True,
        return_dict=True,
    )
print("GraphBigBird forward pass completed. Logits shape:", outputs_graph.logits.shape)

if outputs_graph.encoder_attentions:
    print("Graph Attention Shape:", outputs_graph.encoder_attentions[0].shape)

# %%
