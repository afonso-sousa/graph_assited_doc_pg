# %%
import matplotlib.pyplot as plt
import numpy as np


# %%
class MockBigBird:
    def __init__(self, training=True):
        self.training = training

    def _bigbird_block_rand_mask(
        self,
        from_seq_length,
        to_seq_length,
        from_block_size,
        to_block_size,
        num_rand_blocks,
        last_idx=-1,
    ):
        if from_seq_length // from_block_size != to_seq_length // to_block_size:
            raise ValueError("Error the number of blocks needs to be same!")

        rand_attn = np.zeros(
            (from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32
        )
        # During inference (eval) no randomness
        if not self.training:
            return rand_attn
        middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
        last = to_seq_length // to_block_size - 1
        if last_idx > (2 * to_block_size):
            last = (last_idx // to_block_size) - 1

        r = num_rand_blocks  # shorthand
        for i in range(1, from_seq_length // from_block_size - 1):
            start = i - 2
            end = i
            if i == 1:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[2:last])[:r]
            elif i == 2:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
            elif i == from_seq_length // from_block_size - 3:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -3: should have been sliced till last-3
            elif i == from_seq_length // from_block_size - 2:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -4: should have been sliced till last-4
            else:
                if start > last:
                    start = last
                    rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                elif (end + 1) == last:
                    rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                else:
                    rand_attn[i - 1, :] = np.random.permutation(
                        np.concatenate((middle_seq[:start], middle_seq[end + 1 : last]))
                    )[:r]
        return rand_attn

    def _bigbird_block_graph_mask(
        self,
        from_seq_length,
        to_seq_length,
        from_block_size,
        to_block_size,
        graph_edges,
    ):
        if from_seq_length // from_block_size != to_seq_length // to_block_size:
            raise ValueError("The number of blocks must be the same.")

        num_blocks = from_seq_length // from_block_size
        adj_list = [[] for _ in range(num_blocks)]

        for from_idx, to_idx in graph_edges:
            from_block = from_idx // from_block_size
            to_block = to_idx // to_block_size

            if 0 <= from_block < num_blocks and 0 <= to_block < num_blocks:
                adj_list[from_block].append(to_block)

        max_conn = max((len(l) for l in adj_list), default=0)
        mask = np.full((num_blocks, max_conn), -1, dtype=np.int32)

        for i, conn in enumerate(adj_list):
            mask[i, : len(conn)] = conn

        return mask


# %%
def plot_mask(matrix, title):
    plt.figure(figsize=(8, 6))
    plt.imshow((matrix >= 0).astype(np.int32), cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel("Block connection index")
    plt.ylabel("From block index")
    plt.colorbar(label="Connected (1 = yes)")
    plt.show()


# %%
# Parameters
from_seq_length = 112
to_seq_length = 112
block_size = 16
num_rand_blocks = 2

# Graph edges (directed from -> to)
graph_edges = [(0, 16), (0, 32), (16, 64), (32, 48), (48, 80), (64, 16)]

# %%
model = MockBigBird(training=True)

rand_mask = model._bigbird_block_rand_mask(
    from_seq_length,
    to_seq_length,
    from_block_size=block_size,
    to_block_size=block_size,
    num_rand_blocks=num_rand_blocks,
)

graph_mask = model._bigbird_block_graph_mask(
    from_seq_length,
    to_seq_length,
    from_block_size=block_size,
    to_block_size=block_size,
    graph_edges=graph_edges,
)

# %%
import torch


def create_masks_for_block_sparse_attn(attention_mask: torch.Tensor, block_size: int):
    batch_size, seq_length = attention_mask.size()
    if seq_length % block_size != 0:
        raise ValueError(
            f"Sequence length must be multiple of block size, but sequence length is {seq_length}, while block"
            f" size is {block_size}."
        )

    def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
        """
        Create 3D attention mask from a 2D tensor mask.

        Args:
            from_blocked_mask: 2D Tensor of shape [batch_size,
            from_seq_length//from_block_size, from_block_size].
            to_blocked_mask: int32 Tensor of shape [batch_size,
            to_seq_length//to_block_size, to_block_size].

        Returns:
            float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4, from_block_size,
            3*to_block_size].
        """
        exp_blocked_to_pad = torch.cat(
            [
                to_blocked_mask[:, 1:-3],
                to_blocked_mask[:, 2:-2],
                to_blocked_mask[:, 3:-1],
            ],
            dim=2,
        )
        band_mask = torch.einsum(
            "blq,blk->blqk", from_blocked_mask[:, 2:-2], exp_blocked_to_pad
        )
        band_mask.unsqueeze_(1)
        return band_mask

    blocked_encoder_mask = attention_mask.view(
        batch_size, seq_length // block_size, block_size
    )
    band_mask = create_band_mask_from_inputs(blocked_encoder_mask, blocked_encoder_mask)

    from_mask = attention_mask.view(batch_size, 1, seq_length, 1)
    to_mask = attention_mask.view(batch_size, 1, 1, seq_length)

    return blocked_encoder_mask, band_mask, from_mask, to_mask


batch_size = 2
attention_mask = torch.ones((batch_size, from_seq_length), dtype=torch.float32)

blocked_encoder_mask, band_mask, from_mask, to_mask = (
    create_masks_for_block_sparse_attn(attention_mask, block_size)
)
rand_attn = np.stack(rand_attn, axis=0)
rand_attn = torch.tensor(rand_attn, dtype=torch.long)
rand_attn.unsqueeze_(0)
rand_attn = torch.cat([rand_attn for _ in range(batch_size)], dim=0)

num_windows = from_seq_length // block_size - 2
rand_mask = torch.stack(
    [p1[i1.flatten()] for p1, i1 in zip(blocked_encoder_mask, rand_attn)]
)
rand_mask = rand_mask.view(batch_size, 2, num_windows, num_rand_blocks * block_size)
rand_mask = torch.einsum("blq,bhlk->bhlqk", blocked_encoder_mask[:, 1:-1], rand_mask)

# %%
plot_mask(rand_mask, "Random Attention Mask")
plot_mask(graph_mask, "Graph-Based Attention Mask")

# %%
