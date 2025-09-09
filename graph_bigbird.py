import math
from typing import Optional, Union

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from transformers.models.bigbird_pegasus.configuration_bigbird_pegasus import (
    BigBirdPegasusConfig,
)
from transformers.models.bigbird_pegasus.modeling_bigbird_pegasus import (
    BigBirdPegasusBlockSparseAttention,
    BigBirdPegasusDecoder,
    BigBirdPegasusEncoder,
    BigBirdPegasusEncoderAttention,
    BigBirdPegasusEncoderLayer,
    BigBirdPegasusForConditionalGeneration,
    BigBirdPegasusModel,
    BigBirdPegasusScaledWordEmbedding,
    BigBirdPegasusSelfAttention,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


class GraphBigBirdPegasusBlockSparseAttention(BigBirdPegasusBlockSparseAttention):
    def __init__(self, config, seed=None):
        super().__init__(config, seed)

    def forward(
        self,
        hidden_states,
        band_mask=None,
        from_mask=None,
        to_mask=None,
        from_blocked_mask=None,
        to_blocked_mask=None,
        output_attentions=None,
        graph_edges=None,
    ):
        # Currently this `class` can't be used in decoder.
        batch_size, seqlen, _ = hidden_states.size()
        to_seq_length = from_seq_length = seqlen
        from_block_size = to_block_size = self.block_size

        if from_seq_length % from_block_size != 0:
            raise ValueError(
                "Query sided sequence length must be multiple of block size"
            )

        if to_seq_length % to_block_size != 0:
            raise ValueError(
                "Key/Value sided sequence length must be multiple of block size"
            )

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        context_layer, attention_probs = self.graph_bigbird_block_sparse_attention(
            query_layer,
            key_layer,
            value_layer,
            band_mask,
            from_mask,
            to_mask,
            from_blocked_mask,
            to_blocked_mask,
            self.num_attention_heads,
            self.attention_head_size,
            from_block_size,
            to_block_size,
            batch_size,
            from_seq_length,
            to_seq_length,
            seed=self.seed,
            plan_from_length=None,
            plan_num_rand_blocks=None,
            output_attentions=output_attentions,
            graph_edges=graph_edges,
        )

        context_layer = context_layer.contiguous().view(batch_size, from_seq_length, -1)
        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        return outputs

    def _bigbird_block_graph_mask(
        self,
        from_seq_length,
        to_seq_length,
        from_block_size,
        to_block_size,
        graph_edges_batch,
        n_heads,
    ):
        if from_seq_length // from_block_size != to_seq_length // to_block_size:
            raise ValueError("The number of blocks must be the same.")

        num_blocks = from_seq_length // from_block_size
        batch_size = len(graph_edges_batch)

        all_adj_lists = []
        max_conn_overall = 0

        assert all(
            isinstance(edge, (tuple, list)) and len(edge) == 2
            for graph_edges in graph_edges_batch
            for edge in graph_edges
        ), "Each edge must be a tuple (from_idx, to_idx)"

        for graph_edges in graph_edges_batch:
            adj_list = [[] for _ in range(num_blocks)]
            for from_idx, to_idx in graph_edges:
                from_block = from_idx // from_block_size
                to_block = to_idx // to_block_size

                if 0 <= from_block < num_blocks and 0 <= to_block < num_blocks:
                    adj_list[from_block].append(to_block)
            max_conn = max((len(l) for l in adj_list), default=0)
            max_conn_overall = max(max_conn_overall, max_conn)
            all_adj_lists.append(adj_list)

        mask = np.full((batch_size, num_blocks, max_conn_overall), 0, dtype=np.int32)
        for b, adj_list in enumerate(all_adj_lists):
            for i, conn in enumerate(adj_list):
                mask[b, i, : len(conn)] = conn

        # Expand to [batch_size, n_heads, num_blocks, max_conn]
        mask = np.expand_dims(mask, axis=1)  # [batch_size, 1, num_blocks, max_conn]
        mask = np.repeat(
            mask, n_heads, axis=1
        )  # [batch_size, n_heads, num_blocks, max_conn]

        return torch.tensor(mask, dtype=torch.long)

    @staticmethod
    def _create_rand_mask_from_inputs(
        from_blocked_mask,
        to_blocked_mask,
        graph_attn,
        num_attention_heads,
        batch_size,
        from_seq_length,
        from_block_size,
    ):
        """
        Create 3D attention mask using graph-based `graph_attn`.

        Args:
            from_blocked_mask: [batch_size, num_blocks, from_block_size]
            to_blocked_mask:   [batch_size, num_blocks, to_block_size]
            graph_attn:         [batch_size, num_heads, num_blocks, max_num_connections]
        Returns:
            graph_mask:         [batch_size, num_heads, num_blocks, from_block_size, max_num_connections * to_block_size]
        """
        num_blocks = from_seq_length // from_block_size
        max_conn = graph_attn.shape[-1]

        graph_mask = torch.stack(
            [p1[i1.flatten()] for p1, i1 in zip(to_blocked_mask, graph_attn)]
        )
        graph_mask = graph_mask.view(
            batch_size, num_attention_heads, num_blocks, max_conn * from_block_size
        )
        graph_mask = torch.einsum("blq,bhlk->bhlqk", from_blocked_mask, graph_mask)
        return graph_mask

    def graph_bigbird_block_sparse_attention(
        self,
        query_layer,
        key_layer,
        value_layer,
        band_mask,
        from_mask,
        to_mask,
        from_blocked_mask,
        to_blocked_mask,
        n_heads,
        attention_head_size,
        from_block_size,
        to_block_size,
        batch_size,
        from_seq_len,
        to_seq_len,
        seed,
        plan_from_length,
        plan_num_rand_blocks,
        output_attentions,
        graph_edges,
    ):
        # BigBirdPegasus block-sparse attention as suggested in paper

        # ITC:
        #     global tokens: 2 x block_size
        #     window tokens: 3 x block_size
        #     random tokens: num_rand_tokens x block_size

        # ETC:
        #     global tokens: extra_globals_tokens + 2 x block_size
        #     window tokens: 3 x block_size
        #     random tokens: num_rand_tokens x block_size

        # Note:
        #     1) Currently, ETC is not supported.
        #     2) Window size is fixed to 3 blocks & it can be changed only by
        #     changing `block_size`.
        #     3) Number of global blocks are fixed (2 blocks here) & global tokens can be
        #     controlled only by `block_size`.

        # attention is calculated separately for q[0], q[1], q[2:-2], q[-2], q[-1] in order to use special trick of shifting tokens (for calculating sliding attention)
        # hence following code can be divided into 5 parts.

        if from_seq_len // from_block_size != to_seq_len // to_block_size:
            raise ValueError("Error the number of blocks needs to be same!")

        rsqrt_d = 1 / math.sqrt(attention_head_size)
        bsz = batch_size
        attn_mask_penalty = -10000.0

        np.random.seed(seed)

        assert (
            graph_edges is not None
        ), "Graph edges must be provided for graph-based attention."

        graph_attn = self._bigbird_block_graph_mask(
            from_seq_length=from_seq_len,
            to_seq_length=to_seq_len,
            from_block_size=from_block_size,
            to_block_size=to_block_size,
            graph_edges_batch=graph_edges,
            n_heads=n_heads,
        )
        graph_attn = graph_attn.to(query_layer.device)
        # print("-" * 10)
        # print(graph_attn.shape)
        # print("-" * 10)

        graph_mask = self._create_rand_mask_from_inputs(
            from_blocked_mask,
            to_blocked_mask,
            graph_attn,
            n_heads,
            bsz,
            from_seq_len,
            from_block_size,
        )

        max_conn = graph_attn.shape[-1]

        blocked_query_matrix = query_layer.view(
            bsz, n_heads, from_seq_len // from_block_size, from_block_size, -1
        )
        blocked_key_matrix = key_layer.view(
            bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1
        )
        blocked_value_matrix = value_layer.view(
            bsz, n_heads, to_seq_len // to_block_size, to_block_size, -1
        )

        # preparing block for randn attn
        gathered_key = self.torch_gather_b2(blocked_key_matrix, graph_attn)
        gathered_key = gathered_key.view(
            bsz,
            n_heads,
            to_seq_len // to_block_size,
            max_conn * to_block_size,
            -1,
        )  # [bsz, n_heads, to_seq_len//to_block_size-2, max_conn, to_block_size, -1]
        gathered_value = self.torch_gather_b2(blocked_value_matrix, graph_attn)
        gathered_value = gathered_value.view(
            bsz,
            n_heads,
            to_seq_len // to_block_size,
            max_conn * to_block_size,
            -1,
        )  # [bsz, n_heads, to_seq_len//to_block_size-2, max_conn, to_block_size, -1]

        # 1st PART
        # 1st block (global block) attention scores
        # q[0] x (k[0], k[1], k[2], k[3], k[4] .... )

        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, to_seq_len, -1] ==> [bsz, n_heads, from_block_size, to_seq_len]
        first_product = self.torch_bmm_nd_transpose(
            blocked_query_matrix[:, :, 0], key_layer, ndim=4
        )

        first_product = first_product * rsqrt_d
        first_product += (1.0 - to_mask) * attn_mask_penalty
        first_attn_weights = nn.functional.softmax(
            first_product, dim=-1
        )  # [bsz, n_heads, from_block_size, to_seq_len]

        # [bsz, n_heads, from_block_size, to_seq_len] x [bsz, n_heads, to_seq_len, -1] ==> [bsz, n_heads, from_block_size, -1]
        first_context_layer = self.torch_bmm_nd(first_attn_weights, value_layer, ndim=4)
        first_context_layer.unsqueeze_(2)

        # 2nd PART
        # 2nd block attention scores
        # q[1] x (sliding_keys, random_keys, global_keys)
        # sliding key blocks -> 2nd, 3rd blocks
        # global key blocks -> 1st block

        second_key_mat = torch.cat(
            [
                blocked_key_matrix[:, :, 0],
                blocked_key_matrix[:, :, 1],
                blocked_key_matrix[:, :, 2],
                blocked_key_matrix[:, :, -1],
                gathered_key[:, :, 1],
            ],
            dim=2,
        )  # [bsz, n_heads, (4+max_conn)*to_block_size, -1]
        second_value_mat = torch.cat(
            [
                blocked_value_matrix[:, :, 0],
                blocked_value_matrix[:, :, 1],
                blocked_value_matrix[:, :, 2],
                blocked_value_matrix[:, :, -1],
                gathered_value[:, :, 1],
            ],
            dim=2,
        )  # [bsz, n_heads, (4+max_conn)*to_block_size, -1]

        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, (4+max_conn)*to_block_size, -1] ==> [bsz, n_heads, from_block_size, (4+max_conn)*to_block_size]
        second_product = self.torch_bmm_nd_transpose(
            blocked_query_matrix[:, :, 1], second_key_mat, ndim=4
        )
        second_seq_pad = torch.cat(
            [
                to_mask[:, :, :, : 3 * to_block_size],
                to_mask[:, :, :, -to_block_size:],
                to_mask.new_ones([bsz, 1, 1, max_conn * to_block_size]),
            ],
            dim=3,
        )
        second_rand_pad = torch.cat(
            [
                graph_mask.new_ones([bsz, n_heads, from_block_size, 4 * to_block_size]),
                graph_mask[:, :, 1],  # skip BOS token
            ],
            dim=3,
        )

        second_product = second_product * rsqrt_d
        second_product += (
            1.0 - torch.minimum(second_seq_pad, second_rand_pad)
        ) * attn_mask_penalty
        second_attn_weights = nn.functional.softmax(
            second_product, dim=-1
        )  # [bsz, n_heads, from_block_size, (4+max_conn)*to_block_size]

        # [bsz, n_heads, from_block_size, (4+max_conn)*to_block_size] x [bsz, n_heads, (4+max_conn)*to_block_size, -1] ==> [bsz, n_heads, from_block_size, -1]
        second_context_layer = self.torch_bmm_nd(
            second_attn_weights, second_value_mat, ndim=4
        )

        second_context_layer.unsqueeze_(2)

        # 3rd PART
        # Middle blocks attention scores
        # q[-2:2] x (sliding_keys, random_keys, global_keys)
        # sliding attn is calculated using special trick of shifting tokens as discussed in paper
        # random keys are generated by taking random indices as per `rand_attn`
        # global keys -> 1st & last block

        exp_blocked_key_matrix = torch.cat(
            [
                blocked_key_matrix[:, :, 1:-3],
                blocked_key_matrix[:, :, 2:-2],
                blocked_key_matrix[:, :, 3:-1],
            ],
            dim=3,
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, -1]
        exp_blocked_value_matrix = torch.cat(
            [
                blocked_value_matrix[:, :, 1:-3],
                blocked_value_matrix[:, :, 2:-2],
                blocked_value_matrix[:, :, 3:-1],
            ],
            dim=3,
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, -1]
        middle_query_matrix = blocked_query_matrix[:, :, 2:-2]

        # sliding attention scores for q[-2:2]
        # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1] x [b, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, -1]
        inner_band_product = self.torch_bmm_nd_transpose(
            middle_query_matrix, exp_blocked_key_matrix, ndim=5
        )
        #     ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, 3*to_block_size]
        inner_band_product = inner_band_product * rsqrt_d

        # randn attention scores for q[-2:2]
        # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1] x [bsz, n_heads, from_seq_len//from_block_size-4, max_conn*to_block_size, -1]
        rand_band_product = self.torch_bmm_nd_transpose(
            middle_query_matrix, gathered_key[:, :, 2:-2], ndim=5
        )
        #     ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, max_conn*to_block_size]
        rand_band_product = rand_band_product * rsqrt_d

        # Including 1st block (since it's global)
        first_band_product = torch.einsum(
            "bhlqd,bhkd->bhlqk", middle_query_matrix, blocked_key_matrix[:, :, 0]
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1] x [bsz, n_heads, to_block_size, -1] ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, to_block_size]
        first_band_product = first_band_product * rsqrt_d

        # Including last block (since it's global)
        last_band_product = torch.einsum(
            "bhlqd,bhkd->bhlqk", middle_query_matrix, blocked_key_matrix[:, :, -1]
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1] x [bsz, n_heads, to_block_size, -1] ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, to_block_size]
        last_band_product = last_band_product * rsqrt_d

        # masking padded tokens
        inner_band_product += (1.0 - band_mask) * attn_mask_penalty
        first_band_product += (
            1.0 - to_mask[:, :, :, :to_block_size].unsqueeze(3)
        ) * attn_mask_penalty
        last_band_product += (
            1.0 - to_mask[:, :, :, -to_block_size:].unsqueeze(3)
        ) * attn_mask_penalty
        rand_band_product += (1.0 - graph_mask[:, :, 2:-2]) * attn_mask_penalty

        # completing attention scores matrix for all q[-2:2]
        band_product = torch.cat(
            [
                first_band_product,
                inner_band_product,
                rand_band_product,
                last_band_product,
            ],
            dim=-1,
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, (5+max_conn)*to_block_size]

        # safely doing softmax since attention matrix is completed
        attn_weights = nn.functional.softmax(
            band_product, dim=-1
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, (5+max_conn)*to_block_size]

        # contribution of sliding keys
        # [bsz, n_heads, m//from_block_size-4, from_block_size, 3*to_block_size] x [bsz, n_heads, from_seq_len//from_block_size-4, 3*to_block_size, -1]
        context_layer = self.torch_bmm_nd(
            attn_weights[:, :, :, :, to_block_size : 4 * to_block_size],
            exp_blocked_value_matrix,
            ndim=5,
        )
        #     ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]

        # adding contribution of random keys
        # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, max_conn*to_block_size] x [bsz, n_heads, from_seq_len//from_block_size-4, max_conn*to_block_size, -1]
        context_layer += self.torch_bmm_nd(
            attn_weights[:, :, :, :, 4 * to_block_size : -to_block_size],
            gathered_value[:, :, 2:-2],
            ndim=5,
        )
        #     ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]

        # adding contribution of global keys
        context_layer += torch.einsum(
            "bhlqk,bhkd->bhlqd",
            attn_weights[:, :, :, :, :to_block_size],
            blocked_value_matrix[:, :, 0],
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, to_block_size] x [bsz, n_heads, to_block_size, -1] ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]
        context_layer += torch.einsum(
            "bhlqk,bhkd->bhlqd",
            attn_weights[:, :, :, :, -to_block_size:],
            blocked_value_matrix[:, :, -1],
        )  # [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, to_block_size] x [bsz, n_heads, to_block_size, -1] ==> [bsz, n_heads, from_seq_len//from_block_size-4, from_block_size, -1]

        # 4th PART
        # last 2nd token attention scores
        # q[-2] x (sliding_keys, random_keys, global_keys)
        # sliding key blocks -> last 3 blocks
        # global key block -> 1st block
        # random key block -> based on indices stored in `randn_attn`

        second_last_key_mat = torch.cat(
            [
                blocked_key_matrix[:, :, 0],
                blocked_key_matrix[:, :, -3],
                blocked_key_matrix[:, :, -2],
                blocked_key_matrix[:, :, -1],
                gathered_key[:, :, -2],
            ],
            dim=2,
        )  # [bsz, n_heads, (4+n_random_blocks)*to_block_size, -1]
        second_last_value_mat = torch.cat(
            [
                blocked_value_matrix[:, :, 0],
                blocked_value_matrix[:, :, -3],
                blocked_value_matrix[:, :, -2],
                blocked_value_matrix[:, :, -1],
                gathered_value[:, :, -2],
            ],
            dim=2,
        )  # [bsz, n_heads, (4+r)*to_block_size, -1]

        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, (4+max_conn)*to_block_size, -1] ==> [bsz, n_heads, from_block_size, (4+max_conn)*to_block_size]
        second_last_product = self.torch_bmm_nd_transpose(
            blocked_query_matrix[:, :, -2], second_last_key_mat, ndim=4
        )
        second_last_seq_pad = torch.cat(
            [
                to_mask[:, :, :, :to_block_size],
                to_mask[:, :, :, -3 * to_block_size :],
                to_mask.new_ones([bsz, 1, 1, max_conn * to_block_size]),
            ],
            dim=3,
        )
        second_last_rand_pad = torch.cat(
            [
                graph_mask.new_ones([bsz, n_heads, from_block_size, 4 * to_block_size]),
                graph_mask[:, :, -2],  # skip EOS token
            ],
            dim=3,
        )
        second_last_product = second_last_product * rsqrt_d
        second_last_product += (
            1.0 - torch.minimum(second_last_seq_pad, second_last_rand_pad)
        ) * attn_mask_penalty
        second_last_attn_weights = nn.functional.softmax(
            second_last_product, dim=-1
        )  # [bsz, n_heads, from_block_size, (4+max_conn)*to_block_size]

        # [bsz, n_heads, from_block_size, (4+max_conn)*to_block_size] x [bsz, n_heads, (4+max_conn)*to_block_size, -1] ==> [bsz, n_heads, from_block_size, -1]
        second_last_context_layer = self.torch_bmm_nd(
            second_last_attn_weights, second_last_value_mat, ndim=4
        )
        second_last_context_layer.unsqueeze_(2)

        # 5th PART
        # last block (global) attention scores
        # q[-1] x (k[0], k[1], k[2], k[3], .... )

        # [bsz, n_heads, from_block_size, -1] x [bsz, n_heads, to_seq_len, -1] ==> [bsz, n_heads, from_block_size, to_seq_len]
        last_product = self.torch_bmm_nd_transpose(
            blocked_query_matrix[:, :, -1], key_layer, ndim=4
        )
        last_product = last_product * rsqrt_d
        last_product += (1.0 - to_mask) * attn_mask_penalty
        last_attn_weights = nn.functional.softmax(
            last_product, dim=-1
        )  # [bsz, n_heads, from_block_size, n]

        # [bsz, n_heads, from_block_size, to_seq_len] x [bsz, n_heads, to_seq_len, -1] ==> [bsz, n_heads, from_block_size, -1]
        last_context_layer = self.torch_bmm_nd(last_attn_weights, value_layer, ndim=4)
        last_context_layer.unsqueeze_(2)

        # combining representations of all tokens
        context_layer = torch.cat(
            [
                first_context_layer,
                second_context_layer,
                context_layer,
                second_last_context_layer,
                last_context_layer,
            ],
            dim=2,
        )
        context_layer = context_layer.view((bsz, n_heads, from_seq_len, -1)) * from_mask
        context_layer = torch.transpose(context_layer, 1, 2)

        # this is just for visualizing; forward pass doesn't depend on following code
        if output_attentions:
            # TODO(PVP): need to verify if below code is correct
            attention_probs = torch.zeros(
                bsz,
                n_heads,
                from_seq_len,
                to_seq_len,
                dtype=torch.float,
                device=context_layer.device,
            )

            # 1st query block
            # corresponding to `first_context_layer`
            attention_probs[:, :, :from_block_size, :] = (
                first_attn_weights  # all keys global
            )

            # 2nd query block
            # corresponding to `second_context_layer`
            attention_probs[
                :, :, from_block_size : 2 * from_block_size, : 3 * to_block_size
            ] = second_attn_weights[
                :, :, :, : 3 * to_block_size
            ]  # 1st three key blocks (global + sliding)
            attention_probs[
                :, :, from_block_size : 2 * from_block_size, -to_block_size:
            ] = second_attn_weights[
                :, :, :, 3 * to_block_size : 4 * to_block_size
            ]  # last key block (global)
            # graph keys
            for p1, i1, w1 in zip(range(bsz), graph_attn, second_attn_weights):
                # p1, i1, w1 corresponds to batch_dim i.e. following operation is done for each sequence in batch
                for p2, i2, w2 in zip(range(n_heads), i1, w1):
                    # p2, i2, w2 corresponds to head_dim i.e. following operation is done for each heads
                    attn_probs_view = attention_probs.view(
                        bsz,
                        n_heads,
                        from_seq_len // from_block_size,
                        from_block_size,
                        to_seq_len // to_block_size,
                        to_block_size,
                    )
                    right_slice = w2[:, 4 * to_block_size :]
                    attn_probs_view[p1, p2, 1, :, i2[0]] = right_slice.view(
                        from_block_size, max_conn, to_block_size
                    )

            # Middle query blocks
            # corresponding to `context_layer`
            # sliding keys
            for q_idx in range(from_seq_len // from_block_size - 4):
                attn_probs_view = attention_probs.view(
                    bsz,
                    n_heads,
                    from_seq_len // from_block_size,
                    from_block_size,
                    to_seq_len // to_block_size,
                    to_block_size,
                )[:, :, 2:-2, :, 1:-1, :]
                right_slice = attn_weights[
                    :, :, q_idx, :, to_block_size : 4 * to_block_size
                ]
                attn_probs_view[:, :, q_idx, :, q_idx : q_idx + 3, :] = (
                    right_slice.view(bsz, n_heads, from_block_size, 3, to_block_size)
                )  # inner_band_product
            # global keys (corresponding to 1st key block)
            attention_probs[
                :, :, 2 * from_block_size : -2 * from_block_size, :to_block_size
            ] = attn_weights[:, :, :, :, :to_block_size].view(
                bsz, n_heads, -1, to_block_size
            )  # first_band_product
            # global keys (corresponding to last key block)
            attention_probs[
                :, :, 2 * from_block_size : -2 * from_block_size, -to_block_size:
            ] = attn_weights[:, :, :, :, -to_block_size:].view(
                bsz, n_heads, -1, to_block_size
            )  # last_band_product
            # graph keys
            for p1, i1, w1 in zip(range(bsz), graph_attn, attn_weights):
                # p1, i1, w1 corresponds to batch_dim i.e. following operation is done for each sequence in batch
                for p2, i2, w2 in zip(range(n_heads), i1, w1):
                    # p2, i2, w2 corresponds to head_dim i.e. following operation is done for each heads
                    i2 = i2[1:-2]  # adjust to match intended shape
                    for q_idx in range(
                        1, len(i2) - 1
                    ):  # 2 to -2 to pad for BOS and EOS
                        attn_probs_view = attention_probs.view(
                            bsz,
                            n_heads,
                            from_seq_len // from_block_size,
                            from_block_size,
                            to_seq_len // to_block_size,
                            to_block_size,
                        )
                        right_slice = w2[
                            q_idx - 1, :, 4 * to_block_size : -to_block_size
                        ]
                        attn_probs_view[p1, p2, q_idx + 1, :, i2[q_idx]] = (
                            right_slice.view(from_block_size, max_conn, to_block_size)
                        )

            # Second-last query block
            # corresponding to `second_last_context_layer`
            attention_probs[
                :, :, -2 * from_block_size : -from_block_size, :to_block_size
            ] = second_last_attn_weights[
                :, :, :, :to_block_size
            ]  # 1st key block (global)
            attention_probs[
                :, :, -2 * from_block_size : -from_block_size, -3 * to_block_size :
            ] = second_last_attn_weights[
                :, :, :, to_block_size : 4 * to_block_size
            ]  # last three blocks (global + sliding)
            # graph keys
            for p1, i1, w1 in zip(range(bsz), graph_attn, second_last_attn_weights):
                # p1, i1, w1 corresponds to batch_dim i.e. following operation is done for each sequence in batch
                for p2, i2, w2 in zip(range(n_heads), i1, w1):
                    # p2, i2, w2 corresponds to head_dim i.e. following operation is done for each heads
                    attn_probs_view = attention_probs.view(
                        bsz,
                        n_heads,
                        from_seq_len // from_block_size,
                        from_block_size,
                        to_seq_len // to_block_size,
                        to_block_size,
                    )
                    right_slice = w2[:, 4 * to_block_size :]
                    attn_probs_view[p1, p2, -2, :, i2[-1]] = right_slice.view(
                        from_block_size, max_conn, to_block_size
                    )

            # last query block
            # corresponding to `last_context_layer`
            attention_probs[:, :, -from_block_size:, :] = (
                last_attn_weights  # all keys global
            )

        else:
            attention_probs = None

        return context_layer, attention_probs


class GraphBigBirdPegasusEncoderAttention(BigBirdPegasusEncoderAttention):
    def __init__(self, config, seed=None):
        super().__init__(config, seed)

        if self.attention_type == "block_sparse":
            self.self = GraphBigBirdPegasusBlockSparseAttention(config, seed)

    def set_attention_type(self, value: str):
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        # attention type is already correctly set
        if value == self.attention_type:
            return

        if value == "original_full":
            # copy all weights to new full attention class
            attn_weights = BigBirdPegasusSelfAttention(self.config)
        else:
            # copy all weights to new sparse attention class
            attn_weights = GraphBigBirdPegasusBlockSparseAttention(
                self.config, self.seed
            )

        attn_weights.query = self.self.query
        attn_weights.value = self.self.value
        attn_weights.key = self.self.key
        self.self = attn_weights
        self.attention_type = value

        if not self.training:
            self.self.eval()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        band_mask=None,
        from_mask=None,
        to_mask=None,
        from_blocked_mask=None,
        to_blocked_mask=None,
        graph_edges=None,
    ):
        # Expand dims to enable multiplication in the self-attention module
        head_mask = head_mask.reshape(1, -1, 1, 1) if head_mask is not None else None

        if self.attention_type == "original_full":
            self_outputs = self.self(
                hidden_states,
                attention_mask,
                head_mask,
                output_attentions=output_attentions,
            )
        else:
            self_outputs = self.self(
                hidden_states,
                band_mask,
                from_mask,
                to_mask,
                from_blocked_mask,
                to_blocked_mask,
                output_attentions,
                graph_edges,
            )

        attention_output = self.output(self_outputs[0])
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class GraphBigBirdPegasusEncoderLayer(BigBirdPegasusEncoderLayer):
    def __init__(self, config: BigBirdPegasusConfig, seed=None):
        super().__init__(config, seed)
        self.self_attn = GraphBigBirdPegasusEncoderAttention(config, seed=seed)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        band_mask=None,
        from_mask=None,
        to_mask=None,
        from_blocked_mask=None,
        to_blocked_mask=None,
        output_attentions: bool = False,
        graph_edges=None,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        self_attention_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=layer_head_mask,
            output_attentions=output_attentions,
            band_mask=band_mask,
            from_mask=from_mask,
            to_mask=to_mask,
            from_blocked_mask=from_blocked_mask,
            to_blocked_mask=to_blocked_mask,
            graph_edges=graph_edges,
        )
        hidden_states = self_attention_outputs[0]

        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attention_outputs[1],)

        return outputs


class GraphBigBirdPegasusEncoder(BigBirdPegasusEncoder):
    def __init__(
        self, config: BigBirdPegasusConfig, embed_tokens: Optional[nn.Embedding] = None
    ):
        super().__init__(config, embed_tokens)

        self.layers = nn.ModuleList(
            [
                GraphBigBirdPegasusEncoderLayer(config, seed=i)
                for i in range(config.encoder_layers)
            ]
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        graph_edges=None,
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=hidden_states.device)
        attention_mask = attention_mask.long()

        # in order to use block_sparse attention, sequence_length has to be at least bigger than all global attentions: 2 * block_size
        # + sliding tokens: 3 * block_size
        # + random tokens: 2 * num_random_blocks * block_size
        max_tokens_to_attend = (
            5 + 2 * self.config.num_random_blocks
        ) * self.config.block_size
        if (
            self.attention_type == "block_sparse"
            and input_shape[1] <= max_tokens_to_attend
        ):
            # change attention_type from block_sparse to original_full
            sequence_length = input_shape[1]
            logger.warning(
                "Attention type 'block_sparse' is not possible if sequence_length: "
                f"{sequence_length} <= num global tokens: 2 * config.block_size "
                "+ min. num sliding tokens: 3 * config.block_size "
                "+ config.num_random_blocks * config.block_size "
                "+ additional buffer: config.num_random_blocks * config.block_size "
                f"= {max_tokens_to_attend} with config.block_size "
                f"= {self.config.block_size}, config.num_random_blocks "
                f"= {self.config.num_random_blocks}. "
                "Changing attention type to 'original_full'..."
            )
            breakpoint()
            self.set_attention_type("original_full")

        if self.attention_type == "block_sparse":
            padding_len, hidden_states, attention_mask = self._pad_to_block_size(
                hidden_states, attention_mask
            )
        else:
            padding_len = 0

        # expand attention_mask
        if self.attention_type == "original_full":
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(
                attention_mask, inputs_embeds.dtype
            )
            blocked_encoder_mask = band_mask = from_mask = to_mask = None
        elif self.attention_type == "block_sparse":
            blocked_encoder_mask, band_mask, from_mask, to_mask = (
                self.create_masks_for_block_sparse_attn(attention_mask, self.block_size)
            )
            attention_mask = None
        else:
            raise ValueError(
                f"attention_type can either be original_full or block_sparse, but is {self.attention_type}"
            )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != len(self.layers):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    band_mask=band_mask,
                    from_mask=from_mask,
                    to_mask=to_mask,
                    from_blocked_mask=blocked_encoder_mask,
                    to_blocked_mask=blocked_encoder_mask,
                    output_attentions=output_attentions,
                    graph_edges=graph_edges,
                )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layernorm_embedding(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if padding_len > 0:
            # unpad `sequence_output` because the calling function is expecting a length == input_ids.size(1)
            hidden_states = hidden_states[:, :-padding_len]

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class BigBirdPegasusEncoderAttention(nn.Module):
    def __init__(self, config, seed=None):
        super().__init__()
        self.config = config
        self.seed = seed

        self.attention_type = config.attention_type

        if self.attention_type == "original_full":
            self.self = BigBirdPegasusSelfAttention(config)
        elif self.attention_type == "block_sparse":
            self.self = BigBirdPegasusBlockSparseAttention(config, seed)
        else:
            raise ValueError(
                f"attention_type can either be original_full or block_sparse, but is {self.config.attention_type}"
            )

        self.output = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.use_bias
        )

    def set_attention_type(self, value: str):
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        # attention type is already correctly set
        if value == self.attention_type:
            return

        if value == "original_full":
            # copy all weights to new full attention class
            attn_weights = BigBirdPegasusSelfAttention(self.config)
        else:
            # copy all weights to new sparse attention class
            attn_weights = BigBirdPegasusBlockSparseAttention(self.config, self.seed)

        attn_weights.query = self.self.query
        attn_weights.value = self.self.value
        attn_weights.key = self.self.key
        self.self = attn_weights
        self.attention_type = value

        if not self.training:
            self.self.eval()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        band_mask=None,
        from_mask=None,
        to_mask=None,
        from_blocked_mask=None,
        to_blocked_mask=None,
    ):
        # Expand dims to enable multiplication in the self-attention module
        head_mask = head_mask.reshape(1, -1, 1, 1) if head_mask is not None else None

        if self.attention_type == "original_full":
            self_outputs = self.self(
                hidden_states,
                attention_mask,
                head_mask,
                output_attentions=output_attentions,
            )
        else:
            self_outputs = self.self(
                hidden_states,
                band_mask,
                from_mask,
                to_mask,
                from_blocked_mask,
                to_blocked_mask,
                output_attentions,
            )

        attention_output = self.output(self_outputs[0])
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class GraphBigBirdPegasusModel(BigBirdPegasusModel):
    def __init__(self, config: BigBirdPegasusConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.shared = BigBirdPegasusScaledWordEmbedding(
            vocab_size, config.d_model, padding_idx, embed_scale=embed_scale
        )

        self.encoder = GraphBigBirdPegasusEncoder(config, self.shared)
        self.decoder = BigBirdPegasusDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[list[torch.FloatTensor]] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        graph_edges: Optional[list[tuple]] = None,
    ) -> Union[tuple, Seq2SeqModelOutput]:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Provide for translation and summarization training. By default, the model will create this tensor by
            shifting the `input_ids` to the right, following the paper.
        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            If you want to change padding behavior, you should read
            [`modeling_bigbird_pegasus._prepare_decoder_attention_mask`] and modify to your needs. See diagram 1 in
            [the paper](https://huggingface.co/papers/1910.13461) for more information on the default strategy.
        decoder_head_mask (`torch.Tensor` of shape `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        """
        # different to other models, BigBirdPegasus automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                graph_edges=graph_edges,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class GraphBigBirdPegasusForConditionalGeneration(
    BigBirdPegasusForConditionalGeneration
):
    base_model_prefix = "model"
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
        "lm_head.weight",
    ]
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]

    def __init__(self, config: BigBirdPegasusConfig):
        super().__init__(config)
        self.model = GraphBigBirdPegasusModel(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[list[torch.FloatTensor]] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        graph_edges=None,
    ) -> Union[tuple, Seq2SeqLMOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None:
            if use_cache:
                logger.warning(
                    "The `use_cache` argument is changed to `False` since `labels` is provided."
                )
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        assert (
            graph_edges is not None
        ), "Graph edges must be provided when using GraphBigBirdPegasusForConditionalGeneration"

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            graph_edges=graph_edges,
        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
