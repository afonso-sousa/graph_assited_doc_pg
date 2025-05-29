import torch
from torch import nn
from transformers import LongT5ForConditionalGeneration, LongT5PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import \
    BaseModelOutputWithPastAndCrossAttentions
from transformers.models.longt5.modeling_longt5 import (LongT5Block,
                                                        LongT5LayerNorm)
from transformers.utils import is_torchdynamo_compiling, logging

logger = logging.get_logger(__name__)


class CustomLongT5ForConditionalGeneration(LongT5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    class LongT5Stack(LongT5PreTrainedModel):
        def __init__(self, config, embed_tokens=None):
            super().__init__(config)

            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
            if embed_tokens is not None:
                self.embed_tokens.weight = embed_tokens.weight
            self.is_decoder = config.is_decoder

            self.block = nn.ModuleList(
                [
                    LongT5Block(config, has_relative_attention_bias=bool(i == 0), layer_idx=i)
                    for i in range(config.num_layers)
                ]
            )
            self.final_layer_norm = LongT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
            self.dropout = nn.Dropout(config.dropout_rate)

            self.gradient_checkpointing = False

            # Initialize weights and apply final processing
            self.post_init()

        # Copied from transformers.models.t5.modeling_t5.T5Stack.get_input_embeddings
        def get_input_embeddings(self):
            return self.embed_tokens

        # Copied from transformers.models.t5.modeling_t5.T5Stack.set_input_embeddings
        def set_input_embeddings(self, new_embeddings):
            self.embed_tokens = new_embeddings

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            cache_position=None,
        ):
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if input_ids is not None and inputs_embeds is not None:
                err_msg_prefix = "decoder_" if self.is_decoder else ""
                raise ValueError(
                    f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
                )
            elif input_ids is not None:
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                err_msg_prefix = "decoder_" if self.is_decoder else ""
                raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning_once(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

            if inputs_embeds is None:
                assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
                inputs_embeds = self.embed_tokens(input_ids)

            batch_size, seq_length = input_shape

            # initialize past_key_values
            return_legacy_cache = False
            return_self_attention_cache = False
            if self.is_decoder and (use_cache or past_key_values is not None):
                if isinstance(past_key_values, Cache) and not isinstance(past_key_values, EncoderDecoderCache):
                    return_self_attention_cache = True
                    past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
                elif not isinstance(past_key_values, EncoderDecoderCache):
                    return_legacy_cache = True
                    logger.warning_once(
                        "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.48.0. "
                        "You should pass an instance of `EncoderDecoderCache` instead, e.g. "
                        "`past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`."
                    )
                    past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)
                elif past_key_values is None:
                    past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
            elif not self.is_decoder:
                # do not pass cache object down the line for encoder stack
                # it messes indexing later in decoder-stack because cache object is modified in-place
                past_key_values = None

            past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
            if cache_position is None:
                cache_position = torch.arange(
                    past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
                )

            if attention_mask is None and not is_torchdynamo_compiling():
                # required mask seq length can be calculated via length of past
                mask_seq_length = past_key_values_length + seq_length
                attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

            if self.is_decoder:
                causal_mask = self._update_causal_mask(
                    attention_mask,
                    inputs_embeds,
                    cache_position,
                    past_key_values.self_attention_cache if past_key_values is not None else None,
                    output_attentions,
                )
            # We use local attention in encoder self-attention, otherwise standard self & cross attentions are used
            elif self.config.encoder_attention_type == "graph_local":
                causal_mask = _get_graph_attention_mask(graph_edges, seq_length, inputs_embeds.device)
            else:  # we need to use both local attention mask and standard extended mask for transient-global attention
                causal_mask = attention_mask

            # If a 2D or 3D attention mask is provided for the cross-attention
            # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
                encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
                if encoder_attention_mask is None:
                    encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
                encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_extended_attention_mask = None

            # Prepare head mask if needed
            head_mask = self.get_head_mask(head_mask, self.config.num_layers)
            cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
            all_hidden_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None
            all_cross_attentions = () if (output_attentions and self.is_decoder) else None
            position_bias = None
            encoder_decoder_position_bias = None

            hidden_states = self.dropout(inputs_embeds)

            for i, layer_module in enumerate(self.block):
                layer_head_mask = head_mask[i]
                cross_attn_layer_head_mask = cross_attn_head_mask[i]

                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer_module.forward,
                        hidden_states,
                        causal_mask,
                        position_bias,
                        encoder_hidden_states,
                        encoder_extended_attention_mask,
                        encoder_decoder_position_bias,
                        layer_head_mask,
                        cross_attn_layer_head_mask,
                        None,  # past_key_value is always None with gradient checkpointing
                        use_cache,
                        output_attentions,
                        return_dict,
                        cache_position,
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_bias=position_bias,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_extended_attention_mask,
                        encoder_decoder_position_bias=encoder_decoder_position_bias,
                        layer_head_mask=layer_head_mask,
                        cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                        past_key_value=past_key_values,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        return_dict=return_dict,
                        cache_position=cache_position,
                    )

                # layer_outputs is a tuple with:
                # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
                if use_cache is False:
                    layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

                hidden_states, next_decoder_cache = layer_outputs[:2]

                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
                # (cross-attention position bias), (cross-attention weights)
                position_bias = layer_outputs[2]
                if self.is_decoder and encoder_hidden_states is not None:
                    encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[3],)
                    if self.is_decoder:
                        all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.dropout(hidden_states)

            # Add last layer
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            next_cache = next_decoder_cache if use_cache else None
            if return_self_attention_cache:
                next_cache = past_key_values.self_attention_cache
            if return_legacy_cache:
                next_cache = past_key_values.to_legacy_cache()

            if not return_dict:
                return tuple(
                    v
                    for v in [
                        hidden_states,
                        next_cache,
                        all_hidden_states,
                        all_attentions,
                        all_cross_attentions,
                    ]
                    if v is not None
                )
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
                cross_attentions=all_cross_attentions,
            )
        
    def _get_graph_attention_mask(batch_graph_edges, batch_size, seq_length, device):
        mask = torch.zeros((batch_size, seq_length, seq_length), device=device)
        for i, edges in enumerate(batch_graph_edges):
            for source, target in edges:
                mask[i, source, target] = 1
            mask[i].fill_diagonal_(1)
        return mask