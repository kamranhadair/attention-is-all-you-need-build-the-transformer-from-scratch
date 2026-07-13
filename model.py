"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    """
    Build a token-to-id vocabulary from whitespace-tokenized sentences.

    Special tokens receive ids 0..len(specials)-1 in the given order.
    Remaining unique tokens receive ids starting at len(specials) in order
    of first appearance. Tokens already in specials are not re-added.
    """
    vocab = {tok: i for i, tok in enumerate(specials)}
    next_id = len(specials)

    for sentence in sentences:
        for token in sentence.split():
            if token not in vocab:
                vocab[token] = next_id
                next_id += 1

    return vocab

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    """
    Invert a token-to-id mapping to produce an id-to-token mapping.

    Args:
        token_to_id (dict[str, int]): Mapping from token strings to integer ids.

    Returns:
        dict[int, str]: Mapping from integer ids back to token strings.
    """
    return {id_: token for token, id_ in token_to_id.items()}

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    """
    Convert a whitespace-tokenized sentence into a list of integer token ids.

    Args:
        sentence (str): Input sentence, tokens separated by whitespace.
        token_to_id (dict[str, int]): Mapping from token strings to ids.
                                      Must contain an entry for `unk_token`.
        unk_token (str, optional): The token string used for unknown tokens.
                                   Defaults to '<unk>'.

    Returns:
        list[int]: List of ids corresponding to the tokens in order.
                   Out-of-vocabulary tokens are mapped to the id of `unk_token`.
    """
    # Look up the id of the special unknown token.
    unk_id = token_to_id[unk_token]
    return [token_to_id.get(token, unk_id) for token in sentence.split()]

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    """
    Convert a list of token ids back to their string tokens.

    Args:
        ids (list[int]): Sequence of token ids.
        id_to_token (dict[int, str]): Mapping from ids to token strings.
                                      Must contain all ids in `ids`.

    Returns:
        list[str]: List of token strings in the same order as `ids`.
    """
    return [id_to_token[i] for i in ids]

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    """
    Resize a single id sequence to exactly `max_len`.

    - If the sequence is shorter than `max_len`, pad on the right with `pad_id`.
    - If the sequence is longer than `max_len`, truncate from the right.
    - If the length already equals `max_len`, return the sequence unchanged.

    Args:
        ids (list[int]): Original token id sequence.
        max_len (int): Desired length of the output sequence.
        pad_id (int): Token id used for padding.

    Returns:
        list[int]: Sequence of exactly `max_len` token ids.
    """
    if len(ids) < max_len:
        return ids + [pad_id] * (max_len - len(ids))
    elif len(ids) > max_len:
        return ids[:max_len]
    else:
        return ids

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """
    Convert a list of equal-length padded id sequences into a 2D LongTensor batch.

    Args:
        padded_sequences (list[list[int]]): B sequences, each of length L.

    Returns:
        torch.LongTensor: Tensor of shape (B, L) with dtype torch.long.
    """
    return torch.tensor(padded_sequences, dtype=torch.long)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """
    Scale token embeddings by the square root of the model dimension.

    This replicates the preprocessing step from the original Transformer paper,
    where embeddings are multiplied by √d_model before adding positional encodings.

    Args:
        embeddings (torch.Tensor): Token embeddings, shape (..., d_model).
        d_model (int): Model dimension (last dimension of embeddings).

    Returns:
        torch.Tensor: Scaled embeddings, same shape and dtype as input.
    """
    return embeddings * math.sqrt(d_model)

# Step 8 - compute_positional_div_term
import torch
import math

def compute_positional_div_term(d_model):
    """
    Compute the frequency divisor vector for sinusoidal positional encoding.

    Returns a 1D tensor of length d_model // 2 containing values
    10000^{-2i/d_model} for i = 0, 1, ..., d_model//2 - 1.
    These values are used as the argument multiplier inside sin and cos:
        PE(pos, 2i)   = sin(pos * div_term[i])
        PE(pos, 2i+1) = cos(pos * div_term[i])

    Args:
        d_model (int): Model dimension, must be even.

    Returns:
        torch.FloatTensor: Shape (d_model // 2,), dtype float32.
    """
    # Even indices: 0, 2, 4, ..., d_model-2
    even_indices = torch.arange(0, d_model, 2, dtype=torch.float32)
    # exp( even_indices * (-log(10000) / d_model) )
    div_term = torch.exp(even_indices * (-math.log(10000.0) / d_model))
    return div_term

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """
    Build a column vector of position indices for sinusoidal positional encoding.

    Args:
        max_len (int): Maximum sequence length (number of positions).

    Returns:
        torch.FloatTensor: Shape (max_len, 1), containing floats 0.0 .. max_len-1.0.
    """
    positions = torch.arange(max_len, dtype=torch.float32)
    return positions.view(-1, 1)

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """
    Fill the even-indexed columns of a positional encoding matrix with sine values.

    Args:
        pe (torch.Tensor): Positional encoding matrix of shape (L, D). Modified in-place.
        position (torch.Tensor): Column vector of shape (L, 1) containing position indices.
        div_term (torch.Tensor): Frequency divisor vector of shape (D/2,).

    Returns:
        torch.Tensor: The same tensor `pe` after updating its even columns.
    """
    # Broadcast: (L,1) * (D/2,) -> (L, D/2)
    angles = position * div_term
    # Assign sin values to even columns (0, 2, 4, ...)
    pe[:, 0::2] = torch.sin(angles)
    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    """
    Fill the odd-indexed columns of a positional encoding matrix with cosine values.

    Args:
        pe (torch.Tensor): Positional encoding matrix of shape (L, D). Modified in-place.
        position (torch.Tensor): Column vector of shape (L, 1) containing position indices.
        div_term (torch.Tensor): Frequency divisor vector of shape (D/2,).

    Returns:
        torch.Tensor: The same tensor `pe` after updating its odd columns.
    """
    # Broadcast: (L,1) * (D/2,) -> (L, D/2)
    angles = position * div_term
    # Assign cos values to odd columns (1, 3, 5, ...)
    pe[:, 1::2] = torch.cos(angles)
    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """
    Construct the sinusoidal positional encoding matrix as in the original Transformer.

    Args:
        max_len (int): Maximum sequence length.
        d_model (int): Model dimension (must be even).

    Returns:
        torch.FloatTensor: Tensor of shape (max_len, d_model) containing the positional encodings.
    """
    # Start with a zero matrix of the required shape
    pe = torch.zeros(max_len, d_model, dtype=torch.float32)

    # Frequency divisors for each feature pair
    div_term = compute_positional_div_term(d_model)

    # Column vector of position indices
    position = build_position_index_column(max_len)

    # Fill even columns with sine, odd columns with cosine
    pe = fill_even_indices_with_sin(pe, position, div_term)
    pe = fill_odd_indices_with_cos(pe, position, div_term)

    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embeddings, positional_encoding):
    """
    Add sinusoidal positional encodings to a batch of token embeddings.

    Args:
        embeddings (torch.Tensor): Token embeddings of shape (B, L, d_model).
        positional_encoding (torch.Tensor): Precomputed positional encoding table
                                            of shape (max_len, d_model), with max_len >= L.

    Returns:
        torch.Tensor: Embeddings with positional encodings added, shape (B, L, d_model).
    """
    L = embeddings.shape[1]
    # Take only the rows needed for the current sequence length
    pe_slice = positional_encoding[:L]   # shape (L, d_model)
    # Broadcasting adds the same positional pattern to every batch element
    return embeddings + pe_slice

# Step 14 - build_padding_mask
import torch

def build_padding_mask(ids, pad_id):
    """
    Create a padding mask from a batch of token ids.

    Args:
        ids (torch.Tensor): LongTensor of token ids with shape (B, L).
        pad_id (int): The token id designated for padding.

    Returns:
        torch.Tensor: A boolean tensor of shape (B, 1, 1, L) where True indicates
                      a real content token and False indicates a padding token.
    """
    # Create a boolean mask where True means "not padding"
    mask = (ids != pad_id)
    
    # Reshape to (B, 1, 1, L) for broadcasting with multi-head attention scores (B, H, L_q, L_k)
    return mask.unsqueeze(1).unsqueeze(2)

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """
    Create a lower-triangular boolean causal mask.

    Args:
        seq_len (int): The sequence length.

    Returns:
        torch.Tensor: A boolean tensor of shape (1, 1, seq_len, seq_len) where
                      True indicates that position i can attend to position j,
                      and False indicates it cannot (i.e., j is in the future).
    """
    # torch.tril creates a lower-triangular matrix (including the diagonal)
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    
    # Reshape to (1, 1, seq_len, seq_len) for broadcasting with attention scores (B, H, L, L)
    return mask.unsqueeze(0).unsqueeze(0)

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    """
    Fuse a padding mask and a causal mask into a single decoder mask.

    Args:
        padding_mask (torch.BoolTensor): Shape (B, 1, 1, L), True for real tokens.
        causal_mask (torch.BoolTensor): Shape (1, 1, L, L), True for allowed causal attention.

    Returns:
        torch.BoolTensor: Shape (B, 1, L, L), True only at positions allowed by BOTH masks.
    """
    # Logical AND automatically broadcasts (B, 1, 1, L) & (1, 1, L, L) -> (B, 1, L, L)
    return padding_mask & causal_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """
    Compute the raw attention score matrix via matrix multiplication.

    Args:
        query (torch.Tensor): Tensor of shape (..., Lq, d_k).
        key (torch.Tensor): Tensor of shape (..., Lk, d_k).

    Returns:
        torch.Tensor: Raw attention scores of shape (..., Lq, Lk).
    """
    # Transpose the last two dimensions of the key tensor to get (..., d_k, Lk)
    # then perform batched matrix multiplication with query
    return torch.matmul(query, key.transpose(-2, -1))

# Step 18 - scale_attention_scores
import math
import torch

def scale_attention_scores(scores, d_k):
    """
    Scale raw attention scores by the square root of the per-head dimension d_k.

    Args:
        scores (torch.Tensor): Raw attention scores of shape (..., Lq, Lk).
        d_k (int): The dimensionality of the keys/queries per head.

    Returns:
        torch.Tensor: Scaled attention scores of the same shape (..., Lq, Lk).
    """
    return scores / math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """
    Apply a boolean mask to attention scores, replacing blocked positions with -inf.

    Args:
        scores (torch.Tensor): Attention scores of shape (..., Lq, Lk).
        mask (torch.Tensor): Boolean mask broadcastable to scores shape. 
                             True = keep, False = block.

    Returns:
        torch.Tensor: A new tensor where blocked positions are -inf and kept positions 
                      retain their original score.
    """
    # Invert the mask so True corresponds to positions we want to fill with -inf
    return scores.masked_fill(~mask, float('-inf'))

# Step 20 - softmax_attention_weights
import torch
import torch.nn.functional as F

def softmax_attention_weights(scores):
    """
    Convert masked attention scores into attention weights using softmax.

    Args:
        scores (torch.Tensor): Attention scores of shape (..., Lq, Lk), 
                               potentially containing -inf for masked positions.

    Returns:
        torch.Tensor: Attention weights of shape (..., Lq, Lk). Non-negative entries 
                      that sum to 1 along the last axis. Fully masked rows yield all zeros.
    """
    # Apply standard softmax along the last dimension (key positions)
    attn_weights = F.softmax(scores, dim=-1)
    
    # Handle rows that are entirely masked (-inf). 
    # Softmax of all -inf results in 0/0 = NaN. We replace these NaNs with 0.0.
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
    
    return attn_weights

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """
    Combine attention weights with the value tensor to produce context vectors.

    Args:
        attention_weights (torch.Tensor): Attention weights of shape (..., Lq, Lk).
        value (torch.Tensor): Value tensor of shape (..., Lk, d_v).

    Returns:
        torch.Tensor: Context tensor of shape (..., Lq, d_v).
    """
    # Batched matrix multiplication over the last two dimensions
    return torch.matmul(attention_weights, value)

# Step 22 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Compute Scaled Dot-Product Attention.
    
    Args:
        query (torch.Tensor): Query tensor of shape (..., Lq, d_k).
        key (torch.Tensor): Key tensor of shape (..., Lk, d_k).
        value (torch.Tensor): Value tensor of shape (..., Lk, d_v).
        mask (torch.Tensor, optional): Boolean mask broadcastable to 
                                        (..., Lq, Lk). True = keep, False = block.
                                        Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 
            - context: Output tensor of shape (..., Lq, d_v).
            - attention_weights: Attention weights of shape (..., Lq, Lk).
    """
    # Extract the per-head dimension d_k from the last axis of the query
    d_k = query.shape[-1]
    
    # 1. Compute raw attention scores: (..., Lq, d_k) x (..., d_k, Lk) -> (..., Lq, Lk)
    scores = compute_raw_attention_scores(query, key)
    
    # 2. Scale the scores by 1 / sqrt(d_k)
    scores = scale_attention_scores(scores, d_k)
    
    # 3. Apply mask if provided (replaces blocked positions with -inf)
    if mask is not None:
        scores = mask_attention_scores_with_neg_inf(scores, mask)
        
    # 4. Apply softmax over the key axis to get attention weights
    attn_weights = softmax_attention_weights(scores)
    
    # 5. Mix the value vectors according to the weights
    context = apply_attention_weights_to_values(attn_weights, value)
    
    return context, attn_weights

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(x, num_heads):
    """
    Reshape the last feature dimension into (num_heads, d_k).

    Args:
        x (torch.Tensor): Input tensor of shape (B, L, d_model).
        num_heads (int): Number of attention heads.

    Returns:
        torch.Tensor: Reshaped tensor of shape (B, L, num_heads, d_model // num_heads).
    """
    B, L, d_model = x.shape
    d_k = d_model // num_heads
    return x.reshape(B, L, num_heads, d_k)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(x):
    """
    Swap the head and sequence axes for multi-head attention.

    Args:
        x (torch.Tensor): Head-split tensor of shape (B, L, num_heads, d_k).

    Returns:
        torch.Tensor: Transposed tensor of shape (B, num_heads, L, d_k).
    """
    # Swap the sequence dimension (axis 1) and the head dimension (axis 2)
    return x.transpose(1, 2)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(x):
    """
    Merge the head dimension back into the feature dimension, undoing the multi-head split.

    Args:
        x (torch.Tensor): Tensor of shape (B, num_heads, L, d_k).

    Returns:
        torch.Tensor: Contiguous tensor of shape (B, L, num_heads * d_k).
    """
    # 1. Swap the head and sequence axes: (B, H, L, d_k) -> (B, L, H, d_k)
    x = x.transpose(1, 2)
    
    # 2. Make the tensor contiguous in memory so the reshape reads elements correctly
    x = x.contiguous()
    
    # 3. Flatten the last two dimensions: (B, L, H, d_k) -> (B, L, H * d_k)
    B, L, H, d_k = x.shape
    return x.view(B, L, H * d_k)

# Step 26 - apply_linear_projection
import torch

def apply_linear_projection(x, weight, bias):
    """
    Apply a standard linear projection y = xW^T + b.

    Args:
        x (torch.Tensor): Input tensor of shape (..., in_features).
        weight (torch.Tensor): Weight matrix of shape (out_features, in_features).
        bias (torch.Tensor | None): Bias vector of shape (out_features,), or None.

    Returns:
        torch.Tensor: Projected tensor of shape (..., out_features).
    """
    # Matrix multiply input by the transpose of the weight matrix
    output = x @ weight.T
    
    # Add bias if it is provided
    if bias is not None:
        output = output + bias
        
    return output

# Step 27 - project_to_query_key_value
import torch

def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    """
    Project the input tensor into query, key, and value tensors.

    Args:
        x (torch.Tensor): Input tensor of shape (B, L, d_model).
        w_q (torch.Tensor): Weight matrix for query projection, shape (d_model, d_model).
        b_q (torch.Tensor): Bias vector for query projection, shape (d_model,).
        w_k (torch.Tensor): Weight matrix for key projection, shape (d_model, d_model).
        b_k (torch.Tensor): Bias vector for key projection, shape (d_model,).
        w_v (torch.Tensor): Weight matrix for value projection, shape (d_model, d_model).
        b_v (torch.Tensor): Bias vector for value projection, shape (d_model,).

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            Query, Key, and Value tensors, each of shape (B, L, d_model).
    """
    # Apply independent linear projections to the original input x
    q = apply_linear_projection(x, w_q, b_q)
    k = apply_linear_projection(x, w_k, b_k)
    v = apply_linear_projection(x, w_v, b_v)
    
    return q, k, v

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    """
    Reshape and transpose the Q, K, V tensors into multi-head form.

    Args:
        q (torch.Tensor): Query tensor of shape (B, L, d_model).
        k (torch.Tensor): Key tensor of shape (B, L, d_model).
        v (torch.Tensor): Value tensor of shape (B, L, d_model).
        num_heads (int): Number of attention heads.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            Head-split tensors q_h, k_h, v_h, each of shape (B, num_heads, L, d_k).
    """
    # 1. Split the last dimension into (num_heads, d_k)
    q_split = split_last_dim_into_heads(q, num_heads)
    k_split = split_last_dim_into_heads(k, num_heads)
    v_split = split_last_dim_into_heads(v, num_heads)
    
    # 2. Transpose to move the head axis before the sequence axis
    q_h = transpose_heads_before_sequence(q_split)
    k_h = transpose_heads_before_sequence(k_split)
    v_h = transpose_heads_before_sequence(v_split)
    
    return q_h, k_h, v_h

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    """
    Run scaled dot-product attention on multi-head formatted tensors.

    Args:
        q_h (torch.Tensor): Multi-head query tensor of shape (B, num_heads, Lq, d_k).
        k_h (torch.Tensor): Multi-head key tensor of shape (B, num_heads, Lk, d_k).
        v_h (torch.Tensor): Multi-head value tensor of shape (B, num_heads, Lk, d_v).
        mask (torch.Tensor, optional): Boolean mask broadcastable to (B, num_heads, Lq, Lk).
                                       True = keep, False = block. Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 
            - context: Output tensor of shape (B, num_heads, Lq, d_v).
            - attention_weights: Attention weights of shape (B, num_heads, Lq, Lk).
    """
    # Reuse the standard scaled dot-product attention; matmul and softmax 
    # automatically broadcast over the batch and head dimensions.
    return scaled_dot_product_attention(q_h, k_h, v_h, mask)

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    """
    Merge multi-head context vectors back to the model dimension and apply the output projection.

    Args:
        context (torch.Tensor): Per-head context tensor of shape (B, num_heads, L, d_k).
        w_o (torch.Tensor): Output projection weight matrix of shape (d_model, d_model).
        b_o (torch.Tensor): Output projection bias vector of shape (d_model,).

    Returns:
        torch.Tensor: Projected output tensor of shape (B, L, d_model).
    """
    # 1. Concatenate heads: (B, H, L, d_k) -> (B, L, d_model)
    merged_context = merge_heads_back_to_model_dim(context)
    
    # 2. Apply the output linear projection
    output = apply_linear_projection(merged_context, w_o, b_o)
    
    return output

# Step 31 - assemble_multi_head_attention_forward
import torch

def assemble_multi_head_attention_forward(
    query, key, value, W_q, W_k, W_v, W_o, num_heads, 
    mask=None, b_q=None, b_k=None, b_v=None, b_o=None
):
    """
    Perform the complete multi-head attention forward pass.

    Args:
        query (torch.Tensor): Query input tensor of shape (B, Tq, D).
        key (torch.Tensor): Key input tensor of shape (B, Tk, D).
        value (torch.Tensor): Value input tensor of shape (B, Tk, D).
        W_q (torch.Tensor): Query weight matrix of shape (D, D).
        W_k (torch.Tensor): Key weight matrix of shape (D, D).
        W_v (torch.Tensor): Value weight matrix of shape (D, D).
        W_o (torch.Tensor): Output projection weight matrix of shape (D, D).
        num_heads (int): Number of attention heads.
        mask (torch.Tensor, optional): Attention mask broadcastable to (B, num_heads, Tq, Tk).
        b_q (torch.Tensor, optional): Query bias of shape (D,).
        b_k (torch.Tensor, optional): Key bias of shape (D,).
        b_v (torch.Tensor, optional): Value bias of shape (D,).
        b_o (torch.Tensor, optional): Output projection bias of shape (D,).

    Returns:
        torch.Tensor: Output tensor of shape (B, Tq, D).
    """
    # 1. Project inputs to Query, Key, and Value independently
    q = apply_linear_projection(query, W_q, b_q)
    k = apply_linear_projection(key, W_k, b_k)
    v = apply_linear_projection(value, W_v, b_v)
    
    # 2. Reshape into per-head views: (B, T, D) -> (B, num_heads, T, d_k)
    q_h, k_h, v_h = split_qkv_into_heads(q, k, v, num_heads)
    
    # 3. Run scaled dot-product attention across all heads
    context_h, _ = multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask)
    
    # 4. Merge heads back together and apply the output projection
    output = merge_heads_and_project_output(context_h, W_o, b_o)
    
    return output

# Step 32 - apply_ffn_first_linear_and_relu
import torch


def apply_ffn_first_linear_and_relu(x, w1, b1):
    """First half of the position-wise feed-forward network.

    Projects from d_model up to d_ff and applies ReLU, independently
    at every (batch, seq) position.

    x:  (B, T, d_model)
    w1: (d_model, d_ff)  -- applied directly, no transpose needed
    b1: (d_ff,)
    returns: (B, T, d_ff)
    """
    return torch.relu(x @ w1 + b1)

# Step 33 - apply_ffn_second_linear
import torch


def apply_ffn_second_linear(hidden, w2, b2):
    """Second half of the position-wise feed-forward network.

    Projects from d_ff back down to d_model, applied independently
    at every (batch, seq) position.

    hidden: (B, T, d_ff)
    w2:     (d_ff, d_model)  -- applied directly, no transpose needed
    b2:     (d_model,)
    returns: (B, T, d_model)
    """
    return hidden @ w2 + b2

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    """Position-wise feed-forward sublayer: two linears with ReLU between.

    FFN(x) = ReLU(x @ w1 + b1) @ w2 + b2, applied independently at
    every (batch, seq) position.

    x:  (B, T, d_model)
    w1: (d_model, d_ff), b1: (d_ff,)
    w2: (d_ff, d_model), b2: (d_model,)
    returns: (B, T, d_model)
    """
    hidden = apply_ffn_first_linear_and_relu(x, w1, b1)
    output = apply_ffn_second_linear(hidden, w2, b2)
    return output

# Step 35 - compute_layer_norm_mean_and_variance
import torch


def compute_layer_norm_mean_and_variance(x):
    """Per-feature mean and (biased) variance over the last dimension.

    x: (..., d_model)
    returns: (mean, variance), each of shape (..., 1) so they broadcast
             back against x.
    """
    mean = x.mean(dim=-1, keepdim=True)
    variance = x.var(dim=-1, unbiased=False, keepdim=True)
    return mean, variance

# Step 36 - normalize_and_scale_with_gamma_beta
import torch


def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    """Complete layer normalization: standardize then apply learned affine.

    x: (..., d_model)
    gamma, beta: (d_model,) -- broadcast across all leading axes
    eps: added inside the square root for numerical stability
    returns: (..., d_model)
    """
    mean, variance = compute_layer_norm_mean_and_variance(x)
    x_hat = (x - mean) / torch.sqrt(variance + eps)
    return gamma * x_hat + beta

# Step 37 - apply_residual_add_and_norm (not yet solved)
# TODO: implement

# Step 38 - apply_dropout_with_keep_mask (not yet solved)
# TODO: implement

# Step 39 - encoder_layer_self_attention_sublayer (not yet solved)
# TODO: implement

# Step 40 - encoder_layer_feed_forward_sublayer (not yet solved)
# TODO: implement

# Step 41 - assemble_encoder_layer (not yet solved)
# TODO: implement

# Step 42 - stack_encoder_layers (not yet solved)
# TODO: implement

# Step 43 - decoder_layer_masked_self_attention_sublayer (not yet solved)
# TODO: implement

# Step 44 - decoder_layer_cross_attention_sublayer (not yet solved)
# TODO: implement

# Step 45 - decoder_layer_feed_forward_sublayer (not yet solved)
# TODO: implement

# Step 46 - assemble_decoder_layer (not yet solved)
# TODO: implement

# Step 47 - stack_decoder_layers (not yet solved)
# TODO: implement

# Step 48 - apply_final_output_projection (not yet solved)
# TODO: implement

# Step 49 - tie_output_projection_to_token_embeddings (not yet solved)
# TODO: implement

# Step 50 - apply_log_softmax_over_vocab (not yet solved)
# TODO: implement

# Step 51 - run_transformer_forward (not yet solved)
# TODO: implement

# Step 52 - init_encoder_layer_parameters (not yet solved)
# TODO: implement

# Step 53 - init_decoder_layer_parameters (not yet solved)
# TODO: implement

# Step 54 - init_embedding_and_projection_parameters (not yet solved)
# TODO: implement

# Step 55 - collect_model_parameters_into_list (not yet solved)
# TODO: implement

# Step 56 - shift_targets_right_with_start_token (not yet solved)
# TODO: implement

# Step 57 - compute_noam_learning_rate (not yet solved)
# TODO: implement

# Step 58 - build_uniform_smoothing_distribution (not yet solved)
# TODO: implement

# Step 59 - set_confidence_on_gold_tokens (not yet solved)
# TODO: implement

# Step 60 - zero_pad_column_and_pad_token_rows (not yet solved)
# TODO: implement

# Step 61 - compute_label_smoothed_kl_loss (not yet solved)
# TODO: implement

# Step 62 - average_loss_over_non_pad_tokens (not yet solved)
# TODO: implement

# Step 63 - compute_token_accuracy_ignoring_pad (not yet solved)
# TODO: implement

# Step 64 - initialize_adam_optimizer_state (not yet solved)
# TODO: implement

# Step 65 - update_adam_first_moment (not yet solved)
# TODO: implement

# Step 66 - update_adam_second_moment (not yet solved)
# TODO: implement

# Step 67 - apply_adam_bias_correction (not yet solved)
# TODO: implement

# Step 69 - apply_adam_step_to_all_parameters (not yet solved)
# TODO: implement

# Step 70 - zero_all_parameter_gradients (not yet solved)
# TODO: implement

# Step 71 - compute_batch_training_loss (not yet solved)
# TODO: implement

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

