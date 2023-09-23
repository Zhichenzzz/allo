# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import torch
import math
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, AutoModelForCausalLM
from transformers.pytorch_utils import Conv1D

class GPT2(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layers):
        super(GPT2, self).__init__()
        self.transformer = nn.ModuleList(
            [TransformerBlock(n_embd, n_head, n_embd * 4) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x, kvcache=None):
        if not kvcache:
            kvcache = [None] * len(self.transformer)
        new_kvcache = []
        for block, kvcache_block in zip(self.transformer, kvcache):
            x, updated_cache = block(x, kvcache_block)
            new_kvcache.append(updated_cache)

        x = self.ln_f(x)
        x = self.lm_head(x)
        return x, new_kvcache


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, num_heads, ffn_hidden_dim):
        super(TransformerBlock, self).__init__()

        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, num_heads)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, ffn_hidden_dim, n_embd)

    def forward(self, x, kvcache=None):
        ln_1 = self.ln_1(x)
        attn_output, kvcache_updated = self.attn(ln_1, kvcache)
        out1 = x + attn_output

        ln_2 = self.ln_2(out1)
        ffn_output = self.mlp(ln_2)
        out2 = out1 + ffn_output

        return out2, kvcache_updated


class MLP(nn.Module):
    def __init__(self, n_embd, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.c_fc = Conv1D(hidden_dim, n_embd)
        self.c_proj = Conv1D(output_dim,hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = n_embd // num_heads
        self.c_attn = Conv1D(n_embd*3, n_embd)
        self.c_proj = Conv1D(n_embd, n_embd)


    def mask(self, x, kvcache=None):
        if kvcache:
            causal_mask = 0
        else:
            ones = torch.ones(x.size(1), x.size(1))
            causal_mask = (1 - torch.tril(ones)) * -1e10
        return causal_mask

    def split_heads(self, x):
        # x: (batch_size, seq_len, hidden_size)
        new_shape = x.shape[:-1] + (self.num_heads, -1)
        x = x.view(new_shape)
        # output: (bs, head, seq, hs // head)
        return x.permute(0, 2, 1, 3)

    def scaled_dot_product(self, q, k, v, x, kvcache=None):
        # (bs, head, seq, hs // head)
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        ) + self.mask(x, kvcache)
        # (bs, head, seq, seq)
        attn_probs = F.softmax(attn_score, dim=-1)
        # (bs, head, seq, hs // head)
        attn = torch.matmul(attn_probs, v)
        return attn

    def forward(self, x, kvcache=None):
        # qkv layers
        q, k, v = torch.chunk(self.c_attn(x), 3, dim=-1)
        # split heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        if kvcache:
            new_k, new_v = k, v
            old_k, old_v = kvcache
            k = torch.cat([old_k, new_k], dim=-2)
            v = torch.cat([old_v, new_v], dim=-2)
        # print(q.shape, k.shape, v.shape)
        current_cache = [k, v]

        # core attention
        output = self.scaled_dot_product(q, k, v, x, kvcache)
        # output: (bs, seq, head, hs // head)
        output = output.permute(0, 2, 1, 3)
        output = output.reshape(output.shape[0], output.shape[1], -1)
        output = self.c_proj(output)
        return output, current_cache


class Embedding(nn.Module):
    def __init__(self, vocab_size, n_position, n_embd):
        super(Embedding, self).__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_position, n_embd)

    def forward(self, input_ids, kvcache=None):
        # kvcache = None
        if kvcache is None:
            wpe_out = self.wpe(torch.arange(len(input_ids)))
            input_tensor = input_ids
        else:
            wpe_out = self.wpe(torch.tensor(len(input_ids) - 1))
            input_tensor = [input_ids[-1]]
        input_tensor = torch.tensor(input_tensor, dtype=torch.long)
        print(input_tensor)
        x = self.wte(input_tensor) + wpe_out
        # print(x)
        return x.unsqueeze(0)


def generate(inputs, model, embeddings, n_tokens_to_generate, kvcache=None):
    with torch.no_grad():
        for _ in tqdm(
            range(n_tokens_to_generate), "generating"
        ):  # auto-regressive decode loop
            x = embeddings(inputs, kvcache)
            logits, kvcache = model(x, kvcache)  # model forward pass
            next_id = torch.argmax(logits[0, -1, :]).item()  # greedy sampling
            # print(logits)
            inputs.append(next_id)  # append prediction to input
        return inputs  # only return generated ids


vocab_size = 50257
n_embd = 768
n_head = 12
n_layers = 12
n_position = 1024
batch_size = 2


tokenizer = GPT2Tokenizer.from_pretrained("gpt2", is_split_into_words=True)
input_text = "Hello, my dog is"
in_tokens = tokenizer.encode(input_text)

GPT_model = AutoModelForCausalLM.from_pretrained("gpt2")

module = GPT2(vocab_size, n_embd, n_head, n_layers).eval()
embeddings = Embedding(vocab_size, n_position, n_embd).eval()

dic_from = GPT_model.state_dict()
dic_to = module.state_dict()
dic_emb = embeddings.state_dict()


for k, weight in dic_from.items():
    if "wte" in k or "wpe" in k:
        dic_emb[k[12:]] = weight
    if k.replace("h.", "") in dic_to:
        dic_to[k.replace("h.", "")] = weight

module.load_state_dict(dic_to)
embeddings.load_state_dict(dic_emb)

out = generate(in_tokens, module, embeddings, 10)
generated_text = tokenizer.decode(out)
full_string = "".join(generated_text)
print(full_string)
