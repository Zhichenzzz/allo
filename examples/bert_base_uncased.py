from transformers import AutoConfig, BertLMHeadModel, GPT2LMHeadModel, BertLayer
import torch
import allo
import numpy as np

hidden_size = 768
n_heads = 12
batch_size = 2
intermediate_size = 3072
seq_len = 512

config = AutoConfig.from_pretrained("bert-base-uncased")
model = BertLayer(config)
print(model)
example_inputs = [torch.rand(batch_size, seq_len, hidden_size)]
llvm_mod = allo.frontend.from_pytorch(
    model, example_inputs=example_inputs, verbose=True
)
golden = model(*example_inputs)
np_inputs = [x.detach().numpy() for x in example_inputs]
res = llvm_mod(*np_inputs)
np.testing.assert_allclose(res, golden[0].detach().numpy(), atol=1e-3)
