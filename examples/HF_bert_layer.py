# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import torch
import allo
import pytest
import numpy as np
from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertLayer, BertEncoder

config = AutoConfig.from_pretrained("bert-base-uncased")
bert_layer_module = BertLayer(config).eval()
bert_encoder_module = BertEncoder(config).eval()


@pytest.mark.parametrize("module", [bert_layer_module, bert_encoder_module])
def test_bert_module(module):
    batch_size = 2
    seq_len = 512
    hidden_size = 768

    example_inputs = [torch.rand(batch_size, seq_len, hidden_size)]
    input_names = [
        "input_ids",
    ]
    sig = inspect.signature(module.forward)
    concrete_args = {
        p.name: p.default
        for p in sig.parameters.values()
        if p.name not in input_names and p.default is not inspect.Parameter.empty
    }

    llvm_mod = allo.frontend.from_pytorch(
        module,
        example_inputs=example_inputs,
        concrete_args=concrete_args,
        verbose=False,
    )

    golden = module(*example_inputs)
    np_inputs = [x.detach().numpy() for x in example_inputs]
    res = llvm_mod(*np_inputs)
    np.testing.assert_allclose(res, golden[0].detach().numpy(), atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])