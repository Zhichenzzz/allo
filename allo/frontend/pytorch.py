# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import operator
import inspect
import math
import warnings

try:
    import torch
    from torch import fx
    from torch.nn import functional as F
    from torch.fx.graph_module import GraphModule
    from torch.fx.passes.shape_prop import ShapeProp, TensorMetadata
    from .tracer import AlloTracer
except ImportError:
    pass

from .. import dsl
from ..ir import types
from ..customize import customize


def from_pytorch(model, example_inputs, verbose=False):
    sig = inspect.signature(model.forward)
    input_names = [
        p.name for i, p in enumerate(sig.parameters.values()) if i < len(example_inputs)
    ]
    concrete_args = {
        p.name: p.default for p in sig.parameters.values() if p.name not in input_names
    }
    args = []
    args += example_inputs
    for item in concrete_args.values():
        args.append(item)

    tracer = AlloTracer(model, concrete_args=concrete_args)
    graph = tracer.trace()
    name = (
        model.__class__.__name__
        if isinstance(model, torch.nn.Module)
        else model.__name__
    )
    gm = GraphModule(tracer.root, graph, name)
    ShapeProp(gm).propagate(*args)
    if verbose:
        print(gm.graph)
    global_vars = {}
    for pymod in (types,):
        global_vars.update({item[0]: item[1] for item in inspect.getmembers(pymod)})
    global_vars.update({"dsl": dsl})
    for name, param in gm.named_parameters():
        new_name = "g_" + name.replace(".", "_")
        global_vars.update({new_name: param.detach().numpy()})
    for name, buffer in gm.named_buffers():
        new_name = "g_" + name.replace(".", "_")
        global_vars.update({new_name: buffer.detach().numpy()})
    builder = TorchBuilder(gm, example_inputs)
    code = builder.build()
    s = customize(code, verbose=verbose, global_vars=global_vars)
    mod = s.build()
    if verbose:
        print(s.module)
    return mod


def get_var_name(node):
    return node.name if isinstance(node, fx.Node) else node


def slice_to_string(slice_obj):
    start_str = str(slice_obj.start) if slice_obj.start is not None else ""
    stop_str = str(slice_obj.stop) if slice_obj.stop is not None else ""
    step_str = str(slice_obj.step) if slice_obj.step is not None else ""

    if step_str:
        return f"{start_str}:{stop_str}:{step_str}"
    elif stop_str:
        return f"{start_str}:{stop_str}"
    elif start_str:
        return f"{start_str}:"
    else:
        return ":"


class TorchBuilder:
    def __init__(self, gm, example_inputs):
        self.gm = gm
        self.code = []
        self.input_names = []
        self.input_args = []
        self.input_shapes = []
        self.example_inputs = example_inputs
        self.named_params = gm.named_parameters()
        self.named_buffers = gm.named_buffers()
        self.output = []

    def build(self):
        for node in self.gm.graph.nodes:
            self(node)
        for i, x in enumerate(self.example_inputs):
            if isinstance(x, torch.Tensor):
                self.input_shapes.append(x.shape)
                self.input_args.append(self.input_names[i])
            elif isinstance(x, tuple) or isinstance(x, list):
                input_name = self.input_names[i]
                for num, item in enumerate(x):
                    if isinstance(item, torch.Tensor):
                        self.input_shapes.append(item.shape)
                        self.input_args.append(f"{input_name}_{num}")
                    else:
                        raise NotImplementedError("Unsupported input type")
            elif isinstance(x, int):
                self.input_shapes.append(None)
                self.input_args.append(self.input_names[i])
        args = [
            f"{name}: float32[{', '.join([str(s) for s in shape])}]"
            if shape
            else f"{name}: int32"
            for name, shape in zip(self.input_args, self.input_shapes)
        ]
        # inputs
        res = f"def forward({', '.join(args)})".format()
        # outputs
        res += f" -> ({', '.join(self.output)}):\n"
        # global parameters
        if self.named_params:
            for name, param in self.named_params:
                new_name = name.replace(".", "_")
                res += f"  {new_name}: float32[{', '.join([str(s) for s in param.shape])}] = g_{new_name}\n"
        if self.named_buffers:
            for name, buffer in self.named_buffers:
                new_name = name.replace(".", "_")
                res += f"  {new_name}: float32[{', '.join([str(s) for s in buffer.shape])}] = g_{new_name}\n"
        # function body
        for line in self.code:
            res += f"  {line}\n"
        return res

    def __call__(self, node):
        method = getattr(self, "build_" + node.op)
        ret = method(node)
        if ret:
            self.code.append(ret)
        return ret

    def get_module(self, name):
        return dict(self.gm.named_modules())[name]

    def build_placeholder(self, node):
        self.input_names.append(node.name)

    def build_getattr(self, node):
        pass

    def build_get_attr(self, node):
        for name, buffer in self.gm.named_buffers():
            if node.target == name:
                return f"{node.name} = dsl.copy({node.target})"

    def build_call_module(self, node):
        module = self.get_module(node.target)
        op = {
            torch.nn.Linear: "linear",
            torch.nn.Dropout: "identity",
            torch.nn.GELU: "gelu",
            torch.nn.LayerNorm: "layernorm",
        }.get(type(module), None)
        if op is None:
            raise NotImplementedError("Unsupported module")
        if op == "linear":
            bias = True if module.bias is not None else None
            return getattr(self, "build_linear")(node, bias)
        return getattr(self, f"build_{op}")(node)

    def build_call_function(self, node):
        opcls = {
            operator.add: "add",
            operator.sub: "sub",
            operator.mul: "mul",
            operator.truediv: "div",
            operator.getitem: "getitem",
            torch.matmul: "matmul",
            torch.ones: "ones",
            torch.zeros: "zeros",
            math.sqrt: "sqrt",
            F.softmax: "softmax",
            F.linear: "linear",
            F.gelu: "gelu",
            F.relu: "relu",
            F.dropout: "identity",
            torch.tril: "tril",
            torch.cat: "concat",
            torch.narrow: "narrow",
        }.get(node.target)
        # Only nodes with shape need to be built.
        return (
            getattr(self, f"build_{opcls}")(node)
            if "tensor_meta" in node.meta
            or (opcls == "add" and node.meta["type"] is not torch.Size)
            else None
        )

    def build_call_method(self, node):
        if node.target == "contiguous":
            return self.build_identity(node)
        # Only nodes with shape need to be built.
        return (
            getattr(self, f"build_{node.target}")(node)
            if "tensor_meta" in node.meta
            else None
        )

    def build_output(self, node):
        for output in node.meta["tensor_meta"]:
            if isinstance(output, TensorMetadata):
                output_tensor_meta = output
                shape = str(list(output_tensor_meta.shape))
                dtype = str(output_tensor_meta.dtype)[6:]
                self.output.append(dtype + shape)
            elif isinstance(output, tuple) or isinstance(output, list):
                for item in output:
                    if isinstance(item, TensorMetadata):
                        output_tensor_meta = item
                        shape = str(list(output_tensor_meta.shape))
                        dtype = str(output_tensor_meta.dtype)[6:]
                        self.output.append(dtype + shape)
                    else:
                        raise NotImplementedError("Unsupported output type")
            elif isinstance(output, dict):
                output_tensor_meta = list(output.values())[0]
                shape = str(list(output_tensor_meta.shape))
                dtype = str(output_tensor_meta.dtype)[6:]
                self.output.append(dtype + shape)
            else:
                raise NotImplementedError("Unsupported output type")

        name = get_var_name(node.args[0])
        return_name = (
            str(name)
            .replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
        )
        return f"return ({return_name})"

    def build_add(self, node):
        lhs = get_var_name(node.args[0])
        rhs = get_var_name(node.args[1])
        if node.meta["type"] is int:
            return f"{node.name}: int32 ={lhs} + {rhs}"
        elif node.meta["type"] is float:
            return f"{node.name}: float32 ={lhs} + {rhs}"
        return f"{node.name} = {lhs} + {rhs}"

    def build_sub(self, node):
        lhs = get_var_name(node.args[0])
        rhs = get_var_name(node.args[1])
        return f"{node.name} = {lhs} - {rhs}"

    def build_mul(self, node):
        lhs = get_var_name(node.args[0])
        rhs = get_var_name(node.args[1])
        return f"{node.name} = {lhs} * {rhs}"

    def build_matmul(self, node):
        lhs = get_var_name(node.args[0])
        rhs = get_var_name(node.args[1])
        return f"{node.name} = dsl.matmul({lhs}, {rhs})"

    def build_div(self, node):
        lhs = get_var_name(node.args[0])
        rhs = get_var_name(node.args[1])
        return f"{node.name} = {lhs} / {rhs}"

    def build_softmax(self, node):
        if node.kwargs.get("dim") != -1:
            raise NotImplementedError("Only support softmax on the last dimension")
        inp = get_var_name(node.args[0])
        return f"{node.name} = dsl.softmax({inp})"

    def build_relu(self, node):
        inp = get_var_name(node.args[0])
        return f"{node.name} = dsl.relu({inp})"

    def build_linear(self, node, bias):
        target_name = node.target.replace(".", "_")
        inp = get_var_name(node.args[0])
        weight = get_var_name(target_name + "_weight")
        if bias:
            bias = get_var_name(target_name + "_bias")
            return f"{node.name} = dsl.linear({inp}, {weight}, {bias})"
        return f"{node.name} = dsl.linear({inp}, {weight})"

    def build_gelu(self, node):
        inp = get_var_name(node.args[0])
        return f"{node.name} = dsl.gelu({inp})"

    def build_layernorm(self, node):
        target_name = node.target.replace(".", "_")
        inp = get_var_name(node.args[0])
        weight = get_var_name(target_name + "_weight")
        bias = get_var_name(target_name + "_bias")
        return f"{node.name} = dsl.layernorm({inp}, {weight}, {bias})"

    def build_view(self, node):
        inp = get_var_name(node.args[0])
        shape = tuple(node.meta["tensor_meta"].shape)
        return f"{node.name} = dsl.view({inp}, {shape})"

    def build_reshape(self, node):
        return self.build_view(node)

    def build_permute(self, node):
        inp = get_var_name(node.args[0])
        permutation = node.args[1:]
        return f"{node.name} = dsl.transpose({inp}, {permutation})"

    def build_transpose(self, node):
        # PyTorch only supports transposing two dimensions,
        # https://pytorch.org/docs/stable/generated/torch.transpose.html
        inp = get_var_name(node.args[0])
        shape_len = len(node.meta["tensor_meta"].shape)
        sorted_args = sorted(
            [
                node.args[1] if node.args[1] >= 0 else node.args[1] + shape_len,
                node.args[2] if node.args[2] >= 0 else node.args[2] + shape_len,
            ]
        )
        permutation = list(range(shape_len))
        permutation[sorted_args[0]] = sorted_args[1]
        permutation[sorted_args[1]] = sorted_args[0]
        return f"{node.name} = dsl.transpose({inp}, {tuple(permutation)})"

    def build_identity(self, node):
        inp = get_var_name(node.args[0])
        return f"{node.name} = {inp}"

    def build_ones(self, node):
        shape = tuple(node.meta["tensor_meta"].shape)
        dtype = node.meta["tensor_meta"].dtype
        if str(dtype).startswith("torch."):
            dtype = str(dtype)[6:]
        return f"{node.name} = dsl.ones({shape}, dtype={dtype})"

    def build_zeros(self, node):
        shape = tuple(node.meta["tensor_meta"].shape)
        dtype = node.meta["tensor_meta"].dtype
        if str(dtype).startswith("torch."):
            dtype = str(dtype)[6:]
        return f"{node.name} = dsl.zeros({shape}, dtype={dtype})"

    def build_tril(self, node):
        inp = get_var_name(node.args[0])
        return f"{node.name} = dsl.tril({inp})"

    def build_concat(self, node):
        shape_len = len(node.meta["tensor_meta"].shape)
        tensor_A = get_var_name(node.args[0][0])
        tensor_B = get_var_name(node.args[0][1])
        dim = node.kwargs["dim"] + (node.kwargs["dim"] < 0) * shape_len
        return f"{node.name} = dsl.concat({tensor_A}, {tensor_B}, axis={dim})"

    def build_getitem(self, node):
        if isinstance(node.args[1], tuple):
            args =[]
            for slice_item in node.args[1]:
                if isinstance(slice_item,slice ):
                    args.append(slice_to_string(slice_item))
                else:
                    raise NotImplementedError("Unsupported slice type")
            slc = ", ".join(args)
            return f"{node.name} = {get_var_name(node.args[0])}[{slc}]"               
        elif isinstance(node.args[1], int):
            inp = get_var_name(node.args[0])
            index = node.args[1]
            return f"{node.name} = dsl.copy({inp}_{index})"
