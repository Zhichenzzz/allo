# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module

import os
import numpy as np
from hcl_mlir.ir import (
    Location,
    MemRefType,
    FunctionType,
    TypeAttr,
    FlatSymbolRefAttr,
    ArrayAttr,
    Attribute,
    AffineMapAttr,
    InsertionPoint,
    Module,
)
from hcl_mlir.dialects import (
    hcl as hcl_d,
    func as func_d,
    affine as affine_d,
    memref as memref_d,
    linalg as linalg_d,
)
from hcl_mlir.ir import StringAttr
from hcl_mlir.passmanager import PassManager as mlir_pass_manager
from .ir.transform import create_buffer


def _mlir_lower_pipeline(module, **kwargs):
    hcl_d.loop_transformation(module)
    passes = ["affine-loop-normalize", "cse", "affine-simplify-structures"]
    if "canonicalize" in kwargs:
        passes += ["canonicalize"]
    if "lower_linalg" in kwargs:
        passes += ["convert-linalg-to-affine-loops"]
    pipeline = f'builtin.module(func.func({",".join(passes)}))'
    try:
        with module.context:
            mlir_pass_manager.parse(pipeline).run(module.operation)
        return module
    except Exception as e:
        print("Error: failed to run MLIR lower pipeline, printing module...")
        print(module)
        raise e


def lower_linalg_and_attach_names(module):
    op_names = []
    cnt_loop_nests = 0

    def is_linalg_op(op):
        return isinstance(
            op,
            (
                linalg_d.BatchMatmulOp,
                linalg_d.MatmulOp,
                linalg_d.SoftmaxOp,
                linalg_d.GenericOp,
                linalg_d.FillOp,
                linalg_d.AddOp,
                linalg_d.SubOp,
                linalg_d.DivOp,
                linalg_d.ExpOp,
                linalg_d.LogOp,
                linalg_d.AbsOp,
                linalg_d.TransposeOp,
                linalg_d.BroadcastOp,
            ),
        )

    def annotate_affine_for(op):
        nonlocal cnt_unnamed, cnt_loop_nests
        if isinstance(op, affine_d.AffineForOp):
            if ("loop_name" not in op.attributes) and ("op_name" not in op.attributes):
                if cnt_unnamed == 0:
                    op.attributes["op_name"] = StringAttr.get(op_names[cnt_loop_nests])
                loop_name = f"L_{cnt_unnamed}"
                cnt_unnamed += 1
                op.attributes["loop_name"] = StringAttr.get(loop_name)
            annotate_affine_for(op.body.operations[0])

    with module.context:
        for op in module.body.operations:
            if isinstance(op, func_d.FuncOp):
                func = op
                for op_ in func.entry_block.operations:
                    if is_linalg_op(op_) or isinstance(op_, affine_d.AffineForOp):
                        op_names.append(op_.attributes["op_name"].value)

        _mlir_lower_pipeline(module, lower_linalg=True)
        for op in module.body.operations:
            if isinstance(op, func_d.FuncOp):
                func = op
                for op_ in func.entry_block.operations:
                    cnt_unnamed = 0
                    annotate_affine_for(op_)
                    if isinstance(op_, affine_d.AffineForOp):
                        cnt_loop_nests += 1


def generate_input_output_buffers(top_func, flatten=False):
    with top_func.context, Location.unknown():
        first_op = top_func.entry_block.operations[0]
        new_in_types = []
        for i, arg in enumerate(top_func.arguments):
            create_buffer(arg, f"buf{i}", ip=first_op, flatten=flatten)
            if flatten:
                old_memref = MemRefType(arg.type)
                new_memref = MemRefType.get(
                    (np.prod(old_memref.shape),),
                    old_memref.element_type,
                )
                arg.set_type(new_memref)
                new_in_types.append(new_memref)
            else:
                new_in_types.append(arg.type)
        # find return op
        new_out_types = []
        for op in top_func.entry_block.operations:
            if isinstance(op, func_d.ReturnOp):
                for i, arg in enumerate(op.operands):
                    buf = create_buffer(
                        arg,
                        f"result{i+len(top_func.arguments)}",
                        ip=op,
                        alloc_ip=first_op,
                        flatten=flatten,
                    )
                    # update returnop
                    op.operation.replace_uses_of_with(arg, buf.result)
                    new_out_types.append(buf.result.type)
                break
        func_type = FunctionType.get(new_in_types, new_out_types)
        top_func.attributes["function_type"] = TypeAttr.get(func_type)


def decompose_softmax(module):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_directory, "ir/template/softmax_impl.mlir")
    with open(file_path, "r", encoding="utf-8") as f:
        softmax_module = f.read()
    with module.context, Location.unknown():
        # get all functions from origin module and find the function to replace
        body_op_to_remove = []
        for op in module.body.operations:
            if isinstance(op, func_d.FuncOp) and hasattr(op, "entry_block"):
                # put softmax function into the module
                for body_op in op.entry_block.operations:
                    if isinstance(body_op, linalg_d.SoftmaxOp):
                        generate_softmax(softmax_module, body_op, op)
                        body_op_to_remove.append(body_op)
        # need to erase at the end
        for op in body_op_to_remove:
            op.operation.erase()
        return module


def generate_softmax(module, body_op, init_op):
    op_to_remove = []
    # get softmax function
    softmax_mod = Module.parse(module)
    softmax_func = softmax_mod.body.operations[0]
    softmax_func.attributes["sym_name"] = StringAttr.get(f"softmax_{hash(body_op)}")
    args = softmax_func.arguments
    args[0].set_type(body_op.input.type)
    args[1].set_type(body_op.output.type)
    in_types = [args[0].type, args[1].type]
    out_types = [args[1].type]
    func_type = FunctionType.get(in_types, out_types)
    softmax_func.attributes["function_type"] = TypeAttr.get(func_type)
    softmax_func.move_before(init_op)
    func_d.CallOp(
        [body_op.output.type],
        FlatSymbolRefAttr.get(f"softmax_{hash(body_op)}"),
        [body_op.input, body_op.output],
        ip=InsertionPoint(body_op),
    )
    # Update memref shapes and dtypes in the softmax function
    shape = MemRefType(in_types[0]).shape
    for softmax_op in softmax_func.entry_block.operations:
        if isinstance(softmax_op, memref_d.AllocOp):
            alloc_op = memref_d.AllocOp(
                MemRefType.get(
                    shape[:-1],
                    MemRefType(in_types[0]).element_type,
                ),
                [],
                [],
                ip=InsertionPoint(softmax_op),
            )
            softmax_op.result.replace_all_uses_with(alloc_op.result)
            op_to_remove.append(softmax_op)

        elif isinstance(softmax_op, linalg_d.GenericOp):
            in_str = ", ".join([f"d{i}" for i in range(len(shape))])
            out_str = ", ".join([f"d{i}" for i in range(len(shape) - 1)])

            affine_map_in = AffineMapAttr.parse(f"affine_map<({in_str})->({in_str})>")
            affine_map_out = AffineMapAttr.parse(f"affine_map<({in_str})->({out_str})>")
            iter_types_0 = [Attribute.parse("#linalg.iterator_type<parallel>")] * (
                len(shape) - 1
            ) + [Attribute.parse("#linalg.iterator_type<reduction>")]
            iter_types_1 = [Attribute.parse("#linalg.iterator_type<parallel>")] * len(
                shape
            )
            # Replace indexing_maps and iterator_types of GenericOp in softmax_impl.mlir to match the shape of the input
            if (
                str(softmax_op.attributes["iterator_types"])
                == "[#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]"
            ):
                softmax_op.attributes["indexing_maps"] = ArrayAttr.get(
                    [affine_map_in, affine_map_out]
                )
                softmax_op.attributes["iterator_types"] = ArrayAttr.get(iter_types_0)
            elif (
                str(softmax_op.attributes["iterator_types"])
                == "[#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>]"
            ):
                softmax_op.attributes["indexing_maps"] = ArrayAttr.get(
                    [
                        affine_map_in,
                        affine_map_out,
                        affine_map_in,
                    ]
                )
                softmax_op.attributes["iterator_types"] = ArrayAttr.get(iter_types_1)
            else:
                raise NotImplementedError("Unsupported softmax shape")
    # need to erase at the end
    for op in op_to_remove:
        op.operation.erase()
