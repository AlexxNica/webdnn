import itertools
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.tensor_util import MakeNdarray

from webdnn import ConstantVariable
from webdnn.frontend.constraints import AxisVar, unify_order
from webdnn.frontend.tensorflow.converter import TensorFlowConverter
from webdnn.graph.axis import Axis
from webdnn.graph.operators.concat import Concat
from webdnn.graph.operators.depth2space import Depth2Space
from webdnn.graph.operators.reshape import Reshape
from webdnn.graph.operators.slice import Slice
from webdnn.graph.operators.space2depth import Space2Depth
from webdnn.graph.operators.tile import Tile
from webdnn.graph.operators.transpose import Transpose
from webdnn.graph.operators.zero_padding_2d import ZeroPadding2D
from webdnn.graph.order import Order, OrderNHWC
from webdnn.graph.placeholder import Placeholder
from webdnn.util import console
from webdnn.util.assertion import UnexpectedAndPleaseReportError
from webdnn.util.misc import mul


@TensorFlowConverter.register_handler("BatchMatrixBandPart")
def batch_matrix_band_part_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("BatchMatrixDiag")
def batch_matrix_diag_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("BatchMatrixDiagPart")
def batch_matrix_diag_part_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("BatchMatrixSetDiag")
def batch_matrix_set_diag_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("BatchToSpace")
def batch_to_space_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("BatchToSpaceND")
def batch_to_space_nd_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("Bitcast")
def bitcast_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("CheckNumerics")
def check_numerics_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("Concat")
def concat_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("ConcatOffset")
def concat_offset_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("ConcatV2")
def concat_v2_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    xs = [converter.get_variable(tf_tensor) for tf_tensor in tf_op.inputs]

    axis = xs.pop()
    # TODO
    assert isinstance(axis, ConstantVariable), "[TensorFlowConverter] Dynamic axis concatenation is not supported yet."
    axis = xs[0].order.axes[int(axis.data.flatten()[0])]

    for x0, x1 in itertools.permutations(xs, 2):
        unify_order(x0.order, x1.order)

    y, = Concat(None, axis=axis)(*xs)
    converter.set_variable(tf_op.outputs[0], y)


@TensorFlowConverter.register_handler("Const")
def const_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    # FIXME: should output ConstantVariable?
    tensor = tf_op.outputs[0]
    shape = [Placeholder() if dim.value is None else dim.value for dim in tensor.shape.dims]
    value = MakeNdarray(tf_op.get_attr("value"))

    if len(shape) == 0:
        # Scalar variable
        # noinspection PyTypeChecker
        variable = ConstantVariable(value.reshape([1]), Order([AxisVar()]))

    else:
        # noinspection PyTypeChecker
        variable = ConstantVariable(value, Order([AxisVar() for _ in shape]))

    converter.set_variable(tensor, variable)


@TensorFlowConverter.register_handler("DepthToSpace")
def depth_to_space_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    x = converter.get_variable(tf_op.inputs[0])
    unify_order(x.order, OrderNHWC)

    y, = Depth2Space(None, r=tf_op.get_attr("block_size"))(x)
    converter.set_variable(tf_op.outputs[0], y)


@TensorFlowConverter.register_handler("Dequantize")
def dequantize_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("Diag")
def diag_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("DiagPart")
def diag_part_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("EditDistance")
def edit_distance_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("ExpandDims")
def expand_dims_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    x = converter.get_variable(tf_op.inputs[0])
    dim = converter.get_variable(tf_op.inputs[1])

    if not isinstance(dim, ConstantVariable):
        raise NotImplementedError("[TensorFlowConverter] Operator 'ExpandDims' with dynamic dimension is not supported.")

    dim = dim.data.astype(np.int32).flatten()[0]
    new_shape = list(x.shape)
    new_shape.insert(dim, 1)
    new_axes = list(x.order.axes)
    new_axes.insert(dim, AxisVar())
    y, = Reshape(None, in_order=x.order, out_order=Order(new_axes), out_shape=new_shape)(x)
    converter.set_variable(tf_op.outputs[0], y)


@TensorFlowConverter.register_handler("ExtractImagePatches")
def extract_image_patches_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("FakeQuantWithMinMaxArgs")
def fake_quant_with_min_max_args_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("FakeQuantWithMinMaxArgsGradient")
def fake_quant_with_min_max_args_gradient_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("FakeQuantWithMinMaxVars")
def fake_quant_with_min_max_vars_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("FakeQuantWithMinMaxVarsGradient")
def fake_quant_with_min_max_vars_gradient_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("FakeQuantWithMinMaxVarsPerChannel")
def fake_quant_with_min_max_vars_per_channel_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("FakeQuantWithMinMaxVarsPerChannelGradient")
def fake_quant_with_min_max_vars_per_channel_gradient_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("Fill")
def fill_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("Gather")
def gather_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("GatherNd")
def gather_nd_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("GatherV2")
def gather_v2_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("Identity")
def identity_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    converter.set_variable(tf_op.outputs[0], converter.get_variable(tf_op.inputs[0]))


@TensorFlowConverter.register_handler("ImmutableConst")
def immutable_const_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("InvertPermutation")
def invert_permutation_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("ListDiff")
def list_diff_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("MatrixBandPart")
def matrix_band_part_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("MatrixDiag")
def matrix_diag_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("MatrixDiagPart")
def matrix_diag_part_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("MatrixSetDiag")
def matrix_set_diag_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("MirrorPad")
def mirror_pad_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("MirrorPadGrad")
def mirror_pad_grad_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("OneHot")
def one_hot_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("OnesLike")
def ones_like_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("Pack")
def pack_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    xs = [converter.get_variable(tf_tensor) for tf_tensor in tf_op.inputs]
    i_axis = tf_op.get_attr("axis")

    for x0, x1 in itertools.permutations(xs, 2):
        unify_order(x0.order, x1.order)
        assert x0.shape == x1.shape

    if i_axis == 0:
        concat_axis = xs[0].order.axes[0]

    else:
        concat_axis = xs[0].order.axes[i_axis - 1]

    y, = Concat(None, axis=concat_axis)(*xs)

    new_axes = list(y.order.axes)
    new_axes.insert(i_axis, AxisVar())
    new_order = Order(new_axes)

    if i_axis == 0:
        new_shape = list(y.shape)
        new_shape[i_axis] //= len(xs)
        new_shape.insert(i_axis, len(xs))

    else:
        new_shape = list(y.shape)
        new_shape[i_axis - 1] //= len(xs)
        new_shape.insert(i_axis, len(xs))

    y, = Reshape(None, in_order=y.order, out_order=new_order, out_shape=new_shape)(y)
    converter.set_variable(tf_op.outputs[0], y)


@TensorFlowConverter.register_handler("Pad")
def pad_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    # Zero padding
    # FIXME: currently, determining padding from shape of input / output. Originally, determining by inputs[1] is correct.

    in_var = converter.get_variable(tf_op.inputs[0])
    unify_order(in_var.order, OrderNHWC)  # FIXME: assuming input order as NHWC
    out_tf_var = tf_op.outputs[0]
    # calculate output shape from out_tf_var.shape and in_var.shape
    # ZeroPadding2D operator only accepts padding for H and W axes.
    padding = [0, 0]
    for dim in range(in_var.ndim):
        in_size = in_var.shape[dim]
        out_size = out_tf_var.shape.dims[dim].value
        assert isinstance(in_size, int), "[TensorFlowConverter] Pad: Placeholder for input shape is not supported yet."
        assert isinstance(out_size, int), "[TensorFlowConverter] Pad: Placeholder for output shape is not supported yet."
        axis = in_var.order.axes[dim]
        if axis in [Axis.H, Axis.W]:
            assert (out_size - in_size % 2) != 0, "[TensorFlowConverter] Pad: Uneven padding is not supported yet."
            pad_size = (out_size - in_size) // 2
            if axis == Axis.H:
                padding[0] = pad_size
            elif axis == Axis.W:
                padding[1] = pad_size
        else:
            assert out_size == in_size, "[TensorFlowConverter] Pad: padding for axis other than H and W is not supported yet."
    out_var, = ZeroPadding2D(None, padding=tuple(padding))(in_var)
    converter.set_variable(out_tf_var, out_var)


@TensorFlowConverter.register_handler("PadV2")
def pad_v2_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("ParallelConcat")
def parallel_concat_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("Placeholder")
def placeholder_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    shape = [Placeholder() if dim.size() is -1 else dim.size() for dim in tf_op.get_attr("shape").dim]
    assert all(Placeholder.check_resolved(s) for s in shape), UnexpectedAndPleaseReportError(shape)

    # noinspection PyTypeChecker
    converter.set_variable(tf_op.outputs[0], ConstantVariable(shape, Order([AxisVar for _ in shape])))


@TensorFlowConverter.register_handler("PlaceholderV2")
def placeholder_v2_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("PlaceholderWithDefault")
def placeholder_with_default_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("PreventGradient")
def prevent_gradient_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("QuantizeAndDequantize")
def quantize_and_dequantize_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("QuantizeAndDequantizeV2")
def quantize_and_dequantize_v2_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("QuantizeAndDequantizeV3")
def quantize_and_dequantize_v3_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("QuantizeV2")
def quantize_v2_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("QuantizedConcat")
def quantized_concat_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("QuantizedInstanceNorm")
def quantized_instance_norm_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("QuantizedReshape")
def quantized_reshape_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("Rank")
def rank_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    x = converter.get_variable(tf_op.inputs[0])

    # noinspection PyTypeChecker
    y = ConstantVariable(np.array([x.ndim]), Order([AxisVar()]))
    converter.set_variable(tf_op.outputs[0], y)


@TensorFlowConverter.register_handler("RefIdentity")
def ref_identity_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("Reshape")
def reshape_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    tf_x = tf_op.inputs[0]
    x = converter.get_variable(tf_x)
    shape = converter.get_variable(tf_op.inputs[1])

    assert isinstance(shape, ConstantVariable), NotImplementedError("[TensorFlowConverter] Dynamic shape reshaping is not supported yet.")

    shape = shape.data.astype(int).flatten().tolist()  # type: List[int]
    if -1 in shape:
        i = shape.index(-1)
        shape.remove(-1)
        shape.insert(i, x.size // mul(shape))

    # noinspection PyTypeChecker
    y, = Reshape(None, in_order=x.order, out_order=Order([AxisVar() for _ in shape]), out_shape=shape)(x)
    converter.set_variable(tf_op.outputs[0], y)


@TensorFlowConverter.register_handler("ResourceStridedSliceAssign")
def resource_strided_slice_assign_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("Reverse")
def reverse_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("ReverseSequence")
def reverse_sequence_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("ReverseV2")
def reverse_v2_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("ScatterNd")
def scatter_nd_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("ScatterNdNonAliasingAdd")
def scatter_nd_non_aliasing_add_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("Shape")
def shape_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    x = converter.get_variable(tf_op.inputs[0])
    assert all(Placeholder.check_resolved(s) for s in x.shape), "[TensorFlowConverter] op 'Shape' with dynamic shape is not supported yet. "

    # noinspection PyTypeChecker
    y = ConstantVariable(np.array(x.shape), Order([AxisVar()]))
    converter.set_variable(tf_op.outputs[0], y)


@TensorFlowConverter.register_handler("ShapeN")
def shape_n_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("Size")
def size_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("Slice")
def slice_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    x = converter.get_variable(tf_op.inputs[0])
    begin = converter.get_variable(tf_op.inputs[1])
    size = converter.get_variable(tf_op.inputs[2])

    assert isinstance(begin, ConstantVariable), \
        NotImplementedError("[TensorFlowConverter] Slicing tensor with dynamic index is not supported yet:"
                            f"  type(begin)={type(begin)}")

    assert isinstance(size, ConstantVariable), \
        NotImplementedError("[TensorFlowConverter] Slicing tensor with dynamic index is not supported yet:"
                            f"  type(size)={type(size)}")

    begin = begin.data.astype(int).flatten().tolist()  # type: List[int]
    size = size.data.astype(int).flatten().tolist()  # type: List[int]

    y, = Slice(None, begin=begin, end=begin + size, stride=[1 for _ in begin])(x)

    converter.set_variable(tf_op.outputs[0], y)


@TensorFlowConverter.register_handler("SpaceToBatch")
def space_to_batch_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("SpaceToBatchND")
def space_to_batch_nd_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("SpaceToDepth")
def space_to_depth_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    x = converter.get_variable(tf_op.inputs[0])
    unify_order(x.order, OrderNHWC)

    y, = Space2Depth(None, r=tf_op.get_attr("block_size"))(x)
    converter.set_variable(tf_op.outputs[0], y)


@TensorFlowConverter.register_handler("Split")
def split_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("SplitV")
def split_v_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("Squeeze")
def squeeze_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    squeeze_dims = tf_op.get_attr("squeeze_dims")  # type: List[int]

    in_var = converter.get_variable(tf_op.inputs[0])
    in_var_shape = in_var.shape
    out_var_shape = []  # type: List[int]
    out_var_order = []  # type: List[Axis]
    for dim in range(len(in_var_shape)):
        if dim in squeeze_dims:
            assert in_var_shape[dim] == 1, f"[TensorFlowConverter] {tf_op.type}: dimension to be squeezed have to be 1."
        else:
            out_var_shape.append(in_var_shape[dim])
            out_var_order.append(in_var.order.axes[dim])

    out_var, = Reshape(None, in_order=in_var.order, out_order=Order(out_var_order), out_shape=out_var_shape)(in_var)
    out_tf_var = tf_op.outputs[0]
    converter.set_variable(out_tf_var, out_var)


@TensorFlowConverter.register_handler("StopGradient")
def stop_gradient_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    console.warning("[TensorFlowConverter] StopGradient is ignored.")
    converter.set_variable(tf_op.outputs[0], converter.get_variable(tf_op.inputs[0]))


@TensorFlowConverter.register_handler("StridedSlice")
def strided_slice_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    x = converter.get_variable(tf_op.inputs[0])
    begin = converter.get_variable(tf_op.inputs[1])
    end = converter.get_variable(tf_op.inputs[2])
    stride = converter.get_variable(tf_op.inputs[3])

    assert isinstance(begin, ConstantVariable), \
        NotImplementedError("[TensorFlowConverter] Slicing tensor with dynamic index is not supported yet:"
                            f"  type(begin)={type(begin)}")

    assert isinstance(end, ConstantVariable), \
        NotImplementedError("[TensorFlowConverter] Slicing tensor with dynamic index is not supported yet:"
                            f"  type(end)={type(end)}")

    assert isinstance(stride, ConstantVariable), \
        NotImplementedError("[TensorFlowConverter] Slicing tensor with dynamic index is not supported yet:"
                            f"  type(strides)={type(stride)}")

    begin = begin.data.astype(int).flatten().tolist()  # type: List[int]
    end = end.data.astype(int).flatten().tolist()  # type: List[int]
    stride = stride.data.astype(int).flatten().tolist()  # type: List[int]

    begin_mask = tf_op.get_attr("begin_mask")
    end_mask = tf_op.get_attr("end_mask")
    ellipsis_mask = tf_op.get_attr("ellipsis_mask")
    new_axis_mask = tf_op.get_attr("new_axis_mask")

    d = 0
    while d < x.ndim:
        if (1 << d) & ellipsis_mask:
            begin.pop(d)
            end.pop(d)
            stride.pop(d)

            ellipsis_mask ^= 1 << d
            begin_mask >>= 1
            end_mask >>= 1

            while len(begin) < x.ndim:
                begin.insert(d, 0)
                end.insert(d, x.shape[d])
                stride.insert(d, 1)

                begin_mask <<= 1
                end_mask <<= 1

                d += 1

            continue

        if (1 << d) & begin_mask:
            begin_mask ^= 1 << d
            begin[d] = 0

        if (1 << d) & end_mask:
            end_mask ^= 1 << d
            end[d] = x.shape[d]

        d += 1

    y, = Slice(None, begin=begin, end=end, stride=stride)(x)

    if new_axis_mask > 0:
        new_axes = list(y.order.axes)
        new_shape = list(y.shape)
        for d in reversed(range(0, tf_op.outputs[0].shape.ndims)):
            if new_axis_mask & 1 << d:
                new_axes.insert(d, AxisVar())
                new_shape.insert(d, 1)

        y, = Reshape(y, in_order=y.order, out_order=Order(new_axes), out_shape=new_shape)(y)

    converter.set_variable(tf_op.outputs[0], y)


@TensorFlowConverter.register_handler("StridedSliceAssign")
def strided_slice_assign_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("StridedSliceGrad")
def strided_slice_grad_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("Tile")
def tile_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    x = converter.get_variable(tf_op.inputs[0])
    multiplier = converter.get_variable(tf_op.inputs[1])

    if not isinstance(multiplier, ConstantVariable):
        raise NotImplementedError("[TensorFlowConverter] Operator 'Tile' with dynamic multiplier is not supported yet.")

    multiplier = multiplier.data.astype(int).flatten().tolist()  # type: List[int]
    y, = Tile(None, multiplier=multiplier)(x)

    converter.set_variable(tf_op.outputs[0], y)


@TensorFlowConverter.register_handler("TileGrad")
def tile_grad_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("Transpose")
def transpose_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    x = converter.get_variable(tf_op.inputs[0])
    indices = converter.get_variable(tf_op.inputs[1])

    if not isinstance(indices, ConstantVariable):
        raise NotImplementedError("[TensorFlowConverter] Operator 'Transpose' with dynamic indices is not supported yet.")

    indices = indices.data.astype(int).flatten().tolist()  # type: List[int]
    y, = Transpose(None)(x)
    y.change_order(Order([x.order.axes[i] for i in indices]))

    converter.set_variable(tf_op.outputs[0], y)


@TensorFlowConverter.register_handler("Unique")
def unique_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("UniqueWithCounts")
def unique_with_counts_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("Unpack")
def unpack_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("Where")
def where_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("ZerosLike")
def zeros_like_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")
