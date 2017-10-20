from webdnn.frontend.keras.converter import KerasConverter
from webdnn.frontend.keras.layers.util import do_activation
from webdnn.graph.axis import Axis
from webdnn.graph.operators.convolution2d import Convolution2D
from webdnn.graph.operators.deconvolution2d import Deconvolution2D
from webdnn.graph.operators.resize_2d import Resize2D
from webdnn.graph.operators.zero_padding_1d import ZeroPadding1D
from webdnn.graph.operators.zero_padding_2d import ZeroPadding2D
from webdnn.graph.order import OrderC, OrderNCHW, OrderNHWC, OrderHWCN, OrderNTC, OrderHWNC

try:
    import keras
except ImportError as e:
    pass


# noinspection PyUnusedLocal
@KerasConverter.register_handler("Conv1D")
def _convert_conv1d(converter: KerasConverter, k_op: "keras.layers.Conv1D"):
    # TODO
    raise NotImplementedError('[KerasConverter] keras.layers.Conv1D is not supported')


@KerasConverter.register_handler("Conv2D")
def _convert_conv2d(converter: KerasConverter, k_op: "keras.layers.Conv2D"):
    x = converter.get_variable(converter.get_input_tensor(k_op)[0])

    if k_op.data_format == "channels_first":
        assert x.order == OrderNCHW

    elif k_op.data_format == "channels_last":
        assert x.order == OrderNHWC

    else:
        raise ValueError(f"[KerasConverter] Unknown data format is detected: {k_op.data_format}")

    w = converter.convert_to_constant_variable(k_op.kernel, OrderHWCN)

    ksize = tuple(k_op.kernel_size)
    stride = tuple(k_op.strides)
    dilation_rate = tuple(k_op.dilation_rate)
    if k_op.padding == "valid":
        padding = (0, 0)

    elif k_op.padding == "same":
        # @see https://github.com/tensorflow/tensorflow/blob/e5cf6f0c13b6053e4c58af6a951b204fde263172/tensorflow/python/ops/nn_ops.py#L507-L519
        dilated_ksize = [k + (k - 1) * (d - 1) for k, d in zip(ksize, dilation_rate)]
        pad_extra_shape = [dk - 1 for dk in dilated_ksize]

        if any(p % 2 != 0 for p in pad_extra_shape):
            raise NotImplementedError(f"[KerasConverter] Currently WebDNN doesn't supports different size padding: "
                                      f"  (pad_extra_shape)=f{pad_extra_shape}")

        padding = tuple(p // 2 for p in pad_extra_shape)

    else:
        raise ValueError(f"[KerasConverter] Unknown padding: {k_op.padding}")

    y, = Convolution2D(None, ksize=ksize, stride=stride, padding=padding, dilation_rate=dilation_rate)(x, w)

    if k_op.use_bias:
        b = converter.convert_to_constant_variable(k_op.bias, OrderC)
        y = y + b

    y = do_activation(k_op.activation, y)
    converter.set_variable(converter.get_output_tensor(k_op)[0], y)


# noinspection PyUnusedLocal
@KerasConverter.register_handler("SeparableConv2D")
def _convert_separable_conv2d(converter: KerasConverter, k_op: "keras.layers.SeparableConv2D"):
    # TODO
    raise NotImplementedError('[KerasConverter] keras.layers.SeparableConv2D is not supported')


# noinspection PyUnusedLocal
@KerasConverter.register_handler("Conv2DTranspose")
def _convert_conv2d_transpose(converter: KerasConverter, k_op: "keras.layers.Conv2DTranspose"):
    x = converter.get_variable(converter.get_input_tensor(k_op)[0])
    if k_op.data_format == "channels_first":
        assert x.order == OrderNCHW

    elif k_op.data_format == "channels_last":
        assert x.order == OrderNHWC

    else:
        raise ValueError(f"[KerasConverter] Unknown data format is detected: {k_op.data_format}")

    w = converter.convert_to_constant_variable(k_op.kernel, OrderHWNC)

    ksize = tuple(k_op.kernel_size)
    stride = tuple(k_op.strides)

    if k_op.padding == "valid":
        padding = (0, 0)

    elif k_op.padding == "same":
        # @see https://github.com/tensorflow/tensorflow/blob/e5cf6f0c13b6053e4c58af6a951b204fde263172/tensorflow/python/ops/nn_ops.py#L507-L519
        pad_extra_shape = [k - 1 for k in ksize]

        if any(p % 2 != 0 for p in pad_extra_shape):
            raise NotImplementedError(f"[KerasConverter] Currently WebDNN doesn't supports different size padding: "
                                      f"  (pad_extra_shape)=f{pad_extra_shape}")

        padding = tuple(p // 2 for p in pad_extra_shape)

    w = converter.convert_to_constant_variable(k_op.kernel, OrderHWNC)

    ksize = tuple(k_op.kernel_size)
    stride = tuple(k_op.strides)

    if k_op.padding == "valid":
        padding = (0, 0)

    elif k_op.padding == "same":
        padding = (ksize[0] // 2, ksize[1] // 2)

    else:
        raise ValueError(f"[KerasConverter] Unknown padding: {k_op.padding}")

    y, = Deconvolution2D(None, ksize=ksize, stride=stride, padding=padding)(x, w)
    if k_op.use_bias:
        b = converter.convert_to_constant_variable(k_op.bias, OrderC)
        y = y + b

    y = do_activation(k_op.activation, y)
    converter.set_variable(converter.get_output_tensor(k_op)[0], y)


# noinspection PyUnusedLocal
@KerasConverter.register_handler("Conv3D")
def _convert_conv3d(converter: KerasConverter, k_op: "keras.layers.Conv3D"):
    # TODO
    raise NotImplementedError('[KerasConverter] keras.layers.Conv3D is not supported')


# noinspection PyUnusedLocal
@KerasConverter.register_handler("Cropping1D")
def _convert_cropping1d(converter: KerasConverter, k_op: "keras.layers.Cropping1D"):
    # TODO
    raise NotImplementedError('[KerasConverter] keras.layers.Cropping1D is not supported')


# noinspection PyUnusedLocal
@KerasConverter.register_handler("Cropping2D")
def _convert_cropping2d(converter: KerasConverter, k_op: "keras.layers.Cropping2D"):
    # TODO
    raise NotImplementedError('[KerasConverter] keras.layers.Cropping2D is not supported')


# noinspection PyUnusedLocal
@KerasConverter.register_handler("Cropping3D")
def _convert_cropping3d(converter: KerasConverter, k_op: "keras.layers.Cropping3D"):
    # TODO
    raise NotImplementedError('[KerasConverter] keras.layers.Cropping3D is not supported')


# noinspection PyUnusedLocal
@KerasConverter.register_handler("UpSampling1D")
def _convert_up_sampling1d(converter: KerasConverter, k_op: "keras.layers.UpSampling1D"):
    # TODO
    raise NotImplementedError('[KerasConverter] keras.layers.UpSampling1D is not supported')


# noinspection PyUnusedLocal
@KerasConverter.register_handler("UpSampling2D")
def _convert_up_sampling2d(converter: KerasConverter, k_op: "keras.layers.UpSampling2D"):
    x = converter.get_variable(converter.get_input_tensor(k_op)[0])

    y, = Resize2D(None,
                  axis1=Axis.H, size1=x.shape_dict[Axis.H] * k_op.size[0],
                  axis2=Axis.W, size2=x.shape_dict[Axis.W] * k_op.size[1])(x)

    converter.set_variable(converter.get_output_tensor(k_op)[0], y)


# noinspection PyUnusedLocal
@KerasConverter.register_handler("UpSampling3D")
def _convert_up_sampling2d(converter: KerasConverter, k_op: "keras.layers.UpSampling3D"):
    # TODO
    raise NotImplementedError('[KerasConverter] keras.layers.UpSampling3D is not supported')


@KerasConverter.register_handler("ZeroPadding1D")
def _convert_zero_padding1d(converter: KerasConverter, k_op: "keras.layers.ZeroPadding1D"):
    x = converter.get_variable(converter.get_input_tensor(k_op)[0])

    assert x.order == OrderNTC

    y, = ZeroPadding1D(None, padding=tuple(k_op.padding))(x)
    converter.set_variable(converter.get_output_tensor(k_op)[0], y)


@KerasConverter.register_handler("ZeroPadding2D")
def _convert_zero_padding2d(converter: KerasConverter, k_op: "keras.layers.ZeroPadding2D"):
    x = converter.get_variable(converter.get_input_tensor(k_op)[0])

    padding = k_op.padding
    top = padding[0][0]
    if top != padding[0][1]:
        # FIXME: This condition should be checked in each backend
        raise NotImplementedError(
            "[KerasConverter] In current implementation, Padding size of top and bottom must be same.")

    left = padding[1][0]
    if left != padding[1][1]:
        # FIXME: This condition should be checked in each backend
        raise NotImplementedError(
            "[KerasConverter] In current implementation, Padding size of left and right must be same.")

    y, = ZeroPadding2D(None, (top, left))(x)
    converter.set_variable(converter.get_output_tensor(k_op)[0], y)


# noinspection PyUnusedLocal
@KerasConverter.register_handler("ZeroPadding3D")
def _convert_zero_padding3d(converter: KerasConverter, k_op: "keras.layers.ZeroPadding3D"):
    # TODO
    raise NotImplementedError('[KerasConverter] keras.layers.ZeroPadding3D is not supported')
