import six
import tensorflow as tf

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables


def _add_variable_to_collections(variable, collections_set, collections_name):
    """Adds variable (or all its parts) to all collections with that name."""
    collections = utils.get_variable_collections(collections_set,
                                                 collections_name) or []
    variables_list = [variable]
    if isinstance(variable, tf_variables.PartitionedVariable):
        variables_list = [v for v in variable]
    for collection in collections:
        for var in variables_list:
            if var not in ops.get_collection(collection):
                ops.add_to_collection(collection, var)


@add_arg_scope
def fully_connected_weight_norm(
        inputs,
        num_outputs,
        activation_fn=nn.relu,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=init_ops.zeros_initializer(),
        biases_regularizer=None,
        scale=True,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope=None):
    if not isinstance(num_outputs, six.integer_types):
        raise ValueError('num_outputs should be int or long, got %s.',
                         num_outputs)

    with variable_scope.variable_scope(
            scope, 'fully_connected', [inputs], reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        dtype = inputs.dtype.base_dtype

        if scale:
            g = variable_scope.get_variable(
                'g',
                shape=[num_outputs],
                dtype=dtype,
                initializer=tf.ones_initializer(),
                trainable=trainable)
        else:
            g = 1.
        V = variable_scope.get_variable(
            'V',
            shape=[int(inputs.get_shape()[1]), num_outputs],
            dtype=dtype,
            initializer=weights_initializer,
            regularizer=weights_regularizer,
            trainable=trainable)

        if biases_initializer is not None:
            b = variable_scope.get_variable(
                'biases',
                shape=[num_outputs],
                dtype=dtype,
                initializer=biases_initializer,
                regularizer=biases_regularizer,
                trainable=trainable)
        else:
            b = None

        outputs = tf.matmul(inputs, V)
        scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
        outputs = tf.reshape(scaler, [1, num_outputs]) * outputs

        # Add variables to collections.
        if scale:
            _add_variable_to_collections(g, variables_collections, 'g')
        _add_variable_to_collections(V, variables_collections, 'V')

        if b is not None:
            outputs = nn.bias_add(outputs, b)
            _add_variable_to_collections(b, variables_collections, 'biases')

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)


@add_arg_scope
def conv2d_weight_norm(inputs,
                       num_outputs,
                       kernel_size,
                       stride=1,
                       padding='SAME',
                       data_format=None,
                       rate=1,
                       activation_fn=nn.relu,
                       weights_initializer=initializers.xavier_initializer(),
                       weights_regularizer=None,
                       biases_initializer=init_ops.zeros_initializer(),
                       biases_regularizer=None,
                       scale=True,
                       reuse=None,
                       variables_collections=None,
                       outputs_collections=None,
                       trainable=True,
                       scope=None):
    if data_format not in [None, 'NHWC', 'NCHW']:
        raise ValueError('Invalid data_format: %r' % (data_format, ))

    input_rank = inputs.get_shape().ndims
    if input_rank != 4:
        raise ValueError('Convolution not supported for input with rank',
                         input_rank)

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if isinstance(stride, int):
        stride = (stride, stride)

    with variable_scope.variable_scope(
            scope, 'Conv', [inputs], reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        dtype = inputs.dtype.base_dtype

        if scale:
            g = variable_scope.get_variable(
                'g',
                shape=[num_outputs],
                dtype=dtype,
                initializer=tf.ones_initializer(),
                trainable=trainable)
        else:
            g = 1.
        V = variable_scope.get_variable(
            'V',
            shape=list(kernel_size) +
            [int(inputs.get_shape()[-1]), num_outputs],
            dtype=dtype,
            initializer=weights_initializer,
            regularizer=weights_regularizer,
            trainable=trainable)

        if biases_initializer is not None:
            b = variable_scope.get_variable(
                'biases',
                shape=[num_outputs],
                dtype=dtype,
                initializer=biases_initializer,
                regularizer=biases_regularizer,
                trainable=trainable)
        else:
            b = None

        W = tf.nn.l2_normalize(V, [0, 1, 2])
        if scale:
            W = tf.reshape(g, [1, 1, 1, num_outputs]) * W

        # calculate convolutional layer output
        outputs = tf.nn.conv2d(inputs, W, [1] + list(stride) + [1], padding)

        # Add variables to collections.
        if scale:
            _add_variable_to_collections(g, variables_collections, 'g')
        _add_variable_to_collections(V, variables_collections, 'V')

        if b is not None:
            outputs = nn.bias_add(outputs, b)
            _add_variable_to_collections(b, variables_collections, 'biases')

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)


def deconv_output_length(input_length, filter_size, padding, stride):
    """Determines output length of a transposed convolution given input length.

  Arguments:
      input_length: integer.
      filter_size: integer.
      padding: one of "same", "valid", "full".
      stride: integer.

  Returns:
      The output length (integer).
  """
    if input_length is None:
        return None
    input_length *= stride
    if padding == 'valid':
        input_length += max(filter_size - stride, 0)
    elif padding == 'full':
        input_length -= (stride + filter_size - 2)
    return input_length


@add_arg_scope
def conv2d_subpixel(inputs,
                    scale=2,
                    data_format='NHWC',
                    activation_fn=None,
                    scope=None):

    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('data_format has to be either NCHW or NHWC.')

    if data_format == 'NCHW':
        c_axis, h_axis, w_axis = 1, 2, 3
    else:
        c_axis, h_axis, w_axis = 3, 1, 2

    batch_size = array_ops.shape(inputs)[0]
    with variable_scope.variable_scope(scope, 'Conv2d_subpixel',
                                       [inputs]) as sc:
        inputs = ops.convert_to_tensor(inputs)

        inputs_shape = inputs.get_shape()
        if int(inputs_shape[c_axis]) / (scale**2) % 1 != 0:
            raise ValueError(
                'The number of input channels == (scale x scale) x The number of output channels'
            )

        num_outputs = int(inputs_shape[c_axis]) // (scale**2)

        outputs = tf.split(inputs, scale, c_axis)  #b*h*w*r*r
        outputs = tf.concat(outputs, w_axis)  #b*h*(r*w)*r
        outputs_shape = [batch_size, 0, 0, 0]
        outputs_shape[c_axis] = num_outputs
        outputs_shape[h_axis] = scale * int(inputs_shape[h_axis])
        outputs_shape[w_axis] = scale * int(inputs_shape[w_axis])
        outputs = tf.reshape(outputs, outputs_shape)  # b*(r*h)*(r*w)*c

        if activation_fn is not None:
            outputs = activation_fn(outputs)
    return outputs


@add_arg_scope
def conv2d_transpose_weight_norm(
        inputs,
        num_outputs,
        kernel_size,
        stride=1,
        padding='SAME',
        data_format='NHWC',
        activation_fn=nn.relu,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=init_ops.zeros_initializer(),
        biases_regularizer=None,
        scale=True,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope=None):
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('data_format has to be either NCHW or NHWC.')

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if isinstance(stride, int):
        stride = (stride, stride)

    inputs_shape = array_ops.shape(inputs)
    batch_size = inputs_shape[0]
    if data_format == 'NCHW':
        c_axis, h_axis, w_axis = 1, 2, 3
    else:
        c_axis, h_axis, w_axis = 3, 1, 2

    height, width = inputs_shape[h_axis], inputs_shape[w_axis]
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride

    out_height = deconv_output_length(height, kernel_h,
                                      padding.lower(), stride_h)
    out_width = deconv_output_length(width, kernel_w, padding.lower(),
                                     stride_w)

    if data_format == 'NCHW':
        output_shape = (batch_size, num_outputs, out_height, out_width)
        stride = (1, 1, stride_h, stride_w)
    else:
        output_shape = (batch_size, out_height, out_width, num_outputs)
        stride = (1, stride_h, stride_w, 1)

    with variable_scope.variable_scope(
            scope, 'Conv2d_transpose', [inputs], reuse=reuse) as sc:
        inputs = ops.convert_to_tensor(inputs)
        dtype = inputs.dtype.base_dtype

        g = variable_scope.get_variable(
            'g',
            shape=[num_outputs],
            dtype=dtype,
            initializer=tf.ones_initializer(),
            trainable=trainable)
        V = variable_scope.get_variable(
            'V',
            shape=list(kernel_size) +
            [num_outputs, int(inputs.get_shape()[-1])],
            dtype=dtype,
            initializer=weights_initializer,
            regularizer=weights_regularizer,
            trainable=trainable)

        if biases_initializer is not None:
            b = variable_scope.get_variable(
                'biases',
                shape=[num_outputs],
                dtype=dtype,
                initializer=biases_initializer,
                regularizer=biases_regularizer,
                trainable=trainable)
        else:
            b = None

        W = tf.nn.l2_normalize(V, [0, 1, 3])
        if scale:
            W = tf.reshape(g, [1, 1, num_outputs, 1]) * W

        # calculate convolutional layer output
        output_shape_tensor = array_ops.stack(output_shape)
        outputs = tf.nn.conv2d_transpose(
            inputs,
            W,
            output_shape_tensor,
            stride,
            padding=padding,
            data_format=data_format)

        # Add variables to collections.
        _add_variable_to_collections(g, variables_collections, 'g')
        _add_variable_to_collections(V, variables_collections, 'V')

        if b is not None:
            outputs = nn.bias_add(outputs, b)
            _add_variable_to_collections(b, variables_collections, 'biases')

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)
