import tensorflow as tf

from ..layers import fully_connected_weight_norm


class ModelBase(object):
    def __init__(self, scope):
        self.name = scope.name
        self.scope = scope
        self.summaries = []

    def get_vars(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def reg_loss(self):
        reg_vars = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES, scope=self.name)
        return tf.add_n(reg_vars) if reg_vars else tf.constant(0.0)

    def update_ops(self):
        return tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.name)

    def add_summary(self,
                    name,
                    op,
                    summary_fn=tf.summary.scalar,
                    summary_list=None):
        op = summary_fn(name, op)
        self.add_summary_op(op, summary_list=summary_list)

    def add_summary_op(self, op, summary_list=None):
        if summary_list is None:
            summary_list = [self.summaries]
        elif not isinstance(summary_list, list) or len(
                summary_list) == 0 or not isinstance(summary_list[0], list):
            summary_list = [summary_list]
        for summary in summary_list:
            summary.append(op)


class DistBase(ModelBase):
    def build_prototypes(self, flatten_outputs, activation_fn):
        with tf.variable_scope('outputs'):
            biases_initializer = tf.zeros_initializer(
            ) if self.dist_type.startswith('pcd') else None
            outputs = fully_connected_weight_norm(
                inputs=flatten_outputs,
                num_outputs=self.num_outputs,
                activation_fn=None,
                weights_regularizer=self.regularizer,
                weights_initializer=self.initializer,
                biases_regularizer=self.regularizer,
                biases_initializer=biases_initializer)
            self.outputs = outputs
            self.activations = activation_fn(
                self.outputs) if activation_fn else self.outputs

        # get prototype outputs
        if self.dist_type in {'pcd', 'monomer'}:
            with tf.variable_scope('prototype_outputs'):
                biases_initializer = tf.zeros_initializer(
                ) if self.dist_type.startswith('pcd') else None
                prototype_inputs = flatten_outputs
                prototype_outputs = fully_connected_weight_norm(
                    inputs=prototype_inputs,
                    num_outputs=self.num_outputs * self.num_components,
                    activation_fn=None,
                    weights_regularizer=self.regularizer,
                    weights_initializer=self.initializer,
                    biases_initializer=biases_initializer,
                    biases_regularizer=self.regularizer)

                prototype_activations = activation_fn(
                    prototype_outputs) if activation_fn else prototype_outputs
                self.flat_prototype_activations = prototype_activations
                self.flat_all_activations = tf.concat(
                    [self.activations, self.flat_prototype_activations],
                    axis=-1)
                prototype_activations = tf.reshape(prototype_activations, [
                    -1, self.num_components, self.num_outputs
                ])
                self.prototype_activations = prototype_activations
                if self.gate is not None:
                    self.one_prototype_activations = tf.gather_nd(
                        self.prototype_activations, self.gate)

                self.all_prototype_activations = tf.split(
                    tf.reshape(self.prototype_activations,
                               (-1, self.num_components * self.num_outputs)),
                    self.num_components,
                    axis=1)

        if self.dist_type == 'monomer':
            with tf.variable_scope('monomer_outputs'):
                monomer_outputs = outputs
                monomer_outputs = fully_connected_weight_norm(
                    inputs=monomer_outputs,
                    num_outputs=self.num_components,
                    activation_fn=None,
                    weights_regularizer=self.regularizer,
                    weights_initializer=self.initializer,
                    biases_initializer=None)
                self.monomer_outputs = monomer_outputs
                self.monomer_activations = tf.nn.softmax(self.monomer_outputs)

    def build_dist(self, target):
        with tf.variable_scope(self.scope, reuse=True):
            if self.dist_type == 'monomer':
                diff = tf.subtract(
                    tf.reshape(self.activations, (-1, 1, self.num_outputs)),
                    target.prototype_activations)
                diff = tf.square(diff)
                diff = tf.reduce_sum(diff, axis=-1)
                weighted_dist = tf.reduce_sum(
                    tf.multiply(self.monomer_activations, diff), axis=-1)
                return tf.reshape(weighted_dist, (-1, 1))

            elif self.dist_type == 'siamese':
                diff = tf.subtract(self.activations, target.activations)
                diff = tf.square(diff)
                diff = tf.reduce_sum(diff, axis=-1)
                return tf.reshape(diff, (-1, 1))

            elif self.dist_type.startswith('pcd'):
                v = target.activations
                with tf.variable_scope(self.scope, reuse=True):
                    if self.num_components > 1:
                        diff = tf.subtract(
                            tf.reshape(v, (-1, 1, self.num_outputs)),
                            self.prototype_activations)
                        logits = -tf.reduce_sum(tf.square(diff), axis=-1)
                        scales = tf.nn.softmax(logits)
                        means = tf.reduce_sum(
                            self.prototype_activations * tf.reshape(
                                scales, (-1, self.num_components, 1)),
                            axis=-2)
                        dist = tf.reduce_sum(tf.square(v - means), -1)
                        return tf.reshape(dist, (-1, 1))
                    else:
                        diff = tf.subtract(
                            v,
                            tf.reshape(self.prototype_activations,
                                       (-1, self.num_outputs)))
                        dist = tf.reduce_sum(tf.square(diff), -1)
                        return tf.reshape(dist, (-1, 1))
