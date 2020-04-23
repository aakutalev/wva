import tensorflow as tf
import numpy as np
from tensorflow.python.training.optimizer import Optimizer


def weight_variable(shape):
    # here we use Kaiming He initialization
    stddev = 2. / np.sqrt(shape[0])
    initial = tf.random.truncated_normal(shape=shape, mean=0.0, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


class _SpecializedSGD(tf.compat.v1.train.GradientDescentOptimizer):

    def __init__(self, learning_rate, use_locking=False, name="GradientDescent"):
        super(_SpecializedSGD, self).__init__(learning_rate, use_locking, name)

    def minimize(self, loss, global_step=None, var_list=None,
                 gate_gradients=Optimizer.GATE_OP, aggregation_method=None,
                 colocate_gradients_with_ops=False, name=None,
                 grad_loss=None, attenuators=None):
        """ comments """
        grads_and_vars = self.compute_gradients(
            loss, var_list=var_list, gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss)

        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        if not vars_with_grad:
            raise ValueError(
                "No gradients provided for any variable, check your graph for ops"
                " that do not support gradients, between variables %s and loss %s." %
                ([str(v) for _, v in grads_and_vars], loss))

        if attenuators is None:
            processed_grads_and_vars = grads_and_vars
        else:
            attenuator = iter(attenuators)
            processed_grads_and_vars = []
            for g, v in grads_and_vars:
                if g is None:
                    processed_grads_and_vars.append((g, v))
                else:
                    processed_grads_and_vars.append((tf.multiply(g, next(attenuator)), v))

        return self.apply_gradients(processed_grads_and_vars, global_step=global_step,
                                    name=name)


class Model:

    # learning type
    SGD = 0
    EWC = 1
    WVA = 2

    # acting matrix
    SIGNAL = 0
    FISHER = 1

    def __init__(self, shape=[784, 50, 10]):
        depth = len(shape) - 1
        if depth < 1:
            raise ValueError()

        # These are placeholder that'll be used to feed data
        self.x = tf.placeholder(tf.float32, shape=[None, shape[0]])
        self.labels = tf.placeholder(tf.float32, shape=[None, shape[-1]])

        self.learning_type = None
        self.acting_matrix = None

        # So, this variable could be used to access weights and biases outside the class!
        self.var_list = []

        for ins, outs in zip(shape[:-1], shape[1:]):
            self.var_list.append(weight_variable([ins, outs]))
            self.var_list.append(bias_variable([outs]))

        # Create gradient attenuation values
        #self.wb_attenuation = [tf.ones(tf.shape(v)) for v in self.var_list]
        self.wb_attenuation = [np.zeros(v.shape, dtype=np.float32) for v in self.var_list]

        # Create weight and bias regularization factors
        self.wb_regularization = [np.zeros(v.shape, dtype=np.float32) for v in self.var_list]

        # build-graph
        x = self.x
        adders = []
        outputs = []
        for i in range(depth):
            layer_adders = tf.matmul(x, self.var_list[i*2]) + self.var_list[i*2+1]
            adders.append(layer_adders)
            if i == depth-1:
                outputs.append(tf.nn.softmax(layer_adders))
            else:
                x = tf.nn.relu(layer_adders)
                outputs.append(x)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_adders, labels=self.labels))
        self.correct_preds = (tf.equal(tf.argmax(layer_adders, axis=1), tf.argmax(self.labels, axis=1)))

        # accuracy
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_preds, tf.float32))

        # sampling a random class from softmax
        class_ind = tf.to_int32(tf.multinomial(tf.math.log(outputs[-1]), 1)[0][0])

        # calculating probability gradients
        self.prob_grads = tf.gradients(tf.math.log(outputs[-1][0, class_ind]), self.var_list)

        # calculate passed signals
        os = tf.reduce_sum(tf.abs(self.x), axis=0)

        self.signals = []
        for i in range(depth):
            ws = tf.transpose(tf.multiply(os, tf.transpose(tf.abs(self.var_list[i*2]))))
            self.signals.append(ws)
            #bs = tf.reduce_sum(tf.abs(adders[i]), axis=0)
            #self.signals.append(bs)
            os = tf.reduce_sum(tf.abs(outputs[i]), axis=0)
            self.signals.append(os)

    def open_lesson(self, learning_type=None, acting_matrix=SIGNAL, learning_rate=1.0, lmbda=0.0):
        self.learning_type = learning_type
        self.acting_matrix = acting_matrix

        if learning_type is None or learning_type == Model.SGD:
            self.train_step = _SpecializedSGD(learning_rate).minimize(self.cross_entropy)

        elif learning_type == Model.EWC:
            ewc_loss = self.cross_entropy

            if hasattr(self, "star_vars") and lmbda != 0:
                for v in range(len(self.var_list)):
                    ewc_loss += (lmbda / 2.) * tf.reduce_sum(tf.multiply(
                        self.wb_regularization[v], tf.square(self.var_list[v] - self.star_vars[v])
                    ))
            self.train_step = _SpecializedSGD(learning_rate).minimize(ewc_loss)

        elif learning_type == Model.WVA:
            attenuators = [1./(1. + lmbda * v) for v in self.wb_attenuation]
            #attenuators = [np.exp(-lmbda * v) for v in self.wb_attenuation]
            self.train_step = _SpecializedSGD(learning_rate).minimize(
                self.cross_entropy, attenuators=attenuators
            )

    def close_lesson(self, test_set=None, session=None):
        if self.acting_matrix == Model.SIGNAL:
            acting_matrix = self._compute_signal(test_set, session)
        elif self.acting_matrix == Model.FISHER:
            acting_matrix = self._compute_fisher(test_set, session)
        else:
            acting_matrix = None

        if self.learning_type == Model.EWC:
            self._update_regularization(acting_matrix)

        elif self.learning_type == Model.WVA:
            self._update_attenuation(acting_matrix)

        self._store_weights_and_biases()

    def _update_regularization(self, acting_matrix):
        # update regularization coefficients
        for r, a in zip(self.wb_regularization, acting_matrix):
            r += a

    def _update_attenuation(self, acting_matrix):
        # update attenuation values
        for wba, a in zip(self.wb_attenuation, acting_matrix):
            wba += a

    def _store_weights_and_biases(self):
        # used for saving optimal weights after most recent task training

        # create weight and bias storage for keeping network state
        self.star_vars = [v.eval() for v in self.var_list]

    def _compute_signal(self, test_set, session):
        # compute total absolute signal passed through each connection
        num_samples = len(test_set)
        return [session.run(s, feed_dict={self.x: test_set}) / num_samples for s in self.signals]

    def _compute_fisher(self, test_set, session):
        # computer Fisher information for each parameter

        num_samples = len(test_set)
        # initialize Fisher information for most recent task
        fisher = [np.zeros(self.var_list[v].shape, dtype=np.float32)
            for v in range(len(self.var_list))]

        for i in range(num_samples):
            # compute first-order derivatives
            derivatives = session.run(self.prob_grads, feed_dict={self.x: test_set[i:i+1]})
            # square the derivatives and add to total
            for f, d in zip(fisher, derivatives):
                f += np.square(d)

        # divide totals by number of samples
        for f in fisher:
            f /= num_samples

        return fisher

    def _restore_weights_and_biases(self, sess):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                sess.run(self.var_list[v].assign(self.star_vars[v]))
