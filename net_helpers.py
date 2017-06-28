from keras import backend as K
from keras.engine.topology import Layer

import tensorflow as tf


"""
Helper classes and functions for building the Keras model.
"""


class BprLoss(Layer):

    """
    Implementation of a loss function based on Bayesian Personalized Ranking:
    https://arxiv.org/abs/1205.2618 (Rendle et al 2012).

    Note that this is implemented as a layer, rather than a loss function.
    This is because Keras loss functions only take one input (and labels),
    but BPR loss takes threee inputs: a positive example, a negative
    example, and a user. Also, it has no explicit labels! The objective
    is to push the dot product between user and positive item closer to 1,
    while simultaneously pushing the dot product between user and negative
    item closer to 0.
    """
    
    def __init__(self, **kwargs):
        super(BprLoss, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BprLoss, self).build(input_shape)
        
    def call(self, inputs):
        """
        Inputs are:
         - A batch of positive item vectors (batch size x item dimensionality)
         - A batch of negative item vectors (batch size x item dimensionality)
         - A batch of user vectors (batch size x user dimensionality)
        Note that these can be raw embeddings, straight out of an Embedding
        layer, or they can be pre-processed or composed however you like.
        """
        assert len(inputs) == 3
        pos_item = inputs[0]
        neg_item = inputs[1]
        user = inputs[2]
        
        loss = 1.0 - K.log(K.sigmoid(
            K.sum(user * pos_item, axis=-1, keepdims=True) -
            K.sum(user * neg_item, axis=-1, keepdims=True)))

        return loss


def identity_loss(y_true, y_pred):
    """
    The 'real' loss function for the triplet network is calculated by
    the BprLoss layer, with the 'label' of a pair of items being
    implied by which is positive and which is negative. But, Keras
    still needs a loss function to train the model. So you need to pass
    in a vector of all ones as the label, so this function returns the
    BPR loss unchanged.

    NEVER pass anything but a vector of all ones to the y_true param!
    """
    return K.mean(y_pred * y_true)


class ZeroMaskedEntries(Layer):

    """
    Taken from this github issue to work around a Keras limitation:
    https://github.com/fchollet/keras/issues/1579#issuecomment-238644596

    This layer is called after an Embedding layer.
    It zeros out all of the masked-out embeddings.
    It also swallows the mask without passing it on.
    You can change this to default pass-on behavior as follows:

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)
    """

    def __init__(self, **kwargs):
        self.support_mask = True
        super(ZeroMaskedEntries, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.repeat_dim = input_shape[2]

    def call(self, x, mask=None):
        mask = K.cast(mask, 'float32')
        mask = K.repeat(mask, self.repeat_dim)
        mask = K.permute_dimensions(mask, (0, 2, 1))
        return x * mask

    def compute_mask(self, input_shape, input_mask=None):
        return None


def zero_mask(embeddings):
    """
    Just a convenience function to apply ZeroMaskedEntries
    to the output of an embedding layer.
    """
    return ZeroMaskedEntries()(embeddings)


def mask_aware_mean(masked_embeddings):

    """
    Taken from same github issue as above.

    Computes the mean of a set of embeddings that have been masked
    out by ZeroMaskedEntries.
    """

    # recreate the masks - all zero rows have been masked
    mask = K.not_equal(
        K.sum(K.abs(masked_embeddings), axis=2, keepdims=True),
        0)

    # number of that rows are not all zeros (min 1)
    n = K.sum(K.cast(mask, 'float32'), axis=1, keepdims=False)
    n = K.maximum(n, 1.0)
    
    # compute mean of remaining rows, or all zeroes if no rows present
    return K.sum(masked_embeddings, axis=1, keepdims=False) / n


def mask_aware_mean_output_shape(input_shape):
    """
    Taken from same github issue as above.

    Helper function to compute output shape, required by Lambda layer.
    """
    shape = list(input_shape)
    assert len(shape) == 3 
    return (shape[0], shape[2])




