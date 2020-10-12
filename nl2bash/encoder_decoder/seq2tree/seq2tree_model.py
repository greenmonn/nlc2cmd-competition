"""Sequence-to-tree model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .. import encoder
from ..framework import EncoderDecoderModel
from . import tree_decoder


class Seq2TreeModel(EncoderDecoderModel):
    """Sequence-to-tree models.
    """

    def __init__(self, hyperparams, buckets=None):
        """
        Create the model.
        :param hyperparams: learning hyperparameters
        :param buckets: if not None, train bucket model.
        :param forward_only: if set, we do not construct the backward pass.
        """
        super(Seq2TreeModel, self).__init__(hyperparams, buckets)

    def define_encoder(self, input_keep, output_keep):
        """
        Construct sequence encoder.
        """
        if self.encoder_topology == "rnn":
            self.encoder = encoder.RNNEncoder(
                self.hyperparams, input_keep, output_keep)
        elif self.encoder_topology == "birnn":
            self.encoder = encoder.BiRNNEncoder(
                self.hyperparams, input_keep, output_keep)
        else:
            raise ValueError("Unrecognized encoder type.")

    def define_decoder(self, dim, embedding_dim, use_attention,
                       attention_function, input_keep, output_keep):
        """Construct tree decoders."""
        if self.decoder_topology == "basic_tree":
            self.decoder = tree_decoder.BasicTreeDecoder(
                hyperparams=self.hyperparams,
                scope='tree_decoder', dim=dim,
                embedding_dim=embedding_dim,
                use_attention=use_attention,
                attention_function=attention_function,
                input_keep=input_keep,
                output_keep=output_keep,
                decoding_algorithm=self.token_decoding_algorithm
            )
                # self.hyperparams, dim, self.output_project())
        else:
            raise ValueError("Unrecognized decoder topology: {}."
                             .format(self.decoder_topology))
