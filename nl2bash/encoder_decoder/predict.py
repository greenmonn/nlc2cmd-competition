from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import numpy as np
import pickle
import time
from tqdm import tqdm
import tensorflow as tf
from ..encoder_decoder import data_utils
from ..encoder_decoder import decode_tools
from ..encoder_decoder import graph_utils
from ..encoder_decoder import parse_args
from ..encoder_decoder.seq2seq.seq2seq_model import Seq2SeqModel
from ..encoder_decoder.seq2tree.seq2tree_model import Seq2TreeModel
from .seq2doubleseq.seq2doubleseq_model import Seq2DSeqModel

import os
import sys

if sys.version_info > (3, 0):
    from six.moves import xrange


# Using copyNet

FLAGS = tf.compat.v1.flags.FLAGS
parse_args.define_input_flags()


def define_model(session, forward_only, buckets=None):
    """
    Define tensor graphs.
    """
    if FLAGS.decoder_topology in ['basic_tree']:
        return graph_utils.define_model(
            FLAGS, session, Seq2TreeModel, buckets, forward_only)
    elif FLAGS.decoder_topology in ['rnn']:
        return graph_utils.define_model(
            FLAGS, session, Seq2SeqModel, buckets, forward_only)
    elif FLAGS.decoder_topology in ['doublernn']:
        return graph_utils.define_model(
            FLAGS, session, Seq2DSeqModel, buckets, forward_only)
    else:
        raise ValueError("Unrecognized decoder topology: {}.".format(
            FLAGS.decoder_topology))


def demo(sess, model, invocations, result_cnt, vocab, FLAGS):
    n_batch = len(invocations)
    predictions = [[''] * result_cnt for _ in range(n_batch)]

    for i in range(n_batch):
        sentence = invocations[i]

        # Do not fill argument slots
        batch_outputs, sequence_logits = decode_tools.translate_fun(
            sentence, sess, model, vocab, FLAGS)

        # Use Beam Search
        if batch_outputs:
            top_k_predictions = batch_outputs[0]
            top_k_scores = sequence_logits[0]
            for j in xrange(min(FLAGS.beam_size, result_cnt-1, len(batch_outputs[0]))):
                if len(top_k_predictions) <= j:
                    break
                top_k_pred_tree, top_k_pred_cmd = top_k_predictions[j]
                predictions[i][j] = top_k_pred_cmd
                print('Prediction {}: {} ({}) '.format(
                    j+1, top_k_pred_cmd, top_k_scores[j]))

    return predictions


def prepare_flags():
    FLAGS.data_dir = os.path.join(
        os.path.dirname(__file__), "..", "data", FLAGS.dataset)
    print("Reading data from {}".format(FLAGS.data_dir))

    # set up encoder/decider dropout rate
    if FLAGS.universal_keep >= 0 and FLAGS.universal_keep < 1:
        FLAGS.sc_input_keep = FLAGS.universal_keep
        FLAGS.sc_output_keep = FLAGS.universal_keep
        FLAGS.tg_input_keep = FLAGS.universal_keep
        FLAGS.tg_output_keep = FLAGS.universal_keep
        FLAGS.attention_input_keep = FLAGS.universal_keep
        FLAGS.attention_output_keep = FLAGS.universal_keep

    # adjust hyperparameters for batch normalization
    if FLAGS.recurrent_batch_normalization:
        # larger batch size
        FLAGS.batch_size *= 4
        # larger initial learning rate
        FLAGS.learning_rate *= 10

    if FLAGS.decoder_topology in ['basic_tree']:
        FLAGS.model_root_dir = os.path.join(
            os.path.dirname(__file__), "..", FLAGS.model_root_dir, "seq2tree")
    elif FLAGS.decoder_topology in ['rnn']:
        FLAGS.model_root_dir = os.path.join(
            os.path.dirname(__file__), "..", FLAGS.model_root_dir, "seq2seq")
    elif FLAGS.decoder_topology in ['doublernn']:
        FLAGS.model_root_dir = os.path.join(
            os.path.dirname(__file__), "..", FLAGS.model_root_dir, "seq2doubleseq")
    else:
        raise ValueError("Unrecognized decoder topology: {}."
                         .format(FLAGS.decoder_topology))
    print("Saving models to {}".format(FLAGS.model_root_dir))

    train_set, dev_set, test_set = data_utils.load_data(
        FLAGS, use_buckets=True)

    print("Set dataset parameters")
    vocab = data_utils.load_vocabulary(FLAGS)
    FLAGS.max_sc_length = train_set.max_sc_length if not train_set.buckets else \
        train_set.buckets[-1][0]
    FLAGS.max_tg_length = train_set.max_tg_length if not train_set.buckets else \
        train_set.buckets[-1][1]
    FLAGS.sc_vocab_size = len(vocab.sc_vocab)
    FLAGS.tg_vocab_size = len(vocab.tg_vocab)
    FLAGS.max_sc_token_size = vocab.max_sc_token_size
    FLAGS.max_tg_token_size = vocab.max_tg_token_size

    return train_set, vocab


class Env:
    sess = None
    model = None


def predict_nl2bash(invocations, result_cnt):
    train_set, vocab = prepare_flags()
    buckets = train_set.buckets

    tf.compat.v1.disable_eager_execution()

    if Env.sess == None:
        Env.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                                                        log_device_placement=FLAGS.log_device_placement))

    # Initialize model parameters.
    if Env.model == None:
        Env.model = define_model(Env.sess, forward_only=True, buckets=buckets)

    predictions = demo(Env.sess, Env.model, invocations,
                       result_cnt, vocab, FLAGS)
    return predictions


if __name__ == "__main__":
    predict_nl2bash(["Display all lines containing \"IP_MROUTE\" in the current kernel's compile-time config file."],
                    3)
    predict_nl2bash(["Display all lines containing \"IP_MROUTE\" in the current kernel's compile-time config file."],
                    3)
