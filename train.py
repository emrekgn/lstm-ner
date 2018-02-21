#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from loader import DataLoader
from model import Model
from six.moves import cPickle


def main(args):
    # Load input file, prepare training and validation sets
    data_loader = DataLoader(args.input, args.dim_word, args.pre_emb, args.batch_size, args.lowercase, args.zeros)

    # Save vocabularies
    with open(os.path.join(args.output, 'words_vocab.pkl'), 'wb') as f:
        cPickle.dump(data_loader.word_to_id, f)
    with open(os.path.join(args.output, 'char_vocab.pkl'), 'wb') as f:
        cPickle.dump(data_loader.char_to_id, f)
    with open(os.path.join(args.output, 'tag_vocab.pkl'), 'wb') as f:
        cPickle.dump(data_loader.tag_to_id, f)
    # Save parameters
    with open(os.path.join(args.output, 'args.json'), 'wb') as f:
        cPickle.dump(args, f)

    # Build model
    model = Model(args, data_loader)

    best_score = 0
    niter_without_improvement = 0
    for epoch in range(args.nepochs):
        print("Epoch {:} out of {:}".format(epoch + 1, args.nepochs))
        data_loader.reset_pointer()
        score = model.run_epoch(epoch)
        args.learning_rate *= args.decay_rate
        # early stopping and saving best parameters
        if score >= best_score:
            niter_without_improvement = 0
            model.save_session(args.output)
            best_score = score
            print("New best score: {}".format(score))
        else:
            niter_without_improvement += 1
            if niter_without_improvement >= args.early_stopping:
                print("Early stopping {} epochs without improvement".format(niter_without_improvement))
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu_mem',
        default=0.666,
        type=float,
        help='GPU memory that can be allocated.')
    parser.add_argument(
        '--input',
        default='data/input.txt',
        help='Train data file location.')
    parser.add_argument(
        '--output',
        default='data/output',
        help='Output location. Used to save model config/params and logs.')
    parser.add_argument(
        '--pre_emb',
        default='data/pretrained.txt',
        help='Pretrained embeddings location.')
    parser.add_argument(
        '--save_every',
        default=1000,
        type=int,
        help='Save frequency.')
    parser.add_argument(
        '--early_stopping',
        default=6,
        type=int,
        help='Early stopping if loss does not decrease for the number of this iterations.')
    parser.add_argument(
        '--model',
        default='lstm',
        help='Model of the RNNs (possible values: lstm, gru, rnn).')
    parser.add_argument(
        '--nepochs',
        default=100,
        type=int,
        help='Number of epochs to run.')
    parser.add_argument(
        '--lowercase',
        default=0,
        type=int,
        help='Lowercase words (this will not affect character inputs).')
    parser.add_argument(
        '--zeros',
        default=0,
        type=int,
        help='Replace digits with 0.')
    parser.add_argument(
        '--dropout',
        default=0.5,
        type=float,
        help='Droupout on the input of bi LSTM layer (1.0 = no dropout).')
    parser.add_argument(
        '--lr_method',
        default='sgd',
        help='Learning rate method (possible values: adam, adagrad, sgd, rmsprop).')
    parser.add_argument(
        '--learning_rate',
        default=0.001,
        type=float,
        help='Learning rate (Default 0.001).')
    parser.add_argument(
        '--decay_rate',
        default=1.0,
        type=float,
        help='Learning rate decay (1.0 = no decay).')
    parser.add_argument(
        '--grad_clip',
        default=5,
        type=int,
        help='Gradient clipping (0 = no clipping).')
    parser.add_argument(
        '--dim_char',
        default=25,
        type=int,
        help='Char embedding dimension.')
    parser.add_argument(
        '--hidden_size_char',
        default=25,
        type=int,
        help='Char hidden layer size.')
    parser.add_argument(
        '--dim_word',
        default=200,
        type=int,
        help='Word embedding dimension.')
    parser.add_argument(
        '--hidden_size_lstm',
        default=200,
        type=int,
        help='bi LSTM hidden layer size.')
    parser.add_argument(
        '--batch_size',
        default=5,
        type=int,
        help='Batch size.')
    args, _ = parser.parse_known_args()

    # Validate parameters
    assert os.path.isfile(args.input)
    assert not args.pre_emb or os.path.isfile(args.pre_emb)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    main(args)
