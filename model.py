#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class Model:
    def __init__(self, args, data_loader):
        self.args = args
        self.data_loader = data_loader
        self.nwords = data_loader.word_vocab_size
        self.nchars = data_loader.char_vocab_size
        self.ntags = data_loader.tag_vocab_size

        # Define inputs and hyper parameters
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None], name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None], name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.tags = tf.placeholder(tf.int32, shape=[None, None], name="tags")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name="learning_rate")

        # Pre-trained word embeddings
        with tf.variable_scope("words"):
            # TODO currently, randomly generated. use pretrained word embeddings instead!
            _word_embeddings = tf.get_variable(
                    name="_word_embeddings",
                    dtype=tf.float32,
                    shape=[self.nwords, self.args.dim_word])
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids, name="word_embeddings")

        # Char embeddings
        with tf.variable_scope("chars"):
            # get char embeddings matrix
            _char_embeddings = tf.get_variable(
                    name="_char_embeddings",
                    dtype=tf.float32,
                    shape=[self.nchars, self.args.dim_char])
            char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.char_ids, name="char_embeddings")

            # put the time dimension on axis=1
            s = tf.shape(char_embeddings)
            char_embeddings = tf.reshape(char_embeddings, shape=[s[0]*s[1], s[-2], self.args.dim_char])
            word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

            # bi lstm on chars
            cell_fw = tf.contrib.rnn.LSTMCell(self.args.hidden_size_char, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self.args.hidden_size_char, state_is_tuple=True)
            _output = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, char_embeddings,
                    sequence_length=word_lengths, dtype=tf.float32)

            # read and concat output
            _, ((_, output_fw), (_, output_bw)) = _output
            output = tf.concat([output_fw, output_bw], axis=-1)

            # shape = (batch size, max sentence length, char hidden size)
            output = tf.reshape(output, shape=[s[0], s[1], 2*self.args.hidden_size_char])
            word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        # Concatenate word embbeddings with char embeddings to get final word embeddings
        self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)

        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.args.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.args.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32, shape=[2*self.args.hidden_size_lstm, self.ntags])

            b = tf.get_variable("b", shape=[self.ntags], dtype=tf.float32, initializer=tf.zeros_initializer())
            # TODO incele
            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.args.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.ntags])

        # Prediction (without CRF!)
        #self.tags_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

        # CRF
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
            self.logits, self.tags, self.sequence_lengths)
        self.trans_params = trans_params  # need to evaluate it for decoding
        self.loss = tf.reduce_mean(-log_likelihood)

        # Loss calculation
        #losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.tags)
        #mask = tf.sequence_mask(self.sequence_lengths)
        #losses = tf.boolean_mask(losses, mask)
        #self.loss = tf.reduce_mean(losses)

        tf.summary.scalar("loss", self.loss)

        # Optimization
        _lr_m = self.args.lr_method.lower()

        with tf.variable_scope("train_step"):
            if _lr_m == 'adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            # gradient clipping if clip is positive
            if self.args.grad_clip > 0:
                grads, vs = zip(*optimizer.compute_gradients(self.loss))
                grads, gnorm = tf.clip_by_global_norm(grads, self.args.grad_clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(self.loss)

        # Initialize variables
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        # Tensorboard
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.args.output, self.sess.graph)

    def run_epoch(self, epoch):
        """Performs one complete pass over the train set and evaluate on dev.

        :param data_loader: feeds train and dev data
        :param epoch: current epoch index
        :return: score to select model on, higher is better
        """
        # Iterate over dataset
        for batch_count in range(self.data_loader.num_batches):
            words, chars, tags = self.data_loader.next()
            fd, _ = self.get_feed_dict(words, chars, tags=tags, learning_rate=self.args.learning_rate,
                                       dropout=self.args.dropout)

            _, train_loss, summary = self.sess.run([self.train_op, self.loss, self.merged], feed_dict=fd)

            # Tensorboard
            if batch_count % 10 == 0:
                self.file_writer.add_summary(summary, epoch * self.data_loader.num_batches + batch_count)

        metrics = self.run_evaluate()
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        print(msg)

        return metrics["f1"]

    def get_feed_dict(self, words, chars, tags=None, learning_rate=None, dropout=None):
        word_ids, sequence_lengths = self.pad_sequences(words, 0)
        char_ids, word_lengths = self.pad_sequences(chars, 0, nlevels=2)

        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths,
            self.char_ids: char_ids,
            self.word_lengths: word_lengths
        }

        if tags is not None:
            tags, _ = self.pad_sequences(tags, 0)
            feed[self.tags] = tags

        if learning_rate is not None:
            feed[self.learning_rate] = learning_rate

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths

    def run_evaluate(self):
        """Evaluates performance on test set

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...
        """

        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for batch_count in range(self.data_loader.num_dev_batches):
            words, chars, tags = self.data_loader.next("dev")
            tags_pred, sequence_lengths = self.predict_batch(words, chars)

            for tag, tag_pred, length in zip(tags, tags_pred,
                                             sequence_lengths):
                tag = tag[:length]
                tag_pred = tag_pred[:length]
                accs += [a==b for (a, b) in zip(tag, tag_pred)]

                lab_chunks = set(self.get_chunks(tag, self.data_loader.tag_to_id))
                lab_pred_chunks = set(self.get_chunks(tag_pred,
                                                 self.data_loader.tag_to_id))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        self.data_loader.reset_pointer("dev")

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"acc": 100*acc, "f1": 100*f1}

    def get_chunks(self, seq, tags):
        """Given a sequence of tags, group entities and their position

        Args:
            seq: [4, 4, 0, 0, ...] sequence of labels
            tags: dict["O"] = 4

        Returns:
            list of (chunk_type, chunk_start, chunk_end)

        Example:
            seq = [4, 5, 0, 3]
            tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
            result = [("PER", 0, 2), ("LOC", 3, 4)]

        """
        default = tags["O"]
        idx_to_tag = {idx: tag for tag, idx in tags.items()}
        chunks = []
        chunk_type, chunk_start = None, None
        for i, tok in enumerate(seq):
            # End of a chunk 1
            if tok == default and chunk_type is not None:
                # Add a chunk.
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None

            # End of a chunk + start of a chunk!
            elif tok != default:
                tok_chunk_class, tok_chunk_type = self.get_chunk_type(tok, idx_to_tag)
                if chunk_type is None:
                    chunk_type, chunk_start = tok_chunk_type, i
                elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                    chunk_type, chunk_start = tok_chunk_type, i
            else:
                pass

        # end condition
        if chunk_type is not None:
            chunk = (chunk_type, chunk_start, len(seq))
            chunks.append(chunk)

        return chunks

    def get_chunk_type(self, tok, idx_to_tag):
        """
        Args:
            tok: id of token, ex 4
            idx_to_tag: dictionary {4: "B-PER", ...}

        Returns:
            tuple: "B", "PER"

        """
        tag_name = idx_to_tag[tok]
        tag_class = tag_name.split('-')[0]
        tag_type = tag_name.split('-')[-1]
        return tag_class, tag_type

    def predict_batch(self, words, chars):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length
        """
        fd, sequence_lengths = self.get_feed_dict(words, chars, dropout=1.0)
        # get tag scores and transition params of CRF
        viterbi_sequences = []
        logits, trans_params = self.sess.run(
                [self.logits, self.trans_params], feed_dict=fd)

        # iterate over the sentences because no batching in viterbi_decode
        for logit, sequence_length in zip(logits, sequence_lengths):
            logit = logit[:sequence_length] # keep only the valid steps
            viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params)
            viterbi_sequences += [viterbi_seq]

        return viterbi_sequences, sequence_lengths

    def save_session(self, path):
        """Save session, weights"""
        self.saver.save(self.sess, path)

    def pad_sequences(self, sequences, pad_tok, nlevels=1):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the char to pad with
            nlevels: "depth" of padding, for the case where we have characters ids

        Returns:
            a list of list where each sublist has same length

        """
        if nlevels == 1:
            max_length = max(map(lambda x: len(x), sequences))
            sequence_padded, sequence_length = self._pad_sequences(sequences, pad_tok, max_length)

        elif nlevels == 2:
            max_length_word = max([max(map(lambda x: len(x), seq))
                                   for seq in sequences])
            sequence_padded, sequence_length = [], []
            for seq in sequences:
                # all words are same length now
                sp, sl = self._pad_sequences(seq, pad_tok, max_length_word)
                sequence_padded += [sp]
                sequence_length += [sl]

            max_length_sentence = max(map(lambda x : len(x), sequences))
            sequence_padded, _ = self._pad_sequences(sequence_padded, [pad_tok]*max_length_word, max_length_sentence)
            sequence_length, _ = self._pad_sequences(sequence_length, 0, max_length_sentence)

        return sequence_padded, sequence_length

    @staticmethod
    def _pad_sequences(sequences, pad_tok, max_length):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the char to pad with

        Returns:
            a list of list where each sublist has same length
        """
        sequence_padded, sequence_length = [], []

        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
            sequence_padded += [seq_]
            sequence_length += [min(len(seq), max_length)]

        return sequence_padded, sequence_length
