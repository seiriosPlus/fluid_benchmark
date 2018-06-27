#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Text classification benchmark in Fluid"""
import numpy as np
import sys
import os
import argparse
import time

import paddle
import paddle.fluid as fluid

from config import text_classification_config as conf


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dict_path',
        type=str,
        required=True,
        help="Path of the word dictionary.")
    return parser.parse_args()

def get_place():
    place = fluid.core.CPUPlace() if not conf.use_gpu else fluid.core.CUDAPlace(0)
    return place


def get_reader(word_dict):
    # The training data set.
    train_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.imdb.train(word_dict), buf_size=51200), batch_size=conf.batch_size)

    # The testing data set.
    test_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.imdb.test(word_dict), buf_size=51200), batch_size=conf.batch_size)

    return train_reader, test_reader

def get_optimizer():
    sgd_optimizer = fluid.optimizer.SGD(learning_rate=conf.learning_rate)
    return sgd_optimizer


def inference_network(dict_dim):
    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)
    out = conv_net(data, dict_dim)
    return out


def train_network(dict_dim):
    def true_nn():
        out = inference_network(dict_dim)
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        cost = fluid.layers.cross_entropy(input=out, label=label)
        acc = fluid.layers.accuracy(input=out, label=label)
        avg_cost = fluid.layers.mean(x=cost)
        return [avg_cost, acc]
    return true_nn


def conv_net(
            input,
            dict_dim,
            emb_dim=128,
             window_size=3,
             num_filters=128,
             fc0_dim=96,
             class_dim=2):
    emb = fluid.layers.embedding(input=input, size=[dict_dim, emb_dim], is_sparse=False)

    conv_3 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=num_filters,
        filter_size=window_size,
        act="tanh",
        pool_type="max")

    fc_0 = fluid.layers.fc(input=[conv_3], size=fc0_dim)
    prediction = fluid.layers.fc(input=[fc_0], size=class_dim, act="softmax")
    return prediction

# Load the dictionary.
def load_vocab(filename):
    vocab = {}
    with open(filename) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx
    return vocab


def get_worddict(dict_path):
    word_dict = load_vocab(dict_path)
    word_dict["<unk>"] = len(word_dict)
    dict_dim = len(word_dict)
    return (word_dict, dict_dim)


def as_numpy(tensor):
    if isinstance(tensor, list):
        return [as_numpy(t) for t in tensor]
    assert isinstance(tensor, fluid.core.LoDTensor)
    lod = tensor.lod()
    if len(lod) > 0:
        raise RuntimeError("Some of your fetched tensors hold LoD information. \
            They can not be completely cast to Python ndarray. \
            Please set the parameter 'return_numpy' as 'False' to \
            return LoDTensor itself directly.")
    return np.array(tensor)



def train(dict_path):
    word_dict, dict_dim = get_worddict(dict_path)
    print("[get_worddict] The dictionary size is : %d" % dict_dim)

    training_role = os.getenv("PADDLE_TRAINING_ROLE", "")
    if training_role == "PSERVER":
        train_reader = None
        test_reader = None
    else:
        train_reader, test_reader = get_reader(word_dict)

    trainer = fluid.Trainer(
        train_func=train_network(dict_dim),
        place=get_place(),
        parallel=True,
        optimizer_func=get_optimizer)

    def event_handler(event):
        samples = 25000
        global step_start_time, epoch_start_time, speeds
        global accuracies, losses, t_accuracies, t_losses

        if isinstance(event, fluid.BeginEpochEvent):
            epoch_start_time = time.time()
            losses = []
            accuracies = []
            t_losses = []
            t_accuracies = []

        if isinstance(event, fluid.BeginStepEvent):
            if event.epoch == 0 and event.step == 0:
                speeds = []
            step_start_time = time.time()

        if isinstance(event, fluid.EndStepEvent):
            loss, accuracy = as_numpy(event.metrics)
            losses.append(loss.mean())
            accuracies.append(accuracy.mean())

            t_loss, t_accuracy = trainer.test(reader=test_reader, feed_order=['words', 'label'])
            t_losses.append(t_loss.mean())
            t_accuracies.append(t_accuracy.mean())

        if isinstance(event, fluid.EndEpochEvent):
            epoch_end_time = time.time()
            time_consuming = epoch_end_time-epoch_start_time
            speed = samples/time_consuming
            speeds.append(speed)

            print("Epoch: {0},Time: {1},  Speed: {2}, Avg Speed: {3}, Avg Loss: {4}, Avg accuracy: {5}".format(
                event.epoch, time_consuming, speed, 
                np.array(speeds).mean(),
                np.array(losses).mean(),
                np.array(accuracies).mean()))

            print("Epoch: {0},Test Avg Loss: {1}, Test Avg accuracy: {2}".format(
                event.epoch,  
                np.array(t_losses).mean(),
                np.array(t_accuracies).mean()))

    trainer.train(reader=train_reader, num_epochs=conf.num_passes, 
                                event_handler=event_handler, feed_order=['words', 'label'])


if __name__ == '__main__':
    args = parse_args()
    train(args.dict_path)