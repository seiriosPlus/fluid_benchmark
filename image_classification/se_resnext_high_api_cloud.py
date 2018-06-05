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

import os
import numpy as np
import time
import sys
import tempfile

import paddle as paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

is_gpu = False
learning_rate=0.1
num_passes = 200
batch_size = 40
optimize_choose = 2
checkpoint_hdfs_dir=None
checkpoint_dir=None
model_dir="init_model_path"


def conv_bn_layer(input, num_filters, filter_size, stride=1, groups=1,
                  act=None):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=(filter_size - 1) / 2,
        groups=groups,
        act=None,
        bias_attr=False)
    return fluid.layers.batch_norm(input=conv, act=act)


def squeeze_excitation(input, num_channels, reduction_ratio):
    pool = fluid.layers.pool2d(
        input=input, pool_size=0, pool_type='avg', global_pooling=True)
    squeeze = fluid.layers.fc(input=pool,
                              size=num_channels / reduction_ratio,
                              act='relu')
    excitation = fluid.layers.fc(input=squeeze,
                                 size=num_channels,
                                 act='sigmoid')
    scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
    return scale


def shortcut(input, ch_out, stride):
    ch_in = input.shape[1]
    if ch_in != ch_out:
        if stride == 1:
            filter_size = 1
        else:
            filter_size = 3
        return conv_bn_layer(input, ch_out, filter_size, stride)
    else:
        return input


def bottleneck_block(input, num_filters, stride, cardinality, reduction_ratio):
    conv0 = conv_bn_layer(
        input=input, num_filters=num_filters, filter_size=1, act='relu')
    conv1 = conv_bn_layer(
        input=conv0,
        num_filters=num_filters,
        filter_size=3,
        stride=stride,
        groups=cardinality,
        act='relu')
    conv2 = conv_bn_layer(
        input=conv1, num_filters=num_filters * 2, filter_size=1, act=None)
    scale = squeeze_excitation(
        input=conv2,
        num_channels=num_filters * 2,
        reduction_ratio=reduction_ratio)

    short = shortcut(input, num_filters * 2, stride)

    return fluid.layers.elementwise_add(x=short, y=scale, act='relu')


def SE_ResNeXt(input, class_dim, infer=False, layers=50):
    supported_layers = [50, 152]
    if layers not in supported_layers:
        print("supported layers are", supported_layers, "but input layer is",
              layers)
        exit()
    if layers == 50:
        cardinality = 32
        reduction_ratio = 16
        depth = [3, 4, 6, 3]
        num_filters = [128, 256, 512, 1024]

        conv = conv_bn_layer(
            input=input, num_filters=64, filter_size=7, stride=2, act='relu')
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')
    elif layers == 152:
        cardinality = 64
        reduction_ratio = 16
        depth = [3, 8, 36, 3]
        num_filters = [128, 256, 512, 1024]

        conv = conv_bn_layer(
            input=input, num_filters=64, filter_size=3, stride=2, act='relu')
        conv = conv_bn_layer(
            input=conv, num_filters=64, filter_size=3, stride=1, act='relu')
        conv = conv_bn_layer(
            input=conv, num_filters=128, filter_size=3, stride=1, act='relu')
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

    for block in range(len(depth)):
        for i in range(depth[block]):
            conv = bottleneck_block(
                input=conv,
                num_filters=num_filters[block],
                stride=2 if i == 0 and block != 0 else 1,
                cardinality=cardinality,
                reduction_ratio=reduction_ratio)

    pool = fluid.layers.pool2d(
        input=conv, pool_size=0, pool_type='avg', global_pooling=True)
    if not infer:
        drop = fluid.layers.dropout(x=pool, dropout_prob=0.2)
    else:
        drop = pool
    out = fluid.layers.fc(input=drop, size=class_dim, act='softmax')
    return out


def inference_network():
    class_dim = 1000
    image_shape = [3, 224, 224]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    out = SE_ResNeXt(input=image, class_dim=class_dim, layers=layers)
    return out


def train_network():
    out = inference_network()
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

    return [avg_cost, acc_top1, acc_top5]


def get_optimizer():
    if lr_strategy is None:
        optimizer = fluid.optimizer.Momentum(
            learning_rate=learning_rate,
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))
    else:
        bd = lr_strategy["bd"]
        lr = lr_strategy["lr"]

        choose = optimize_choose

        if choose == 1:
            print("use optimizer Momentum with learning_rate=fluid.layers.piecewise_decay")
            optimizer = fluid.optimizer.Momentum(
                learning_rate=fluid.layers.piecewise_decay(
                    boundaries=bd, values=lr),
                momentum=0.9,
                regularization=fluid.regularizer.L2Decay(1e-4))
        elif choose == 2:
            print("use optimizer Momentum with learning_rate=learning_rate")
            optimizer = fluid.optimizer.Momentum(
                learning_rate=learning_rate,
                momentum=0.9,
                regularization=fluid.regularizer.L2Decay(1e-4))
        else:
            print("use optimizer SGD")
            optimizer = fluid.optimizer.SGD(learning_rate=learning_rate)

    return optimizer


def get_place():
    place = core.CPUPlace() if not is_gpu else core.CUDAPlace(0)
    return place


def get_reader():
    train_reader = paddle.batch(
        paddle.dataset.flowers.train(), batch_size=batch_size)
    test_reader = paddle.batch(
        paddle.dataset.flowers.test(), batch_size=batch_size)
    return train_reader, test_reader


def put_ckpt_hdfs():
    print "put %s to %s" % (checkpoint_dir, checkpoint_hdfs_dir)
    serial = fluid.io.get_latest_checkpoint_serial(checkpoint_dir)
    if serial < 0:
        print "serial: %s, there is no need to put to hdfs." % serial
    dirname = fluid.io._get_serial_dir(checkpoint_dir, serial)
    print "serial: %s, will put %s to hdfs." % (serial, dirname)

def get_ckpt_hdfs():
    print "get %s from %s" % (checkpoint_hdfs_dir, checkpoint_dir)

def get_ckpt_config():
    if not checkpoint_hdfs_dir:
        return None
    get_ckpt_hdfs()

    max_num = os.getenv("CHECKPOINT_MAX_NUM", 3)
    epoch_interval = os.getenv("CHECKPOINT_EPOCH_INTERVAL", 1)
    step_interval = os.getenv("CHECKPOINT_STEP_INTERVAL", 10)
    config = fluid.CheckpointConfig(checkpoint_dir, max_num, epoch_interval, step_interval)
    return config


def train():
    ckpt_config = get_ckpt_config()
    print "config checkpoint finish ..."

    trainer = fluid.Trainer(
        train_func=train_network,
        place=get_place(),
        optimizer=get_optimizer(),
        param_path=model_dir,
        checkpoint_config=ckpt_config)

    train_reader, test_reader = get_reader()

    def event_handler(event):
        if isinstance(event, fluid.BeginStepEvent):
            pass

        if isinstance(event, fluid.EndStepEvent):
            loss, acc1, acc5 = event.metrics
            print("Epoch {0}, Step {1}, loss {2}, acc1 {3}, acc5 {4} time {5}".format(
                event.epoch, event.step, loss[0], acc1[0], acc5[0], "%2.2f sec" % 0.00))

            if ckpt_config:
                put_ckpt_hdfs()

    trainer.train(reader=train_reader, num_epochs=10,
                                event_handler=event_handler, feed_order=['image', 'label'])

    # if models_dir:
    #     trainer.save_params(".outputs/models")


if __name__ == '__main__':
    batch_size = int(os.getenv("BATCH_SIZE"), 20)
    optimize_choose = int(os.getenv("OPTIMIZE_CHOOSE"),3)
    is_gpu = True if os.getenv("PLACE", "CPU") == "GPU" else False
    
    checkpoint_hdfs_dir = os.getenv("CHECKPOINT_HDFS_DIR", "")
    checkpoint_hdfs_dir = checkpoint_hdfs_dir.strip()
    checkpoint_dir = tempfile.mktemp()

    print "batch_size: ", batch_size
    print "optimize_choose: ", optimize_choose
    print "is_gpu: ", "True" if is_gpu else "False"
    print "checkpoint_hdfs_dir: ", checkpoint_hdfs_dir    
    print "checkpoint_dir: ", checkpoint_dir

    epoch_points = [30, 60, 90]
    total_images = 8789
    step = int(total_images / batch_size + 1)
    bd = [e * step for e in epoch_points]
    lr = [0.1, 0.01, 0.001, 0.0001]
    lr_strategy = {"bd": bd, "lr": lr}
    # layers: 50, 152
    layers = 50

    print "Start  Train ..."
    train()
    print "Finish Train ..."
