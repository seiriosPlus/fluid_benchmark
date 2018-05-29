import os
import sys
import argparse
import time

import paddle as paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.profiler as profiler
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--local',
    type=str2bool,
    default=True,
    help='Whether to run as local mode.')

parser.add_argument(
    '--batch_size',
    type=int,
    default=40,
    help='Whether to run as local mode.')

parser.add_argument(
    '--device',
    type=str,
    default='CPU',
    choices=['CPU', 'GPU'],
    help="The device type.")

parser.add_argument('--device_id', type=int, default=0, help="The device id.")

parser.add_argument(
    '--num_passes',
    type=int,
    default=1,
    help='Whether to run as local mode.')

parser.add_argument(
    '--accuracy',
    type=str2bool,
    default=False,
    help='accuracy mode, in this mode, will split input data')

parser.add_argument(
    "--ps_hosts",
    type=str,
    default="",
    help="Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
parser.add_argument(
    "--task_index", type=int, default=0, help="Index of task within the job")

args = parser.parse_args()


parameter_attr = ParamAttr(initializer=MSRA())


def conv_bn_layer(input,
                  filter_size,
                  num_filters,
                  stride,
                  padding,
                  channels=None,
                  num_groups=1,
                  act='relu',
                  use_cudnn=True):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=padding,
        groups=num_groups,
        act=None,
        use_cudnn=use_cudnn,
        param_attr=parameter_attr,
        bias_attr=False)
    #return fluid.layers.batch_norm(input=conv, act=act)
    return conv 


def depthwise_separable(input, num_filters1, num_filters2, num_groups, stride,
                        scale):
    """
    """
    depthwise_conv = conv_bn_layer(
        input=input,
        filter_size=3,
        num_filters=int(num_filters1 * scale),
        stride=stride,
        padding=1,
        num_groups=int(num_groups * scale),
        use_cudnn=False)

    pointwise_conv = conv_bn_layer(
        input=depthwise_conv,
        filter_size=1,
        num_filters=int(num_filters2 * scale),
        stride=1,
        padding=0)
    return pointwise_conv


def mobile_net(img, class_dim, scale=1.0):

    # conv1: 112x112
    tmp = conv_bn_layer(
        img,
        filter_size=3,
        channels=3,
        num_filters=int(32 * scale),
        stride=2,
        padding=1)

    # 56x56
    tmp = depthwise_separable(
        tmp,
        num_filters1=32,
        num_filters2=64,
        num_groups=32,
        stride=1,
        scale=scale)

    tmp = depthwise_separable(
        tmp,
        num_filters1=64,
        num_filters2=128,
        num_groups=64,
        stride=2,
        scale=scale)

    # 28x28
    tmp = depthwise_separable(
        tmp,
        num_filters1=128,
        num_filters2=128,
        num_groups=128,
        stride=1,
        scale=scale)

    tmp = depthwise_separable(
        tmp,
        num_filters1=128,
        num_filters2=256,
        num_groups=128,
        stride=2,
        scale=scale)

    # 14x14
    tmp = depthwise_separable(
        tmp,
        num_filters1=256,
        num_filters2=256,
        num_groups=256,
        stride=1,
        scale=scale)

    tmp = depthwise_separable(
        tmp,
        num_filters1=256,
        num_filters2=512,
        num_groups=256,
        stride=2,
        scale=scale)

    # 14x14
    for i in range(5):
        tmp = depthwise_separable(
            tmp,
            num_filters1=512,
            num_filters2=512,
            num_groups=512,
            stride=1,
            scale=scale)
    # 7x7
    tmp = depthwise_separable(
        tmp,
        num_filters1=512,
        num_filters2=1024,
        num_groups=512,
        stride=2,
        scale=scale)

    tmp = depthwise_separable(
        tmp,
        num_filters1=1024,
        num_filters2=1024,
        num_groups=1024,
        stride=1,
        scale=scale)

    tmp = fluid.layers.pool2d(
        input=tmp,
        pool_size=0,
        pool_stride=1,
        pool_type='avg',
        global_pooling=True)

    tmp = fluid.layers.fc(input=tmp,
                          size=class_dim,
                          act='softmax',
                          param_attr=parameter_attr)
    return tmp


def local_train(learning_rate, batch_size, num_passes, model_save_dir='model'):
    class_dim = 102
    image_shape = [3, 224, 224]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    out = mobile_net(image, class_dim=class_dim)

    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    optimizer = fluid.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(5 * 1e-5))
#    optimizer = fluid.optimizer.SGD(learning_rate=learning_rate)
    opts = optimizer.minimize(avg_cost)

    b_size_var = fluid.layers.create_tensor(dtype='int64')
    b_acc_var = fluid.layers.accuracy(input=out, label=label, total=b_size_var)

    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        inference_program = fluid.io.get_inference_program(
            target_vars=[b_acc_var, b_size_var])

    place = core.CPUPlace() if args.device == 'CPU' else core.CUDAPlace(
        args.device_id)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    train_reader = paddle.batch(
        paddle.dataset.flowers.train(), batch_size=batch_size)
    test_reader = paddle.batch(
        paddle.dataset.flowers.test(), batch_size=batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    train_pass_acc_evaluator = fluid.average.WeightedAverage()
    test_pass_acc_evaluator = fluid.average.WeightedAverage()

    train_proc = fluid.default_main_program()

    for pass_id in range(num_passes):
        start = time.clock()
        train_pass_acc_evaluator.reset()
        for batch_id, data in enumerate(train_reader()):
            loss, acc, size = exe.run(
                train_proc,
                feed=feeder.feed(data),
                fetch_list=[avg_cost, b_acc_var, b_size_var])
            train_pass_acc_evaluator.add(value=acc, weight=size)
            print("Pass {0}, batch {1}, loss {2}, acc {3}".format(
                pass_id, batch_id, loss[0], acc[0]))

        test_pass_acc_evaluator.reset()
        for data in test_reader():
            acc, size = exe.run(
                inference_program,
                feed=feeder.feed(data),
                fetch_list=[b_acc_var, b_size_var])
            test_pass_acc_evaluator.add(value=acc, weight=size)
        print("End pass {0}, train_acc {1}, test_acc {2}, cost {3} second".format(
           pass_id,
           train_pass_acc_evaluator.eval(), test_pass_acc_evaluator.eval(), time.clock() - start))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def accuracy_data(trainers, trainer_id, datas):
    partitions = list(chunks(range(len(datas)), len(datas)/trainers))

    data = []

    for id in partitions[trainer_id]:
        data.append(datas[id])

    return data


def cluster_train(learning_rate, batch_size, num_passes, model_save_dir='model'):
    class_dim = 102
    image_shape = [3, 224, 224]
    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    out = mobile_net(image, class_dim=class_dim)
    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    optimizer = fluid.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(5 * 1e-5))
    optimize_ops, params_grads = optimizer.minimize(avg_cost)

    place = core.CPUPlace() if args.device == 'CPU' else core.CUDAPlace(
        args.device_id)
    exe = fluid.Executor(place)

    standalone = int(os.getenv("STANDALONE", 0))

    if standalone:
        pserver_endpoints = os.getenv("PSERVERS")
        trainers = int(os.getenv("TRAINERS"))
        current_endpoint = os.getenv("SERVER_ENDPOINT")
        trainer_id = int(os.getenv("PADDLE_INIT_TRAINER_ID"))
        training_role = os.getenv("TRAINING_ROLE", "TRAINER")

    else:
        pport = os.getenv("PADDLE_INIT_PSERVER_PORT", "6174")
        tport = os.getenv("PADDLE_INIT_TRAINER_PORT", "6174")
        pserver_ips = os.getenv("PADDLE_INIT_PSERVERS")  # ip,ip...
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, pport]))
        pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
        trainers = int(os.getenv("TRAINERS"))
        trainer_id = int(os.getenv("PADDLE_INIT_TRAINER_ID"))
        training_role = os.getenv("TRAINING_ROLE", "TRAINER")

        if training_role == "PSERVER":
            current_endpoint =  os.getenv("POD_IP") + ":" + pport
        else:
            current_endpoint =  os.getenv("POD_IP") + ":" + tport

    print("pserver_endpoints: {0}, trainers: {1}, current_endpoint: {2}, trainer_id: {3}, training_role: {4}".format(pserver_endpoints, trainers,current_endpoint,trainer_id,training_role))

    t = fluid.DistributeTranspiler()
    t.transpile(
        optimize_ops,
        params_grads,
        trainer_id=trainer_id,
        pservers=pserver_endpoints,
        trainers=trainers)

    if training_role == "PSERVER":
        pserver_prog = t.get_pserver_program(current_endpoint)
        pserver_startup = t.get_startup_program(current_endpoint, pserver_prog)
        exe.run(pserver_startup)
        exe.run(pserver_prog)
    elif training_role == "TRAINER":
        b_size_var = fluid.layers.create_tensor(dtype='int64')
        b_acc_var = fluid.layers.accuracy(input=out, label=label, total=b_size_var)
        inference_program = fluid.default_main_program().clone()
        with fluid.program_guard(inference_program):
            inference_program = fluid.io.get_inference_program(
                target_vars=[b_acc_var, b_size_var])
    
        exe.run(fluid.default_startup_program())
    
        train_reader = paddle.batch(
            paddle.dataset.flowers.train(), batch_size=batch_size)
        test_reader = paddle.batch(
            paddle.dataset.flowers.test(), batch_size=batch_size)
        feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

        train_proc = t.get_trainer_program()
    
        train_pass_acc_evaluator = fluid.average.WeightedAverage()
        test_pass_acc_evaluator = fluid.average.WeightedAverage()

        # with profiler.profiler(args.device, 'total', "profilers.log"):

        for pass_id in range(num_passes):
            start = time.clock()
            train_pass_acc_evaluator.reset()
            for batch_id, data in enumerate(train_reader()):

                if args.accuracy: 
                    data = accuracy_data(trainers, trainer_id, data)

                loss, acc, size = exe.run(
                    train_proc,
                    feed=feeder.feed(data),
                    fetch_list=[avg_cost, b_acc_var, b_size_var])
                train_pass_acc_evaluator.add(value=acc, weight=size)
                print("Pass {0}, batch {1}, loss {2}, acc {3}".format(
                    pass_id, batch_id, loss[0], acc[0]))

            test_pass_acc_evaluator.reset()
            for data in test_reader():
                acc, size = exe.run(
                    inference_program,
                    feed=feeder.feed(data),
                    fetch_list=[b_acc_var, b_size_var])
                test_pass_acc_evaluator.add(value=acc, weight=size)
            print("End pass {0}, train_acc {1}, test_acc {2}, cost {3} second".format(
               pass_id,
               train_pass_acc_evaluator.eval(), test_pass_acc_evaluator.eval(), 
               time.clock() - start))


def print_arguments():
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == '__main__':
    print_arguments()
    if args.local:
        local_train(learning_rate=0.005, batch_size=args.batch_size, num_passes=args.num_passes)
    else:
        cluster_train(learning_rate=0.005, batch_size=args.batch_size, num_passes=args.num_passes)

