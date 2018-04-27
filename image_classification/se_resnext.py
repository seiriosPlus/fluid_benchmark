import os
import numpy as np
import time
import sys
import paddle as paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

is_accuracy = False
is_debug = False
is_gpu = False
is_cluster = False 
is_parallelexecutor = False
standalone = False

learning_rate=0.1
num_passes = 200
batch_size = 40
optimize_choose = 2

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
    #return fluid.layers.layer_norm(input=conv, act=act)
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


def serial_executor(
                    init_model=None,
                    model_save_dir='model',
                    parallel=True,
                    use_nccl=True,
                    lr_strategy=None,
                    layers=50):

    class_dim = 1000
    image_shape = [3, 224, 224]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    out = SE_ResNeXt(input=image, class_dim=class_dim, layers=layers)
    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    size_var = fluid.layers.create_tensor(dtype='int64')
    acc_var = fluid.layers.accuracy(input=out, label=label, total=size_var)

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

    optimize_ops, params_grads = optimizer.minimize(avg_cost)

    inference_program = fluid.default_main_program().clone()
    with fluid.program_guard(inference_program):
        inference_program = fluid.io.get_inference_program(
            [avg_cost, acc_var, size_var])

    train_reader = paddle.batch(
        paddle.dataset.flowers.train(), batch_size=batch_size)
    test_reader = paddle.batch(
        paddle.dataset.flowers.test(), batch_size=batch_size)

    place = core.CPUPlace() if not is_gpu else core.CUDAPlace(0)
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    if not is_cluster:
        exe.run(fluid.default_startup_program())
        train_proc = fluid.default_main_program()
    else:
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
        else:
            exe.run(fluid.default_startup_program())
            train_proc = t.get_trainer_program()


    for pass_id in range(num_passes):
        start = time.time()
        train_info = [[], [], []]
        test_info = [[], [], []]

        for batch_id, data in enumerate(train_reader()):
            t1 = time.time()

            if is_accuracy:
                data = accuracy_data(trainers, trainer_id, data)

            loss, acc1, acc5 = exe.run(
                train_proc,
                feed=feeder.feed(data),
                fetch_list=[avg_cost, acc_var, size_var])
            t2 = time.time()
            period = t2 - t1
            train_info[0].append(loss[0])
            train_info[1].append(acc1[0])
            train_info[2].append(acc5[0])
            print("Pass {0}, batch {1}, loss {2}, \
                   acc {3}, size {4} time {5}"
                                               .format(pass_id, \
                   batch_id, loss[0], acc1[0], acc5[0], \
                   "%2.2f sec" % period))
            sys.stdout.flush()

            if is_debug:
                print("Just for Debug, break loop, to speed up test")
                break

        train_loss = np.array(train_info[0]).mean()
        train_acc1 = np.array(train_info[1]).mean()
        train_acc5 = np.array(train_info[2]).mean()

        for data in test_reader():
            t1 = time.time()
            acc1, acc5 = exe.run(
                inference_program,
                feed=feeder.feed(data),
                fetch_list=[acc_var, size_var])
            t2 = time.time()
            period = t2 - t1
            test_info[0].append(0.0)
            test_info[1].append(acc1[0])
            test_info[2].append(acc5[0])
            if batch_id % 10 == 0:
                print("Pass {0},testbatch {1},loss {2}, \
                       acc {3}, size {4},time {5}"
                                                  .format(pass_id, \
                       batch_id, 0.0, acc1[0], acc5[0], \
                       "%2.2f sec" % period))
                sys.stdout.flush()
            if is_debug:
                print("Just for Debug, break loop, to speed up test")
                break

        test_loss = np.array(test_info[0]).mean()
        test_acc1 = np.array(test_info[1]).mean()
        test_acc5 = np.array(test_info[2]).mean()

        print("End pass {0}, train_loss {1}, train_acc {2}, train_acc_size {3}, \
               test_loss {4}, test_acc {5}, test_acc_size {6}, time period sec {7}"
                                                           .format(pass_id, \
              train_loss, train_acc1, train_acc5, test_loss, test_acc1, \
              test_acc5, time.time()-start))
        sys.stdout.flush()

        model_path = os.path.join(model_save_dir, str(pass_id))
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        fluid.io.save_persistables(exe, model_path)


def parallel_executor(
                    init_model=None,
                    model_save_dir='model',
                    parallel=True,
                    use_nccl=True,
                    lr_strategy=None,
                    layers=50):

    class_dim = 1000
    image_shape = [3, 224, 224]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    out = SE_ResNeXt(input=image, class_dim=class_dim, layers=layers)
    cost = fluid.layers.cross_entropy(input=out, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)
    test_program = fluid.default_main_program().clone(for_test=True)

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

    optimize_ops, params_grads = optimizer.minimize(avg_cost)

    train_reader = paddle.batch(
        paddle.dataset.flowers.train(), batch_size=batch_size)
    test_reader = paddle.batch(
        paddle.dataset.flowers.test(), batch_size=batch_size)

    if not is_cluster:
        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

        train_exe = fluid.ParallelExecutor(use_cuda=True, loss_name=avg_cost.name)
        test_exe = fluid.ParallelExecutor(use_cuda=True, main_program=test_program, share_vars_from=train_exe)
    else:
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

        print("pserver_endpoints: {0}, trainers: {1}, current_endpoint: {2}, trainer_id: {3}, training_role: {4}".format(pserver_endpoints, 
                  trainers,current_endpoint,trainer_id,training_role))

        t = fluid.DistributeTranspiler()
        t.transpile(
            optimize_ops,
            params_grads,
            trainer_id=trainer_id,
            pservers=pserver_endpoints,
            trainers=trainers)

        if training_role == "PSERVER":

            place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(current_endpoint, pserver_prog)
            exe.run(pserver_startup)
            exe.run(pserver_prog)
        else:

            place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

            train_proc = t.get_trainer_program()
            train_exe = fluid.ParallelExecutor(use_cuda=True, main_program=train_proc, loss_name=avg_cost.name)
            test_exe = fluid.ParallelExecutor(use_cuda=True, main_program=test_program, share_vars_from=train_exe)

    fetch_list = [avg_cost.name, acc_top1.name, acc_top5.name]

    for pass_id in range(num_passes):
        pass_t1 = time.time()

        train_info = [[], [], []]
        test_info = [[], [], []]
        for batch_id, data in enumerate(train_reader()):
            t1 = time.time()
            loss, acc1, acc5 = train_exe.run(fetch_list, feed=feeder.feed(data))
            t2 = time.time()
            period = t2 - t1
            loss = np.mean(np.array(loss))
            acc1 = np.mean(np.array(acc1))
            acc5 = np.mean(np.array(acc5))
            train_info[0].append(loss)
            train_info[1].append(acc1)
            train_info[2].append(acc5)
            if batch_id % 10 == 0:
                print("Pass {0}, trainbatch {1}, loss {2}, \
                       acc1 {3}, acc5 {4} time {5}".format(pass_id, \
                       batch_id, loss, acc1, acc5, "%2.2f sec" % period))
                sys.stdout.flush()

            if is_debug:
                print "DEBUG ..."
                break

        train_loss = np.array(train_info[0]).mean()
        train_acc1 = np.array(train_info[1]).mean()
        train_acc5 = np.array(train_info[2]).mean()
        for data in test_reader():
            t1 = time.time()
            loss, acc1, acc5 = test_exe.run(fetch_list,
                                            feed=feeder.feed(data))
            t2 = time.time()
            period = t2 - t1
            loss = np.mean(np.array(loss))
            acc1 = np.mean(np.array(acc1))
            acc5 = np.mean(np.array(acc5))
            test_info[0].append(loss)
            test_info[1].append(acc1)
            test_info[2].append(acc5)
            if batch_id % 10 == 0:
                print("Pass {0},testbatch {1},loss {2}, \
                       acc1 {3},acc5 {4},time {5}"
                                                  .format(pass_id, \
                       batch_id, loss, acc1, acc5, \
                       "%2.2f sec" % period))
                sys.stdout.flush()
            if is_debug:
                print "DEBUG ..."
                break

        test_loss = np.array(test_info[0]).mean()
        test_acc1 = np.array(test_info[1]).mean()
        test_acc5 = np.array(test_info[2]).mean()

        print("End pass {0}, train_loss {1}, train_acc1 {2}, train_acc5 {3}, \
               test_loss {4}, test_acc1 {5}, test_acc5 {6}, cost time {7}"
                                                           .format(pass_id, \
              train_loss, train_acc1, train_acc5, test_loss, test_acc1, \
              test_acc5, time.time() - pass_t1))
        sys.stdout.flush()

        model_path = os.path.join(model_save_dir, str(pass_id))
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        fluid.io.save_persistables(exe, model_path)


if __name__ == '__main__':

    standalone = int(os.getenv("STANDALONE", 0))
    batch_size = int(sys.argv[1]) 
    optimize_choose = int(sys.argv[2])
    is_debug = int(sys.argv[3])
    is_gpu = True if sys.argv[4] == "GPU" else False
    is_accuracy = int(sys.argv[5]) 
    is_cluster = int(sys.argv[6])
    is_parallelexecutor = int(sys.argv[7])

    print "batch_size: ", batch_size
    print "optimize_choose: ", optimize_choose
    print "is_debug    : ", "True" if is_debug else "False"
    print "is_gpu      : ", "True" if is_gpu else "False"
    print "is_accuracy : ", "True" if is_accuracy else "False"
    print "is_cluster  : ", "True" if is_cluster else "False"
    print "standalone  : ", "True" if standalone else "False"
    print "is_parallelexecutor : ", "True" if is_parallelexecutor else "False"

    epoch_points = [30, 60, 90]
    total_images = 8789
    step = int(total_images / batch_size + 1)
    bd = [e * step for e in epoch_points]
    lr = [0.1, 0.01, 0.001, 0.0001]

    lr_strategy = {"bd": bd, "lr": lr}

    use_nccl = True
    # layers: 50, 152
    layers = 50

    if not is_parallelexecutor:
        serial_executor(
                    init_model=None,
                    model_save_dir='model',
                    parallel=False,
                    use_nccl=True,
                    lr_strategy=lr_strategy,
                    layers=layers)
    else:
        if not is_gpu:
            print "parallel executor need GPU, eixt"
            sys.exit(1) 

        parallel_executor(
                    init_model=None,
                    model_save_dir='model',
                    parallel=False,
                    use_nccl=True,
                    lr_strategy=lr_strategy,
                    layers=layers)
