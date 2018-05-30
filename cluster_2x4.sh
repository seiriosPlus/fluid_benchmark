#!/bin/bash

echo "WARNING: This script only for run PaddlePaddle Fluid on one node..."
echo "WARNING: You must to modify train.py manual..."
echo ""

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export PADDLE_PSERVER_PORTS=36001,36002
export PADDLE_PSERVER_PORT_ARRAY=(36001 36002)
export PADDLE_PSERVERS=2
export PADDLE_IP=127.0.0.1
export PADDLE_TRAINERS=4

if [ "$1" = "ps" ]
then
    export PADDLE_TRAINING_ROLE=PSERVER    
    export GLOG_v=0
    export GLOG_logtostderr=1

    for((i=0;i<$PADDLE_PSERVERS;i++))
    do
        cur_port=${PADDLE_PSERVER_PORT_ARRAY[$i]}
        echo "PADDLE WILL START PSERVER "$cur_port
        CUR_PORT=$cur_port PADDLE_TRAINER_ID=$i stdbuf -oL python /models/image_classification/se_resnext_high_api.py 4 2 0 CPU 0 1 0 &> pserver.$i &
    done
fi

if [ "$1" = "tr" ]
then
    export PADDLE_TRAINING_ROLE=TRAINER
    export GLOG_v=0
    export GLOG_logtostderr=1

    for((i=0;i<$PADDLE_TRAINERS;i++))
    do
        echo "PADDLE WILL START Trainer "$i
        PADDLE_TRAINER_ID=$i stdbuf -oL python /models/image_classification/se_resnext_high_api.py 4 2 0 CPU 0 1 0 &> trainerlog.$i &
    done
fi