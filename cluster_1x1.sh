#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

if [ "$1" = "local" ]
then
    GLOG_v=0 GLOG_logtostderr=1 stdbuf -oL python /models/image_classification/se_resnext_high_api.py 4 2 0 CPU 0 1 0 &> trainerlog.0 &
    exit 0
fi

export PADDLE_PSERVER_PORT=36001
export PADDLE_PSERVER_IPS=127.0.0.1
export PADDLE_TRAINERS=1
export PADDLE_CURRENT_IP=127.0.0.1
export PADDLE_TRAINER_ID=0

if [ "$1" = "ps" ]
then
    export PADDLE_TRAINING_ROLE=PSERVER
     
    export GLOG_v=3
    export GLOG_logtostderr=1

    echo "PADDLE WILL START PSERVER ..."
    stdbuf -oL python /models/image_classification/se_resnext_high_api.py 4 2 0 CPU 0 1 0 &> pserver.0 &
fi

if [ "$1" = "tr" ]
then
    export PADDLE_TRAINING_ROLE=TRAINER

    export GLOG_v=3
    export GLOG_logtostderr=1

    echo "PADDLE WILL START TRAINER ..."
    stdbuf -oL python /models/image_classification/se_resnext_high_api.py 4 2 0 CPU 0 1 0 &> trainerlog.0 &
fi