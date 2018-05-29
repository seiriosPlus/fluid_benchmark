#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

export TRAINERS=1
export STANDALONE=1

export PSERVERS=127.0.0.1:36001

if [ "$1" = "local" ]
then
    GLOG_v=0 GLOG_logtostderr=1 stdbuf -oL python /models/image_classification/se_resnext_high_api.py 4 2 0 CPU 0 1 0 &> trainerlog.0 &
    exit 0
fi

GLOG_v=0 GLOG_logtostderr=1  SERVER_ENDPOINT=127.0.0.1:36001 TRAINING_ROLE=PSERVER PADDLE_INIT_TRAINER_ID="0" stdbuf -oL python /models/image_classification/se_resnext.py 4 2 0 CPU 0 1 0 &> pserver.0 &

echo "Sleep to wait Pserver linten on ..."
sleep 25

GLOG_v=0 GLOG_logtostderr=1 SERVER_ENDPOINT=127.0.0.1:39001 TRAINING_ROLE=TRAINER PADDLE_INIT_TRAINER_ID="0" stdbuf -oL python /models/image_classification/se_resnext.py 4 2 0 CPU 0 1 0 &> trainerlog.0 &