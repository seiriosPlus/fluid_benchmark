#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/lib

export GLOG_v=4
export GLOG_logtostderr=1

export TRAINERS=1
export STANDALONE=1

export PSERVERS=127.0.0.1:36001


SERVER_ENDPOINT=127.0.0.1:36001 TRAINING_ROLE=PSERVER PADDLE_INIT_TRAINER_ID="0" stdbuf -oL python /models/image_classification/se_resnext_cluster.py 4 3 1 CPU  &> pserver.0 &

SERVER_ENDPOINT=127.0.0.1:39001 TRAINING_ROLE=TRAINER PADDLE_INIT_TRAINER_ID="0" stdbuf -oL python /models/image_classification/se_resnext_cluster.py 4 3 1 CPU  &> trainerlog.0 &
