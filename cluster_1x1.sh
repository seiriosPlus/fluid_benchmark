#!/bin/bash

export GLOG_v=4
export GLOG_logtostderr=1

export PADDLE_INIT_PSERVERS=127.0.0.1
export POD_IP=127.0.0.1
export PADDLE_INIT_PSERVER_PORT=36001
export PADDLE_INIT_TRAINER_PORT=39001
export TRAINERS=1

TRAINING_ROLE=PSERVER PADDLE_INIT_TRAINER_ID="0" stdbuf -oL python /models/image_classification/mobilenet.py --local 0 &> pserver.one &
TRAINING_ROLE=TRAINER PADDLE_INIT_TRAINER_ID="0" stdbuf -oL python /models/image_classification/mobilenet.py --local 0 &> trainer.one &
