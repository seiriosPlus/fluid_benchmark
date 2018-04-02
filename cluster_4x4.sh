#!/bin/bash

PSERVERS=""
for i in {6170..6173}
do
  if [ "${PSERVERS}" == "" ]; then
    PSERVERS="127.0.0.1:${i}"
  else
    PSERVERS="${PSERVERS},127.0.0.1:${i}"
  fi
done
export PSERVERS=$PSERVERS
export TRAINERS=4
export TRAINING_ROLE=PSERVER
export GLOG_v=4
export GLOG_logtostderr=1

echo "PSERVERS:" $PSERVERS

for i in {0..3}
do
  port="617${i}"
  SERVER_ENDPOINT="127.0.0.1:${port}" PADDLE_INIT_TRAINER_ID="${i}" stdbuf -oL python /models/image_classification/mobilenet.py --local 0 &> pserver.${i} &
done
