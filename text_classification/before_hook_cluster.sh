shdir=$(dirname $(readlink -f ${BASH_SOURCE[0]}))

pwd

echo "we are running before_hook.sh"  
echo ""

echo "********  PaddleCloud Move Dataset To Local ********"
mkdir -p /root/.cache/paddle/dataset/imdb/
mv train/* /root/.cache/paddle/dataset/imdb/
echo "******  PaddleCloud Download Dataset To Local END ********"
echo ""

echo "********  PaddleCloud Thridparty Install PaddlePaddle ***"
pushd ./thirdparty
pwd

pkg="paddlepaddle-0.12.0-cp27-cp27mu-linux_x86_64.cpu.whl"
if [ -e $pkg ];then
    echo "pip install new paddle  =============" 
    pip list
    pip uninstall -y paddlepaddle-gpu 
    pip uninstall -y paddlepaddle-cpu
    pip uninstall -y paddlepaddle
    pip install $pkg
fi
echo "****  PaddleCloud Thridparty Install PaddlePaddle END***"
echo ""


echo "********  Rename Cluster Env to PaddleFluid Env ********"

export PADDLE_TRAINING_ROLE=$TRAINING_ROLE
export PADDLE_PSERVER_PORT=$PADDLE_PORT
export PADDLE_PSERVER_IPS=$PADDLE_PSERVERS
export PADDLE_TRAINERS=$PADDLE_TRAINERS_NUM
export PADDLE_CURRENT_IP=$POD_IP
export PADDLE_TRAINER_ID=$PADDLE_TRAINER_ID

echo "******  Rename Cluster Env to PaddleFluid Env END ******"
echo ""

echo "****************** FINISH  ALL ***********************"