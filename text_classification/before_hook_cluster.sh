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

echo "****************** FINISH  ALL ***********************"