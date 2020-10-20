set -e
cd "$(dirname "$0")"

export BAZEL_RUNID=$RANDOM
echo $BAZEL_RUNID

cd /root/maddpg
pip install -e .

cd /root/maddpg/maddpg
python -m pytest
