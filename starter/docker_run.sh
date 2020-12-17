set -e

export num_agent=20

cd /root/maddpg

pip install -e .

python -m maddpg.run --save_rate=1000 --num_env 10  --env_batch_size 100 \
    --warm_up 3500 --num_agent $(num_agent) \
    --enable_prioritized_replay
