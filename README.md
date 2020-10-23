## maddpg
Distribution MADDPG using SEED_RL paradigm

### 参考
- [MADDPG](https://github.com/openai/maddpg)
- [SEED_RL](https://github.com/google-research/seed_rl)

### Features
并没有完全实现seed rl的设计， 使用ZeroMQ的`server/client`模式来进行收集采集数据，这种方式
已经加速度比较明显， TODO: batch inference + recurrent states 实现stream效果

### 快速开始
#### ubuntu/mac docker
    - 测试 `make test`
    - simeple: TODO
#### windows anaconda
    - 测试
    - simple

### 设计TODO

### 关键代码说明

### 实验
- 速度: TODO
- 收敛: TODO
