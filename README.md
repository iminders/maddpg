## maddpg ![Test](https://github.com/iminders/maddpg/workflows/Test/badge.svg) ![Experiment](https://github.com/iminders/maddpg/workflows/Experiment/badge.svg)
Distribution MADDPG using SEED_RL paradigm

### 参考
- [MADDPG](https://github.com/openai/maddpg)
- [SEED_RL](https://github.com/google-research/seed_rl)
- [tf2rl](https://github.com/keiohta/tf2rl)
- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf)
### Features
- [ ] 加速
    - [x] 使用zmq `server/client`(并没有实现seed rl的设计stream gRPC)，zmq方式
已经加速度比较明显
    - [ ] TODO: stream gRPC batch inference + recurrent states 实现stream效果
- [ ] 随机过程:
    - [x] `tensorflow uniform` 是目前`simple`使用的随机过程
    - [ ] ou(OrnsteinUhlenbeckProcess): TODO
    - [ ] GaussianProcess(原:DiagGaussianPdType, Box 连续空间): TODO
    - [ ] SoftCategoricas(原:SoftCategoricalPdType, Discrete离散空间): TODO
    - [ ] SoftMultiCategorical(原:SoftMultiCategoricalPdType, MultiDiscrete连续空间): TODO
    - [ ] Bernoulli(原:BernoulliPdType， 多二值变量连续空间): TODO

### 快速开始
#### ubuntu/mac
    - 安装 `make install`
    - 单元测试 `make test`
    - 场景测试simple: `make run num_agent=3`

#### 源码安装
- Known dependencies: Python 3, OpenAI gym (0.10.5), tensorflow (2.3.0)
- 安装[Multi-Agent Particle Environments (MPE)](https://github.com/iminders/multiagent-particle-envs)
- To install, `cd` into the root directory and type `pip install -e .`

### 设计TODO

### 关键代码说明
- 核心算法实现: `maddpg.agents.maddpg.agent`, `maddpg.agents.maddpg.base`
- NN tf2实现: `maddpg.nets.actor`, `maddpg.nets.critic`, `maddpg.nets.mpl`
- 运行进程
    - 探索进程: `maddpg.explorer`
    - 学习进程: `maddpg.learner`
    - 参数设置: `maddpg.arguments`
    - 运行入口: `maddpg.run`
    - 公共模块: 环境，常数，日志，云存储等, `maddpg.common.*`
- 其他:
    zmq: `experiments.zmq`, zmq server/client模式回归测试

### 实验
- 速度:
    - 纯CPU环境
        - `simple agent_num=3`, 平均batch时间约为原版的1/3, [运行日志](TODO)
        - `simple agent_num=20`, TODO

- 收敛:
    - 纯CPU环境
        - `simple agent_num=3`, 比原版更优, [运行日志](TODO)
        - `simple agent_num=20`, TODO
