## maddpg
Distribution MADDPG using SEED_RL paradigm

### 参考
- [MADDPG](https://github.com/openai/maddpg)
- [SEED_RL](https://github.com/google-research/seed_rl)
- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf)
### Features
- [ ] 加速
    - [x] zmq `server/client`(并没有实现seed rl的设计stream gRPC)，zmq方式
已经加速度比较明显
    - [ ] TODO: stream gRPC batch inference + recurrent states 实现stream效果
- [ ] 随机过程(`maddpg.distributions`):
    - [x] `tensorflow uniform` 是目前`simple`使用的随机过程
    - [ ] ou(OrnsteinUhlenbeckProcess): TODO
    - [ ] GaussianProcess(原:DiagGaussianPdType, Box 连续空间)
    - [ ] SoftCategoricas(原:SoftCategoricalPdType, Discrete离散空间)
    - [ ] SoftMultiCategorical(原:SoftMultiCategoricalPdType, MultiDiscrete连续空间)
    - [ ] Bernoulli(原:BernoulliPdType， 多二值变量连续空间)

### 快速开始
#### ubuntu/mac
    - 安装 `make install`
    - 单元测试 `make test`
    - 场景测试simple: `make run num_agent=3`

#### 源码安装
- Known dependencies: Python 3, OpenAI gym (0.10.5), tensorflow (2.3.0)
- 安装[Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs)
- To install, `cd` into the root directory and type `pip install -e .`

### 设计TODO

### 关键代码说明

### 实验
- 速度: TODO
- 收敛: TODO
