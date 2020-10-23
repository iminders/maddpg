## maddpg
Distribution MADDPG using SEED_RL paradigm

### 参考
- [MADDPG](https://github.com/openai/maddpg)
- [SEED_RL](https://github.com/google-research/seed_rl)

### Features
- [ ] 加速
    - [x] zmq `server/client`(并没有实现seed rl的设计stream gRPC)，这种方式
已经加速度比较明显
    - [ ] TODO: stream gRPC batch inference + recurrent states 实现stream效果
- [ ] 随机过程(`maddpg.distributions`):
    - [x] ou(OrnsteinUhlenbeckProcess)
    - [x] GaussianProcess(原:DiagGaussianPdType, Box 连续空间)
    - [ ] SoftCategoricas(原:SoftCategoricalPdType, Discrete离散空间)
    - [ ] SoftMultiCategorical(原:SoftMultiCategoricalPdType, MultiDiscrete连续空间)
    - [ ] Bernoulli(原:BernoulliPdType， 多二值变量连续空间)

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
