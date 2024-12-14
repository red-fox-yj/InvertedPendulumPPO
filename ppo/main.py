import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =========================================
# 自定义网格世界环境 (不依赖gym)
# =========================================

class GridWorldEnv:
    def __init__(self, grid_size=5, max_steps=50):
        self.grid_size = grid_size     # 网格大小，如5表示5x5
        self.max_steps = max_steps     # 每个episode的最大步数
        self.reset()                   # 重置环境状态

    def reset(self):
        # 智能体起点在(0,0)
        self.x, self.y = 0, 0
        self.steps = 0                 # 步数计数器清零
        return self._get_obs()

    def _get_obs(self):
        # 将状态(x,y)归一化到[0,1]区间，以便神经网络输入
        # 如网格为5x5，则位置最大坐标为4，因此归一化用(x/4, y/4)
        return np.array([self.x / (self.grid_size - 1),
                         self.y / (self.grid_size - 1)], dtype=np.float32)

    def step(self, action):
        # 动作定义：0:上, 1:右, 2:下, 3:左
        # 首先记录旧位置，用于检查动作是否越界
        old_x, old_y = self.x, self.y
        if action == 0: # 上
            self.x = max(self.x - 1, 0)
        elif action == 1: # 右
            self.y = min(self.y + 1, self.grid_size - 1)
        elif action == 2: # 下
            self.x = min(self.x + 1, self.grid_size - 1)
        elif action == 3: # 左
            self.y = max(self.y - 1, 0)

        self.steps += 1

        # 奖励逻辑：
        # 若到达终点(4,4)，则奖励1.0并结束
        # 否则每步-0.01作为惩罚
        # 若超过max_steps仍未到达目标，也结束episode，但无正向奖励
        done = False
        if self.x == self.grid_size - 1 and self.y == self.grid_size - 1:
            reward = 1.0
            done = True
        else:
            reward = -0.01

        if self.steps >= self.max_steps:
            # 超过最大步数就强制结束
            done = True

        return self._get_obs(), reward, done, {}

# =========================================
# PPO相关代码
# =========================================

class PolicyValueNet(nn.Module):
    """
    策略-价值网络共用底层特征：
    输入：状态(s)，这里为2维特征(x_norm, y_norm)
    输出：策略对应动作的logits(长度为动作数4)和状态价值
    """
    def __init__(self, input_dim=2, hidden_dim=64, action_dim=4):
        super(PolicyValueNet, self).__init__()
        # 两层MLP作为特征提取
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # 策略头输出action的logits（未归一化概率）
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        # 价值头输出状态价值
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch, 2]
        h = self.fc(x)
        logits = self.policy_head(h)   # [batch, action_dim]
        value = self.value_head(h)     # [batch, 1]
        return logits, value

def select_action(model, state):
    """
    给定当前状态，从模型中选取一个动作：
    1. 前向计算策略logits和价值
    2. 基于softmax分布采样动作
    3. 返回动作、对应的log_prob、和状态价值估计
    """
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits, value = model(state_t)
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob, value.item()

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    使用广义优势估计(GAE)计算优势函数：
    A_t = δ_t + γλδ_{t+1} + ... 
    其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)
    此处values最后会有一个values[T]为下一个状态的价值(episode结束时为0)
    """
    advantages = []
    gae = 0
    # 从后往前计算GAE
    for t in reversed(range(len(rewards))):
        # dones[t]表示在t步是否结束episode，若结束则下一个价值算0
        next_value = 0 if dones[t] else values[t+1]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * (0 if dones[t] else gae)
        advantages.insert(0, gae)
    return advantages

def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages, 
               clip_range=0.2, epochs=10, batch_size=64):
    """
    使用PPO算法对策略-价值网络进行更新的函数。

    参数说明：
    - model: 策略-价值联合网络（前向输出：给定state输出action logits和state value）
    - optimizer: 优化器(如Adam)，对model参数进行梯度下降更新
    - states: 收集到的一批状态数据，形状为[N, state_dim]
    - actions: 对应states采取的动作数据，形状为[N]
    - old_log_probs: 执行这些actions时旧策略对该动作的log概率，形状为[N]
    - returns: 对应状态的回报（return），形状为[N]
    - advantages: 对应状态的优势值(Advantage)，形状为[N]
    - clip_range: PPO截断范围，如0.2表示ratio在[0.8, 1.2]之外会被截断
    - epochs: 对同一批数据进行多少次迭代训练
    - batch_size: 每次迭代训练使用多大批量的数据

    函数流程：
    1. 对advantages进行标准化
    2. 多个epochs迭代：
       - 随机打乱数据
       - 按照batch_size分批训练：
         * 计算当前策略下log_probs和value
         * 计算ratio并使用PPO剪切目标函数
         * 同时优化策略和价值函数，并加入熵鼓励探索
    """

    # 对优势(advantages)进行标准化，有助于训练稳定
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # 数据集大小
    dataset_size = len(states)

    # 迭代多个epochs来对同一批数据反复训练
    for _ in range(epochs):

        # 生成一个包含[0, 1, 2, ..., dataset_size-1]的索引数组
        indices = np.arange(dataset_size)

        # 将indices打乱，以对数据进行随机抽样，从而提高泛化能力
        np.random.shuffle(indices)

        # 按batch_size将数据分成若干小批次训练
        for start in range(0, dataset_size, batch_size):
            # 当前小批次的结束索引
            end = start + batch_size

            # 当前小批次的索引切片
            idx = indices[start:end]

            # 从states、actions、old_log_probs、returns、advantages中取出对应小批数据，并转为PyTorch张量
            s = torch.tensor(states[idx], dtype=torch.float32)
            a = torch.tensor(actions[idx], dtype=torch.long)
            old_lp = torch.tensor(old_log_probs[idx], dtype=torch.float32)
            R = torch.tensor(returns[idx], dtype=torch.float32)
            Adv = torch.tensor(advantages[idx], dtype=torch.float32)

            # 前向传播：获取当前策略对这些状态的输出
            # logits为动作的未归一化对数几率，values为状态价值
            logits, values = model(s)

            # values形状可能是[batch_size, 1]，用squeeze(-1)转为[batch_size]
            values = values.squeeze(-1)

            # 计算当前策略下的动作概率分布
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            # 计算当前策略下选取a动作的log概率
            log_probs = dist.log_prob(a)

            # ratio = 当前策略动作概率 / 旧策略动作概率 = exp(log_probs - old_lp)
            ratio = torch.exp(log_probs - old_lp)

            # surr1 = ratio * Adv，即PPO目标的原始形式
            surr1 = ratio * Adv

            # surr2 = clip(ratio, 1-clip_range, 1+clip_range) * Adv
            # 使用截断的ratio限制策略更新幅度
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * Adv

            # 策略损失使用PPO目标的最小值，以达到对策略更新幅度的限制
            # 我们要最大化优势，因此损失取负号
            policy_loss = -torch.min(surr1, surr2).mean()

            # 价值函数损失：使用预测的values和真实的returns比较，采用均方误差(MSE)
            value_loss = (R - values).pow(2).mean()

            # 熵(entropy)衡量策略分布的随机性，越大表示越分散，有利于探索
            # 我们希望策略保持一定随机性，因此在损失中减去熵项(加强熵)，
            # 从而鼓励策略不因过度确定而陷入局部最优
            entropy_loss = dist.entropy().mean()

            # 最终损失函数：
            # policy_loss是我们真正想优化的(带clip的策略更新目标)
            # value_loss帮助价值函数逼近真实回报
            # entropy_loss鼓励探索（故要减去这个项）
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

            # 清空梯度缓存
            optimizer.zero_grad()

            # 反向传播计算梯度
            loss.backward()

            # 根据梯度更新参数
            optimizer.step()

def collect_trajectories(model, env, num_episodes=10, gamma=0.99, lam=0.95):
    """
    收集指定数量的episode数据，用于PPO更新：
    1. 运行num_episodes个episode，记录状态、动作、奖励、价值预测、log_prob等数据
    2. 最后使用GAE计算优势，并计算returns = advantages + values
    3. 返回用于训练的numpy数据数组
    """
    states_list = []
    actions_list = []
    rewards_list = []
    dones_list = []
    values_list = []
    log_probs_list = []

    for _ in range(num_episodes):
        s = env.reset()
        ep_states = []
        ep_actions = []
        ep_rewards = []
        ep_values = []
        ep_log_probs = []
        ep_dones = []
        done = False

        # 运行单个episode
        while not done:
            a, lp, v = select_action(model, s)
            next_s, r, done, _ = env.step(a)

            ep_states.append(s)
            ep_actions.append(a)
            ep_rewards.append(r)
            ep_values.append(v)
            ep_log_probs.append(lp.item())
            ep_dones.append(done)

            s = next_s

        # 追加一个额外的value用来计算GAE中的next_value
        # episode已结束，下个状态价值设为0
        ep_values.append(0.0)

        # 将该episode数据加入整体数据集中
        states_list.extend(ep_states)
        actions_list.extend(ep_actions)
        rewards_list.extend(ep_rewards)
        dones_list.extend(ep_dones)
        values_list.extend(ep_values)
        log_probs_list.extend(ep_log_probs)

    # 使用GAE计算优势
    advantages = compute_gae(rewards_list, values_list, dones_list, gamma, lam)

    # returns = advantages + values（values_list多了一个结尾0要去掉）
    returns = [adv + v for adv, v in zip(advantages, values_list[:-1])]

    return (np.array(states_list, dtype=np.float32),
            np.array(actions_list, dtype=np.int64),
            np.array(log_probs_list, dtype=np.float32),
            np.array(returns, dtype=np.float32),
            np.array(advantages, dtype=np.float32))

# =========================================
# 主训练流程
# =========================================
env = GridWorldEnv(grid_size=5, max_steps=50)     # 创建5x5网格环境
model = PolicyValueNet()                          # 初始化策略价值网络
optimizer = optim.Adam(model.parameters(), lr=3e-4)

num_iterations = 2000       # 迭代次数
episodes_per_iter = 10      # 每次迭代采集的episode数量

for i in range(num_iterations):
    # 收集一批数据用于训练
    states, actions, old_log_probs, returns, advantages = collect_trajectories(model, env, episodes_per_iter)
    # 使用PPO进行更新
    ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages, clip_range=0.2, epochs=4, batch_size=64)

    if (i+1) % 100 == 0:
        # 每100次迭代测试智能体的表现
        test_rewards = []
        for _ in range(20):
            s = env.reset()
            total_reward = 0
            done = False
            while not done:
                # 测试时使用贪心策略：选择概率最高的动作
                with torch.no_grad():
                    s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                    logits, v = model(s_t)
                    probs = torch.softmax(logits, dim=-1)
                    a = torch.argmax(probs, dim=-1).item()
                s, r, done, _ = env.step(a)
                total_reward += r
            test_rewards.append(total_reward)
        # 输出测试平均奖励
        print(f"Iteration {i+1}: Test Average Reward = {np.mean(test_rewards):.3f}")

# 随着训练进行，智能体应逐渐学会以较少的步数到达(4,4)处，以获得+1奖励并减少-0.01的累积惩罚。
# 测试平均奖励应逐渐接近1。