import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =========================================
# 自定义网格世界环境 (不依赖gym)
# =========================================

class GridWorldEnv:
    def __init__(self, grid_size=5, max_steps=50):
        self.grid_size = grid_size     # 网格大小，例如5表示5x5的网格
        self.max_steps = max_steps     # 每个episode的最大步数
        self.reset()                   # 重置环境状态

    def reset(self):
        # 智能体起点在(0,0)
        self.x, self.y = 0, 0
        self.steps = 0                 # 步数计数器清零
        return self._get_obs()

    def _get_obs(self):
        # 将位置(x,y)归一化到[0,1]区间，以便NN输入
        # 如网格为5x5，则坐标最大为4，x/(4)=x/(grid_size-1)
        return np.array([self.x / (self.grid_size - 1),
                         self.y / (self.grid_size - 1)], dtype=np.float32)

    def step(self, action):
        # 动作定义：0:上, 1:右, 2:下, 3:左
        # 基于动作更新坐标，越界则停在边界
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
        # 到达目标(4,4)奖励1并结束
        # 否则每步-0.01作为惩罚
        # 若超过max_steps仍未到达目标也结束episode
        done = False
        if self.x == self.grid_size - 1 and self.y == self.grid_size - 1:
            reward = 1.0
            done = True
        else:
            reward = -0.01

        if self.steps >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

# =========================================
# 策略-价值网络（共享特征）
# 输入：状态(s)为2维
# 输出：动作logits（用于产生策略分布）和状态价值
# =========================================

class PolicyValueNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, action_dim=4):
        super(PolicyValueNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.fc(x)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value

def select_action(model, state):
    """
    根据当前模型策略分布对给定状态进行采样选行动作。
    返回：动作、该动作的log概率以及价值估计。
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
    使用GAE计算优势A_t:
    A_t = δ_t + γλδ_{t+1} + ...
    其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)
    values额外包含一个终止状态后的价值(或者episode结束时为0)，以便计算delta。
    """
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        next_value = 0 if dones[t] else values[t+1]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * (0 if dones[t] else gae)
        advantages.insert(0, gae)
    return advantages

def collect_trajectories(model, env, num_episodes=10, gamma=0.99, lam=0.95):
    """
    收集若干episode的数据(状态、动作、奖励、优势等)用于后续训练。
    返回数组形式的states, actions, old_log_probs, returns, advantages。
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

        # 运行一个episode
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

        ep_values.append(0.0)  # episode结束后下个状态价值0

        # 整合episode数据到全局轨迹中
        states_list.extend(ep_states)
        actions_list.extend(ep_actions)
        rewards_list.extend(ep_rewards)
        dones_list.extend(ep_dones)
        values_list.extend(ep_values)
        log_probs_list.extend(ep_log_probs)

    # 计算优势和returns
    advantages = compute_gae(rewards_list, values_list, dones_list, gamma, lam)
    returns = [adv + v for adv, v in zip(advantages, values_list[:-1])]

    return (np.array(states_list, dtype=np.float32),
            np.array(actions_list, dtype=np.int64),
            np.array(log_probs_list, dtype=np.float32),
            np.array(returns, dtype=np.float32),
            np.array(advantages, dtype=np.float32))

# =========================================
# 以下为TRPO的核心实现
# =========================================

def flat_params(model):
    """
    将模型的参数展平为一个一维张量，以便进行向量化操作。
    """
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_params(model, new_params):
    """
    将向量化的新参数分配回模型中对应的参数位置。
    """
    old_params = flat_params(model)
    # 计算参数更新量delta
    delta = new_params - old_params
    idx = 0
    for p in model.parameters():
        size = p.numel()
        # 用delta对应部分加到当前参数上
        p.data = p.data + delta[idx:idx+size].view(p.size())
        idx += size

def get_log_probs_and_value(model, states, actions):
    """
    给定状态和动作，计算模型下这些动作的log概率和状态价值。
    """
    logits, values = model(states)
    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    log_probs = dist.log_prob(actions)
    return log_probs, values.squeeze(-1), dist

def get_loss(model, states, actions, advantages, old_log_probs):
    """
    TRPO的目标函数: L = E[ (πθ(a|s) / π_old(a|s)) * A(s,a) ]
    = E[ exp(logπθ(a|s)-old_log_probs)*A(s,a) ]
    """
    log_probs, values, dist = get_log_probs_and_value(model, states, actions)
    ratio = torch.exp(log_probs - old_log_probs)
    loss = (ratio * advantages).mean()
    return loss

def get_fvp(model, states, vector, damping=0.1):
    """
    Fisher-Vector Product计算:
    近似KL散度关于参数的二阶导数并与向量vector相乘。
    """
    logits, _ = model(states)
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)

    # KL散度 w.r.t old参数约等于 E[ sum(π_old(a|s)*logπ_current(a|s)) ]
    # 这里简化处理为对当前策略self-KL近似，因为严格TRPO需对旧策略分布固定
    # 为演示起见：
    kl = (probs.detach() * log_probs).sum(dim=-1).mean()

    # 对kl求梯度，获得一阶导数
    grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
    flat_grads = torch.cat([g.view(-1) for g in grads])

    # 对 flat_grads 与 vector 点积，再对参数求梯度得到二阶导数近似(Fv)
    grad_vector = (flat_grads * vector).sum()
    hv = torch.autograd.grad(grad_vector, model.parameters())
    hv = torch.cat([h.view(-1) if h is not None else torch.zeros_like(flat_params(model)) for h in hv])

    # 添加damping以提高数值稳定性
    return hv + damping * vector

def conjugate_gradients(model, states, b, nsteps=10, damping=0.1, residual_tol=1e-10):
    """
    共轭梯度法求解Fx = g的近似解，用于求自然梯度方向。
    b为g，F为Fisher矩阵近似，通过Fvp计算。
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rr = r.dot(r)

    for i in range(nsteps):
        fvp = get_fvp(model, states, p, damping)
        alpha = rr / p.dot(fvp)
        x += alpha * p
        r -= alpha * fvp
        new_rr = r.dot(r)
        if new_rr < residual_tol:
            break
        beta = new_rr / rr
        p = r + beta * p
        rr = new_rr
    return x

def line_search(model, f, x, fullstep, max_backtracks=10, accept_ratio=0.1):
    """
    对更新方向fullstep进行线搜索:
    从fullstep开始，如果不满足条件(如KL超限或性能未提高)，就递减步长，反复尝试。
    如果最终没有找到合适的步长，则返回原参数不变。
    """
    fval = f(x)
    for stepfrac in 0.5**np.arange(max_backtracks):
        xnew = x + stepfrac * fullstep
        val = f(xnew)
        # 如果目标函数有提高（val >= fval * accept_ratio表示至少提高一定比例）
        # 则接受该步长
        if val >= fval * accept_ratio:
            return xnew
    return x  # 未找到满意步长则保持不变

def trpo_update(model, states, actions, old_log_probs, returns, advantages, old_logits, max_kl=0.01, damping=0.1):
    """
    TRPO更新步骤：
    1. 计算梯度g = ∇L(θ_old)
    2. 使用共轭梯度求解F * x = g，得到自然梯度方向x
    3. 计算步长alpha，使得KL约束最大为max_kl
    4. 在线搜索中寻找合适的update step
    """
    states_t = torch.tensor(states, dtype=torch.float32)
    actions_t = torch.tensor(actions, dtype=torch.long)
    old_lp_t = torch.tensor(old_log_probs, dtype=torch.float32)
    returns_t = torch.tensor(returns, dtype=torch.float32)
    advantages_t = torch.tensor(advantages, dtype=torch.float32)

    # 标准化优势有助于稳定训练
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    # 清除梯度
    for p in model.parameters():
        p.grad = None

    # 计算当前策略下loss及其梯度
    loss = get_loss(model, states_t, actions_t, advantages_t, old_lp_t)
    grads = torch.autograd.grad(loss, model.parameters())
    g = torch.cat([grad.view(-1) for grad in grads])

    # 使用共轭梯度求解出自然梯度方向x
    x = conjugate_gradients(model, states_t, g, nsteps=10, damping=damping)

    # 计算xFx，用于确定更新步长alpha
    def Fvp(v):
        return get_fvp(model, states_t, v, damping)
    xFx = (x * Fvp(x)).sum()

    # alpha = sqrt(2 * delta / x^T F x)
    alpha = torch.sqrt(2 * max_kl / xFx)
    fullstep = alpha * x

    old_params = flat_params(model)

    def surrogate(params):
        # 给定新参数params，设置给model并计算loss
        set_params(model, params)
        return get_loss(model, states_t, actions_t, advantages_t, old_lp_t).item()

    # 在更新方向上进行线搜索
    new_params = line_search(model, surrogate, old_params, fullstep)
    set_params(model, new_params)

# =========================================
# 主训练流程 (使用TRPO)
# =========================================

env = GridWorldEnv(grid_size=5, max_steps=50)
model = PolicyValueNet()

# TRPO中一般将策略与价值分离，这里简单用同一网络，并用Adam训练价值函数部分
optimizer = optim.Adam(model.parameters(), lr=3e-4)

num_iterations = 200
episodes_per_iter = 10

for i in range(num_iterations):
    # 收集交互数据
    states, actions, old_log_probs, returns, advantages = collect_trajectories(model, env, episodes_per_iter)

    # 更新价值函数(简单地MSE回归)
    # TRPO通常单独训练价值网络，这里为简化仅示意
    states_t = torch.tensor(states, dtype=torch.float32)
    returns_t = torch.tensor(returns, dtype=torch.float32)
    for _ in range(5):
        _, values = model(states_t)
        value_loss = (returns_t - values.squeeze(-1)).pow(2).mean()
        optimizer.zero_grad()
        value_loss.backward()
        optimizer.step()

    # 保存旧策略的logits用于计算KL散度（实际中需要进行KL校验，这里不完全实现）
    with torch.no_grad():
        old_logits, _ = model(states_t)

    # 使用TRPO更新策略
    trpo_update(model, states, actions, old_log_probs, returns, advantages, old_logits.detach(), max_kl=0.01, damping=0.1)

    # 每隔20次评估策略表现
    if (i+1) % 20 == 0:
        test_rewards = []
        for _ in range(20):
            s = env.reset()
            total_reward = 0
            done = False
            while not done:
                with torch.no_grad():
                    s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                    logits, _ = model(s_t)
                    probs = torch.softmax(logits, dim=-1)
                    a = torch.argmax(probs, dim=-1).item() # 测试时取Greedy动作
                s, r, done, _ = env.step(a)
                total_reward += r
            test_rewards.append(total_reward)
        print(f"Iteration {i+1}: Test Average Reward = {np.mean(test_rewards):.3f}")