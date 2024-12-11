import argparse
import datetime
import time
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gym
from torch import nn
import numpy as np
import os
from pathlib import Path
from utils import *

# PPOMemory 类，存储和采样训练数据
class PPOMemory:
    def __init__(self, batch_size):
        # 存储用于PPO的状态、动作、概率、值、奖励等信息
        self.states, self.probs, self.vals, self.actions, self.rewards, self.dones = [], [], [], [], [], []
        self.batch_size = batch_size  # 每次采样的批大小

    def sample(self):
        # 对采样的数据进行批量划分，便于后续的SGD优化
        batch_step = np.arange(0, len(self.states), self.batch_size)
        indices = np.arange(len(self.states), dtype=np.int64)
        np.random.shuffle(indices)  # 随机打乱样本
        batches = [indices[i:i + self.batch_size] for i in batch_step]
        return np.array(self.states), np.array(self.actions), np.array(self.probs), \
               np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches

    def push(self, state, action, probs, vals, reward, done):
        # 将状态、动作、概率、值、奖励等信息存入内存
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        # 清空存储的数据
        self.states, self.probs, self.actions, self.rewards, self.dones, self.vals = [], [], [], [], [], []

# Actor 网络，用于选择动作，基于策略
class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim):
        super(Actor, self).__init__()
        # 构建一个简单的三层全连接神经网络（MLP）作为策略网络
        self.actor = nn.Sequential(
            nn.Linear(n_states, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions), nn.Softmax(dim=-1)  # Softmax输出动作的概率分布
        )

    def forward(self, state):
        # 根据输入的状态，输出一个动作的概率分布
        dist = self.actor(state)
        return Categorical(dist)  # 返回一个Categorical分布对象

# Critic 网络，用于评估状态的价值
class Critic(nn.Module):
    def __init__(self, n_states, hidden_dim):
        super(Critic, self).__init__()
        # 构建一个简单的三层全连接神经网络作为价值网络
        self.critic = nn.Sequential(
            nn.Linear(n_states, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出状态的价值（一个标量）
        )

    def forward(self, state):
        # 根据输入的状态，输出该状态的价值
        return self.critic(state)

# PPO 算法主体
class PPO:
    def __init__(self, n_states, n_actions, cfg):
        # 初始化PPO算法的相关参数
        self.gamma, self.policy_clip, self.device = cfg['gamma'], cfg['policy_clip'], cfg['device']
        # 初始化策略网络（Actor）和值网络（Critic）
        self.actor, self.critic = Actor(n_states, n_actions, cfg['hidden_dim']).to(self.device), Critic(n_states, cfg['hidden_dim']).to(self.device)
        # 使用Adam优化器分别优化Actor和Critic网络
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg['critic_lr'])
        # 创建一个PPOMemory对象用于存储交互数据
        self.memory = PPOMemory(cfg['batch_size'])

    def choose_action(self, state):
        # 根据当前状态选择动作
        state = torch.tensor([state], dtype=torch.float).to(self.device)  # 将状态转为tensor并送到设备
        dist = self.actor(state)  # 得到动作分布
        value = self.critic(state)  # 得到状态价值
        action = dist.sample()  # 从分布中采样一个动作
        probs = dist.log_prob(action)  # 得到动作的对数概率
        action = action.item()  # 将动作转为数值
        value = value.item()  # 将状态价值转为数值
        return action, probs.item(), value  # 返回动作、对数概率和状态价值

    def update(self):
        # 更新策略和价值网络
        for _ in range(self.n_epochs):
            # 从内存中采样数据
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.sample()
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            
            # 计算优势函数（Advantage），使用广义优势估计（GAE）
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * vals_arr[k + 1] * (1 - int(dones_arr[k])) - vals_arr[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            advantage = torch.tensor(advantage).to(self.device)  # 转为tensor并送到GPU
            
            # 对每个批次进行更新
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)
                dist = self.actor(states)
                critic_value = torch.squeeze(self.critic(states))  # 预测的状态价值
                new_probs = dist.log_prob(actions)  # 计算新策略下的对数概率
                prob_ratio = new_probs.exp() / old_probs.exp()  # 计算概率比率
                weighted_probs = advantage[batch] * prob_ratio  # 加权概率
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()  # 最大化actor损失
                returns = advantage[batch] + vals_arr[batch]  # 返回值
                critic_loss = (returns - critic_value) ** 2  # 最小化critic损失
                total_loss = actor_loss + 0.5 * critic_loss  # 总损失
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        
        self.memory.clear()  # 清空内存

    def save_model(self, path):
        # 保存训练好的模型
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, 'ppo_actor.pt'))
        torch.save(self.critic.state_dict(), os.path.join(path, 'ppo_critic.pt'))

    def load_model(self, path):
        # 加载训练好的模型
        self.actor.load_state_dict(torch.load(os.path.join(path, 'ppo_actor.pt')))
        self.critic.load_state_dict(torch.load(os.path.join(path, 'ppo_critic.pt')))

# 训练函数
def train(arg_dict, env, agent):
    startTime = time.time()
    print(f"环境名: {arg_dict['env_name']}, 算法名: {arg_dict['algo_name']}, Device: {arg_dict['device']}")
    print("开始训练智能体......")
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    steps = 0
    for i_ep in range(arg_dict['train_eps']):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            if arg_dict['train_render']:
                env.render()  # 渲染环境
            action, prob, val = agent.choose_action(state)  # 智能体选择动作
            state_, reward, done, _ = env.step(action)  # 执行动作并获取下一状态和奖励
            steps += 1
            ep_reward += reward
            agent.memory.push(state, action, prob, val, reward, done)  # 存储交互信息
            if steps % arg_dict['update_fre'] == 0:
                agent.update()  # 更新网络
            state = state_
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            print(f"回合：{i_ep + 1}/{arg_dict['train_eps']}，奖励：{ep_reward:.2f}")
    print('训练结束 , 用时: ' + str(time.time() - startTime) + " s")
    env.close()  # 关闭环境
    return {'episodes': range(len(rewards)), 'rewards': rewards}  # 返回训练过程中的回合和奖励


# 测试函数
def test(arg_dict, env, agent):
    startTime = time.time()
    print("开始测试智能体......")
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(arg_dict['test_eps']):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            if arg_dict['test_render']:
                env.render()  # 渲染环境
            action, prob, val = agent.choose_action(state)  # 智能体选择动作
            state_, reward, done, _ = env.step(action)  # 执行动作并获取下一状态和奖励
            ep_reward += reward
            state = state_
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        print('回合：{}/{}, 奖励：{}'.format(i_ep + 1, arg_dict['test_eps'], ep_reward))
    print("测试结束 , 用时: " + str(time.time() - startTime) + " s")
    env.close()  # 关闭环境
    return {'episodes': range(len(rewards)), 'rewards': rewards}


# 创建环境和智能体
def create_env_agent(arg_dict):
    env = gym.make(arg_dict['env_name'])
    all_seed(env, seed=arg_dict["seed"])  # 设置环境的随机种子
    try:
        n_states = env.observation_space.n  # 离散空间
    except AttributeError:
        n_states = env.observation_space.shape[0]  # 连续空间
    n_actions = env.action_space.n  # 获取动作空间的维度
    print(f"状态数: {n_states}, 动作数: {n_actions}")
    arg_dict.update({"n_states": n_states, "n_actions": n_actions})  # 更新参数字典
    agent = PPO(n_states, n_actions, arg_dict)  # 创建 PPO 智能体
    return env, agent


# 主函数
if __name__ == '__main__':
    # 防止报错 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # 获取当前路径
    curr_path = os.path.dirname(os.path.abspath(__file__))
    # 获取当前时间
    curr_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    
    # 相关参数设置
    parser = argparse.ArgumentParser(description="hyper parameters")
    parser.add_argument('--algo_name', default='PPO', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='CartPole-v0', type=str, help="name of environment")
    parser.add_argument('--continuous', default=False, type=bool, help="if PPO is continuous")  
    parser.add_argument('--train_eps', default=200, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=20, type=int, help="episodes of testing")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--batch_size', default=5, type=int)  # mini-batch SGD中的批量大小
    parser.add_argument('--n_epochs', default=4, type=int)
    parser.add_argument('--actor_lr', default=0.0003, type=float, help="learning rate of actor net")
    parser.add_argument('--critic_lr', default=0.0003, type=float, help="learning rate of critic net")
    parser.add_argument('--gae_lambda', default=0.95, type=float)
    parser.add_argument('--policy_clip', default=0.2, type=float)  # PPO-clip中的clip参数，一般是0.1~0.2左右
    parser.add_argument('--update_fre', default=20, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--device', default='cpu', type=str, help="cpu or cuda")
    parser.add_argument('--seed', default=520, type=int, help="seed")
    parser.add_argument('--show_fig', default=False, type=bool, help="if show figure or not")
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
    parser.add_argument('--train_render', default=False, type=bool, help="Whether to render the environment during training")
    parser.add_argument('--test_render', default=True, type=bool, help="Whether to render the environment during testing")
    args = parser.parse_args()

    default_args = {'result_path': f"{curr_path}/outputs/{args.env_name}/{curr_time}/results/",
                    'model_path': f"{curr_path}/outputs/{args.env_name}/{curr_time}/models/",
                    }
    # 将参数转化为字典 type(dict)
    arg_dict = {**vars(args), **default_args}
    print("算法参数字典:", arg_dict)

    # 创建环境和智能体
    env, agent = create_env_agent(arg_dict)
    
    # 传入算法参数、环境、智能体，然后开始训练
    res_dic = train(arg_dict, env, agent)
    print("算法返回结果字典:", res_dic)
    
    # 保存相关信息
    agent.save_model(path=arg_dict['model_path'])
    save_args(arg_dict, path=arg_dict['result_path'])
    save_results(res_dic, tag='train', path=arg_dict['result_path'])
    plot_rewards(res_dic['rewards'], arg_dict, path=arg_dict['result_path'], tag="train")

    # =================================================================================================
    # 创建新环境和智能体用来测试
    print("=" * 300)
    env, agent = create_env_agent(arg_dict)
    # 加载已保存的智能体
    agent.load_model(path=arg_dict['model_path'])
    res_dic = test(arg_dict, env, agent)
    save_results(res_dic, tag='test', path=arg_dict['result_path'])
    plot_rewards(res_dic['rewards'], arg_dict, path=arg_dict['result_path'], tag="test")
