import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical, Normal
from collections import OrderedDict
import copy

class PolicyNetwork(nn.Module):
    """정책 네트워크: 상태를 입력받아 행동 분포를 출력"""
    def __init__(self, state_dim, action_dim, hidden_dim=64, continuous=False):
        super(PolicyNetwork, self).__init__()
        self.continuous = continuous
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        if continuous:
            self.fc_mean = nn.Linear(hidden_dim, action_dim)
            self.fc_logstd = nn.Linear(hidden_dim, action_dim)
        else:
            self.fc_out = nn.Linear(hidden_dim, action_dim)
        
        self.relu = nn.ReLU()
        
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        
        if self.continuous:
            mean = self.fc_mean(x)
            log_std = self.fc_logstd(x)
            log_std = torch.clamp(log_std, -20, 2)
            return mean, log_std
        else:
            return self.fc_out(x)
    
    def get_action(self, state):
        if self.continuous:
            mean, log_std = self.forward(state)
            std = log_std.exp()
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        else:
            logits = self.forward(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action).unsqueeze(-1)
        
        return action, log_prob
    
    def get_log_prob(self, state, action):
        if self.continuous:
            mean, log_std = self.forward(state)
            std = log_std.exp()
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        else:
            logits = self.forward(state)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(action).unsqueeze(-1)
        
        return log_prob

class ValueNetwork(nn.Module):
    """가치 네트워크: 상태 가치 함수 V(s)"""
    def __init__(self, state_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc_out(x)

class MAMLPPO:
    """MAML-PPO 알고리즘 구현"""
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        lr_inner=0.1,
        lr_outer=0.001,
        gamma=0.99,
        gae_lambda=0.95,
        eps_clip=0.2,
        k_epochs=4,
        continuous=False
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_inner = lr_inner  # α: inner loop learning rate
        self.lr_outer = lr_outer  # β: outer loop learning rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        # 메타 정책과 가치 네트워크
        self.meta_policy = PolicyNetwork(state_dim, action_dim, continuous=continuous)
        self.meta_value = ValueNetwork(state_dim)
        
        # Outer loop optimizer
        self.policy_optimizer = optim.Adam(self.meta_policy.parameters(), lr=lr_outer)
        self.value_optimizer = optim.Adam(self.meta_value.parameters(), lr=lr_outer)
        
    def compute_gae(self, rewards, values, dones, next_value):
        """Generalized Advantage Estimation (GAE) 계산
        
        A_t = δ_t + (γλ)δ_{t+1} + ... + (γλ)^{T-t+1}δ_{T-1}
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values
        
        return advantages, returns
    
    def inner_loop_update(self, policy, value, trajectories):
        """Inner loop: 태스크별 적응 (gradient descent)
        θ'_i = θ - α∇_θL_Ti(f_θ)
        """
        states = trajectories['states']
        actions = trajectories['actions']
        rewards = trajectories['rewards']
        dones = trajectories['dones']
        
        # 가치 예측
        with torch.no_grad():
            values = value(states).squeeze()
            next_value = value(trajectories['next_states'][-1]).item()
        
        # GAE 계산
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Inner loop gradient 계산 및 업데이트
        old_log_probs = policy.get_log_prob(states, actions).detach()
        
        for _ in range(self.k_epochs):
            # PPO loss 계산
            log_probs = policy.get_log_prob(states, actions)
            ratio = torch.exp(log_probs - old_log_probs)
            
            surr1 = ratio * advantages.unsqueeze(-1)
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages.unsqueeze(-1)
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Gradient 계산
            policy_grad = torch.autograd.grad(policy_loss, policy.parameters(), create_graph=True)
            
            # Inner loop 파라미터 업데이트
            updated_params = OrderedDict()
            for (name, param), grad in zip(policy.named_parameters(), policy_grad):
                updated_params[name] = param - self.lr_inner * grad
            
            # 업데이트된 파라미터 적용
            for name, param in policy.named_parameters():
                param.data = updated_params[name]
        
        # Value function 업데이트
        for _ in range(self.k_epochs):
            value_pred = value(states).squeeze()
            value_loss = nn.MSELoss()(value_pred, returns)
            
            value_grad = torch.autograd.grad(value_loss, value.parameters(), create_graph=True)
            
            updated_value_params = OrderedDict()
            for (name, param), grad in zip(value.named_parameters(), value_grad):
                updated_value_params[name] = param - self.lr_inner * grad
            
            for name, param in value.named_parameters():
                param.data = updated_value_params[name]
        
        return policy, value
    
    def meta_update(self, task_batch):
        """Outer loop: 메타 파라미터 업데이트
        θ = θ - β∇_θ Σ_Ti L_Ti(f_θ'i)
        """
        meta_policy_loss = 0
        meta_value_loss = 0
        
        for task_trajectories in task_batch:
            # 각 태스크에 대해 adapted parameters 계산
            task_policy = copy.deepcopy(self.meta_policy)
            task_value = copy.deepcopy(self.meta_value)
            
            # Inner loop 적응
            adapted_policy, adapted_value = self.inner_loop_update(
                task_policy, task_value, task_trajectories['support']
            )
            
            # Query set에서 메타 loss 계산
            query_traj = task_trajectories['query']
            states = query_traj['states']
            actions = query_traj['actions']
            rewards = query_traj['rewards']
            dones = query_traj['dones']
            
            with torch.no_grad():
                values = adapted_value(states).squeeze()
                next_value = adapted_value(query_traj['next_states'][-1]).item()
            
            advantages, returns = self.compute_gae(rewards, values, dones, next_value)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO loss on query set
            old_log_probs = adapted_policy.get_log_prob(states, actions).detach()
            log_probs = adapted_policy.get_log_prob(states, actions)
            ratio = torch.exp(log_probs - old_log_probs)
            
            surr1 = ratio * advantages.unsqueeze(-1)
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages.unsqueeze(-1)
            
            meta_policy_loss += -torch.min(surr1, surr2).mean()
            
            # Value loss on query set
            value_pred = adapted_value(states).squeeze()
            meta_value_loss += nn.MSELoss()(value_pred, returns)
        
        # 메타 파라미터 업데이트
        meta_policy_loss /= len(task_batch)
        meta_value_loss /= len(task_batch)
        
        self.policy_optimizer.zero_grad()
        meta_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_policy.parameters(), 0.5)
        self.policy_optimizer.step()
        
        self.value_optimizer.zero_grad()
        meta_value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_value.parameters(), 0.5)
        self.value_optimizer.step()
        
        return meta_policy_loss.item(), meta_value_loss.item()
    
    def adapt_to_new_task(self, support_trajectories, num_adapt_steps=5):
        """새로운 태스크에 대한 few-shot 적응"""
        adapted_policy = copy.deepcopy(self.meta_policy)
        adapted_value = copy.deepcopy(self.meta_value)
        
        for _ in range(num_adapt_steps):
            adapted_policy, adapted_value = self.inner_loop_update(
                adapted_policy, adapted_value, support_trajectories
            )
        
        return adapted_policy, adapted_value


# 사용 예제
def collect_trajectories(env, policy, num_episodes=10):
    """환경에서 trajectory 수집"""
    trajectories = {
        'states': [],
        'actions': [],
        'rewards': [],
        'next_states': [],
        'dones': []
    }
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, _ = policy.get_action(state_tensor)
            
            next_state, reward, done, _ = env.step(action.item())
            
            trajectories['states'].append(state)
            trajectories['actions'].append(action.item())
            trajectories['rewards'].append(reward)
            trajectories['next_states'].append(next_state)
            trajectories['dones'].append(done)
            
            state = next_state
    
    # Convert to tensors
    for key in trajectories:
        trajectories[key] = torch.FloatTensor(trajectories[key])
    
    return trajectories


def train_maml_ppo(env_class, num_tasks=10, meta_iterations=1000):
    """MAML-PPO 학습 메인 루프"""
    
    # 환경 정보
    sample_env = env_class()
    state_dim = sample_env.observation_space.shape[0]
    action_dim = sample_env.action_space.n  # discrete action space
    
    # MAML-PPO 초기화
    maml_ppo = MAMLPPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_inner=0.1,
        lr_outer=0.001,
        continuous=False
    )
    
    for iteration in range(meta_iterations):
        # 태스크 배치 샘플링
        task_batch = []
        
        for _ in range(num_tasks):
            # 새로운 태스크 환경 생성
            task_env = env_class()
            
            # Support와 Query trajectory 수집
            support_traj = collect_trajectories(
                task_env, maml_ppo.meta_policy, num_episodes=5
            )
            query_traj = collect_trajectories(
                task_env, maml_ppo.meta_policy, num_episodes=5
            )
            
            task_batch.append({
                'support': support_traj,
                'query': query_traj
            })
        
        # 메타 업데이트
        policy_loss, value_loss = maml_ppo.meta_update(task_batch)
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Policy Loss = {policy_loss:.4f}, Value Loss = {value_loss:.4f}")
    
    return maml_ppo


# 환경 클래스 정의 (예: CartPole variants)
class TaskEnvironment:
    def __init__(self):
        # 태스크별 파라미터 변경 (예: 중력, 카트 질량 등)
        pass

# MAML-PPO 학습
maml_agent = train_maml_ppo(
    env_class=TaskEnvironment,
    num_tasks=10,  # 배치당 태스크 수
    meta_iterations=1000  # 메타 학습 반복 횟수
)

# 새로운 태스크에 적응
new_task_env = TaskEnvironment()
support_data = collect_trajectories(new_task_env, maml_agent.meta_policy, num_episodes=5)
adapted_policy, adapted_value = maml_agent.adapt_to_new_task(support_data, num_adapt_steps=5)