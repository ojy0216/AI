import sys
import gym
import pylab
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

PATH = 'torch_model_450'


# 정책 신경망과 가치 신경망 생성
class A2C(nn.Module):
    def __init__(self, state_size, action_size):
        super(A2C, self).__init__()
        self.actor_fc = nn.Linear(state_size, 24)
        self.actor_out = nn.Linear(24, action_size)
        nn.init.uniform_(self.actor_out.weight, -1e-3, 1e-3)

        self.critic_fc1 = nn.Linear(state_size, 24)
        self.critic_fc2 = nn.Linear(24, 24)
        self.critic_out = nn.Linear(24, 1)
        nn.init.uniform_(self.critic_out.weight, -1e-3, 1e-3)

    def forward(self, x):
        actor_x = torch.tanh(self.actor_fc(x))
        policy = f.softmax(self.actor_out(actor_x), dim=1)

        critic_x = torch.tanh(self.critic_fc1(x))
        critic_x = torch.tanh(self.critic_fc2(critic_x))
        value = self.critic_out(critic_x)

        return policy, value


# 카트폴 예제에서의 액터-크리틱(A2C) 에이전트
class A2CAgent:
    def __init__(self, state_size, action_size):
        self.render = True

        # 행동의 크기 정의
        self.action_size = action_size
        self.state_size = state_size

        # 정책신경망과 가치신경망 생성
        self.model = A2C(self.state_size, self.action_size)

        self.model.load_state_dict(torch.load(PATH))

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        with torch.no_grad():
            policy, _ = self.model(state)
            policy = np.array(policy[0])
        return np.random.choice(self.action_size, 1, p=policy)[0]


if __name__ == "__main__":
    # CartPole-v1 환경, 최대 타임스텝 수가 500
    env = gym.make('CartPole-v1')
    # 환경으로부터 상태와 행동의 크기를 받아옴
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 액터-크리틱(A2C) 에이전트 생성
    agent = A2CAgent(state_size, action_size)

    scores, episodes = [], []
    score_avg = 0

    num_episode = 10
    for e in range(num_episode):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        state = torch.tensor(state, dtype=torch.float)

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            next_state = torch.tensor(next_state, dtype=torch.float)

            score += reward
            state = next_state

            if done:
                print("episode: {:3d} | score: {:3d}".format(e, int(score)))
