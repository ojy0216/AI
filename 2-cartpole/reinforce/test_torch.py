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

PATH = 'torch_model'


# 상태가 입력, 각 행동의 확률이 출력인 인공신경망 생성
class REINFORCE(nn.Module):
    def __init__(self, state_size, action_size):
        super(REINFORCE, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc_out = nn.Linear(24, action_size)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        policy = f.softmax(self.fc_out(x), 1)
        return policy


# 카트폴 예제에서의 REINFORCE 에이전트
class REINFORCEAgent:
    def __init__(self, state_size, action_size):
        self.render = True

        # 행동의 크기 정의
        self.action_size = action_size
        self.state_size = state_size

        # 정책신경망과 가치신경망 생성
        self.model = REINFORCE(state_size, action_size)
        self.model.load_state_dict(torch.load(PATH))

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        with torch.no_grad():
            policy = self.model(state)[0]
            policy = np.array(policy)
        return np.random.choice(self.action_size, 1, p=policy)[0]


if __name__ == '__main__':
    # CartPole-v1 환경, 최대 타임스텝 수가 500
    env = gym.make('CartPole-v1')
    # 환경으로부터 상태와 행동의 크기를 받아옴
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # REINFORCE 에이전트 생성
    agent = REINFORCEAgent(state_size, action_size)

    scores, episodes = [], []
    score_avg = 0

    EPISODES = 10
    for e in range(EPISODES):
        done = False
        score = 0
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        state = torch.tensor(state, dtype=torch.float)

        while not done:
            if agent.render:
                env.render()

            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)

            # 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            next_state = torch.tensor(next_state, dtype=torch.float)

            score += reward

            state = next_state

            if done:
                # 에피소드마다 학습 결과 출력
                print("episode: {:3d} | score: {:3d}".format(e, int(score)))
