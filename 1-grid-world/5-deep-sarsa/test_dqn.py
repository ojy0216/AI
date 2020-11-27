import random
import numpy as np
from environment import Env
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

PATH = './torch_model_dqn'


# 상태가 입력, 큐함수가 출력인 인공신경망 생성
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc_out = nn.Linear(30, action_size)
        # nn.init.uniform_(self.fc_out.weight, -1e-3, 1e-3)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        q = self.fc_out(x)
        return q


# DQN 에이전트
class DQNAgent:
    def __init__(self, state_size, action_size):
        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # 모델과 타깃 모델 생성
        self.model = DQN(state_size, action_size)
        self.model.load_state_dict(torch.load(PATH))

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        with torch.no_grad():
            q_value = self.model(state)
        return torch.argmax(q_value, 1).item()


if __name__ == "__main__":
    env = Env(render_speed=0.00001)
    state_size = 15
    action_space = [0, 1, 2, 3, 4]
    action_size = len(action_space)
    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []

    EPISODES = 10
    for e in range(EPISODES):
        done = False
        score = 0
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        state = torch.tensor(state, dtype=torch.float)

        while not done:
            env.render()

            # 현재 상태로 행동을 선택
            action = agent.get_action(state)
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            next_state = torch.tensor(next_state, dtype=torch.float)

            score += reward
            state = next_state

            if done:
                # 에피소드마다 학습 결과 출력
                print("episode: {:3d} | score: {:.3f} ".format(e, score))
