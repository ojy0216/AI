import random
import numpy as np
from environment import Env
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

PATH = 'torch_model'


# 딥살사 인공신경망
class DeepSARSA(nn.Module):
    def __init__(self, state_size, action_size):
        super(DeepSARSA, self).__init__()
        self.fc1 = nn.Linear(state_size, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc_out = nn.Linear(30, action_size)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        q = self.fc_out(x)
        return q


# 그리드월드 예제에서의 딥살사 에이전트
class DeepSARSAgent:
    def __init__(self, state_size, action_size):
        # 상태의 크기와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        self.epsilon = 0.01
        self.model = DeepSARSA(self.state_size, self.action_size)
        self.model.load_state_dict(torch.load(PATH))

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_values = self.model(state)
            return torch.argmax(q_values, 1)


if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env(render_speed=0.05)
    state_size = 15
    action_space = [0, 1, 2, 3, 4]
    action_size = len(action_space)
    agent = DeepSARSAgent(state_size, action_size)

    scores, episodes = [], []

    EPISODES = 10
    for e in range(EPISODES):
        score = 0
        done = False
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        state = torch.tensor(state, dtype=torch.float)

        while not done:
            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)

            # 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            next_state = torch.tensor(next_state, dtype=torch.float)

            state = next_state
            score += reward

            if done:
                # 에피소드마다 학습 결과 출력
                print("episode: {:3d} | score: {:3d}".format(e, score))