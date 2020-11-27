import os
import sys
import pylab
from environment import Env
import random
import numpy as np
from collections import deque
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


# 그리드 월드 DQN 에이전트
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = False

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000

        # 리플레이 메모리, 최대 크기 2000
        self.memory = deque(maxlen=2000)

        # 모델과 타깃 모델 생성
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.eval()

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # 타깃 모델 초기화
        self.update_target_model()

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(state)
            return torch.argmax(q_value, 1).item()

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def one_hot_encoding(self, length, x):
        tmp = torch.zeros((length, 1))
        tmp[x] = 1
        return tmp

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = torch.tensor([np.array(sample[0][0]) for sample in mini_batch], dtype=torch.float)
        actions = torch.tensor([sample[1].item() for sample in mini_batch])
        rewards = torch.tensor([sample[2] for sample in mini_batch])
        next_states = torch.tensor([np.array(sample[3][0]) for sample in mini_batch], dtype=torch.float)
        dones = torch.tensor([sample[4] for sample in mini_batch])

        dones_int = torch.tensor([1 if don else 0 for don in dones])

        # 현재 상태에 대한 모델의 큐함수
        predicts = self.model(states)
        one_hot_actions = self.one_hot_encoding(len(actions), actions)
        predicts = torch.sum(torch.mul(predicts, one_hot_actions), axis=1)

        # 다음 상태에 대한 타깃 모델의 큐함수
        with torch.no_grad():
            target_predicts = self.target_model(next_states)

        # 벨만 최적 방정식을 이용한 업데이트 타깃
        max_q = torch.amax(target_predicts, 1)
        targets = rewards + (1 - dones_int) * self.discount_factor * max_q

        # 오류함수를 줄이는 방향으로 모델 업데이트
        self.optimizer.zero_grad()
        loss = self.criterion(targets, predicts)
        # loss = f.smooth_l1_loss(targets, predicts)
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    env = Env(render_speed=0.001)
    state_size = 15
    action_space = [0, 1, 2, 3, 4]
    action_size = len(action_space)
    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []
    score_avg = 0

    EPISODES = 300
    for e in range(EPISODES):
        done = False
        score = 0
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

            score += reward

            # 리플레이 메모리에 샘플 <s, a, r, s'> 저장
            agent.append_sample(np.array(state), np.array(action), reward, np.array(next_state), done)

            # 매 타임스텝마다 학습
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            state = next_state

            if done:
                # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                agent.update_target_model()
                # 에피소드마다 학습 결과 출력
                print("episode: {:3d} | score: {:3.2f} | memory length: {:4d} | epsilon: {:.4f}".format(
                    e, score, len(agent.memory), agent.epsilon))

                torch.save(agent.model.state_dict(), PATH)
