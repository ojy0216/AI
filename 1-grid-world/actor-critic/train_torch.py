import sys
import gym
import pylab
import numpy as np
from environment import Env
import os
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

PATH = 'torch_model_450'


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
        self.render = False

        # 행동의 크기 정의
        self.action_size = action_size
        self.state_size = state_size

        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        # 정책신경망과 가치신경망 생성
        self.model = A2C(state_size, action_size)
        # 최적화 알고리즘 설정, 미분값이 너무 커지는 현상을 막기 위해 clipnorm 설정
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        with torch.no_grad():
            policy, _ = self.model(state)
            policy = np.array(policy[0])
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def one_hot_encoding(self, length, x):
        tmp = torch.zeros((1, length))
        tmp[0][x] = 1
        return tmp

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self, state, action, reward, next_state, done):
        policy, value = self.model(state)
        _, next_value = self.model(next_state)
        target = reward + (1 - done) * self.discount_factor * next_value[0]

        # 정책 신경망 오류 함수 구하기
        one_hot_action = self.one_hot_encoding(self.action_size, action)
        action_prob = torch.sum(torch.mul(one_hot_action, policy))
        cross_entropy = -torch.log(action_prob + 1e-5)
        # with torch.no_grad():
        advantage = target - value[0]
        actor_loss = torch.sum(cross_entropy * advantage)

        # 가치 신경망 오류 함수 구하기
        # torch.nn.utils.clip_grad_norm(self.critic.parameters(), 5)
        critic_loss = 0.5 * torch.pow(target - value[0], 2)
        critic_loss = torch.sum(critic_loss)

        # 하나의 오류 함수로 만들기
        loss = 0.2 * actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss


if __name__ == '__main__':
    # 환경 생성
    env = Env(render_speed=0.001)
    state_size = 15
    action_space = [0, 1, 2, 3, 4]
    action_size = len(action_space)

    # 액터-크리틱(A2C) 에이전트 생성
    agent = A2CAgent(state_size, action_size)

    scores, episodes = [], []

    num_episode = 200
    for e in range(num_episode):
        done = False
        score = 0
        loss_list = []
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

            reward -= 0.1
            score += reward

            # 매 타임스텝마다 학습
            loss = agent.train_model(state, action, reward, next_state, done)
            loss_list.append(loss)
            state = next_state

            if done:
                # 에피소드마다 학습 결과 출력
                print("episode: {:3d} | score: {:3.2f} | loss: {:.3f}".format(
                    e, score, torch.mean(torch.stack(loss_list))))

                # 에피소드마다 학습 결과 그래프로 저장
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.xlabel("episode")
                pylab.ylabel("score")
                pylab.savefig("torch_graph.png")

    torch.save(agent.model.state_dict(), PATH)
