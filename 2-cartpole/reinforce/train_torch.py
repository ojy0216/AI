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
        self.render = False

        # 행동의 크기 정의
        self.action_size = action_size
        self.state_size = state_size

        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        # 정책신경망과 가치신경망 생성
        self.model = REINFORCE(state_size, action_size)
        # 최적화 알고리즘 설정, 미분값이 너무 커지는 현상을 막기 위해 clipnorm 설정
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.states, self.actions, self.rewards = [], [], []

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        with torch.no_grad():
            policy = self.model(state)[0]
            policy = np.array(policy)
        return np.random.choice(self.action_size, 1, p=policy)[0]

# 반환값 계산
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # 한 에피소드 동안의 상태, 행동, 보상을 저장
    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    # 정책신경망 업데이트
    def train_model(self):
        discounted_rewards = torch.tensor(self.discount_rewards(self.rewards), dtype=torch.float)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)

        # 크로스 엔트로피 오류함수 계산
        policies = self.model(torch.stack(self.states))
        actions = torch.tensor(self.actions, dtype=torch.float)
        action_prob = torch.sum(torch.mul(policies, actions), dim=1)
        cross_entropy = -torch.log(action_prob + 1e-5)
        loss = torch.sum(cross_entropy * discounted_rewards)
        entropy = -policies * torch.log(policies)

        # 오류함수를 줄이는 방향으로 모델 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.states, self.actions, self.rewards = [], [], []
        return torch.mean(entropy)


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

    EPISODES = 1000
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

            agent.append_sample(state, action, reward)

            score += reward
            reward = 0.1 if not done or score == 500 else -1

            state = next_state

            if done:
                # 에피소드마다 정책신경망 업데이트
                entropy = agent.train_model()
                # 에피소드마다 학습 결과 출력
                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                print("episode: {:3d} | score avg: {:3.2f} | entropy: {:.3f}".format(
                    e, score_avg, entropy))

                scores.append(score_avg)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.xlabel("episode")
                pylab.ylabel("average score")
                pylab.savefig("torch_graph.png")

                # 이동 평균이 470 이상일 때 종료
                if score_avg > 470:
                    torch.save(agent.model.state_dict(), PATH)
                    sys.exit()

    torch.save(agent.model.state_dict(), PATH)
