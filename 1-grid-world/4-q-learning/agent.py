import numpy as np
import random
from environment import Env
from collections import defaultdict

EPISODE_ROUND = 100
INIT_EPSILON = 0.9

class QLearningAgent:
    def __init__(self, actions):
        self.actions = actions
        self.step_size = 0.01
        self.discount_factor = 0.9
        self.epsilon = INIT_EPSILON
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # <s, a, r, s'> 샘플로부터 큐함수 업데이트
    def learn(self, state, action, reward, next_state):
        state, next_state = str(state), str(next_state)
        q_1 = self.q_table[state][action]
        # 벨만 최적 방정식을 사용한 큐함수의 업데이트
        q_2 = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.step_size * (q_2 - q_1)

    # 큐함수에 의거하여 입실론 탐욕 정책에 따라서 행동을 반환
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 무작위 행동 반환
            action = np.random.choice(self.actions)
        else:
            # 큐함수에 따른 행동 반환
            state = str(state)
            q_list = self.q_table[state]
            action = arg_max(q_list)
        return action

    def set_epsilon(self, k):
        self.epsilon = INIT_EPSILON / k


# 큐함수의 값에 따라 최적의 행동을 반환
def arg_max(q_list):
    max_idx_list = np.argwhere(q_list == np.amax(q_list))
    max_idx_list = max_idx_list.flatten().tolist()
    return random.choice(max_idx_list)


if __name__ == "__main__":
    env = Env()
    agent = QLearningAgent(actions=list(range(env.n_actions)))
    step = 0
    episode_num = 1
    reward_list = []

    for episode in range(EPISODE_ROUND):
        state = env.reset()

        while True:
            # 게임 환경과 상태를 초기화
            env.render()
            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)
            # 행동을 취한 후 다음 상태, 보상 에피소드의 종료여부를 받아옴
            next_state, reward, done = env.step(action)
            # <s,a,r,s'>로 큐함수를 업데이트
            agent.learn(state, action, reward, next_state)

            state = next_state
            
            # 모든 큐함수를 화면에 표시
            env.print_value_all(agent.q_table)

            step += 1

            if done:
                # Terminal state 이외에는 reward 가 0 이므로 1번만 계산
                episode_reward = reward * (agent.discount_factor ** step)
                print("Episode[{}] : Epsilon = {}, Terminal state = {}, Step = {}, Reward return = {}".
                      format(episode_num, agent.epsilon, next_state, step, episode_reward))
                step = 0
                episode_num += 1
                reward_list.append((agent.epsilon, episode_reward))
                agent.set_epsilon(episode_num)
                break

    np.save('decaying-e-greedy', reward_list)
