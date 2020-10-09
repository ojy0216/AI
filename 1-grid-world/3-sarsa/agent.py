import numpy as np
import random
from collections import defaultdict
from environment import Env

EPISODE_ROUND = 100
INIT_EPSILON = 0.1


class SARSAgent:
    def __init__(self, actions):
        self.actions = actions
        self.step_size = 0.01
        self.discount_factor = 0.9
        self.epsilon = INIT_EPSILON
        # 0을 초기값으로 가지는 큐함수 테이블 생성
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # <s, a, r, s', a'>의 샘플로부터 큐함수를 업데이트
    def learn(self, state, action, reward, next_state, next_action):
        state, next_state = str(state), str(next_state)
        current_q = self.q_table[state][action]
        next_state_q = self.q_table[next_state][next_action]
        td = reward + self.discount_factor * next_state_q - current_q
        new_q = current_q + self.step_size * td
        self.q_table[state][action] = new_q

    # 입실론 탐욕 정책에 따라서 행동을 반환
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
    agent = SARSAgent(actions=list(range(env.n_actions)))
    step = 0
    episode_num = 1
    reward_list = []

    for episode in range(EPISODE_ROUND):
        # 게임 환경과 상태를 초기화
        state = env.reset()
        # 현재 상태에 대한 행동을 선택
        action = agent.get_action(state)


        while True:
            env.render()

            # 행동을 위한 후 다음상태 보상 에피소드의 종료 여부를 받아옴
            next_state, reward, done = env.step(action)
            # 다음 상태에서의 다음 행동 선택
            next_action = agent.get_action(next_state)
            # <s,a,r,s',a'>로 큐함수를 업데이트
            agent.learn(state, action, reward, next_state, next_action)

            state = next_state
            action = next_action

            step += 1

            # 모든 큐함수를 화면에 표시
            env.print_value_all(agent.q_table)

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

    np.save('decaying_e-greedy', reward_list)
