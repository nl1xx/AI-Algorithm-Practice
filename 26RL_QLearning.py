# 宝藏在最右边的位置。通过训练(随机选择向左还是向右)去获得这个宝藏

import time
import numpy as np
import pandas as pd

N_STATES = 6
ACTIONS = ["left", "right"]
EPSILON = 0.9  # 探索率
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 15
FRESH_TIME = 0.3
TerminalFlag = "terminal"


def build_q_table(n_states, actions):
    return pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions
    )


def choose_action(state, q_table):
    state_table = q_table.loc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_table == 0).all()):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_table.idxmax()  # .idxmax()返回的是最大值的索引
    return action_name


def get_env_feedback(S, A):
    if A == "right":
        if S == N_STATES - 2:  # s=4终止
            S_, R = TerminalFlag, 1
        else:
            S_, R = S + 1, 0
    else:
        S_, R = max(0, S - 1), 0
    return S_, R


def update_env(S, episode, step_counter):
    env_list = ["-"] * (N_STATES - 1) + ["T"]
    if S == TerminalFlag:
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print(interaction)
        time.sleep(2)
    else:
        env_list[S] = '0'
        interaction = ''.join(env_list)
        print(interaction)
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]

            if S_ != TerminalFlag:
                q_target = R + GAMMA * q_table.loc[S_, :].max()
            else:
                q_target = R
                is_terminated = True
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            S = S_
            update_env(S, episode, step_counter + 1)
            step_counter += 1
    return q_table


if __name__ == '__main__':
    q_table = rl()
    print(q_table)
