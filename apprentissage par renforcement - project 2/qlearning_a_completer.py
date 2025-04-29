import random

def playEpisode(Q, s0, actions, observe_next_state, observe_reward, isEnd, epsilon=0.1, alpha=0.1, horizon=1000, gamma=0.9):
    # set cursor to initial state
    state = s0
    h = 0
    total_rewards = 0

    # step when end state reached or if more than "horizon" number of actions
    while not isEnd(state) and h < horizon:
        # epsilon-greedy action selection: explore or exploit
        if random.random() < epsilon:
            # Explore: choose a random action
            action = random.choice(actions)
            typeAction = "explore"
        else:
            # Exploit: choose the best action based on Q-values
            action = max(Q[state], key=Q[state].get)  # Best action based on Q-values
            typeAction = "exploit"

        next_state = observe_next_state(state, action)
        reward = observe_reward(state, action)
        total_rewards += reward
        h += 1

        # initialize new state in Q matrix if not yet exists
        if next_state not in Q:
            Q[next_state] = {a: 0.0 for a in actions}

        # update Q-value using the Q-learning formula
        max_future_q = max(Q[next_state].values())  # max Q-value for the next state
        Q[state][action] += alpha * (reward + gamma * max_future_q - Q[state][action])

        # print trace: state, action, next_state, reward
        print(f"trace {h}; state: {state}; action: {action} ({typeAction}); next state: {next_state}; reward: {reward}")

        # move to next state
        state = next_state

    return Q, total_rewards


def printQ(title, Q):
    print(title)
    print("\nQ_TABLE")
    print("\n".join([state + ": " + ", ".join([action + ": " + str(round(Q[state][action], 2)) for action in Q[state]]) for state in Q]))
    print("\nBEST ACTIONS")
    print(str({state: max(Q[state], key=Q[state].get) for state in Q}))
