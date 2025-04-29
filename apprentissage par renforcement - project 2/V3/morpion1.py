import random

def initialize_board():
    return [" " for _ in range(9)]

def display_board(board):
    print("\n".join(["|".join(board[i:i+3]) for i in range(0, 9, 3)]))
    print()

def available_actions(board):
    actions = []
    for i, spot in enumerate(board):
        if spot == " ":
            actions.append(i)
    return actions

def next_state(board, action, player):
    new_board = board.copy()
    new_board[action] = player
    return new_board

def is_winner(board, player):
    wins = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)]

    for win in wins:
        i, j, k = win
        if board[i] == player and board[j] == player and board[k] == player:
            return True
    return False

def is_full(board):
    for spot in board:
        if spot == " ":
            return False
    return True

def is_terminal(board):
    if is_winner(board, "X") or is_winner(board, "O") or is_full(board):
        return True
    return False

def reward(board):
    if is_winner(board, "X"):
        return 1
    return 0

# Q-learning agent for X
def play_episode(Q, epsilon=0.1, alpha=0.5, gamma=0.9):
    board = initialize_board()
    state = tuple(board)
    trace = []

    while not is_terminal(board):
        actions = available_actions(board)

        # Epsilon-greedy strategy
        if random.random() < epsilon or state not in Q:
            action = random.choice(actions)
        else:
            max_action = None
            max_value = float('-inf')
            for a in actions:
                if Q[state][a] > max_value:
                    max_value = Q[state][a]
                    max_action = a
            action = max_action

        # Play X's move
        new_board = next_state(board, action, "X")
        new_state = tuple(new_board)

        r = reward(new_board)

        if new_state not in Q:
            Q[new_state] = {a: 0.0 for a in available_actions(new_board)}

        # Update Q-table
        if not is_terminal(new_board):
            next_max = max(Q[new_state].values()) if Q[new_state] else 0
        else:
            next_max = 0

        if state not in Q:
            Q[state] = {a: 0.0 for a in actions}

        Q[state][action] += alpha * (r + gamma * next_max - Q[state][action])

        trace.append((state, action, r))

        if is_terminal(new_board):
            break

        # O plays randomly
        o_actions = available_actions(new_board)
        if o_actions:
            o_action = random.choice(o_actions)
            board_after_o = next_state(new_board, o_action, "O")
        else:
            board_after_o = new_board

        board = board_after_o
        state = tuple(board)

    return Q, trace

def printQ(Q):
    for state in Q:
        print("State:", state)
        for action in Q[state]:
            print(f"  Action {action}: {round(Q[state][action], 2)}")

if __name__ == "__main__":
    Q = {}
    episodes = 10000
    for ep in range(episodes):
        Q, trace = play_episode(Q)
        if ep % 1000 == 0:
            print(f"Episode {ep} completed.")

    printQ(Q)