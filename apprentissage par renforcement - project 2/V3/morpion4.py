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
    # Give a reward for X's win and a penalty for O's win
    if is_winner(board, "X"): # X wins!
        return 1  # Reward for X winning
    elif is_winner(board, "O"): # O wins!
        return -1  # Penalty for X losing
    else: # It's a draw!
        return 0  # Neutral reward for a draw

# Q-learning agent for X
def play_episode(Q, epsilon=0.1, alpha=0.5, gamma=0.9):
    board = initialize_board()
    state = tuple(board)
    trace = []
    result = None  # Track result of the episode ('win', 'loss', or 'draw')

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

        r = reward(new_board)  # Reward after X's move

        # Initialize Q-table for new states if necessary
        if new_state not in Q:
            Q[new_state] = {a: 0.0 for a in available_actions(new_board)}

        # Update Q-table using the reward (r)
        if not is_terminal(new_board):  # If the game isn't over yet, look ahead
            next_max = max(Q[new_state].values()) if Q[new_state] else 0
        else:
            next_max = 0

        if state not in Q:
            Q[state] = {a: 0.0 for a in actions}

        # Q-value update for X
        Q[state][action] += alpha * (r + gamma * next_max - Q[state][action])

        trace.append((state, action, r))

        # O plays randomly
        o_actions = available_actions(new_board)
        if o_actions:
            o_action = random.choice(o_actions)
            board_after_o = next_state(new_board, o_action, "O")
        else:
            board_after_o = new_board

        print("board: ", new_board,"board after o: ", board_after_o, "is winner", is_winner(new_board, "X"), "is winner after o", is_winner(board_after_o, "X"))
        board = board_after_o
        state = tuple(board)
    # Determine the result of the game
    if is_winner(board, "X"):
        result = 'win'
    elif is_winner(board, "O"):
        result = 'loss'
    else:
        result = 'draw'

    return Q, result, trace

def printQ(Q):
    best_actions = {}
    for state in Q:
        if Q[state]:  # Check if the state has actions in the Q-table
            best_action = max(Q[state], key=Q[state].get)
            best_actions[state] = best_action
        else:
            best_actions[state] = None  # No best action if the state has no available actions
    print("Best actions:")
    for state, action in best_actions.items():
        if action is not None:
            print(f"State {state}: Best action = {action}, is winner: {is_winner(state, 'X')}")
        else:
            print(f"State {state}: No best action (Q-table is empty)")

if __name__ == "__main__":
    Q = {}
    episodes = 3000
    wins, losses, draws = 0, 0, 0

    for ep in range(episodes):
        Q, result, trace = play_episode(Q)
        if result == 'win':
            wins += 1
        elif result == 'loss':
            losses += 1
        else:
            draws += 1

        # Gradually decrease epsilon to exploit more as the agent learns
        epsilon = max(0.1, 1 - ep / (episodes / 2))

        if ep % 100 == 0:  # Print progress every 100 episodes
            print(f"Episode {ep} completed.")

    # Final results
    print("\nFinal Q-values:")
    printQ(Q)

    print("\nFinal Score after {} episodes:".format(episodes))
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Draws: {draws}")
    print(f"Win rate: {wins / episodes * 100:.2f}%")
    print(f"Loss rate: {losses / episodes * 100:.2f}%")
    print(f"Draw rate: {draws / episodes * 100:.2f}%")
    print("\nTraining completed.")