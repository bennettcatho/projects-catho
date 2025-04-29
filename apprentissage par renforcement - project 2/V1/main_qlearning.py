import qlearning_a_completer as qlearning
import grid6 as game


# Evaluate if the simulation reached an end state.
print("\n----------------------------")
print("Simple Q-Learning")
print("----------------------------\n")

# Initialize state-action matrice Q to zero
Q= { game.s0 : { a:0.0 for a in game.actions } }
qlearning.printQ("Initial: ", Q)

# Perform n episodes/iterations of Q-learning
for i in range(1000):
  Q, trace= qlearning.playEpisode(Q, 
                                  game.s0,
                                  game.actions,
                                  game.observe_next_state, 
                                  game.observe_reward, 
                                  game.isEnd)
  print("Iteration " + str(i) + " finished.")


  
qlearning.printQ("Result: ", Q)


# plot number of actions per episode
# plot sum rewards per episode

