# an oracle for grid of 6 positions
import random

actions = ["up", "right", "down", "left"]
s0 = "s0"

# observe the next state given the current state and the taken action
# the observed next state respects the environment evolution and its probablies
def observe_next_state(state, action):
    next_states = {}
    prob_deviation = 0.2
    if (state == "s0" and action == "down") :
        next_states["s4"] = 1.0 - prob_deviation
        next_states[state] = prob_deviation
    elif (state == "s1" and action == "down") :
        next_states["s3"] = 1.0 - prob_deviation
        next_states[state] = prob_deviation
    elif (state == "s1" and action == "up") :
        next_states["s5"] = 1.0 - prob_deviation
        next_states[state] = prob_deviation
    elif (state == "s2" and action == "left") :
        next_states["s3"] = 1.0 - prob_deviation
        next_states[state] = prob_deviation
    elif (state == "s3" and action == "left") :
        next_states["s4"] = 1.0 - prob_deviation
        next_states[state] = prob_deviation
    elif (state == "s3" and action == "up") :
        next_states["s1"] = 1.0 - prob_deviation
        next_states[state] = prob_deviation
    elif (state == "s3" and action == "right") :
        next_states["s2"] = 1.0 - prob_deviation
        next_states[state] = prob_deviation
    elif (state == "s4" and action == "down") :
        next_states["s5"] = 1.0 - prob_deviation
        next_states[state] = prob_deviation
    elif (state == "s4" and action == "right") :
       next_states["s3"] = 1.0 - prob_deviation
       next_states[state] = prob_deviation
    elif (state == "s4" and action == "up") :
       next_states["s0"] = 1.0 - prob_deviation
       next_states[state] = prob_deviation
    else :
        next_states[state]=1.0

    return random.choices(list(next_states.keys()), weights=list(next_states.values()))[0]

# observe the reward given the current state and the taken action
def observe_reward(state, action):
    if (state == "s4" and action == "down") :
        return 10
    elif (state == "s1" and action == "up") :
        return 100
    elif (state == "s5") :
        return 0
    else:
        return -1
    
def isEnd(state):
    return (state=="s5")