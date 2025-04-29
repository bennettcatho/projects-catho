import random;

def playEpisode(Q, s0, actions, observe_next_state, observe_reward, isEnd, epsilon= 0.1, alpha= 0.1, horizon= 1000, gamma=0.9):
    # set cursor to initial state
    state = s0 
    h = 0
    rewards = 0

    # step when end state reached or if more than "horizon" number of actions 
    while not isEnd(state) and h < horizon :
        # precise type of action 
        typeAction = "explore"
        
        # replace with exploit/explore depending on epsilon (to be replaced and completed)
        action= random.choice( actions )
        
        
        next_state = observe_next_state(state, action)    
        reward = observe_reward(state, action)
        h+= 1
        
        # initialize new state in Q matrix if not yet exists (to be completed)
        

        # update Q matrix (to be completed)


        # print trace : state, action, next_state, reward 
        print ("trace " + str(h) +  "; state : " +  state + "; action : " + action + "(" + typeAction + "); next state : " + next_state + "; reward : " + str(reward) )

        # move to next state
        state= next_state

    return Q, rewards


def printQ(title, Q) :
    print(title)
    print( "\nQ_TABLE")
    print( "\n".join( [ state + ": "+ ", ".join( [ action+": "+ str(round( Q[state][action], 2 )) for action in Q[state] ] ) for state in Q ] ) )
    print( "\nBEST ACTIONS")
    print( str({ state: max(Q[state], key= Q[state].get) for state in Q }) )


