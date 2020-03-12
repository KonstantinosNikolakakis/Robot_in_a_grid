### Monte-Carlo policy evaluation for a fixed policy: FIXED_POLICY
### Output: the table of values of the policy

import numpy as np
import matplotlib.pyplot as plt
from grid_enviroment import Grid
from policy_evaluation import print_value, print_policy

########################### Grid variables ############################################################################
HEIGHT = 3 # dimension Y of the grid
WIDTH = 4 # dimension X of the grid
START = (2,0) # robot's starting position 
WALL  = {(1,1)} # wall position
TARGET = (0,3)	# target position
HOLES = {(1,3)} # holes' position


################# Policy Iteration ####################################################################################



GAMMA = 0.9

global states, actions

FIXED_POLICY = {
    	(0, 0): (0, 1),
    	(0, 1): (0, 2),
    	(0, 2): (0, 3),
    	(1, 0): (0, 0),
    	(2, 0): (1, 0),
    	(1, 2): (1, 3),
    	(2, 1): (2, 2),
    	(2, 2): (2, 3),
    	(2, 3): (1, 3),
  		}

def random_next_state(a,s): # introduce randomness to the reaction of the robot
	p=np.random.random()
	if p<0.5: # 0.5 to act as based on the action, and 0.5 to choose another direction
		return a 
	else:
		temp=list(actions[states.index(s)])
		temp.remove(a)
		random_index=np.random.choice(len(temp))
		return temp[random_index]


def play_game(grid,policy):
	states= list(policy.keys()) # exclude final states
	grid.set_state(states[np.random.choice(len(states))]) # randomly pick a starting state

	s=grid.current_state()
	states_rewards=[(s,0)] # initialize a list for the tuples (state,reward)

	while not grid.game_over(): # play the game with random initial state according to FIXED_POLICY
		a=policy[s]
		#a=random_next_state(a,s) # remove comment to introduce randomness see random_next_state
		r=grid.make_move(a,False,False)
		s=grid.current_state()
		states_rewards.append((s,r)) # store the reward for each state

	# move backwards to find the values for the specific episode
	G=0
	states_returns=[] # initialize a list for the tuples (state, returns)
	terminal_state=True
	
	for s,r in reversed(states_rewards):
		
		if terminal_state: # trminal state has no return
			terminal_state=False
		else:
			states_returns.append((s,G))
		G= r + GAMMA * G
	states_returns.reverse()
	return states_returns


if __name__=="__main__":
		
	grid=Grid(HEIGHT,WIDTH,START,WALL,TARGET,HOLES)
	penalty=-0.1
	target_gain=1 # gain at target
	hole_cost=-1 # cost to fall in a hole
	grid.set_rewards(penalty,hole_cost,target_gain) # initialize the penalty for each move
	states,actions=grid.State_space_Generator()
	s=grid.draw_grid()
	
	

	V={} #initialize the values and the returns
	returns={}


	for s in states: # initialize the values to 0
		if len(actions[states.index(s)])!=0:
			returns[s]=[0]
		else:
			V[s]=0
	
	for t in range(5000):

		states_returns = play_game(grid,FIXED_POLICY) # play an episode, and get the returns for each state 
		
		seen_states=set()

		for s,G in states_returns:

			if s not in seen_states: # we consider only the fisrt time that we see the state s 
				returns[s].append(G) # update the list of the returns with key s
				V[s]=np.mean(returns[s]) # update the values
				seen_states.add(s) # update the states that we have seen already

	print("values")
	print_value(V,grid)