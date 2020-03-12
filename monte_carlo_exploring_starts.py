### Monte-Carlo Exploring Starts: Control problem
### Output: The optimal path 

import numpy as np
import matplotlib.pyplot as plt
from grid_enviroment import Grid
from policy_evaluation import print_value, print_policy

########################### Grid variables ############################################################################
HEIGHT = 15  # dimension Y of the grid
WIDTH = 15	# dimension X of the grid
START = (7,2) # robot's starting position 
WALL  = {(2,4),(2,3),(3,4),(3,3),(5,2),(5,3),(5,4),(9,9),(10,9)} # wall position
TARGET = (6,10)	# target position
HOLES = {(3,9),(3,5),(3,6),(3,7),(3,8),(4,9),(5,9),(6,9),(7,9),(8,9),(7,10),(7,11),(7,12)} # holes' position
################# Policy Iteration ####################################################################################

global states, actions
 
GAMMA = 0.9

def play_game(grid,policy,non_terminal_states): 
	########### #Exploring starts ########################
	grid.set_state(non_terminal_states[np.random.choice(len(non_terminal_states))]) # randomly pick a starting non-terminal state
	s=grid.current_state() # store the coordinates

	random_index= np.random.choice(len(actions[states.index(s)])) #randomly pick an action from the possible actions
	a=actions[states.index(s)][random_index]
	########### #Exploring starts ########################

	states_actions_rewards=[(s,a,0)] # initialize a list for the tuples (state,actions,reward)

	seen_states=set() # keep tracking of the states that we have already seen in this episode 
	seen_states.add(s)
	number_of_moves=0

	# play the game to collect the rewards for each move
	while True: # play the game with random initial state according to the policy
		old_s = grid.current_state()
		r=grid.make_move(a,False,False)
		number_of_moves+=1
		s=grid.current_state()
		
		if s in seen_states: # if the state is already seen penalize the corresponding action to encourage exclusion of loops in the path 
			reward = -10. / number_of_moves
			states_actions_rewards.append((s, None, reward)) # store the reward of the pair (s,a)
			break
		elif grid.game_over():
			states_actions_rewards.append((s, None, r)) # store the reward of the pair (s,a)
			break
		else:
			a=policy[s]
			states_actions_rewards.append((s, a, r)) # store the reward of the pair (s,a)
		seen_states.add(s) # update the already seen states


	# move backwards to find the values for the specific episode
	G=0
	states_actions_returns=[] # initialize a list for the tuples (state, returns) to store the returns
	terminal_state=True
	
	for s,a,r in reversed(states_actions_rewards): # get each triplet (s,a,r) in the reverse order
		
		if terminal_state: # terminal state has no return
			terminal_state=False
		else:
			states_actions_returns.append((s,a,G)) # update returns
		G= r + GAMMA * G # Bellman equation
	states_actions_returns.reverse() # reverse the returns to put them in the original order
	return states_actions_returns # return the returns of the episode

def max_argmax_dict(d): # input: dictionary, output: the maximum value and its key
	max_key = None
	max_value = float('-inf')

	for k, v in d.items():
		if v > max_value:
			max_key = k
			max_value = v
	return max_key, max_value # return the max value and its key

if __name__=="__main__":
		
	grid=Grid(HEIGHT,WIDTH,START,WALL,TARGET,HOLES) # set the enviroment
	penalty=-0.1 # penalty for each non final move
	target_gain=100 # gain at target
	hole_cost=-100 # cost to fall in a hole
	grid.set_rewards(penalty,hole_cost,target_gain) # initialize the penalty for each move
	states,actions=grid.State_space_Generator() #generate the states, actions
	
	non_terminal_states=[] 
	for s in states: 
		if (len(actions[states.index(s)])!=0):
			non_terminal_states.append(s)  # list with non-terminal states

	s=grid.draw_grid() # present the grid
	
	### initialize the policy randomly ###
	policy={}
	policy_list=[]
	for s in states:
		if (len(actions[states.index(s)])!=0): # Check for terminal or unreachable positions, they have no further action
			random_index = np.random.choice(np.arange(len(actions[states.index(s)]))) #Choose randomnly one of the allowed next positions
			policy_list.append(actions[states.index(s)][random_index]) 
		else: 
			policy_list.append('     ') # Terminal states have no further action

	policy=dict(zip(states,policy_list)) # Create a dictionary keys: position, value: next position
	print("The initial random policy is:")
	print_policy(policy,grid) # Print the initial policy
	print("")
	#######################################
	
	Q={} #initialize Q(s,a) and the returns
	returns={}

	for s in states: # initialize the Q's and returns to 0
		if len(actions[states.index(s)])!=0:
			Q[s]={}
			for a in actions[states.index(s)]:
				Q[s][a]=0
				returns[(s,a)]=[]
		else: # terminal and unreachable states
			pass
	
	deltas=[] # to store the maximum change in the Q table
	for t in range(100000):
		if t%1000==0: # print the counter of the episodes
			print(t)
		greatest_change=0 # to check convergence
		states_actions_returns = play_game(grid,policy,non_terminal_states) # play an episode, and get the returns for each state 
		
		seen_actions_states=set()

		for s,a,G in states_actions_returns:

			#if s not in seen_actions_states: # First visit Monte-Carlo, we consider only the fisrt time that we see the state s 
				old_Q= Q[s][a]
				returns[(s,a)].append(G) # update the list of the returns with key s
				Q[s][a]=np.mean(returns[(s,a)]) # update the values
				seen_actions_states.add((s,a)) # update the states that we have seen already
				greatest_change = max(greatest_change,np.abs(old_Q - Q[s][a]))
		deltas.append(greatest_change)

		# update policy after each episode
		for s in policy.keys():
			if (len(actions[states.index(s)])!=0): # Check for terminal or unreachable positions, they have no further action
				policy[s] = max_argmax_dict(Q[s])[0] # update the policy by finding the maximum Q

	print(deltas)
	plt.plot(deltas)
	plt.title("Maximum change among episodes in the Q(s,a) table")
	plt.xlabel("Number of episodes")
	plt.ylabel("Maximum change in the Q table")
	plt.show()
	print_policy(policy,grid)
	
	############ Print the states move by move for the optimal policy ###########
	
	position=START 
	grid.set_state(position) # Set the position as the starting position
	print("Starting state:")
	grid.draw_grid() # Draw the grid


	while True: # While the state is not a terminal state
		a=policy[position]		# Choose the next action based on the optimal policy
		grid.make_move(a,False,True)	# Make the move according to the optimal policy, True = draw the grid
		if (len(actions[states.index(position)])==0): # If the state is the terminal then break
			break
		position=a
	##############################################################################

	### Present a summary of the starting state and the optimal path
	grid_visual=Grid(HEIGHT, WIDTH, START, WALL, TARGET, HOLES) # object grid_visual for the summary	
	print("The starting state is: ")
	grid_visual.draw_grid()

	print("The path of the optimal policy is:")
	grid.draw_grid_path()
	###############################################################################