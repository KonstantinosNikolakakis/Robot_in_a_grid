### Value iteration algorithm for a robot. The enviroment is grid of dimensions HEIGHT x WIDTH
### Grid variables: HEIGHT, WIDTH, START, TARGET, WALL, HOLES
### The program finds the optimal policy (minimum distance) that avoids the holes 'O' and the walls 'W' and reaches the target position 'T' 
### Output: The optimal policy and the optimal values V*

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
THRESHOLD=10e-4 # threshold to test convergence of the value 
GAMMA=0.9 # forgetting factor



if __name__=="__main__":
		
	grid=Grid(HEIGHT, WIDTH, START, WALL, TARGET, HOLES) # initialize the enviroemnt
	penalty=-0.1 # penalize each move
	grid.set_rewards(penalty) # generate the resward for all states
	states,actions=grid.State_space_Generator() # generate the state space and the corresponding actions
	
	grid.draw_grid() # draw the grid at the initial state
	
	print("Rewards for reaching the positions (nan corresponds to unreachable positions):")
	print(grid.rewards) # print the rewards
	print("")

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


	### initialize the values V(s) randomly ####
	V={} 
	for s in states: # Initialize the values to 0
		if (len(actions[states.index(s)])!=0): # Check for terminal or unreachable positions, they have no further action
			V[s]=np.random.random()
		else:
			V[s]=0
	print("The values are initialized randomly, terminal and unreachable positions have value 0:")
	print_value(V,grid) # Print the initial values
	print("")
	############################################
	
	############# Value Iteration ################################
	
	while True: # Repeat until values convergence
		greatest_change=0 # to set the current change and compare it to the THRESHOLD
		for s in states:
			old_V=V[s] # Store the old value of state s
			new_v = float('-inf') # tracking the max value for the corresponding state s

			if (len(actions[states.index(s)])!=0): # Check for terminal or unreachable positions, they have no further action
				for a in actions[states.index(s)]:
					grid.set_state(s)		# Update the state 
					r=grid.make_move(a,False,False) # Make move, False = do not print the grid
					v = r + GAMMA * V[grid.current_state()] # Bellman equation
					if (v > new_v): # Find the max 
						new_v= v
				V[s]=new_v
				greatest_change=max(greatest_change,np.abs(old_V-V[s]))

		if (greatest_change<THRESHOLD): 
			break 

	for s in policy.keys():
		if (len(actions[states.index(s)])!=0): # Check for terminal or unreachable positions, they have no further action
			best_action = None # best action at the state s
			greatest_value = float('-inf')

			for a in actions[states.index(s)]:
				grid.set_state(s)
				r=grid.make_move(a,False,False)
				v= r+GAMMA* V[grid.current_state()]
				if (v > greatest_value):
					greatest_value=v
					best_action=a 
			policy[s]=best_action

	##############################################################

	print("The values of the optimal policy are:")
	print_value(V,grid)
	print("")

	print("The optimal policy is:")
	print_policy(policy,grid)
	print("")

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
