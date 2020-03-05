### Policy iteration algorithm for a robot. The enviroment is grid of dimensions HEIGHT x WIDTH
### Grid variables: HEIGHT, WIDTH, START, TARGET, WALL, HOLES
### The program find the optimal policy (minimum distance) that avoids the holes 'O' and the walls 'W' and reaches the target position 'T' 
### Output: The optimal policy and the optimal values V*

import numpy as np
import matplotlib.pyplot as plt
from grid_enviroment import Grid
from policy_evaluation import print_value, print_policy

########################### Grid variables ############################################################################
HEIGHT = 15  # dimension Y of the grid
WIDTH = 15	# dimension X of the grid
START = (7,2) # robot's starting position 
WALL  = {(2,4),(2,3),(3,4),(3,3),(5,2),(5,3),(9,9),(5,4),(10,9)} # wall position
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
	
	############# Policy Iteration ################################

	while True: # repeat until policy does not change: there is no policy that can imporve the cost function
		
		### Policy Evaluation starts here ###		
		while True: # Repeat until values "convergence": policy evaluation step

			greatest_change=0 # to set the current change and compare it to the THRESHOLD
			for s in states:
				old_V=V[s] # Store the old value of state s

				if (len(actions[states.index(s)])!=0): # Check for terminal or unreachable positions, they have no further action
					a=policy[s]
					grid.set_state(s)		# Update the state 
					r=grid.make_move(a,False,False) # Make move, False = do not print the grid
					V[s] = r + GAMMA * V[grid.current_state()] # Bellman equation
					greatest_change=max(greatest_change,np.abs(old_V-V[s]))

			if (greatest_change<THRESHOLD):
				break # Break the inner while and proceed to policy improvement
		###Policy Evaluation ends here ###

		### Policy Improvement starts here ###
		is_new_policy_different_than_the_old = True
		for s in states:
			if (len(actions[states.index(s)])!=0): # Check for terminal or unreachable positions, they have no further action
				old_a=policy[s] # Store the previous policy for state s
				new_a=None	# Initialize the new policy for state s
				largest_value = float('-inf')	# Initialize the largest value to compare policies

				for a in actions[states.index(s)]: # For all possible actions at state s
					grid.set_state(s)	# Update the state
					r=grid.make_move(a,False,False) # Make move, False = do not print the grid
					v=r+GAMMA*V[grid.current_state()] # Find V, Bellman equation
					if (v > largest_value):  # Check for improvement
						largest_value = v # We find a larger value
						new_a = a 			# New better policy candidate
				policy[s]=new_a 	# Update the policy to be the best so far

				if new_a != old_a: # If the actions of all the states remain the same then the policy is the optimal 
					is_new_policy_different_than_the_old = False 


		if (is_new_policy_different_than_the_old):
			break # Break the outter while, the policy has converged

		### Policy Improvement ends here ###

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