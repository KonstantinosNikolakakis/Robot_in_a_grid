### Policy evaluation for a uniform random policy and a fixed policy: FIXED_POLICY
### Output: the tables of values for the two cases

import numpy as np
import matplotlib.pyplot as plt
from grid_enviroment import Grid
import math

################# Policy evaluation ####################################################################################
THRESHOLD=10e-4 # threshold for convergence

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


def print_value(V,g): # prints the table of values
	for i in range(g.height):
		print("____________________________")
		for j in range(g.width):
			v=V.get((i,j),0) #if the element (i,j) does not exist in V retuns 0
			if (v>=0):
				print(" %.2f|" %v, end = '')
			else:
				print("%.2f|" % v, end = '')
		print("")
	print("____________________________")
	print("")

def print_policy(P,g): # prints the table of actions
	for i in range(g.height):
		print("_________________________________________________")
		for j in range(g.width):
			a=P.get((i,j), '     ')
			if (a!='     '):
				a_string= "("+str(a[0])+","+str(a[1])+")"
			else:
				a_string=a
			print("  %s  |" % a_string, end = '')
		print("")


if __name__=="__main__":
		
	grid=Grid(3,4,(2,0),{(1,1)},(0,3),{(1,3)})
	penalty=-0.1
	grid.set_rewards(penalty)
	states,actions=grid.State_space_Generator()
	s=grid.draw_grid()
	
	print("Rewards for reaching the positions:")
	print(grid.rewards)
	print("")

	### Policy evaluation for uniformly random actions ###

	V={} 
	for s in states: # initialize the values to 0
		V[s]=0
	gamma = 1.0 # discount factor

	# repeat until convergence
	while True:
		greatest_change=0 # tracking of the change

		for s in states:
			old_V=V[s] # store the previos value

			#We update V[s] only if the state is not terminal 
			if (len(actions[states.index(s)])!=0): # Actions at the terminal states are empty lists
				acc_new_v=0 # we accumulate the value while iterating for all posible actions
				prob_of_action= 1.0/len(actions[states.index(s)]) # uniform
				
				for a in actions[states.index(s)]: # itarate through the possible actions
					grid.set_state(s)
					r=grid.make_move(a,False,False) # make move without printing the grid or the path 
					#if (r==-0.1): r=0 #uncomment this line to give 0 penalty per move, excluding the terminal states T, O
					acc_new_v +=  prob_of_action * (r + gamma * V[grid.current_state()]) # Bellman equation
				V[s]=acc_new_v
				greatest_change = max(greatest_change, np.abs(old_V-V[s])) # update the change

		if (greatest_change < THRESHOLD): # convergence condition
			break
	print("The values of the uniform random policy are:")	
	print_value(V,grid)		
	#####################################################
	

	### Policy evaluation of the fixed policy FIXED_POLICY ###
	V = {}
	for s in states: # initialize V(s) = 0
		V[s] = 0

	gamma = 0.9 # discount factor

	while True: # repeat until convergence
		greatest_change = 0 # tracking the change of the values
		for s in states:
			old_v = V[s] # store the previous value

      		# V(s) only has value if it's not a terminal state
			if s in FIXED_POLICY: # avoid terminal sates
				a = FIXED_POLICY[s] # find the correspondin action of the state
				grid.set_state(s)
				r = grid.make_move(a,False,False) # make move without printing the grid or the path

				#if (r==-0.1): r=0 #uncomment this line to give 0 penalty per move, excluding the terminal states T, O
				
				V[s] = r + gamma * V[grid.current_state()] # Bellman equation
				greatest_change = max(greatest_change, np.abs(old_v - V[s])) # update the change
				
		if greatest_change < THRESHOLD: # condition for convergence
			break
	print("The fixed policy is the following:")
	print_policy(FIXED_POLICY, grid) 
	print("")

	print("The values of the fixed policy are:")
	print_value(V, grid)	