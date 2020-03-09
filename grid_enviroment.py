### Class Grid: Definition of the enviroment
### The class provides functions that: initialize the enviroment, interact with agent, present the grid and the path of the policy  
 
import numpy as np
import matplotlib.pyplot as plt


#Input: 1) dimensions of the grid: height x width, 2) starting position: start 
#		3) walls, holes, and target positions: obj_pos, hole, target
class Grid:
	def __init__(self,height,width,start,obj_pos,target,hole): 
		self.width=width
		self.height=height
		self.i=start[0]
		self.j=start[1]
		self.obj_pos=obj_pos
		self.target=target
		self.hole=hole

		### Initializing the Grid as string for the output
		self.sgrid=  ["" for x in range(self.height)]
		for row in range(self.height):
			for col in range(self.width):
				if start==(row,col): self.sgrid[row]=self.sgrid[row]+("R ")
				elif self.target==(row,col): self.sgrid[row]=self.sgrid[row]+("T ")
				elif (row,col) in self.hole: self.sgrid[row]=self.sgrid[row]+("O ")
				elif (row,col) in self.obj_pos: self.sgrid[row]=self.sgrid[row]+("W ")
				else: self.sgrid[row]=self.sgrid[row]+(". ")

		### Initializing the trace of the path
		self.sgrid_path=  ["" for x in range(self.height)]
		for row in range(self.height):
			for col in range(self.width):
				if start==(row,col): self.sgrid_path[row]=self.sgrid_path[row]+("S  ")
				elif self.target==(row,col): self.sgrid_path[row]=self.sgrid_path[row]+("T  ")
				elif (row,col) in self.hole: self.sgrid_path[row]=self.sgrid_path[row]+("O  ")
				elif (row,col) in self.obj_pos: self.sgrid_path[row]=self.sgrid_path[row]+("W  ")
				else: self.sgrid_path[row]=self.sgrid_path[row]+(".  ")

		self.move_number=0 # count number of moves so far, we use it to visualize the traces of the path 

	def set_state(self,s): # update the position of the robot
		self.i=s[0]
		self.j=s[1]

	def current_state(self): # returns the current position
		return (self.i,self.j)

	def draw_grid(self): # draw the grid at the current state
		for row in range(self.height):
			print(self.sgrid[row])
		print("")
		return self.sgrid

	def draw_grid_path(self): # draw the trace of the path at the current state
		for row in range(self.height):
			print(self.sgrid_path[row])
		print("")
		return self.sgrid_path
	
	def possible_actions(self,cor_i,cor_j): #find the set of possible actions at (cor_i,cor_j)
		actions=[]
		if ((cor_i,cor_j)==self.target):
				return actions
		elif ((cor_i,cor_j) in self.hole):
				return actions
		if (-1<cor_i-1 and (cor_i-1,cor_j) not in self.obj_pos):
			actions.append((cor_i-1,cor_j))
		if (cor_i+1<self.height and (cor_i+1,cor_j) not in self.obj_pos):
			actions.append((cor_i+1,cor_j))
		if (cor_j+1<self.width and (cor_i,cor_j+1) not in self.obj_pos):
			actions.append((cor_i,cor_j+1))
		if (-1<cor_j-1 and (cor_i,cor_j-1) not in self.obj_pos):
			actions.append((cor_i,cor_j-1))
		return actions

	def update_grid(self,prev_pos,next_pos,move_number): # update the grid and the trace of the path after a move
		self.sgrid[prev_pos[0]]=self.sgrid[prev_pos[0]][:2*prev_pos[1]] + "." + self.sgrid[prev_pos[0]][2*prev_pos[1]+1:]
		self.sgrid[next_pos[0]]=self.sgrid[next_pos[0]][:2*next_pos[1]] + "R" + self.sgrid[next_pos[0]][2*next_pos[1]+1:]
		
		self.sgrid_path[next_pos[0]]=self.sgrid_path[next_pos[0]][:3*next_pos[1]] +str(move_number//10) +str(move_number%10)+" " + self.sgrid_path[next_pos[0]][3*next_pos[1]+3:]

	# We use make make_move while training the agent and to present the optimal policy at the end
	def make_move(self,action,verbose_grid,verbose_path): # verbose_grid = True to print the grid, verbose_path = True to print the path
		if action in self.possible_actions(self.i,self.j): # Check if the move is valid
			if verbose_grid or verbose_path: 
				self.move_number=self.move_number+1 # increase the number of moves so far
				self.update_grid((self.i,self.j),action,self.move_number) # update the grid and the path
				print("The next move is:", action)
			self.i=action[0] # make the move
			self.j=action[1] # make the move

			if verbose_grid: # Print the grid if requested
				print("The next move is: ", action)
				self.draw_grid()
				if ((self.i,self.j)==self.target): 
					print("Done! The robot reached the target.")
					print("")
				elif ((self.i,self.j) in self.hole):
					print("The Robot was destroyed")
					print("")
				elif ((self.i,self.j) in self.obj_pos): print("invalid action:", action)
					
			if verbose_path: # Print the path if requested
				print("The path is:")
				self.draw_grid_path()
				if ((self.i,self.j)==self.target): 
					print("Done! The robot reached the target.")
					print("")
				elif ((self.i,self.j) in self.hole):
					print("The Robot was destroyed")
					print("")
				elif ((self.i,self.j) in self.obj_pos): print("invalid action:", action)
					

		return self.rewards[self.i][self.j] # returns the reward for the specific move
	
	### State_space_Generator() creates the state space and the posiible actions for each state
	def State_space_Generator(self): # returns a tuple of two lists: list1,list2 the possible positions and the possible actions for each position
		state_space_positions=[]
		state_space_moves=[]
		for k in range(self.height):
			for l in range(self.width):
				if ((k,l) not in self.obj_pos): # the positions of any wall is not a possible state
					state_space_positions.append((k,l))
					state_space_moves.append(self.possible_actions(k,l))
		return state_space_positions,state_space_moves

	### set_rewards(): creates the rewards for each action
	def set_rewards(self,penalty):
		self.rewards= penalty*np.ones((self.height,self.width))
		for x in range(self.height):
			for y in range(self.width):
				if ((x,y)== self.target): self.rewards[x][y]=+1
				elif ((x,y) in self.hole): self.rewards[x][y]=-1
				elif ((x,y) in self.obj_pos):	self.rewards[x][y]=None # the position of walls are not reachable
		return self.rewards

	def game_over(self):
		game_over=False
		if ((self.i,self.j)==self.target) or ((self.i,self.j) in self.hole):
			game_over=True
		return game_over
			