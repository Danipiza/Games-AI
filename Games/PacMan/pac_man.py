import random
import math




"""

Args:
    filename (string) : name of the used maze.
"""
class Pacman:

    """
    MOVEMENTS KEYS:
    "up"    or "w": UP
    "right" or "d": RIGHT
    "down"  or "s": DOWN
    "left"  or "a": LEFT
    """

    def __init__(self,file_name):
        self.file_name=file_name
        self.version=int(file_name[-7]) 
        self.win_condition=132 if self.version==1 else 21
        
        # -------------------------------------------------------------------------------------------------------------------
        # --- CONSTANTS -----------------------------------------------------------------------------------------------------
        
        self.EMPTY  =0
        self.WALL   =1
        self.COIN   =2
        self.POWER  =3
        self.AGENT  =4        
        self.RED    =5
        self.PINK   =6
        self.BLUE   =7
        self.ORANGE =8

        # actions.
        self.UP     ='up'
        self.RIGHT  ='right'        
        self.DOWN   ='down' 
        self.LEFT   ='left'       
        

        self.actions=[self.UP,self.RIGHT, self.DOWN, self.LEFT]

        # directions. 
        # 0: UP 
        # 1: RIGHT 
        # 2: DOWN 
        # 3: LEFT
        self.mX=[-1,0,1,0]
        self.mY=[0,1,0,-1]
                

        self.ghosts_colors  =[5,6,7,8]
        self.state_ticks    =[60,30,30]

        # -------------------------------------------------------------------------------------------------------------------       
        # --- VARIABLES -----------------------------------------------------------------------------------------------------
        
        # state.
        # 0: CHASE          (chase certain targets)
            # RED:      target = agent position 
            # PINK:     target = agent position + 4 cells in the agent direction (up is an exeption, also add 4 to the left)
            # BLUE:     target = tmp + vector from red ghost to tmp.       
            #   where 
            #       tmp = agent position + 2 cells in the agent direction (up is an exeption, also add 2 to the left)                     
            # ORANGE:   target = if distante to agent > 8 -> agent. otherwise -> his scatter point
        # 1: SCATTER        (chase scatter point. borders of the maze)
        # 2: FRIGHTENED     (runaway from the aget)
        self.state=1
        
        self.scatter_targets=[]

        # number of ticks in the execution.
        self.exec_tick=0
        # number of ticks in the actual state.
        self.count_state=0

        # agent.
        self.agent_pos=None
        self.agent_dir=1
        self.agent_coins=0   

        # ghosts.        
        self.n_ghosts=4 if self.version==1 else 1
        if self.n_ghosts==4:
            self.ghosts_pos=[[0,0] for _ in range(4)]
            self.ghosts_dir=[1,2,0,0]
            self.ghosts_house=[False,True,True,True]
            # queue, for the leaving order. 0th: ghost id. 1th: home leaving tick
            self.ghost_inHouse=[[1,3],[2,6],[3,9]]
        else:
            self.ghosts_pos=[[0,0]]
            self.ghosts_dir=[3]
            self.ghosts_house=[False]
            # queue, for the leaving order. 0th: ghost id. 1th: home leaving tick
            self.ghost_inHouse=[]
        
        # maze.
        self.maze=[]    # used for the walls, agent and ghosts positions in the GUI                
        self.n=0        # number of rows
        self.m=0        # number of coloumns

        # finalization variable
        self.end=False

        self.reset(True,None,None,None,None)
    
        

        #self.execute()



    def step(self, accion):                
        eat=self.move_agent(self.actions[accion])
        self.move_ghosts()
        
        next_state=self.get_state()

        aux=""
        
        
        reward=-1           # move without eating

        if eat==1:          # power        
            reward=5        
            aux="POWER"
        elif eat==2:        # coin
            reward=10      
            aux="COIN"
        elif eat==3:        # portal
            reward=20      
            aux="PORTAL"
        elif eat==4:        # ghost
            reward=100     
            aux="GHOST"
        elif eat==5:        # has been eaten by a ghost
            reward=-100    
            aux="PIERDE"
        
        # has been eaten by a ghost
        if self.end==True:
            if self.agent_coins!=self.win_condition: 
                reward=-100
                aux="PIERDE"
            else:                 
                reward=1000
                aux="GANA"
        
        a="N"
        if accion==1: a="E"
        elif accion==2: a="S"
        elif accion==3: a="W"
        print("{}\tTick={}  \tState={}  \tCoins={}  \t{}\tAgent= {}\tGhost= {}".format(a,self.exec_tick, 
                                                                                                 self.count_state, self.agent_coins,aux,
                                                                                                 self.agent_pos, self.ghosts_pos[0]))
        
        

        
        return next_state, reward, self.end
      
    def get_state(self):

        state=[]
        for row in self.maze:
            state.extend(row)

        state.append(self.agent_pos[0])
        state.append(self.agent_pos[1])
        #state.append(self.agent_coins) # not useful
        for i in range(self.n_ghosts):
            state.append(self.ghosts_pos[i][0])
            state.append(self.ghosts_pos[i][1])

        return state
    

    """
    Reseting the class variables.
    """
    def reset(self,init,positions,coins,states,dirs):

        self.exec_tick=0

        self.state=1
        self.count_state=0
        
        self.agent_pos=None
        self.agent_dir=1
        self.agent_coins=0   
        
        if self.n_ghosts==4:
            self.ghosts_pos=[[0,0] for _ in range(4)]
            self.ghosts_dir=[1,2,0,0]
            self.ghosts_house=[False,True,True,True]
            # queue, for the leaving order. 0th: ghost id. 1th: home leaving tick
            self.ghost_inHouse=[[1,3],[2,6],[3,9]]
        else:
            self.ghosts_pos=[[0,0]]
            self.ghosts_dir=[3]
            self.ghosts_house=[False]
            # queue, for the leaving order. 0th: ghost id. 1th: home leaving tick
            self.ghost_inHouse=[]
        
        self.maze=[]
        self.read_maze()
        self.n=len(self.maze)
        self.m=len(self.maze[0])

        if self.version==1:
            self.scatter_targets=[[0,self.m],[0,0],[self.n,self.m],[self.n,0]]
        else:  self.scatter_targets=[[0,4]]

        self.end=False

        if init==False:
            idx=self.version
            if idx>0:
                self.ghosts_house=[False,False,False,False]
                self.ghost_inHouse=[]

                self.exec_tick=states[idx]            
                self.count_state=states[idx]
                self.agent_coins=coins[idx]

                if self.version==1:
                    self.salida_fants=[5,10]
                else: self.salida_fants=[2,5]
                
                for g in range(self.n_ghosts):
                    self.ghosts_pos[g][0]=positions[idx][g][0]
                    self.ghosts_pos[g][1]=positions[idx][g][1]
                    self.ghosts_dir[g]=dirs[idx][g]



        return self.get_state()
    
    """
    Reading the maze from a .txt file. 
    Also search for the positions of the agent and ghosts
    """
    def read_maze(self):                

        with open(self.file_name, 'r') as file:        
            for x, line in enumerate(file):
                row=list(map(int, line.split()))
                self.maze.append([0 for _ in range(len(row))])
                
                for y in range(len(row)):                                          
                    
                    if row[y]==self.AGENT: 
                        self.maze[x][y]=0
                        self.agent_pos=[x,y]
                    
                    elif row[y]==self.RED: 
                        self.maze[x][y]=0
                        self.ghosts_pos[0]=[x,y]                    
                        self.salida_fants=[x,y]

                    elif row[y]==self.PINK: 
                        self.maze[x][y]=0
                        self.ghosts_pos[1]=[x,y]
                        self.ghosts_house[1]=[x,y]
                        

                    elif row[y]==self.BLUE: 
                        self.maze[x][y]=0
                        self.ghosts_pos[2]=[x,y]
                        self.ghosts_house[2]=[x,y]

                    elif row[y]==self.ORANGE: 
                        self.maze[x][y]=0
                        self.ghosts_pos[3]=[x,y]
                        self.ghosts_house[3]=[x,y]

                    else: 
                        self.maze[x][y]=row[y]

    """
    Moving the agent.
    eat ghosts or is eaten by a ghost
    
    Args:
        mov (string): action executed by the agent.

    Return: 
        Coin (int): 1, if the agent eat a coin. 0, otherwise.        
    """          
    def move_agent(self, mov):
        x=self.agent_pos[0]
        y=self.agent_pos[1] 
        
        ret=0
        
        

        # -------------------------------------------------------------------------------------------------------------------
        # --- MOVE ----------------------------------------------------------------------------------------------------------

        # POWER is only reachable by up and down actions

        if mov==self.UP:            
            if x>0 and self.maze[x-1][y]!=1:                # dest position != wall
                if self.maze[x-1][y]==self.COIN:    # COIN. adds a coin
                    self.maze[x-1][y]=0
                    ret=2

                elif self.maze[x-1][y]==self.POWER: # POWER. change the game state to FRIGHTENED
                    self.maze[x-1][y]=0
                    self.state=2
                    self.count_state=0
                    ret=1

                    # change the directions of the ghosts
                    for i in range(self.n_ghosts):                        
                        if not self.ghosts_house[i]:
                            self.ghosts_dir[i]+=2
                            self.ghosts_dir[i]%=4                        
                
                
                x-=1

        elif mov==self.DOWN:
            if x<len(self.maze)-1 and self.maze[x+1][y]!=1: # dest position != wall
                if self.maze[x+1][y]==self.COIN:    # COIN. adds a coin
                    self.maze[x+1][y]=0
                    ret=2

                elif self.maze[x+1][y]==self.POWER: # POWER. change the game state to FRIGHTENED
                    self.maze[x+1][y]=0
                    self.state=2
                    self.count_state=0
                    ret=1

                    for i in range(self.n_ghosts):
                        if not self.ghosts_house[i]:
                            self.ghosts_dir[i]+=2
                            self.ghosts_dir[i]%=4
                
                
                x+=1

        elif mov==self.LEFT:
            if y>0 and self.maze[x][y-1]!=1:                # dest position != wall
                if self.maze[x][y-1]==self.COIN:    # COIN. adds a coin
                    self.maze[x][y-1]=0
                    ret=2 
                elif self.maze[x][y-1]==self.POWER: # POWER. change the game state to FRIGHTENED
                    self.maze[x][y-1]=0
                    self.state=2
                    self.count_state=0
                    ret=1

                    for i in range(self.n_ghosts):
                        if not self.ghosts_house[i]:
                            self.ghosts_dir[i]+=2
                            self.ghosts_dir[i]%=4               
                
                y-=1
            elif y==0 and self.portal_gates(x):           # "portal"                                                 
                y=self.m-1
                ret=3

        elif mov==self.RIGHT:
            if y<self.m-1 and self.maze[x][y+1]!=1:         # dest position != wall
                if self.maze[x][y+1]==self.COIN:    # COIN. adds a coin
                    self.maze[x][y+1]=0
                    ret=2                
                
                y+=1
            elif y==self.m-1 and self.portal_gates(x):    # "portal"                                             
                y=0
                ret=3
        
        # -------------------------------------------------------------------------------------------------------------------
        # --- EAT/LOSE ------------------------------------------------------------------------------------------------------

        if self.state==2:   # eat.
            eaten=[]
            for i in range(self.n_ghosts):
                if self.ghosts_pos[i][0]==x and self.ghosts_pos[i][1]==y:
                    eaten.append(i)
            
            tmp=1
            for i in eaten:
                ret=4

                # move the eaten ghost to the house cell.
                self.ghosts_pos[i][0]=self.salida_fants[0]+2
                self.ghosts_pos[i][1]=self.salida_fants[1]

                # add to the queue of ghosts in house
                # they leave in 3 ticks intervals
                self.ghost_inHouse.append([i,self.exec_tick+(3*tmp)])
                self.ghosts_house[i]=[self.ghosts_pos[i][0],self.ghosts_pos[i][1]]

                tmp+=1
        else:               # lose.
            for i in range(self.n_ghosts):
                if self.ghosts_pos[i][0]==x and self.ghosts_pos[i][1]==y: 
                    
                    ret=5
                    self.end=True                    
                    

        # if the player hasnt lose, he moves 
        if self.end!=True:            
            self.agent_pos=[x,y]
            self.agent_coins+=(1 if ret==2 else 0)
            
        return ret
   
    """
    Moving the ghosts.
    Also change the game state
    """
    def move_ghosts(self):
        
        # -------------------------------------------------------------------------------------------------------------------
        # --- MOVE ----------------------------------------------------------------------------------------------------------

        for i in range(self.n_ghosts):
            self.move_ghost(i) 
        
        

        # -------------------------------------------------------------------------------------------------------------------
        # --- STATE ---------------------------------------------------------------------------------------------------------

        

        # a ghost leaves the house if is his time
        if len(self.ghost_inHouse)!=0 and self.exec_tick==self.ghost_inHouse[0][1]:
            idx=self.ghost_inHouse.pop(0)[0]
            
            self.ghosts_house[idx]=False

            self.ghosts_pos[idx][0]=self.salida_fants[0]
            self.ghosts_pos[idx][1]=self.salida_fants[1]
        
        # increases the execution ticks
        self.exec_tick+=1
        self.count_state+=1

        
        
        if self.count_state==self.state_ticks[self.state]:                     
            if self.state==0: 
                self.state=1
                print("New state: SCATTER",end="")
            else: 
                self.state=0
                print("New state: CHASE",end="")

            for i in range(self.n_ghosts):
                if not self.ghosts_house[i]:
                    self.ghosts_dir[i]+=2
                    self.ghosts_dir[i]%=4
            
            print("\nTurn 180º all ghosts")

            # reset
            self.count_state=0
    
    """
    Moving a ghost. (if the ghost is in the house, he doesnt moves)
    eat the player or is eaten by the player.

    If the current state is FRIGHTENED, moves one cell in two ticks

    Args:     
        ghost (int): index of the ghost.
    
    """
    def move_ghost(self, ghost):
        
        x=0
        y=0
        dir=0

        color=self.RED
        if ghost==1: color=self.PINK
        elif ghost==2: color=self.BLUE
        elif ghost==3: color=self.ORANGE

        aux_x=0
        aux_y=0
        
        
        if not self.ghosts_house[ghost] and (not(self.state==2 and self.count_state%2==0)):
            
            dir=self.ghosts_dir[ghost]            
            x=self.ghosts_pos[ghost][0]
            y=self.ghosts_pos[ghost][1]

            if dir==0 or dir==2: aux_y=1
            else: aux_x=1
            
            

            

            # intersection, moves in the same direction.            
            if self.maze[x+aux_x][y+aux_y]==1 and self.maze[x-aux_x][y-aux_y]==1:
                if dir==0: x-=1
                elif dir==1: y+=1
                elif dir==2: x+=1
                else: y-=1

            else: 
                if self.state==2: # FRIGHTENED. moves randomly in the intersection
                    
                    # the oposite direction is not taken into account

                    opcs=[]
                    for k in range(4):
                        tmp_x=x+self.mX[k]
                        tmp_y=y+self.mY[k]
                        
                        # wall or oposite actual dir
                        if k==((dir+2)%4) or self.maze[tmp_x][tmp_y]==1: continue 
                        else: opcs.append(k)

                    # random.
                    opc=random.randint(0,len(opcs)-1)

                    x+=self.mX[opcs[opc]]
                    y+=self.mY[opcs[opc]]
                    self.ghosts_dir[ghost]=opcs[opc]


                else:

                    target=[0,0]

                    # CHASE (in the constructor is the information of how each ghost choose a target)
                    if self.state==0: 

                        if ghost==0:                                # RED. 
                            target[0]=self.agent_pos[0]
                            target[1]=self.agent_pos[1]
                        elif ghost==1:                              # PINK
                            target[0]=self.agent_pos[0]
                            target[1]=self.agent_pos[1]

                            if self.agent_dir==0: 
                                target[0]-=4
                                target[1]-=4
                            elif self.agent_dir==1: target[1]+=4
                            elif self.agent_dir==2: target[0]+=4
                            else: target[1]-=4
                            
                        elif ghost==2:                              # BLUE
                            tmp=[0,0]
                            tmp[0]=self.agent_pos[0]
                            tmp[1]=self.agent_pos[1]

                            if self.agent_dir==0: 
                                tmp[0]-=2
                                tmp[1]-=2
                            elif self.agent_dir==1: tmp[1]+=2
                            elif self.agent_dir==2: tmp[0]+=2
                            else: tmp[1]-=2

                            dif_x=tmp[0]-self.ghosts_pos[0][0]
                            dif_y=tmp[1]-self.ghosts_pos[0][1]
                            target[0]=tmp[0]+dif_x
                            target[1]=tmp[1]+dif_y
                        
                        elif ghost==3:                              # ORANGE
                            dist_manhattan=abs(self.agent_pos[0]-self.ghosts_pos[3][0])
                            dist_manhattan+=abs(self.agent_pos[1]-self.ghosts_pos[3][1])

                            if dist_manhattan<8: target=self.scatter_targets[3]
                            else: target=self.agent_pos

                    # SCATTER
                    else: target=self.scatter_targets[ghost]
                        
                    dist=float("inf")
                    tmp_x=0
                    tmp_y=0
                    dir_idx=0
                    tmp=0
                    for k in range(4):
                        tmp_x=x+self.mX[k]
                        tmp_y=y+self.mY[k]

                        # wall or oposite actual dir
                        if k==((dir+2)%4) or self.maze[tmp_x][tmp_y]==1: continue 
                        else: tmp=self.distance_cells(target,[tmp_x, tmp_y])
                        
                        if dist>tmp:
                            dist=tmp
                            dir_idx=k
                    
                    self.ghosts_dir[ghost]=dir_idx
                    
                    x+=self.mX[dir_idx]
                    y+=self.mY[dir_idx]
            
            # "portals"
            if y==self.m and self.portal_gates(x): y=0
            if y==-1 and self.portal_gates(x): y=self.m-1
            
            
            
            # collides with the agent
            if self.agent_pos[0]==x and self.agent_pos[1]==y:
                
                # FRIGHTENED. is eaten
                if self.state==2:       

                                    

                    self.ghosts_pos[ghost][0]=self.salida_fants[0]+2
                    self.ghosts_pos[ghost][1]=self.salida_fants[1]

                    self.ghost_inHouse.append([ghost,self.exec_tick+3])
                    self.ghosts_house[ghost]=[self.ghosts_pos[ghost][0],self.ghosts_pos[ghost][1]]
                # OTHERWIESE. eats the player
                else: 
                    self.ghosts_pos[ghost][0]=x
                    self.ghosts_pos[ghost][1]=y
                    
                    self.end=True     
            else:
                self.ghosts_pos[ghost][0]=x
                self.ghosts_pos[ghost][1]=y
    
    """
    Calculates the distance of two points given by parameters
           
    Args:
        a (float[][]): first point.
        b (float[][]): second point.
    
    Return: 
        distance (float): distance between the two points.
    """
    def distance_cells(self, a, b):               
        return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
   
    """
    Printing a matrix giving by parameter.
    
    Args:
        matrix (int[][]): 2-Dimensional matrix array.
    """
    def print_matrix(self, matrix):
        
        n=len(matrix)
        m=len(matrix[0])
        for x in range(n):
            for y in range(m):
                print(matrix[x][y], end=" ")
            print()

    def portal_gates(self, x):
        if self.version==0: return (x==5 or x==9)
        else: return x==4

  
    
    
