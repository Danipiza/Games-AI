import pygame
import sys
import os
import random
import math
import time


# GHOSTS AI
# https://www.youtube.com/watch?v=ataGotQ7ir8&ab_channel=RetroGameMechanicsExplained

# MANUAL
# https://archive.org/details/Pac-Man_1981_Atari/page/n5/mode/2up

class PacmanGUI:
    
    """
    MOVEMENTS KEYS:
    "up"    or "w": UP
    "right" or "d": RIGHT
    "down"  or "s": DOWN
    "left"  or "a": LEFT
    """


    def __init__(self,file_name):
        self.file_name=file_name
        
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
        self.LEFT   ='left'
        self.DOWN   ='down'        
        self.RIGHT  ='right'

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
        self.ghosts_pos=[[0,0] for _ in range(4)]
        self.ghosts_dir=[1,2,0,0]
        self.ghosts_house=[False,True,True,True]
        # queue, for the leaving order. 0th: ghost id. 1th: home leaving tick
        self.ghost_inHouse=[[1,3],[2,6],[3,9]]
        
        
        
        # maze.
        self.maze           =[] # used for the walls, agent and ghosts positions in the GUI
        self.coins_matrix   =[] # used for the coins in the GUI         
        self.n=0                # number of rows
        self.m=0                # number of coloumns

        # finalization variable
        self.end=False

        self.reset()
    
        # screen config
        self.cell_size=30
        self.height=self.n*self.cell_size
        self.width=self.m*self.cell_size

        # init pygame
        pygame.init()       
        self.screen=pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Pac-Man') 

        # images.
        self.empty_img  =[]
        self.coin_img   =[]
        self.power_img  =[]
        self.walls_imgs =[]
        self.agent_imgs =[]
        self.ghosts_imgs=[]
        self.load_images(self.cell_size)

        self.execute()

    """
    Reseting the class variables.

    :type self: class    
    :rtype: None
    """
    def reset(self):

        self.exec_tick=0

        self.state=1
        self.count_state=0
        
        self.agent_pos=None
        self.agent_dir=1
        self.agent_coins=0   

        self.ghosts_pos=[[0,0] for _ in range(4)]
        self.ghosts_dir=[1,2,0,0]
        self.ghosts_house=[False,True,True,True]
        self.ghost_inHouse=[[1,3],[2,6],[3,9]]
        
        self.maze=[]
        self.coins_matrix=[]
        self.read_maze()
        self.n=len(self.maze)
        self.m=len(self.maze[0])

        self.scatter_targets=[[0,self.m],[0,0],[self.n,self.m],[self.n,0]]

        self.end=False

      

    """
    Reading the maze from a .txt file. 
    Also search for the positions of the agent and ghosts
    
    :type self: class        
    :rtype: None
    """
    def read_maze(self):  
        tmp=0

        # -------------------------------------------------------------------------------------------------------------------
        # --- READING -------------------------------------------------------------------------------------------------------

        with open(self.file_name, 'r') as file:        
            for line in file:
                row=list(map(int, line.split()))
                self.maze.append([0 for _ in range(len(row))])
                self.coins_matrix.append([0 for _ in range(len(row))])
                
                # remove the coins from the maze. 
                # "self.coins_matrix" is in charge of the coins
                for i in range(len(row)):   
                    if row[i]==2 or row[i]==3: self.maze[tmp][i]=0
                    else: self.maze[tmp][i]=row[i]
                                
                for i in range(len(row)):                    
                    self.coins_matrix[tmp][i]=row[i]                
                
                tmp+=1
                               
        
        #self.print_maze(self.maze)


        # -------------------------------------------------------------------------------------------------------------------
        # --- POSITIONS -----------------------------------------------------------------------------------------------------

        tmp=0
        # once all are located, break the loop
        for x in range(len(self.maze)):
            for y in range(len(self.maze[0])):                    
                if self.maze[x][y]==self.AGENT: 
                    self.agent_pos=[x,y]
                    tmp+=1
                    if tmp==5: break  

                elif self.maze[x][y]==self.RED: 
                    self.ghosts_pos[0]=[x,y]
                    self.salida_fants=[x,y]
                    tmp+=1
                    if tmp==5: break
                elif self.maze[x][y]==self.PINK: 
                    self.ghosts_pos[1]=[x,y]
                    self.ghosts_house[1]=[x,y]
                    tmp+=1
                    if tmp==5: break
                elif self.maze[x][y]==self.BLUE: 
                    self.ghosts_pos[2]=[x,y]
                    self.ghosts_house[2]=[x,y]
                    tmp+=1
                    if tmp==5: break
                elif self.maze[x][y]==self.ORANGE: 
                    self.ghosts_pos[3]=[x,y]
                    self.ghosts_house[3]=[x,y]
                    tmp+=1
                    if tmp==5: break
        
        
        
                    

    """
    Printing a matrix giving by parameter.
    
    :type self: class 
    :type matrix: int[][]       
    :rtype: None
    """
    def print_matrix(self, matrix):
        
        n=len(matrix)
        m=len(matrix[0])
        for x in range(n):
            for y in range(m):
                if matrix[x][y]<0: 
                    print(-1, end=" ")
                else: print(matrix[x][y], end=" ")
            print()

    

    """
    Moving the agent.
    eat ghosts or is eaten by a ghost
    
    return: 1, if the agent eat a coin. 
            0, otherwise
    
    :type self: class 
    :type mov: string
    :rtype: int
    """
    def move_agent(self, mov):
        x=self.agent_pos[0]
        y=self.agent_pos[1] 
        
        coin=0
        
        # increases the execution ticks
        self.exec_tick+=1

        # -------------------------------------------------------------------------------------------------------------------
        # --- MOVE ----------------------------------------------------------------------------------------------------------

        # POWER is only reachable by up and down actions

        if mov==self.UP:
            if x>0 and self.maze[x-1][y]>=0:                # dest position != wall
                if self.coins_matrix[x-1][y]==self.COIN:    # COIN. adds a coin
                    self.coins_matrix[x-1][y]=0
                    coin=1
                elif self.coins_matrix[x-1][y]==self.POWER: # POWER. change the game state to FRIGHTENED
                    self.coins_matrix[x-1][y]=0
                    self.state=2
                    self.count_state=0
                    
                    # change the directions of the ghosts
                    for i in range(4):                        
                        if not self.ghosts_house[i]:
                            self.ghosts_dir[i]+=2
                            self.ghosts_dir[i]%=4                        
                
                self.maze[x][y]=self.EMPTY
                x-=1

        elif mov==self.DOWN:
            if x<len(self.maze)-1 and self.maze[x+1][y]>=0: # dest position != wall
                if self.coins_matrix[x+1][y]==self.COIN:    # COIN. adds a coin
                    self.coins_matrix[x+1][y]=0
                    coin=1
                elif self.coins_matrix[x+1][y]==self.POWER: # POWER. change the game state to FRIGHTENED
                    self.coins_matrix[x+1][y]=0
                    self.state=2
                    self.count_state=0
                    for i in range(4):
                        if not self.ghosts_house[i]:
                            self.ghosts_dir[i]+=2
                            self.ghosts_dir[i]%=4
                
                self.maze[x][y]=self.EMPTY
                x+=1

        elif mov==self.LEFT:
            if y>0 and self.maze[x][y-1]>=0:                # dest position != wall
                if self.coins_matrix[x][y-1]==self.COIN:    # COIN. adds a coin
                    self.coins_matrix[x][y-1]=0
                    coin=1
                
                self.maze[x][y]=self.EMPTY
                y-=1
            elif y==0 and (x==5 or x==9):           # "portal"                                 
                self.maze[x][y]=self.EMPTY
                y=self.m-1

        elif mov==self.RIGHT:
            if y<self.m-1 and self.maze[x][y+1]>=0:         # dest position != wall
                if self.coins_matrix[x][y+1]==self.COIN:    # COIN. adds a coin
                    self.coins_matrix[x][y+1]=0
                    coin=1
                
                self.maze[x][y]=self.EMPTY
                y+=1
            elif y==self.m-1 and (x==5 or x==9):    # "portal"                             
                self.maze[x][y]=self.EMPTY
                y=0
        
        # -------------------------------------------------------------------------------------------------------------------
        # --- EAT/LOSE ------------------------------------------------------------------------------------------------------

        if self.state==2:   # eat.
            eaten=[]
            for i in range(4):
                if self.ghosts_pos[i][0]==x and self.ghosts_pos[i][1]==y:
                    eaten.append(i)
            
            tmp=1
            for i in eaten:
                # move the eaten ghost to the house cell.
                self.ghosts_pos[i][0]=self.salida_fants[0]+2
                self.ghosts_pos[i][1]=self.salida_fants[1]

                # add to the queue of ghosts in house
                # they leave in 3 ticks intervals
                self.ghost_inHouse.append([i,self.exec_tick+(3*tmp)])
                self.ghosts_house[i]=[self.ghosts_pos[i][0],self.ghosts_pos[i][1]]

                tmp+=1
        else:               # lose.
            for i in range(4):
                if self.ghosts_pos[i][0]==x and self.ghosts_pos[i][1]==y: 
                    self.end=True
                    self.maze[self.agent_pos[0]][self.agent_pos[1]]=self.EMPTY

        # if the player hasnt lose, he moves 
        if self.end!=True:
            self.maze[x][y]=self.AGENT
            self.agent_pos=[x,y]
            self.agent_coins+=coin
            
        return coin


    """
    Moving the ghosts.
    Also change the game state
           
    :type self: class     
    :rtype: None
    """
    def move_ghosts(self):
        
        # -------------------------------------------------------------------------------------------------------------------
        # --- MOVE ----------------------------------------------------------------------------------------------------------
                    
        self.move_ghost(0) 
        self.move_ghost(1)
        self.move_ghost(2)
        self.move_ghost(3)
        
        

        # -------------------------------------------------------------------------------------------------------------------
        # --- STATE ---------------------------------------------------------------------------------------------------------

        self.count_state+=1
        
        print("State=", self.count_state, "\tCoins=",self.agent_coins)

        for i in range(4):
            print(self.ghosts_pos[i][0],self.ghosts_pos[i][1])
        
        if self.count_state==self.state_ticks[self.state]:                     
            if self.state==0: 
                self.state=1
                print("New state: SCATTER",end="")
            else: 
                self.state=0
                print("New state: CHASE",end="")

            for i in range(4):
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
           
    :type self: class     
    :type ghost: int
    :rtype: None
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
            
            self.maze[x][y]=self.EMPTY

            

            # intersection, moves in the same direction.
            if self.maze[x+aux_x][y+aux_y]<0 and self.maze[x-aux_x][y-aux_y]<0:
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
                        if k==((dir+2)%4) or self.maze[tmp_x][tmp_y]<0: continue 
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
                        if k==((dir+2)%4) or self.maze[tmp_x][tmp_y]<0: continue 
                        else: tmp=self.distance_cells(target,[tmp_x, tmp_y])
                        
                        if dist>tmp:
                            dist=tmp
                            dir_idx=k
                    
                    self.ghosts_dir[ghost]=dir_idx
                    
                    x+=self.mX[dir_idx]
                    y+=self.mY[dir_idx]
            
            # "portals"
            if y==self.m and (x==5 or x==9): y=0
            if y==-1 and (x==5 or x==9): y=self.m-1
            
            
            
            # collides with the agent
            if self.agent_pos[0]==x and self.agent_pos[1]==y:
                
                # FRIGHTENED. is eaten
                if self.state==2:       

                    self.maze[x][y]=self.EMPTY                   

                    self.ghosts_pos[ghost][0]=self.salida_fants[0]+2
                    self.ghosts_pos[ghost][1]=self.salida_fants[1]

                    self.ghost_inHouse.append([ghost,self.exec_tick+3])
                    self.ghosts_house[ghost]=[self.ghosts_pos[ghost][0],self.ghosts_pos[ghost][1]]
                # OTHERWIESE. eats the player
                else: 
                    self.ghosts_pos[ghost][0]=x
                    self.ghosts_pos[ghost][1]=y
                    self.maze[x][y]=color
                    self.end=True     
            else:
                self.ghosts_pos[ghost][0]=x
                self.ghosts_pos[ghost][1]=y
                self.maze[x][y]=color
        
    """
    Calculates the distance of two points given by parameters
           
    :type self: class     
    :type a: int[]
    :type b: int[]
    :rtype: int
    """
    def distance_cells(self, a, b):               
        return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

    # --------------------------------------------------------------------------------
    # --- GUI ------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    """
    Main loop.

    Waits for a key event, and execute an iteration. (moves the agent and ghosts).
    Prints the maze in the GUI.

    MOVEMENTS KEYS:
    "up"    or "w": UP
    "right" or "d": RIGHT
    "down"  or "s": DOWN
    "left"  or "a": LEFT

    Close the window for exiting the game.
           
    :type self: class  
    :rtype: int
    """
    def execute(self):       

        #self.print_maze()
         
        mov=None

        while True:
            
            while not self.end:            
                
                # -------------------------------------------------------------------------------------------------------------------
                # --- MOVE ----------------------------------------------------------------------------------------------------------
                
                # event = key pressed
                
                for event in pygame.event.get():
                    if event.type==pygame.QUIT: # ends the execution.
                        pygame.quit()
                        sys.exit()
                    elif event.type==pygame.KEYDOWN: # key pressed              
                        if event.key==pygame.K_UP or event.key==pygame.K_w: 
                            mov=self.UP                    
                            self.agent_dir=0
                        elif event.key==pygame.K_RIGHT or event.key==pygame.K_d:
                            mov=self.RIGHT 
                            self.agent_dir=1
                        elif event.key==pygame.K_DOWN or event.key==pygame.K_s:  
                            mov=self.DOWN    
                            self.agent_dir=2
                        elif event.key==pygame.K_LEFT or event.key==pygame.K_a:  
                            mov=self.LEFT    
                            self.agent_dir=3
                        else: mov=None
                        
                        # run an iteration if the key pressed is binded 
                        if mov!=None:             
                                            
                            self.move_agent(mov)
                            # check for end condition
                            if self.agent_coins==132: self.end=True
                            
                            if self.end!=True:  # no end condition, continues
                                
                                self.move_ghosts()
                                
                                # a ghost leaves the house if is his time
                                if len(self.ghost_inHouse)!=0 and self.exec_tick==self.ghost_inHouse[0][1]:
                                    idx=self.ghost_inHouse.pop(0)[0]

                                    self.maze[self.ghosts_house[idx][0]][self.ghosts_house[idx][1]]=self.EMPTY
                                    self.maze[self.salida_fants[0]][self.salida_fants[1]]=self.ghosts_colors[idx]
                                    self.ghosts_house[idx]=False

                                    self.ghosts_dir[idx]=0

                                    self.ghosts_pos[idx][0]=self.salida_fants[0]
                                    self.ghosts_pos[idx][1]=self.salida_fants[1]
                                
                                # update ghost positions (from lower to higher priority)
                                i=3
                                while i>=0:
                                    self.maze[self.ghosts_pos[i][0]][self.ghosts_pos[i][1]]=self.ghosts_colors[i]
                                    i-=1
                                # update agent position
                                if self.end!=True:
                                    self.maze[self.agent_pos[0]][self.agent_pos[1]]=self.AGENT
                            
                # -------------------------------------------------------------------------------------------------------------------
                # --- MAZE ----------------------------------------------------------------------------------------------------------           


                # paint the maze
                self.GUI_maze()

            # -------------------------------------------------------------------------------------------------------------------
            # --- END MESSAGE ---------------------------------------------------------------------------------------------------  

        
            if self.agent_coins==132:   # win condition
                print("\nYOU WIN!!!\n")
                for _ in range(3):
                    self.GUI_message(1)
                    pygame.display.flip()

                    time.sleep(1)

                    self.GUI_maze()
                    pygame.display.flip()

                    time.sleep(0.33)
                    
            else:                       # lose condition
                print("\nGAME OVER\n")

                for _ in range(3):
                    self.GUI_message(0)
                    pygame.display.flip()

                    time.sleep(1)

                    self.GUI_maze()
                    pygame.display.flip()

                    time.sleep(0.33)
                    


            pygame.display.flip()
                    

            self.reset()
    

    """
    Printing in the GUI, the actual state of the maze.
           
    :type self: class  
    :rtype: int
    """
    def GUI_maze(self):
        self.screen.fill((0, 0, 0))  # cleans the screen
                
        for x, row in enumerate(self.maze):
            self.GUI_line(x, row)
            
        # update
        pygame.display.flip()
    

    """
    Printing in the GUI, a message
           
    :type self: class     
    :type t: int
    :rtype: int
    """
    def GUI_message(self, t):

        aux=0 if t==0 else 1

        for x, row in enumerate(self.maze):
            
            if x!=9: self.GUI_line(x, row)
            
            # message row
            else: 

                tmp=0

                for y, cell in enumerate(row):
                    if (y>=6+aux and y<=9) or (y>=11 and y<=14-aux):
                        image=self.message_imgs[t][tmp]
                        tmp+=1
                    else:
                        image=self.GUI_cell(x,y,cell)
                    
                    # draw in the GUI, the actual position of the maze
                    self.screen.blit(image, (y*self.cell_size, x*self.cell_size))
    

    """
    Printing in the GUI, a line
           
    :type self: class     
    :type x: int
    :type row: int[]
    :rtype: None
    """
    def GUI_line(self, x, row):
        for y, cell in enumerate(row):
            image=self.GUI_cell(x,y,cell)

            # draw in the GUI, the actual position of the maze
            self.screen.blit(image, (y*self.cell_size, x*self.cell_size))
            
    
    """
    Printing in the GUI, a cell
           
    :type self: class     
    :type x: int
    :type y: int
    :type cell: int
    :rtype: image
    """
    def GUI_cell(self, x, y, cell):
        
        # walls
        if cell<0: image=self.walls_imgs[abs(cell)-1]  
            
        # ghosts exit door.
        elif cell==1: image=self.walls_imgs[-1]
        
        # in "self.maze" there are no coins. "self.coins_matrix" store the coins and power.
        elif cell==self.EMPTY: 
            if self.coins_matrix[x][y]==self.COIN:      image=self.coin_img
            elif self.coins_matrix[x][y]==self.POWER:   image=self.power_img
            else:                                       image=self.empty_img                                        
        
        # agent.         
        elif cell==self.AGENT: image=self.agent_imgs[self.agent_dir]
        
        # ghost.
        else:
            # FRIGHTENED mode.
            if self.state==2:
                if self.count_state<20: image=self.ghosts_imgs[-1][0]
                
                # blinks to advice the player is finishing the state
                else: 
                    if self.count_state%2==0:   image=self.ghosts_imgs[-1][1]
                    else:                       image=self.ghosts_imgs[-1][0]
            
            # normal mode.
            else:
                if cell==self.RED:      image=self.ghosts_imgs[0][self.ghosts_dir[0]]
                elif cell==self.PINK:   image=self.ghosts_imgs[1][self.ghosts_dir[1]]
                elif cell==self.BLUE:   image=self.ghosts_imgs[2][self.ghosts_dir[2]]
                else:                   image=self.ghosts_imgs[3][self.ghosts_dir[3]]
    
        return image
        
    """
    Loading all the game images. And scale all of them to the same size   
           
    :type self: class     
    :type size: int
    :rtype: int
    """
    def load_images(self, size):

        # -------------------------------------------------------------------------------------------------------------------
        # --- LOAD ----------------------------------------------------------------------------------------------------------      
        
        empty=pygame.image.load('images/empty.png').convert_alpha()    
        coin=pygame.image.load('images/coin.png').convert_alpha()
        power=pygame.image.load('images/power.png').convert_alpha()        
        
        walls=[]        
        names=["images/walls/wall_0.png","images/walls/wall_01.png","images/walls/wall_1.png","images/walls/wall_02.png",
               "images/walls/wall_2.png","images/walls/wall_03.png","images/walls/wall_3.png","images/walls/wall_012.png",
               "images/walls/wall_12.png","images/walls/wall_013.png","images/walls/wall_13.png","images/walls/wall_023.png",
               "images/walls/wall_23.png","images/walls/wall_123.png","images/walls/wall.png","images/walls/z0.png",
               "images/walls/z1.png","images/walls/z2.png","images/walls/z3.png","images/walls/z4.png","images/walls/ghost_exit.png"]
        for wall in names:
            walls.append(pygame.image.load(wall).convert_alpha())
               

        agent=[]
        names=['images/pacman_up.png','images/pacman_right.png','images/pacman_down.png','images/pacman_left.png']
        for dir in names:
            agent.append(pygame.image.load(dir).convert_alpha())
        

        names=[["images/red_up.png","images/red_right.png","images/red_down.png","images/red_left.png"],
               ["images/pink_up.png","images/pink_right.png","images/pink_down.png","images/pink_left.png"],
               ["images/blue_up.png","images/blue_right.png","images/blue_down.png","images/blue_left.png"],
               ["images/orange_up.png","images/orange_right.png","images/orange_down.png","images/orange_left.png"],
               ["images/fright_ghost1.png","images/fright_ghost2.png"]]
        ghosts=[]
        for ghost in names:
            tmp=[]
            for dir in ghost:
                tmp.append(pygame.image.load(dir).convert_alpha())
            ghosts.append(tmp)

        names=[["images/messages/G.png","images/messages/A.png","images/messages/M.png","images/messages/E.png",
                "images/messages/O.png","images/messages/V.png","images/messages/E.png","images/messages/R.png"],                
                ["images/messages/Y.png","images/messages/O_2.png","images/messages/U.png",
                "images/messages/W.png","images/messages/I.png","images/messages/N.png"]]
        messages=[]
        for message in names:
            tmp=[]
            for char in message:
                tmp.append(pygame.image.load(char).convert_alpha())
            messages.append(tmp)

        # -------------------------------------------------------------------------------------------------------------------
        # --- SCALE ---------------------------------------------------------------------------------------------------------      

        self.empty_img=pygame.transform.scale(empty, (size, size))    
        self.coin_img=pygame.transform.scale(coin, (size, size))
        self.power_img=pygame.transform.scale(power, (size, size))

        self.walls_imgs=[]
        self.agent_imgs=[]
        self.ghosts_imgs=[]
        self.message_imgs=[]

        for w in walls:
            self.walls_imgs.append(pygame.transform.scale(w, (size, size)))
        
        for dir in range(4):
            self.agent_imgs.append(pygame.transform.scale(agent[dir], (size, size)))

        
        for g in ghosts:
            tmp=[]
            for dir in g:
                tmp.append(pygame.transform.scale(dir, (size, size)))
                
            self.ghosts_imgs.append(tmp)

        
        for message in messages:
            tmp=[]
            for char in message:
                tmp.append(pygame.transform.scale(char, (size, size)))
                
            self.message_imgs.append(tmp)


if __name__ == "__main__":    
    env=PacmanGUI(os.path.join("data", "env.txt"))
    
    
