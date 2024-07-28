import pygame
import sys
import os
import random
import math
import time

# MANUAL
# https://archive.org/details/Space_Invaders_1978_Atari/page/n3/mode/2up


class SpaceInvadersGUI:
    
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
        

        self.EMPTY_0=0
        self.EMPTY_1=1
        self.EMPTY_N1=-1
        self.EMPTY_N2=-2
        self.EMPTY_N3=-3
        self.EMPTY_N4=-4
        self.EMPTY_N5=-5
        self.EMPTY_N6=-6

        self.WALL   =2        
        self.AGENT  =3        
        
        self.ALIEN_1=4
        self.ALIEN_2=5
        self.ALIEN_3=6
        self.ALIEN_4=7
        self.ALIEN_5=8
        self.ALIEN_6=9

        

        # actions.
        self.NOOP   ='noop'
        self.LEFT   ='left'               
        self.RIGHT  ='right'
        self.SHOOT  ='shoot'

        # directions. 
        # 0: NOOP
        # 1: LEFT
        # 2: RIGHT 
        # 3: SHOOT
        self.mY=[0,-1,1,0]
                
        # -------------------------------------------------------------------------------------------------------------------       
        # --- GUI -----------------------------------------------------------------------------------------------------
    
        # screen config
        # prev_gui
        """
        self.height=682
        self.width=812
        """
        self.height=840
        self.width=1288

        # init pygame
        pygame.init()       
        self.screen=pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Space Invaders') 

        # images.        
        self.wall_img   =None
        self.agent_img  =None
        self.bg_imgs    =[]
        self.aliens_imgs=[]
        """self.load_images(self.cell_size)"""
        self.load_images(56,40)
        
        """self.state_ticks    =[60,30,30]"""

        # -------------------------------------------------------------------------------------------------------------------       
        # --- VARIABLES -----------------------------------------------------------------------------------------------------

        self.life=0

        # 0: RIGHT, 1: LEFT        
        self.state=0
        
        # agent.
        self.agent_pos=None        
        self.agent_points=0 
        self.agent_l_limit=0  
        self.agent_r_limit=0

        # ghosts.        
        self.aliens_row=[]
        self.aliens_l_limit=0  
        self.aliens_r_limit=0
        """self.aliens_dir=1"""
                
        # maze.
        self.space=[]         
        self.n=0                # number of rows
        self.m=0                # number of columns

        # finalization variable
        self.end=False

        self.reset()

        

        self.execute()

    """
    Reseting the class variables.

    :type self: class    
    :rtype: None
    """
    def reset(self):
        
        self.life=3
        self.end=False

        self.reset_life()

        

        

    def reset_life(self):
        
        self.state=1        
        
        self.agent_pos=None        
        self.agent_points=0 
        self.agent_l_limit=0  
        self.agent_r_limit=0  

        # prev_gui
        """
        self.aliens_row=[1,3,5,7,9,11]
        """
        self.aliens_row=[3,5,7,9,11,13]
        self.aliens_l_limit=0  
        self.aliens_r_limit=0
        
        self.space=[]        
        self.read_maze()
        self.n=len(self.space)
        self.m=len(self.space[0])
        print(self.m)

        


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
                self.space.append([0 for _ in range(len(row))])
                
                
                # remove the coins from the maze. 
                
                for i in range(len(row)):   
                    self.space[tmp][i]=row[i]
                    
                                
                       
                
                tmp+=1
                               
        
        self.print_matrix(self.space)


        
        # -------------------------------------------------------------------------------------------------------------------
        # --- POSITIONS -----------------------------------------------------------------------------------------------------
        # prev_gui
        """
        self.agent_pos=[17,4]
        self.agent_l_limit=4  
        self.agent_r_limit=36
        self.aliens_l_limit=0  
        self.aliens_r_limit=40
        """
        self.agent_pos=[19,2]
        self.agent_l_limit=2  
        self.agent_r_limit=20
        self.aliens_l_limit=0  
        self.aliens_r_limit=22
                
        
        
        
                    

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
                print(matrix[x][y], end=" ")
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
        
        
        

        # -------------------------------------------------------------------------------------------------------------------
        # --- MOVE ----------------------------------------------------------------------------------------------------------

        if mov==self.LEFT:
            if self.agent_l_limit<y:                
                tmp=self.space[x][y-1]
                # prev_gui
                """self.space[x][y-1]=3"""
                self.space[x][y-1]=1
                self.space[x][y]=tmp
                """tmp=self.space[x][0]
                for j in range(0,self.m-1):
                    self.space[x][j]=self.space[x][j+1]
                self.space[x][self.m-1]=tmp"""
                
                self.agent_pos[1]-=1 
                self.print_matrix(self.space)           

        elif mov==self.RIGHT:
            if self.agent_r_limit>y:         
                print(x,y)
                tmp=self.space[x][y+1]
                # prev_gui
                """self.space[x][y+1]=3"""
                self.space[x][y+1]=1
                self.space[x][y]=tmp

                """tmp=self.space[x][self.m-1]
                for j in range(self.m-1,0,-1):                        
                    self.space[x][j]=self.space[x][j-1]
                self.space[x][0]=tmp"""
                
                self.agent_pos[1]+=1
            
        
        # -------------------------------------------------------------------------------------------------------------------
        # --- SHOOT ---------------------------------------------------------------------------------------------------------

        elif mov==self.SHOOT:
            print("SHOOT")
        
        # -------------------------------------------------------------------------------------------------------------------
        # --- RECEIVE A SHOOT -----------------------------------------------------------------------------------------------

        


    """
    Moving the ghosts.
    Also change the game state
           
    :type self: class     
    :rtype: None
    """
    def move_aliens(self):

        # an alien collide with the limit. moves two rows        
        moves=1
        cell=0
        if self.state==1: cell=self.m-1        

        for row in self.aliens_row:
            if self.space[row][cell]>1: 
                moves=0
                break

        # not collide. moves one column to its direction
        if moves==1:
            if self.state==0: # left
                for row in self.aliens_row:
                    tmp=self.space[row][0]
                    for y in range(0,self.m-1):
                        self.space[row][y]=self.space[row][y+1]
                    self.space[row][self.m-1]=tmp

            else: # right                
                for row in self.aliens_row:
                    tmp=self.space[row][self.m-1]
                    for y in range(self.m-1,0,-1):                        
                        self.space[row][y]=self.space[row][y-1]
                    self.space[row][0]=tmp
        else:
            print("Moves 2 rows below. and changes the direction")
            self.state=(self.state+1)%2
            walls=0
            ground=0
            """print("ANTES", self.aliens_row)"""
            for row in range(5,-1,-1):
                if self.aliens_row[row]!=-1:                    
                    for y in range(self.m):
                        tmp=self.space[self.aliens_row[row]+2][y]
                        self.space[self.aliens_row[row]+2][y]=self.space[self.aliens_row[row]][y]
                        self.space[self.aliens_row[row]][y]=tmp
                    self.aliens_row[row]+=2
                    # prev_gui
                    """
                    if(self.aliens_row[row]==15): walls=1
                    elif(self.aliens_row[row]==17): ground=1
                    """
                    if(self.aliens_row[row]==17): walls=1
                    elif(self.aliens_row[row]==19): ground=1
            
            """print("DESPUES", self.aliens_row, "\t", walls, ground)"""

            if walls==1:
                for y in range(self.m):
                    # prev_gui
                    """
                    if self.space[3][y]==2: self.space[3][y]=-8
                    """
                    if self.space[5][y]==2: self.space[5][y]=0

            if ground==1:
                self.life-=1
                if self.life==0: 
                    print("GAME OVER")
                    self.end=True   
                    
                else:
                    print("LOSE ONE LIFE")
                    self.reset_life()
            
        """print("DESPUES", self.aliens_row)"""
        

  
   
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
                        if event.key==pygame.K_RIGHT or event.key==pygame.K_d:
                            mov=self.RIGHT                             
                        elif event.key==pygame.K_LEFT or event.key==pygame.K_a:  
                            mov=self.LEFT    
                            
                        else: mov=None
                        
                        # run an iteration if the key pressed is binded 
                        if mov!=None:             
                            self.move_aliens()             
                            self.move_agent(mov)

                            print("\n")
                            self.print_matrix(self.space)
                            
                            # check for end condition
                            #if self.agent_coins==132: self.end=True
                            
                            
                                
                                
                                
                                
                            
                # -------------------------------------------------------------------------------------------------------------------
                # --- MAZE ----------------------------------------------------------------------------------------------------------           


                # paint the maze
                self.GUI_space()


            # -------------------------------------------------------------------------------------------------------------------
            # --- END MESSAGE ---------------------------------------------------------------------------------------------------  
            """
            if self.agent_coins==132:   # win condition
                print("\nYOU WIN!!!\n")
                for _ in range(3):
                    self.GUI_message(1)
                    pygame.display.flip()

                    time.sleep(1)

                    self.GUI_space()
                    pygame.display.flip()

                    time.sleep(0.33)
                    
            else:                       # lose condition
                print("\nGAME OVER\n")

                for _ in range(3):
                    self.GUI_message(0)
                    pygame.display.flip()

                    time.sleep(1)

                    self.GUI_space()
                    pygame.display.flip()

                    time.sleep(0.33)
                    
            pygame.display.flip()
                    
            """
                    


            
                    

            self.reset()
    

    """
    Printing in the GUI, the actual state of the maze.
           
    :type self: class  
    :rtype: int
    """
    def GUI_space(self):
        self.screen.fill((0, 0, 0))  # cleans the screen
        contador_y=0
        """self.GUI_line(self.space[1],contador_y)"""
        
        for _, row in enumerate(self.space):
            aux=self.GUI_line(row,contador_y)
            contador_y+=aux
        
        # update
        pygame.display.flip()
    

    """
    Printing in the GUI, a line
           
    :type self: class     
    :type x: int
    :type row: int[]
    :rtype: None
    """
    def GUI_line(self, row, height):
        """print("\n\n", row)"""
        contador_x=0
        
        for _, cell in enumerate(row):
            
            image=self.GUI_cell(cell)                                              
            
            # draw in the GUI, the actual position of the maze
            self.screen.blit(image, (contador_x, height))
            contador_x+=image.get_width()
        
        """print(contador_x, contador_y)"""

        return image.get_height()
            
    
    """
    Printing in the GUI, a cell
           
    :type self: class     
    :type x: int
    :type y: int
    :type cell: int
    :rtype: image
    """
    def GUI_cell(self, cell):
        
           
        # prev_gui 
        """
        # BACKGROUND    
        if cell<0: image=self.bg_imgs[1][abs(cell)-1]
        elif cell==0: image=self.bg_imgs[0][0]
        elif cell==1: image=self.bg_imgs[0][1]

        # WALL
        elif cell==2: image=self.wall_img
        # AGENT
        elif cell==3: image=self.agent_img

        # ALIENS
        else: image=self.aliens_imgs[cell-4][self.state]
        """

        if cell<=0: image=self.bg_imgs[abs(cell)]
        # AGENT
        elif cell==1: image=self.agent_img
        # WALL
        elif cell==2: image=self.wall_img               

        # ALIENS
        else: image=self.aliens_imgs[cell-4][self.state]
        
    
        return image

    
        
    """
    Loading all the game images. And scale all of them to the same size   
           
    :type self: class     
    :type size: int
    :rtype: int
    """
    def load_images(self, scale_x,scale_y):

        # -------------------------------------------------------------------------------------------------------------------
        # --- LOAD ----------------------------------------------------------------------------------------------------------      
        
        
        
        
        wall=pygame.image.load('images/2_wall.png').convert_alpha()
        agent=pygame.image.load('images/3_agent.png').convert_alpha()
        
        # prev_gui
        """
        names=[["images/0_bg.png","images/1_bg.png"],
               ["images/n1_bg.png","images/n2_bg.png","images/n3_bg.png",
                "images/n4_bg.png","images/n5_bg.png","images/n6_bg.png",
                "images/n7_bg.png","images/n8_bg.png"]]
        backgrounds=[]
        for bgs in names:
            tmp=[]
            for x in bgs:
                tmp.append(pygame.image.load(x).convert_alpha())
            backgrounds.append(tmp)                
        """
        
        names=["images/0_bg.png","images/1_bg.png","images/2_bg.png","images/3_bg.png"]
        backgrounds=[]
        for bg in names:
            backgrounds.append(pygame.image.load(bg).convert_alpha())

        """print("ANCHURA:",backgrounds[0][0].get_height())"""

        names=[["images/4_alien_l.png","images/4_alien_r.png"],
               ["images/5_alien_l.png","images/5_alien_r.png"],
               ["images/6_alien_l.png","images/6_alien_r.png"],
               ["images/7_alien.png","images/7_alien.png"],
               ["images/8_alien.png","images/8_alien.png"],
               ["images/9_alien_l.png","images/9_alien_r.png"],
               ["images/10_alien.png","images/10_alien.png"]]
        aliens=[]
        for alien in names:
            tmp=[]
            for dir in alien:
                tmp.append(pygame.image.load(dir).convert_alpha())
            aliens.append(tmp)
        
        """print(backgrounds[0][0].get_width())
        print(aliens[5][1].get_height())"""
        
        

        # MESSAGES
        """
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
            """

        """self.wall_img   =wall        
        self.agent_img  =agent
        self.bg_imgs    =backgrounds
        self.aliens_imgs=aliens"""

        # -------------------------------------------------------------------------------------------------------------------
        # --- SCALE ---------------------------------------------------------------------------------------------------------      
        self.wall_img   =None        
        self.agent_img  =None
        self.bg_imgs    =[]
        self.aliens_imgs=[]
        

        # SCALES
        
        self.wall_img=pygame.transform.scale(wall, (scale_x, scale_y))
        self.agent_img=pygame.transform.scale(agent, (scale_x, scale_y))
        
        for bg in backgrounds:                       
            self.bg_imgs.append(pygame.transform.scale(bg, (scale_x, scale_y)))
                
        
        for alien in aliens:
            tmp=[]
            for dir in alien:
                tmp.append(pygame.transform.scale(dir, (scale_x, scale_y)))
                
            self.aliens_imgs.append(tmp)
        
        
        
        # MESSAGES
        """
        for message in messages:
            tmp=[]
            for char in message:
                tmp.append(pygame.transform.scale(char, (size, size)))
                
            self.message_imgs.append(tmp)
            """


if __name__ == "__main__":    
    env=SpaceInvadersGUI(os.path.join("data", "env2.txt"))
    
    
