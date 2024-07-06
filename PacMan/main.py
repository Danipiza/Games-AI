import pygame
import sys
import os
import random
import math
import time


# GHOSTS AI
# https://www.youtube.com/watch?v=ataGotQ7ir8&ab_channel=RetroGameMechanicsExplained

class Pacman:

    def __init__(self,archivo, GUI):
        self.archivo=archivo

        self.mX=[-1,0,1,0]
        self.mY=[0,1,0,-1]

        # Elementos del tablero
        self.VACIO=0
        self.MURO=1
        self.MONEDA=2
        self.POWER=3
        self.AGENTE=4
        # fantasmas
        self.ROJO=5
        self.ROSA=6
        self.AZUL=7
        self.NARANJA=8
        self.COLORES_FANTS=[5,6,7,8]

        # Acciones
        self.ARRIBA='arriba'
        self.ABAJO='abajo'
        self.IZQUIERDA='izquierda'
        self.DERECHA='derecha'
        
        self.acciones = [self.ARRIBA, self.DERECHA, self.ABAJO, self.IZQUIERDA]

        self.state_ticks=[60,30,30]

        
        
        self.reset()

        self.scatter_targets=[[0,self.m],[0,0],[self.n,0],[self.n,self.m]]
        

        if GUI==True:
            # conf: ventana
            self.tam_celda=30
            self.alto=self.n*self.tam_celda
            self.ancho=self.m*self.tam_celda

            # init pygame
            pygame.init()       
            self.pantalla=pygame.display.set_mode((self.ancho, self.alto))
            pygame.display.set_caption('Pac-Man') 

            # Imagenes
            self.empty_img=[]
            self.coin_img=[]
            self.power_img=[]
            self.walls_imgs=[]
            self.agente_imgs=[]
            self.ghosts_imgs=[]
            self.cargar_imagenes(self.tam_celda)

            self.mostrar_laberinto()

    def reset(self):
        self.cont=0
        # 0: CHASE, 1: SCATTER, 2: FRIGHTENED
        self.state=1
        self.cont_state=0

        # Agente
        self.posicion_agente=None
        self.direccion=1
        self.monedas=0   

        # Fantasmas
        self.posicion_fants=[[0,0] for _ in range(4)]
        self.direccion_fants=[1,2,0,0]
        self.casa_fants=[False,True,True,True]
        self.en_casa=[[1,3],[2,6],[3,9]]
        
        # Laberinto
        self.laberinto=[]
        self.monedas_matriz=[]
        self.leer_laberinto(self.archivo)
        self.n=len(self.laberinto)
        self.m=len(self.laberinto[0])

        self.fin=False

        
        
   

    

    # Leer el laberinto desde un archivo
    def leer_laberinto(self, archivo):  
        cont=0

        with open(archivo, 'r') as file:        
            for line in file:
                row=list(map(int, line.split()))
                self.laberinto.append([0 for _ in range(len(row))])
                self.monedas_matriz.append([0 for _ in range(len(row))])
                
                for i in range(len(row)):   
                    if row[i]==2: self.laberinto[cont][i]=0
                    else: self.laberinto[cont][i]=row[i]
                
                for i in range(len(row)):                    
                    self.monedas_matriz[cont][i]=row[i]
                
                
                cont+=1
                
                
        
        self.imprime_laberinto()

        cont=0
        for x in range(len(self.laberinto)):
            for y in range(len(self.laberinto[0])):                    
                if self.laberinto[x][y]==self.AGENTE: 
                    self.posicion_agente=[x,y]
                    cont+=1
                    if cont==5: break  

                elif self.laberinto[x][y]==self.ROJO: 
                    self.posicion_fants[0]=[x,y]
                    self.salida_fants=[x,y]
                    cont+=1
                    if cont==5: break
                elif self.laberinto[x][y]==self.ROSA: 
                    self.posicion_fants[1]=[x,y]
                    self.casa_fants[1]=[x,y]
                    cont+=1
                    if cont==5: break
                elif self.laberinto[x][y]==self.AZUL: 
                    self.posicion_fants[2]=[x,y]
                    self.casa_fants[2]=[x,y]
                    cont+=1
                    if cont==5: break
                elif self.laberinto[x][y]==self.NARANJA: 
                    self.posicion_fants[3]=[x,y]
                    self.casa_fants[3]=[x,y]
                    cont+=1
                    if cont==5: break
        
        
        
                    

    
    def imprime_laberinto(self):
        print("Laberinto")

        for x in range(len(self.laberinto)):
            for y in range(len(self.laberinto[0])):
                if self.laberinto[x][y]<0: 
                    print(-1, end=" ")
                else: print(self.laberinto[x][y], end=" ")
            print()

    def imprime_monedas(self):
        print("Matriz de Monedas")

        for x in range(len(self.monedas_matriz)):
            for y in range(len(self.monedas_matriz[0])):
                if self.monedas_matriz[x][y]<0: 
                    print(-1, end=" ")
                else: print(self.monedas_matriz[x][y], end=" ")
            print()

    # Mueve al agente
    def mover_agente(self, mov):
        x=self.posicion_agente[0]
        y=self.posicion_agente[1] 
        #m=len(self.laberinto[0])
        moneda=0
        
        self.cont+=1

        if mov==self.ARRIBA:
            if x>0 and self.laberinto[x-1][y]>=0:
                if self.monedas_matriz[x-1][y]==self.MONEDA: 
                    self.monedas_matriz[x-1][y]=0
                    moneda=1
                elif self.monedas_matriz[x-1][y]==self.POWER: 
                    self.monedas_matriz[x-1][y]=0
                    self.state=2
                    self.cont_state=0
                    for i in range(4):                        
                        if not self.casa_fants[i]:
                            self.direccion_fants[i]+=2
                            self.direccion_fants[i]%=4                        
                self.laberinto[x][y]=self.VACIO
                x-=1
        elif mov==self.ABAJO:
            if x<len(self.laberinto)-1 and self.laberinto[x+1][y]>=0:
                if self.monedas_matriz[x+1][y]==self.MONEDA: 
                    self.monedas_matriz[x+1][y]=0
                    moneda=1
                elif self.monedas_matriz[x+1][y]==self.POWER: 
                    self.monedas_matriz[x+1][y]=0
                    self.state=2
                    self.cont_state=0
                    for i in range(4):
                        if not self.casa_fants[i]:
                            self.direccion_fants[i]+=2
                            self.direccion_fants[i]%=4
                self.laberinto[x][y]=self.VACIO
                x+=1
        elif mov==self.IZQUIERDA:
            if y>0 and self.laberinto[x][y-1]>=0:
                if self.monedas_matriz[x][y-1]==self.MONEDA: 
                    self.monedas_matriz[x][y-1]=0
                    moneda=1
                self.laberinto[x][y]=self.VACIO
                y-=1
            elif y==0 and (x==5 or x==9): # TODO
                if self.monedas_matriz[x][self.m-1]==self.MONEDA: 
                    self.monedas_matriz[x][self.m-1]=0
                    moneda=1
                self.laberinto[x][y]=self.VACIO
                y=self.m-1
        elif mov==self.DERECHA:
            if y<self.m-1 and self.laberinto[x][y+1]>=0:
                if self.monedas_matriz[x][y+1]==self.MONEDA: 
                    self.monedas_matriz[x][y+1]=0
                    moneda=1
                self.laberinto[x][y]=self.VACIO
                y+=1
            elif y==self.m-1 and (x==5 or x==9): # TODO
                if self.monedas_matriz[x][0]==self.MONEDA: 
                    self.monedas_matriz[x][0]=0
                    moneda=1
                self.laberinto[x][y]=self.VACIO
                y=0
        
        if self.state==2:
            comidos=[]
            for i in range(4):
                if self.posicion_fants[i][0]==x and self.posicion_fants[i][1]==y:
                    comidos.append(i)
            
            contador=1
            for i in comidos:
                self.posicion_fants[i][0]=self.salida_fants[0]+2
                self.posicion_fants[i][1]=self.salida_fants[1]
                self.en_casa.append([i,self.cont+(3*contador)])
                self.casa_fants[i]=[self.posicion_fants[i][0],self.posicion_fants[i][1]]

                contador+=1
        else:
            for i in range(4):
                if self.posicion_fants[i][0]==x and self.posicion_fants[i][1]==y: 
                    self.fin=True
                    self.laberinto[self.posicion_agente[0]][self.posicion_agente[1]]=self.VACIO

        if self.fin!=True:
            self.laberinto[x][y]=self.AGENTE
            self.posicion_agente=[x,y]
            self.monedas+=moneda
        return moneda

    def mover_fants(self):
        
                          
        self.mover_fantasma(0) 
        self.mover_fantasma(1)
        self.mover_fantasma(2)
        self.mover_fantasma(3)


            

        self.cont_state+=1
        print("Estado=", self.cont_state, "\tMonedas=",self.monedas)
        if self.cont_state==self.state_ticks[self.state]:          
            
            if self.state==0: 
                self.state=1
                print("NUEVO ESTADO: SCATTER",end="")
            else: 
                self.state=0
                print("NUEVO ESTADO: CHASE",end="")

            for i in range(4):
                if not self.casa_fants[i]:
                    self.direccion_fants[i]+=2
                    self.direccion_fants[i]%=4
            
            print("\tGIRA 180º")

            self.cont_state=0

    def mover_fantasma(self, fantasma):
        x=0
        y=0
        dir=0
        color=self.ROJO
        if fantasma==1: color=self.ROSA
        elif fantasma==2: color=self.AZUL
        elif fantasma==3: color=self.NARANJA

        aux_x=0
        aux_y=0
        
        
        if not self.casa_fants[fantasma] and (not(self.state==2 and self.cont_state%2==0)):
            """print("SE MUEVE:",self.COLORES_FANTS[fantasma], "POS:", self.posicion_fants[fantasma])"""
            dir=self.direccion_fants[fantasma]
            x=self.posicion_fants[fantasma][0]
            y=self.posicion_fants[fantasma][1]
            if dir==0 or dir==2: aux_y=1
            else: aux_x=1
            
            self.laberinto[x][y]=self.VACIO

            """print(self.posicion_agente)"""

            # Se mueve para la direccion si no esta en una interseccion
            if self.laberinto[x+aux_x][y+aux_y]<0 and self.laberinto[x-aux_x][y-aux_y]<0:                                
                """print("MUEVE")"""

                if dir==0: x-=1
                elif dir==1: y+=1
                elif dir==2: x+=1
                else: y-=1
            
            
                
            else: 
                if self.state==2: # FRIGHTENED
                    """print("FRIGHTENED")"""
                    opcs=[]
                    for k in range(4):
                        tmp_x=x+self.mX[k]
                        tmp_y=y+self.mY[k]
                        if k==((dir+2)%4) or self.laberinto[tmp_x][tmp_y]<0: continue # no puede ir para atras
                        opcs.append(k)
                    opc=random.randint(0,len(opcs)-1)
                    x+=self.mX[opcs[opc]]
                    y+=self.mY[opcs[opc]]
                    self.direccion_fants[fantasma]=opcs[opc]


                else:
                    target=[0,0]
                    if self.state==0: # CHASE
                        """print("CHASE")"""
                        if fantasma==0: 
                            target[0]=self.posicion_agente[0]
                            target[1]=self.posicion_agente[1]
                        elif fantasma==1: 
                            target[0]=self.posicion_agente[0]
                            target[1]=self.posicion_agente[1]

                            if self.direccion==0: 
                                target[0]-=4
                                target[1]-=4
                            elif self.direccion==1: target[1]+=4
                            elif self.direccion==2: target[0]+=4
                            else: target[1]-=4
                            
                        elif fantasma==2: 
                            tmp=[0,0]
                            tmp[0]=self.posicion_agente[0]
                            tmp[1]=self.posicion_agente[1]

                            if self.direccion==0: 
                                tmp[0]-=2
                                tmp[1]-=2
                            elif self.direccion==1: tmp[1]+=2
                            elif self.direccion==2: tmp[0]+=2
                            else: tmp[1]-=2

                            dif_x=tmp[0]-self.posicion_fants[0][0]
                            dif_y=tmp[1]-self.posicion_fants[0][1]
                            target[0]=tmp[0]+dif_x
                            target[1]=tmp[1]+dif_y
                        
                        elif fantasma==3:
                            dist_manhattan=abs(self.posicion_agente[0]-self.posicion_fants[3][0])
                            dist_manhattan+=abs(self.posicion_agente[1]-self.posicion_fants[3][1])

                            if dist_manhattan<8: target=self.scatter_targets[3]
                            else: target=self.posicion_agente


                    else: # SCATTER
                        """print("SCATTER")"""
                        target=self.scatter_targets[fantasma]
                        
                    dist=float("inf")
                    tmp_x=0
                    tmp_y=0
                    dir_idx=0
                    tmp=0
                    for k in range(4):
                        tmp_x=x+self.mX[k]
                        tmp_y=y+self.mY[k]
                        if k==((dir+2)%4) or self.laberinto[tmp_x][tmp_y]<0: continue # no puede ir para atras
                        tmp=self.distancia_celda(target,[tmp_x, tmp_y])
                        """print("pos= {},{}\tdist= {}".format(tmp_x, tmp_y, tmp))"""
                        if dist>tmp:
                            dist=tmp
                            dir_idx=k
                    """print("direccion:", dir_idx)"""
                    self.direccion_fants[fantasma]=dir_idx
                    
                    x+=self.mX[dir_idx]
                    y+=self.mY[dir_idx]
            
            # "portales"
            if y==self.m and (x==5 or x==9): y=0
            if y==-1 and (x==5 or x==9): y=self.m-1
            
            """print(self.m, x, y)"""
            self.posicion_fants[fantasma][0]=x
            self.posicion_fants[fantasma][1]=y
            self.laberinto[x][y]=color

            if self.posicion_agente[0]==x and self.posicion_agente[1]==y:
                if self.state==2:
                    self.laberinto[x][y]=self.VACIO                   

                    self.posicion_fants[fantasma][0]=self.salida_fants[0]+2
                    self.posicion_fants[fantasma][1]=self.salida_fants[1]

                    self.en_casa.append([fantasma,self.cont+3])
                    self.casa_fants[fantasma]=[self.posicion_fants[fantasma][0],self.posicion_fants[fantasma][1]]
                else: self.fin=True
        """else:
            print("no MUEVE:",self.COLORES_FANTS[fantasma], "POS:", self.posicion_fants[fantasma])"""
    
    def distancia_celda(self, a, b):               
        return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

    # --------------------------------------------------------------------------------
    # --- GUI ------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    # Función para inicializar pygame y mostrar el laberinto en una ventana
    def mostrar_laberinto(self):       

        #self.imprime_laberinto()
         
        mov=None
        imagen=None
        while True:

            # bucle principal
            while not self.fin:            
                # evento = pulsa alguna tecla
                for evento in pygame.event.get():
                    if evento.type==pygame.QUIT: # cerrar GUI
                        pygame.quit()
                        sys.exit()
                    elif evento.type==pygame.KEYDOWN: # pulsa tecla                
                        if evento.key==pygame.K_UP: 
                            mov=self.ARRIBA                    
                            self.direccion=0
                        elif evento.key==pygame.K_RIGHT: 
                            mov=self.DERECHA 
                            self.direccion=1
                        elif evento.key==pygame.K_DOWN: 
                            mov=self.ABAJO    
                            self.direccion=2
                        elif evento.key==pygame.K_LEFT: 
                            mov=self.IZQUIERDA    
                            self.direccion=3
                        else: mov=None
                        
                        if mov!=None:                             
                            self.mover_agente(mov)
                            
                            self.mover_fants()
                            
                            if len(self.en_casa)!=0 and self.cont==self.en_casa[0][1]:
                                idx=self.en_casa.pop(0)[0]

                                self.laberinto[self.casa_fants[idx][0]][self.casa_fants[idx][1]]=self.VACIO
                                self.laberinto[self.salida_fants[0]][self.salida_fants[1]]=self.COLORES_FANTS[idx]
                                self.casa_fants[idx]=False

                                self.posicion_fants[idx][0]=self.salida_fants[0]
                                self.posicion_fants[idx][1]=self.salida_fants[1]
                            
                            # fantasmas
                            i=3
                            while i>=0:
                                self.laberinto[self.posicion_fants[i][0]][self.posicion_fants[i][1]]=self.COLORES_FANTS[i]
                                i-=1
                            # agente
                            if self.fin!=True:
                                self.laberinto[self.posicion_agente[0]][self.posicion_agente[1]]=self.AGENTE
                            
                            if self.monedas==132: self.fin=True


                # dibuja el laberinto
                self.GUI_laberinto()
            
            if self.fin==True:
                if self.monedas==132: 
                    print("\nGANASTE\n")
                    for _ in range(3):
                        self.GUI_mensaje(1)
                        pygame.display.flip()
                        time.sleep(1)
                        self.GUI_laberinto()
                        time.sleep(0.33)
                        pygame.display.flip()
                else: 
                    print("\nPERDISTE\n")

                    for _ in range(3):
                        self.GUI_mensaje(0)
                        pygame.display.flip()
                        time.sleep(1)
                        self.GUI_laberinto()
                        time.sleep(0.33)
                        pygame.display.flip()


                pygame.display.flip()
                    

                self.reset()
    
    def GUI_laberinto(self):
        self.pantalla.fill((0, 0, 0))  # limpia antes de dibujar
        
        # recorre el laberinto
        for x, fila in enumerate(self.laberinto):
            for y, celda in enumerate(fila):
                if celda<0:
                    imagen=self.walls_imgs[abs(celda)-1]  
                elif celda==1: imagen=self.walls_imgs[-1]
                elif celda==self.VACIO: 
                    if self.monedas_matriz[x][y]==self.MONEDA: imagen=self.coin_img
                    elif self.monedas_matriz[x][y]==self.POWER: imagen=self.power_img
                    else: imagen=self.empty_img                                        
                elif celda==self.POWER: imagen=self.power_img
                elif celda==self.AGENTE: imagen=self.agente_imgs[self.direccion] 
                else:
                    if self.state==2:
                        if self.cont_state<20: imagen=self.ghosts_imgs[-1][0]
                        else:
                            if self.cont_state%2==0: imagen=self.ghosts_imgs[-1][1]
                            else: imagen=self.ghosts_imgs[-1][0]
                    else:
                        if celda==self.ROJO: imagen=self.ghosts_imgs[0][self.direccion_fants[0]]
                        elif celda==self.ROSA: imagen=self.ghosts_imgs[1][self.direccion_fants[1]]
                        elif celda==self.AZUL: imagen=self.ghosts_imgs[2][self.direccion_fants[2]]
                        else: imagen=self.ghosts_imgs[3][self.direccion_fants[3]]
                

                self.pantalla.blit(imagen, (y*self.tam_celda, x*self.tam_celda))

        pygame.display.flip()
    
    def GUI_mensaje(self, tipo):
        aux=0 if tipo==0 else 1

        for x, fila in enumerate(self.laberinto):
            if x!=9:
                for y, celda in enumerate(fila):
                    if celda<0:
                        imagen=self.walls_imgs[abs(celda)-1]  
                    elif celda==1: imagen=self.walls_imgs[-1]
                    elif celda==self.VACIO: 
                        if self.monedas_matriz[x][y]==self.MONEDA: imagen=self.coin_img
                        elif self.monedas_matriz[x][y]==self.POWER: imagen=self.power_img
                        else: imagen=self.empty_img                                        
                    elif celda==self.POWER: imagen=self.power_img
                    elif celda==self.AGENTE: imagen=self.agente_imgs[self.direccion] 
                    else:
                        if self.state==2:
                            if self.cont_state<20: imagen=self.ghosts_imgs[-1][0]
                            else:
                                if self.cont_state%2==0: imagen=self.ghosts_imgs[-1][1]
                                else: imagen=self.ghosts_imgs[-1][0]
                        else:
                            if celda==self.ROJO: imagen=self.ghosts_imgs[0][self.direccion_fants[0]]
                            elif celda==self.ROSA: imagen=self.ghosts_imgs[1][self.direccion_fants[1]]
                            elif celda==self.AZUL: imagen=self.ghosts_imgs[2][self.direccion_fants[2]]
                            else: imagen=self.ghosts_imgs[3][self.direccion_fants[3]]
                    self.pantalla.blit(imagen, (y*self.tam_celda, x*self.tam_celda))
            else:
                contador=0
                for y, celda in enumerate(fila):
                    if (y>=6+aux and y<=9) or (y>=11 and y<=14-aux):
                        imagen=self.message_imgs[tipo][contador]
                        contador+=1
                    else:
                        if celda<0:
                            imagen=self.walls_imgs[abs(celda)-1]  
                        elif celda==1: imagen=self.walls_imgs[-1]
                        elif celda==self.VACIO: 
                            if self.monedas_matriz[x][y]==self.MONEDA: imagen=self.coin_img
                            elif self.monedas_matriz[x][y]==self.POWER: imagen=self.power_img
                            else: imagen=self.empty_img                                        
                        elif celda==self.POWER: imagen=self.power_img
                        elif celda==self.AGENTE: imagen=self.agente_imgs[self.direccion] 
                        else:
                            if self.state==2:
                                if self.cont_state<20: imagen=self.ghosts_imgs[-1][0]
                                else:
                                    if self.cont_state%2==0: imagen=self.ghosts_imgs[-1][1]
                                    else: imagen=self.ghosts_imgs[-1][0]
                            else:
                                if celda==self.ROJO: imagen=self.ghosts_imgs[0][self.direccion_fants[0]]
                                elif celda==self.ROSA: imagen=self.ghosts_imgs[1][self.direccion_fants[1]]
                                elif celda==self.AZUL: imagen=self.ghosts_imgs[2][self.direccion_fants[2]]
                                else: imagen=self.ghosts_imgs[3][self.direccion_fants[3]]
                    self.pantalla.blit(imagen, (y*self.tam_celda, x*self.tam_celda))
        

    def cargar_imagenes(self, tam):
        # leer las imagenes
        vacio=pygame.image.load('imagenes/empty.png').convert_alpha()    
        moneda=pygame.image.load('imagenes/coin.png').convert_alpha()
        power=pygame.image.load('imagenes/power.png').convert_alpha()
        
        
        walls=[]
        
        names=["imagenes/walls/wall_0.png","imagenes/walls/wall_01.png","imagenes/walls/wall_1.png","imagenes/walls/wall_02.png",
               "imagenes/walls/wall_2.png","imagenes/walls/wall_03.png","imagenes/walls/wall_3.png","imagenes/walls/wall_012.png",
               "imagenes/walls/wall_12.png","imagenes/walls/wall_013.png","imagenes/walls/wall_13.png","imagenes/walls/wall_023.png",
               "imagenes/walls/wall_23.png","imagenes/walls/wall_123.png","imagenes/walls/wall.png","imagenes/walls/z0.png",
               "imagenes/walls/z1.png","imagenes/walls/z2.png","imagenes/walls/z3.png","imagenes/walls/z4.png","imagenes/walls/ghost_exit.png"]
        for wall in names:
            walls.append(pygame.image.load(wall).convert_alpha())
               

        agente=[]
        agente.append(pygame.image.load('imagenes/pacman_up.png').convert_alpha())
        agente.append(pygame.image.load('imagenes/pacman_right.png').convert_alpha())
        agente.append(pygame.image.load('imagenes/pacman_down.png').convert_alpha())
        agente.append(pygame.image.load('imagenes/pacman_left.png').convert_alpha())

        names=[["imagenes/red_up.png","imagenes/red_right.png","imagenes/red_down.png","imagenes/red_left.png"],
               ["imagenes/pink_up.png","imagenes/pink_right.png","imagenes/pink_down.png","imagenes/pink_left.png"],
               ["imagenes/blue_up.png","imagenes/blue_right.png","imagenes/blue_down.png","imagenes/blue_left.png"],
               ["imagenes/orange_up.png","imagenes/orange_right.png","imagenes/orange_down.png","imagenes/orange_left.png"],
               ["imagenes/fright_ghost1.png","imagenes/fright_ghost2.png"]]
        ghosts=[]
        for color in names:
            tmp=[]
            for name in color:
                tmp.append(pygame.image.load(name).convert_alpha())
            ghosts.append(tmp)

        names=[["imagenes/messages/G.png","imagenes/messages/A.png","imagenes/messages/M.png","imagenes/messages/E.png",
                "imagenes/messages/O.png","imagenes/messages/V.png","imagenes/messages/E.png","imagenes/messages/R.png"],                
                ["imagenes/messages/Y.png","imagenes/messages/O_2.png","imagenes/messages/U.png",
                "imagenes/messages/W.png","imagenes/messages/I.png","imagenes/messages/N.png"]]
        messages=[]
        for message in names:
            tmp=[]
            for char in message:
                tmp.append(pygame.image.load(char).convert_alpha())
            messages.append(tmp)

        # escalar para que tengan el mismo tamaño
        self.empty_img=pygame.transform.scale(vacio, (tam, tam))    
        self.coin_img=pygame.transform.scale(moneda, (tam, tam))
        self.power_img=pygame.transform.scale(power, (tam, tam))

        self.walls_imgs=[]
        self.agente_imgs=[]
        for w in walls:
            self.walls_imgs.append(pygame.transform.scale(w, (tam, tam)))
        
        for i in range(4):
            self.agente_imgs.append(pygame.transform.scale(agente[i], (tam, tam)))

        self.ghosts_imgs=[]
        for g in ghosts:
            tmp=[]
            for dir in g:
                tmp.append(pygame.transform.scale(dir, (tam, tam)))
                
            self.ghosts_imgs.append(tmp)

        self.message_imgs=[]
        for message in messages:
            tmp=[]
            for char in message:
                tmp.append(pygame.transform.scale(char, (tam, tam)))
                
            self.message_imgs.append(tmp)


if __name__ == "__main__":    
    env=Pacman(os.path.join("datos", "env.txt"),True)
    
    
