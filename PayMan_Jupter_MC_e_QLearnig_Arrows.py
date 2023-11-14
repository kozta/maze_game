import gymnasium as gym
import numpy as np
import pygame
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import time


class MazeGameEnv(gym.Env):
    def __init__(self, maze, alpha=0.1, gamma=0.99, epsilon=0.1):
        super(MazeGameEnv, self).__init__()
        self.maze = np.array(maze)
        self.start_pos = tuple(np.argwhere(self.maze == 'S')[0]) #argwhere encontra o elemento 'S' de start no labirinto.[0] é para armazenar a tupla na posição 0 
        self.goal_pos = tuple(np.argwhere(self.maze == 'G')[0]) #argwhere encontra o elemento 'G' de goal no labirinto.[0] é para armazenar a tupla na posição 0 
        self.current_pos = self.start_pos
        self.num_rows, self.num_cols = self.maze.shape # shape fornece as dimensões do array

        self.alpha = alpha  # Taxa de aprendizado
        self.gamma = gamma  # Fator de desconto
        self.epsilon = epsilon  # Probabilidade de exploração

        # Define ação como Discrete com 4 ações (cima, baixo, esquerda, direita)
        self.action_space = gym.spaces.Discrete(4)

        # Defina o espaço de observação como uma tupla com o número de linhas e colunas
        self.observation_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(self.num_rows), gym.spaces.Discrete(self.num_cols)))

        # Inicialize o ambiente Pygame
        pygame.init()

        # Defina cores
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLACK = (0, 0, 0)
        self.PURPLE = (255, 0, 255)
        self.BLUE = (0, 0, 255)

    def reset(self):
        self.current_pos = self.start_pos
        return self.current_pos

    def step(self, action, reward=1):
        if action == 0:  # Cima
            self.move('up')
        elif action == 1:  # Baixo
            self.move('down')
        elif action == 2:  # Esquerda
            self.move('left')
        elif action == 3:  # Direita
            self.move('right')

        obs = self.current_pos
        #
        if (reward == 1):
            # reward = 1 if self.current_pos == self.goal_pos else 0
            distance_to_goal = abs(self.current_pos[0] - self.goal_pos[0]) + abs(self.current_pos[1] - self.goal_pos[1])
            reward = 1 / (distance_to_goal + 1)
            done = self.current_pos == self.goal_pos
        else:
            reward = ""
            done = ""

        info = {}

        return obs, reward, done, info

    def move(self, action):
        new_pos = list(self.current_pos)

        if action == 'up':
            new_pos[0] -= 1
        elif action == 'down':
            new_pos[0] += 1
        elif action == 'left':
            new_pos[1] -= 1
        elif action == 'right':
            new_pos[1] += 1

        new_pos = tuple(new_pos)

        if self.is_valid_position(new_pos[0], new_pos[1]):
            self.current_pos = new_pos

    def is_valid_position(self, row, col):
        return 0 <= row < self.num_rows and 0 <= col < self.num_cols and self.maze[row, col] != '#'

    def render(self, mode='human', path=None):
        if mode == 'human':
            self.draw_maze(path)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

        if mode == 'blank':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

    def play_path(self, path, delay=1):
        for position in path:
            self.current_pos = position
            self.render(mode='human', path=path)
            time.sleep(delay)

    def draw_maze(self, path=None):
        self.cell_size = 50
        self.screen_width = self.num_cols * self.cell_size
        self.screen_height = self.num_rows * self.cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.screen.fill(self.WHITE)

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size

                if self.maze[row, col] == '#':
                    pygame.draw.rect(self.screen, self.BLACK, (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.maze[row, col] == 'S':
                    pygame.draw.rect(self.screen, self.GREEN, (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.maze[row, col] == 'G':
                    pygame.draw.rect(self.screen, self.RED, (cell_left, cell_top, self.cell_size, self.cell_size))

        if path:
            for i in range(len(path) - 1):
                start = path[i]
                end = path[i + 1]
                self.draw_arrow(start, end)

        current_row, current_col = self.current_pos
        
        pygame.draw.rect(self.screen, self.PURPLE,
                         (current_col * self.cell_size, current_row * self.cell_size, self.cell_size, self.cell_size))

        # AS DUAS LINHAS ACIMA FORAM COMENTADAS PARA QUE ENTRE AS DUAS LINHAS ABAIXO 
        # FAZENDO A EXIBIÇÃO DA IMAGEM DA SETA
        #arrow_img = pygame.image.load('RDLU.png')      
        #self.screen.blit(arrow_img,(current_col * self.cell_size, current_row * self.cell_size))
        
        

    def draw_arrow(self, start, end):
        start_x = start[1] * self.cell_size + self.cell_size // 2
        start_y = start[0] * self.cell_size + self.cell_size // 2
        end_x = end[1] * self.cell_size + self.cell_size // 2
        end_y = end[0] * self.cell_size + self.cell_size // 2

        pygame.draw.line(self.screen, self.BLUE, (start_x, start_y), (end_x, end_y), 2)

    def close(self):
        pygame.quit()

maze = [
    ['S','.','.','.','.','.'],
    ['.','.','.','.','.','.'],
    ['.','.','.','.','.','.'],
    ['.','.','.','.','.','.'],
    ['.','.','.','#','.','.'],
    ['.','.','.','.','.','G']
]



maze = [
    ['S','.','.','.','.','.'],
    ['.','.','.','#','.','.'],
    ['.','.','.','#','.','.'],
    ['.','.','.','#','.','.'],
    ['.','#','#','#','.','.'],
    ['.','.','.','.','.','G']
]

maze = [
    ['S', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '#', '.', '#', '.', '.', '.'],
    ['.', '.', '.', '.', '#', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '#', '.', '.', '.', '#', '.'],
    ['.', '.', '.', '.', '#', '.', '.', '.', '.', '.'],
    ['.', '#', '#', '#', '#', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '#', '.', '.', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['.', '#', '.', '.', '.', '.', '.', '.', '.', 'G'],
]

env = MazeGameEnv(maze)

done = False
state = env.reset()
env.render()

while not done:
    action = env.action_space.sample()  # Exemplo: selecione uma ação aleatória
    print(action)
    state, reward, done, _ = env.step(action)
    print(state, reward, done)
    env.render()
    pygame.display.set_caption('Roteiro Randômico')
    pygame.time.delay(10)
print("Done!")
env.close()


class MonteCarlo:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, exploration_prob=0.1):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob

        
        
        
        self.mc_acoe = [0, 1, 2, 3]
        self.mc_reward = -1
        self.VI = np.zeros((env.num_cols, env.num_rows))
        print (self.VI)
        self.otimo = []
        self.converg = 80  #o que significa este 80 ?
        
        self.dest = []
        self.dest.append(list(state))
        self.gamm_arr = {(x, y): list() for x in range(env.num_cols) for y in range(env.num_rows)}
        self.vabs = {(x, y): list() for x in range(env.num_cols) for y in range(env.num_rows)}
        self.sta = [[x, y] for x in range(env.num_cols) for y in range(env.num_rows)]

    # A função epoca vai guardar toda a sequencia de estados utilizados ate chegar ao alvo
    # a partir de uma saída(inicio) randômica. Ou seja, gera o episodio.
    def epocas(self):
        begin_state = env.start_pos
        epocas = []
        
        while True:
            # Se estado igual destino encerra a epoca
            if list(begin_state) in self.dest:
                self.env.reset()
                return epocas
            # Seleciona de forma randomica uma ação (cima , esquerda, ...   )
            action = random.choice(self.mc_acoe)

            # Retorna os novos estados que podem ser utilizados
            # A função step retorna 4 valores sendo o primeioro new_state
            # o segundo é o reward de acordo com a distancia do goal. O valor é entre 0 e 1. Quanto mais perto do goal maior o valor
            # o terceiro parametro é "done" que não sei o que é e o quarto é "info" que tambem não vi utilidade ainda
            # na função step devemos colocar o epsilon que determina o ambiente estocastico onde ele ira escolher o destino de acordo com essa opção
            new_state, _, _, _ = self.env.step(action, 0)  # a função step valida tambem se tem parede ou borda

            ## A função 'step' chama a função 'move' que é a responsável por atualizar 'current_pos' que é a variavel que atualiza a tela  em render>
            env.render()
            pygame.display.set_caption('Aprendendo - Monte Carlo')
            pygame.time.delay(1) # tempo do delay em milisegundos
            ## Fim do renderizar o grid



            # Cria lista das epocas(estado), armazenando [posição inicial, a ação, a recompensa-que aqui no MC está sendo unica-, posição final]
            epocas.append([list(begin_state), action, self.mc_reward, list(new_state)])

            begin_state = new_state

    def MC(self, mc_interactions):
        conv_epoc = []
        for z in tqdm(range(mc_interactions)):
            epoc = self.epocas()
            ga = 0
            for i, passo in enumerate(epoc[::-1]):
                ga = env.gamma * ga + passo[2]
                # Começa pelas epocas proximas ao destino
                if passo[0] not in [x[0] for x in epoc[::-1][len(epoc) - i:]]:
                    # Pega posicao que vai ser calculada
                    pos = (passo[0][0], passo[0][1])
                    # Monta uma lista de gamma para posição
                    self.gamm_arr[pos].append(ga)
                    # Calcula a media de todos os gammas da posicao
                    newValue = np.average(self.gamm_arr[pos])
                    # Monta uma lista na posição com o calculo do valor absoluto da diferença entre as posições - antiga e nova
                    self.vabs[pos[0], pos[1]].append(np.abs(self.VI[pos[0], pos[1]] - newValue))
                    # lista com os valores de interacao
                    self.VI[pos[0], pos[1]] = newValue
                
            
           
            ##Função para popular com as setas do resultado final 
            ##---------------------------------------------------
            #self.population_maze()  
            #pygame.display.update()
            ##---------------------------------------------------        
                        
            if z >= self.converg:  # convergencia de acordo com o grafico
                conv_epoc.append(epoc)

            if z in [100, 200, 300, 400, 500, 600, 700, 800, 900,  mc_interactions - 1]:
                print("Iteration {}".format(z + 1))
                print(self.VI)
                #Função para popular com as setas do resultado final 
                #---------------------------------------------------
                env.close()
                self.population_maze()  
                pygame.display.set_caption("Iteration {}".format(z + 1))
                pygame.display.update()
                pygame.time.delay(9999)
                #---------------------------------------------------      
                

            

      
        env.close()
        #Função para popular com as setas do resultado final 
        #---------------------------------------------------
        env.close()
        self.population_maze() 
        pygame.display.set_caption('Ultimo Resultado - Monte Carlo')
        pygame.display.update()
        #---------------------------------------------------        
        
        
        
        
        plt.figure(figsize=(20, 10))
        all_series = [list(x)[:100] for x in self.vabs.values()]
        for series in all_series:
            plt.plot(series)

        list(map(len, conv_epoc))

        def get_line_len(conv_epoc):
            return len(conv_epoc), conv_epoc

        #list_converg recebe organizado por ordem de tamnho ...
        list_converg = sorted(list(map(get_line_len, conv_epoc)))

        print(f'O menor número de passos atingidos em {mc_interactions} interações foram : {list_converg[0][0]}')

        print(
            f'Ocorreu na epoca  {(self.converg + list_converg[0][0])} e é considerado como o resultado ótimo para este treinamento')

        steps = list_converg[0][1]

        bob_volta_casa = []

        for item in steps:
            bob_volta_casa.append((item[3][0], item[3][1]))

        return bob_volta_casa


    def population_maze(self, path=None):
        # Defina cores
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLACK = (0, 0, 0)
        self.PURPLE = (255, 0, 255)
        self.BLUE = (0, 0, 255)
        self.maze = np.array(maze)
        self.num_rows, self.num_cols = self.maze.shape # shape fornece as dimensões do array
        self.cell_size = 50
        self.screen_width = self.num_cols * self.cell_size
        self.screen_height = self.num_rows * self.cell_size
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.screen.fill(self.WHITE)

        for row in range(self.num_rows):
            for col in range(self.num_cols):
                cell_left = col * self.cell_size
                cell_top = row * self.cell_size

                if self.maze[row, col] == '#':
                    pygame.draw.rect(self.screen, self.BLACK, (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.maze[row, col] == 'S':
                    pygame.draw.rect(self.screen, self.GREEN, (cell_left, cell_top, self.cell_size, self.cell_size))
                elif self.maze[row, col] == 'G':
                    pygame.draw.rect(self.screen, self.RED, (cell_left, cell_top, self.cell_size, self.cell_size))
        
        
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                
                        arrow_img = pygame.image.load('Q.png')
                        Figura = ''
                        
                        if self.VI[row, col] != 0:
                            
                            lista_figura = []
                            lista_figura_identifica = []
                            
                            #verifica a direita
                            if col < self.num_cols -1:
                                   
                                   if (self.VI[row, col + 1] != 0 or self.maze[row, col +1] == 'G') and self.VI[row, col + 1] > self.VI[row, col]:
                                       valor_R = self.VI[row, col + 1]
                                       lista_figura.append(valor_R)
                                       lista_figura_identifica.append(['R', valor_R]) 
                                       Figura = Figura + 'R' 
                                       #print ("Direita é maior")
                            
                            #verifica abaixo     
                            if row < self.num_rows -1:
                                    
                                    if (self.VI[row + 1, col] != 0  or self.maze[row + 1, col] == 'G') and (self.VI[row + 1, col] > self.VI[row, col]):
                                       valor_D  = self.VI[row + 1, col]
                                       lista_figura.append(valor_D)
                                       lista_figura_identifica.append(['D', valor_D])
                                       Figura = Figura + 'D' 
                                       #print ("Abaixo é maior")
                                
                            #verifica esquerda     
                            if col > 0:
                                
                                if (self.VI[row, col - 1] != 0 or self.maze[row, col -1] == 'G') and self.VI[row, col -1] > self.VI[row, col] :
                                       valor_L = self.VI[row, col -1]
                                       lista_figura.append(valor_L)    
                                       lista_figura_identifica.append(['L', valor_L])
                                       Figura = Figura + 'L' 
                                       #print ("Esquerda é maior")
                            #verifica acima     
                            if row > 0:
                                    
                                    if (self.VI[row - 1, col] != 0 or self.maze[row -1, col] == 'G') and self.VI[row - 1, col] > self.VI[row, col] :
                                       valor_U = self.VI[row - 1, col]
                                       lista_figura.append(valor_U)
                                       lista_figura_identifica.append(['U', valor_U])
                                       Figura = Figura + 'U'
                                       #print ("Acima é maior")
                                       
                            
                            
                            #Escolhe a seta a ser exibida
                            
                            #verifica se tem valores repetidos na lista
                             #---------------------------------------------------------------------------
                             
                            Figura = ''
                            select_max_indice = -1
                            indices_repetidos = []
                            #lista_figura = [valor_R, valor_D, valor_L, valor_U]
                            #lista_figura_identifica = (['R', valor_R] , ['D', valor_D] , ['L', valor_L] , ['U', valor_U])

                            #verifica se tem valores repetidos na lista
                            for i, elemento in enumerate(lista_figura):
                                   if elemento in lista_figura[i+1:]:
                                      indices_repetidos.append(lista_figura.index(elemento)) 
                                      
                            print (len(indices_repetidos), " Valor repetido")         
                            if len(indices_repetidos) > 0:
                                #verifica se o valor repetido é o max valor da lista
                                select_max_indice = lista_figura.index(max(lista_figura))  
                                try:
                                     indices_repetidos.index(select_max_indice)
                                     #print (indices_repetidos.index(select_max_indice), " ACHEI O MAX DENTRO DE INDICES REPETIDOS")
                                     #se entrou aqui é pq o valor repetido é tambm o máx valor da lista
                                     #então seleciona todos esses que são repetidos para montar a Figura
                                     montar_figura = [igs for igs, item in enumerate(lista_figura) if item == lista_figura[select_max_indice]]
                                     print(montar_figura, "montar_figura")
                                     for g in range(len(montar_figura)):
                                         #Verifica a direcao que o indice representa na lista_figura_identifica
                                         qual_indice = montar_figura[g]
                                         direcao, valor = lista_figura_identifica[qual_indice]
                                         Figura = Figura + direcao
                                         
                                            
                                     #print (montar_figura[g])
                                    
                                
                                except:
                                   
                                   
                                   #Verifica a direcao que o indice representa na lista_figura_identifica
                                   direcao, valor = lista_figura_identifica[select_max_indice]
                                   Figura = Figura + direcao
                                   
                                   
                                    
                            else:       
                                    # se não tem valores repetidos. então pega apenas o max valor da lista    
                                   print(Figura, " Figura")
                                   try:
                                       select_max_indice = lista_figura.index(max(lista_figura))
                                       print(lista_figura.index(max(lista_figura)), " primeira")
                                       #Verifica a direcao que o indice representa na lista_figura_identifica
                                       direcao, valor = lista_figura_identifica[select_max_indice]
                                       Figura = Figura + direcao
                                   except:                                 
                                       Figura = Figura 
                             
                             #----------------------------------------------------------------------------
                                        
                            
                            
                            
                            
                            if Figura == 'RDLU':
                               
                                                              
                                arrow_img = pygame.image.load('RDLU.png')      
                                                                                             
                               
                            elif Figura == 'RD':
                               
                                arrow_img = pygame.image.load('RD.png')      
                                
                            elif Figura == 'RU':    
                                
                                arrow_img = pygame.image.load('RU.png')      
                            
                            elif Figura == 'RDL':
                                
                                arrow_img = pygame.image.load('RDL.png')      
                            
                            elif Figura == 'RDU':
                                
                                arrow_img = pygame.image.load('RDU.png')         
                                
                            elif Figura == 'DLU':    
                                arrow_img = pygame.image.load('DLU.png')         
                            
                            elif Figura == 'RLU':    
                                arrow_img = pygame.image.load('RLU.png')
                            
                            elif Figura == 'DL':
                                arrow_img = pygame.image.load('DL.png')
                            
                            elif Figura == 'LU':
                                arrow_img = pygame.image.load('LU.png')
                            
                            elif Figura == 'R':
                                arrow_img = pygame.image.load('R.png')
                            
                            elif Figura == 'D':
                                arrow_img = pygame.image.load('D.png')
                            
                            elif Figura == 'L':
                                arrow_img = pygame.image.load('L.png')
                            
                            elif Figura == 'U':
                                arrow_img = pygame.image.load('U.png')
                   
                            if row == 5 and col == 4:
                               print ("Figura ", Figura) 
                               
                            self.screen.blit(arrow_img,(col * self.cell_size, row * self.cell_size))        
        

env = MazeGameEnv(maze, alpha=0.1, gamma=0.9, epsilon=0.1)

conf = MonteCarlo(env)

optimal_path = conf.MC(500)

print("Optimal Path:", optimal_path)



