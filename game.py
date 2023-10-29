#import argparse

from tqdm import tqdm
import random
import gym
from gym import spaces
import numpy as np
import pygame
import sys
import matplotlib.pyplot as plt

class MazeGameEnv(gym.Env):
    def __init__(self, maze, alpha=0.1, gamma=0.99, epsilon=0.1):
        super(MazeGameEnv, self).__init__()
        self.maze = np.array(maze)
        self.start_pos = tuple(np.argwhere(self.maze == 'S')[0])
        self.goal_pos = tuple(np.argwhere(self.maze == 'G')[0])
        self.mc_acoe = [0,1,2,3]
        #self.mc_acoe= [[-1,0],[1,0],[0,1],[0,-1]]
        self.mc_reward = -1
        self.current_pos = self.start_pos
        self.num_rows, self.num_cols = self.maze.shape
        self.VI = np.zeros((self.num_cols, self.num_rows))
        #self.VI = np.zeros((10, 10))
        
        

        self.alpha = alpha  # Taxa de aprendizado
        self.gamma = gamma  # Fator de desconto
        self.epsilon = epsilon  # Probabilidade de exploração

        # Define ação como Discrete com 4 ações (cima, baixo, esquerda, direita)
        self.action_space = spaces.Discrete(4)

        # Defina o espaço de observação como uma tupla com o número de linhas e colunas
        self.observation_space = spaces.Tuple((spaces.Discrete(self.num_rows), spaces.Discrete(self.num_cols)))

        # Crie e inicialize a tabela Q
        self.Q_table = {}
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                state = (row, col)    #destino
                self.Q_table[state] = np.zeros(self.action_space.n)
        
        self.dest = []
        self.dest.append(list(state))
        self.gamm_arr = {(x, y):list() for x in range(self.num_cols) for y in range(self.num_rows)}
        self.vabs = {(x, y):list() for x in range(self.num_cols) for y in range(self.num_rows)}
        self.sta = [[x, y] for x in range(self.num_cols) for y in range(self.num_rows)] 

        print(self.Q_table)

        # Inicialize o ambiente Pygame
        pygame.init()

        # Defina cores
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLACK = (0, 0, 0)
        self.PURPLE = (255, 0, 255)

    def epocas (self):
        begin_state = self.start_pos  
        epocas = []
        while True:
            # Se estado igual destino encerra a epoca
            if list(begin_state) in self.dest:
                return epocas
            # Seleciona de forma randomica uma ação (cima , esquerda, ...   )
            action = random.choice(self.mc_acoe)
            
            # Retorna os novos estados que podem ser utilizados
            new_state, _, _, _ = self.step(action, 0)  # valida se tem parede ou borda
            # Cria lista das epocas
            epocas.append([list(begin_state), action, self.mc_reward, list(new_state)])

            begin_state = new_state
            #print (begin_state)
    
    # Monte Carlo
    def MC (self, mc_interactions):
        for z in tqdm(range(mc_interactions)):
            epoc = self.epocas()
            ga = 0
            vi_ant = None
            count = 0
            for i, passo in enumerate(epoc[::-1]):
                ga = self.gamma*ga + passo[2]
                # Começa pelas epocas proximas ao destino
                if passo[0] not in [x[0] for x in epoc[::-1][len(epoc)-i:]]:
                    # Pega posicao que vai ser calculada
                    pos = (passo[0][0], passo[0][1])
                    # Monta uma lista de gamma para posição  
                    self.gamm_arr[pos].append(ga)
                    # Calcula a media de todos os gammas da posicao
                    newValue = np.average(self.gamm_arr[pos])
                    # Monta uma lista na posição com o calculo do valor absoluto da diferença entre as posições - antiga e nova
                    self.vabs[pos[0], pos[1]].append(np.abs(self.VI[pos[0], pos[1]]-newValue))
                    # lista com os valores de interacao 
                    self.VI[pos[0], pos[1]] = newValue
                    
            if np.array_equiv(self.vabs, vi_ant):
                #cont += 1
                #if cont == 5:    
                print (f'convergiu !!! Epoca{i} {self.VI}')
            #else:
            #    cont = 0
            
            if z in [100,200,300, 400, mc_interactions-1]:
                print("Iteration {}".format(z+1))
                print(self.VI)
                print("")
                        
            vabs_ant = self.vabs
        
        plt.figure(figsize=(20,10))
        all_series = [list(x)[:100] for x in self.vabs.values()]
        for series in all_series:
            plt.plot(series)
                

    def train_q_learning(self, num_episodes):
        for episode in range(num_episodes):
            state = self.reset()
            done = False

            while not done:
                # Escolha a ação com base na política epsilon-greedy
                if np.random.rand() < self.epsilon:
                    action = self.action_space.sample()  # Ação aleatória
                else:
                    action = np.argmax(self.Q_table[state])  # Ação com maior valor Q

                new_state, reward, done, _ = self.step(action)

                # Atualize a tabela Q com base na recompensa
                self.Q_table[state][action] = (1 - self.alpha) * self.Q_table[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q_table[new_state]))

                state = new_state

    def find_optimal_path(self):
        state = self.start_pos
        optimal_path = [state]

        while state != self.goal_pos:
            action = np.argmax(self.Q_table[state])
            self.move(action)
            state = self.current_pos
            optimal_path.append(state)

        return optimal_path

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
        info = {}
        #
        if(reward == 1):
            #reward = 1 if self.current_pos == self.goal_pos else 0
            distance_to_goal = abs(self.current_pos[0] - self.goal_pos[0]) + abs(self.current_pos[1] - self.goal_pos[1])
            reward = 1 / (distance_to_goal + 1)
            done = self.current_pos == self.goal_pos

        else :
            reward = ""
            done = ""
            
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

        if self.is_valid_position(new_pos):
            self.current_pos = new_pos

    def is_valid_position(self, pos):
        row, col = pos

        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols:
            return False

        if self.maze[row, col] == '#':
            return False

        return True

    def render(self, mode='blank'):
        if mode == 'human':
            self.draw_maze()
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        if mode == 'blank':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

    def draw_maze(self):
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

        current_row, current_col = self.current_pos
        pygame.draw.rect(self.screen, self.PURPLE, (current_col * self.cell_size, current_row * self.cell_size, self.cell_size, self.cell_size))

    def close(self):
        pygame.quit()
        
    # variaveis monte carlo
    # https://towardsdatascience.com/reinforcement-learning-rl-101-with-python-e1aa0d37d43b
#    mc_gam = 0.6      # desconto do gama
#    mc_recom = -1     # recompensa
#    mc_col = 18
#    mc_lin = 10
#    mc_dest= [[16,3]] # destino
#    mc_acoe= [[-1,0],[1,0],[0,1],[0,-1]] # possiveis ações do entregador
#    mc_inte= 5000     # numero de interações

    # Estado inicial
#    Vari = np.zeros((mc_col, mc_lin))


#    returns = {(x, y):list() for x in range(mc_col) for y in range(mc_lin)}

#    deltas = {(x, y):list() for x in range(mc_col) for y in range(mc_lin)}

#    estado = [[x, y] for x in range(mc_col) for y in range(mc_lin)] 

#    def epocas ():
#        estado_inicial = random.choice(estado[1:-1])
#        epocas = []
#        while True:
#            if list(estado_inicial) in mc_dest:
#                return epocas
#            acao = random.choice(mc_acoe)
#            estado_final = np.array(estado_inicial) + numpy.array(acao)
#            #if mc_dest[0] in list(estado_final):
#            if -1 in list(estado_final) or estado_final[0] == 18 or estado_final[1] == 10:
#                estado_final = estado_inicial
#            epocas.append([list(estado_inicial), acao, mc_recom, list(estado_final)])
#            estado_inicial = estado_final
            
#    def start():
#        for z in tqdm(range(mc_inte)):
#            epoc = epocas()
#            ga = 0
#            for i, passo in enumerate(epoc[::-1]):
#                ga = mc_gam*ga + passo[2]
#                if passo[0] not in [x[0] for x in epoc[::-1][len(epoc)-i:]]:
#                    idx = (passo[0][0], passo[0][1])
#                    returns[idx].append(ga)
#                    newValue = numpy.average(returns[idx])
#                    deltas[idx[0], idx[1]].append(numpy.abs(Vari[idx[0], idx[1]]-newValue))
#                    Vari[idx[0], idx[1]] = newValue
                    
#            if z in [0,1,2,9, 99, mc_inte-1]:
#                print("Iteration {}".format(z+1))
#                print(Vari)
#                print("")


#def parse_args():
#    """
#    Parse arguments from command line input
#    """
#    parser = argparse.ArgumentParser(description='Learning parameters')
#    parser.add_argument('--agent', type=str, default='monte_carlo_es',
#                        help='The agent among monte_carlo_es, '
##                             'on_policy_first_visit_mc_control, '
 #                            'q_learning, sarsa.',
 #                       choices=['monte_carlo_es',
 #                                'on_policy_first_visit_mc_control',
#                                 'q_learning',
#                                 'sarsa'])
#    args, unknown = parser.parse_known_args()
#    return args


#def game():
#    # Argumento informado no comando
#    args = parse_args()

#    # Selecao do argumento
#    if args.agent == 'monte_carlo':
#        agent = MonteCarlo()
#    elif args.agent == 'q_learning':
#        agent = QLearning()
#    elif args.agent == 'sarsa':
#        agent = Sarsa()

#    agent.train()


#if __name__ == '__main__':
#    main()



if __name__ == "__main__":
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

    # ######################################
    # # TRECHO CHAMADA MONTE CARLO
    env = MazeGameEnv(maze, alpha=0.1, gamma=0.6, epsilon=0.1)
    env.MC(5000)

    # # Encontre o caminho ótimo do estado inicial ao estado de destino
    # optimal_path = env.find_optimal_path()
    # print("Optimal Path:", optimal_path)
    # ######################################

    # ######################################
    # # TRECHO CHAMADA QLEARN
    # env = MazeGameEnv(maze, alpha=0.1, gamma=0.99, epsilon=0.1)
    # env.train_q_learning(num_episodes=100)

    # # Encontre o caminho ótimo do estado inicial ao estado de destino
    # optimal_path = env.find_optimal_path()
    # print("Optimal Path:", optimal_path)
    # ######################################

    ######################################
    # TRECHO BASE - BLOCO RANDOMICO
    #env = MazeGameEnv(maze)
    #done = False
    #state = env.reset()
    #env.render()

    #while not done:
    #    action = env.action_space.sample()  # Exemplo: selecione uma ação aleatória
    #    state, reward, done, _ = env.step(action)
    #    print(state, reward, done)
    #    env.render()
    #    #pygame.time.delay(500)
    ######################################

    print("Done!")
    env.close()
