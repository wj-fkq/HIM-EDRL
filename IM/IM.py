from time import process_time_ns

import numpy as np, os, time, random
import torch
from numpy.ma.core import indices
# from requests.packages import target
from sympy.physics.units import current
from torch.optim import Adam
import torch.nn as nn
# from torch_geometric.typing import weight

import Replay_Memory
from Influence_Propagation import Env
from Evolutionary_Algorithm import EA
from DDQN_Model import DDQN
import Mod_Utils as Utils

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
generation = 80  # iterations


class Agent:
    def __init__(self, env,
                 per_alpha=0.6,          # PER α
                 gamma=0.95,             # 折扣因子
                 n_step=3,               # n-step
                 beta_mode='fixed',      # 'fixed' 或 'anneal'
                 beta_fixed=1.0,         # 固定 β 的值
                 beta0=0.4,              # 退火起点
                 beta_final=1.0,         # 退火终点
                 beta_frames=100000):    # 退火步数
        self.is_cuda = True  # Set to use CUDA
        self.is_memory_cuda = True  # Set to use GPU graphics memory
        self.batch_size = 512  # Set batch size
        self.use_done_mask = True
        self.pop_size = 100  # Set population size
        self.buffer_size = 10000  # Set cache pool size
        self.alpha = float(per_alpha)
        self.beta_mode = beta_mode
        self.beta_fixed = float(beta_fixed)
        self.beta_start = float(beta0)
        self.beta_final = float(beta_final)
        self.beta_frames = int(beta_frames)
        self.discount_gamma = float(gamma)
        self.n_step = int(n_step)
        self.randomWalkTimes = 20  # Set the number of random point selections based on DQN
        self.learningTimes = 3  # Set the number of times to accelerate DQN training based on DRL technology
        self.dim = env.dim  # Set the dimension of the DQN input layer
        self.env = env  # Initialize the impact propagation environment
        self.evalStep = 1  # Set DQN point selection times based on the number of seed nodes
        self.evolver = EA(self.pop_size)  # initialization

    def initPop(self):
        # Initialize DQN population
        self.pop = []
        for _ in range(self.pop_size):
            self.pop.append(DDQN(self.dim).cuda())
            # self.pop.append(DDQN(self.dim))
        # Initialize the fitness array corresponding to the DQN population
        self.all_fitness = []
        # Turn off gradients and put in eval mode
        for dqn in self.pop:
            dqn.eval()
        # Initialize optimal DDQN
        self.rl_agent = DDQN(self.dim)
        self.target_agent = DDQN(self.dim)
        self.sync_target_network()
        # Initial DDQN parameters
        self.gamma = self.discount_gamma  # Set update ratio
        self.optim = Adam(self.rl_agent.parameters(), lr=0.001)  # Set learners
        self.loss = nn.MSELoss()  # Set to use mean square error as the loss function
        self.replay_buffer = Replay_Memory.PrioritizedReplayMemory(self.buffer_size, alpha=self.alpha, n_step=self.n_step, gamma=self.discount_gamma)  # Initialize buffer pool
        # Initialize tracker parameters
        self.num_games = 0
        self.num_frames = 0
        self.gen_frames = 0

    def current_beta(self):
        if self.beta_mode == 'fixed':
            return self.beta_fixed
        # 线性退火
        frac = min(1.0, self.num_frames / float(self.beta_frames))
        return self.beta_start + frac * (self.beta_final - self.beta_start)
    
    def sync_target_network(self):
        self.target_agent.load_state_dict(self.rl_agent.state_dict())

    def soft_update_target_netmork(self, tau=0.01):
        for target_param, param in zip(self.target_agent.parameters(), self.rl_agent.parameters()):
            target_param.data.copy_(tau*param.data+(1.0-tau)*target_param.data)

    # Store training quadruple data into a cache pool based on CUDA technology #
    def add_experience(self, state, action, next_state, reward, done):
        reward = Utils.to_tensor(np.array([reward])).unsqueeze(0)
        if self.is_cuda: reward = reward.cuda()
        if self.use_done_mask:
            done = Utils.to_tensor(np.array([done]).astype('uint8')).unsqueeze(0)
            if self.is_cuda: done = done.cuda()

        self.replay_buffer.push(state, action, next_state, reward, done, td_error=1.0)

    # Select seed nodes based on the node score output by DQN and calculate the fitness value, while caching the quadruple data during the selection process #
    def evaluate(self, net, store_transition=True):
        total_reward = 0.0
        state = self.env.reset()
        state = Utils.to_tensor(state).unsqueeze(0)
        if self.is_cuda:
            state = state.cuda()
        done = False
        seeds = []
        while not done:
            if store_transition: self.num_frames += 1; self.gen_frames += 1
            Qvalues = net.forward(state)
            Qvalues = Qvalues.reshape((Qvalues.numel(),))
            sorted, indices = torch.sort(Qvalues, descending=True)

            actionNum = 0

            for i in range(state.shape[1]):
                if state[0][indices[i]][0].item() >= 0:
                    actionNum += 1
                    actionInt = indices[i].item()
                    seeds.append(actionInt)
                    action = torch.tensor([actionInt])
                    next_state, reward, done = self.env.step(actionInt)
                    next_state = Utils.to_tensor(next_state).unsqueeze(0)

                    if self.is_cuda:
                        next_state = next_state.cuda()
                    total_reward += reward
                    if store_transition:
                        self.add_experience(state.cpu(), action, next_state.cpu(), reward, done)
                    state = next_state
                    if actionNum == self.evalStep or done:
                        break

        if store_transition: self.num_games += 1

        return total_reward, seeds

    # Copy the network weights of the source DQN to the network weights of the target DQN #
    def rl_to_evo(self, rl_net, evo_net):
        for target_param, param in zip(evo_net.parameters(), rl_net.parameters()):
            target_param.data.copy_(param.data)

    # Evaluate the fitness value of the evolved DDQN population #
    def evaluate_all(self):
        self.all_fitness = []
        t1 = time.time()
        for net in self.pop:
            fitness, _ = self.evaluate(net)

            self.all_fitness.append(fitness)
        print("all:", self.all_fitness)
        best_train_fitness = max(self.all_fitness)
        print("fitness_init:", best_train_fitness)
        t2 = time.time()
        print("evaluate finished.    cost time:", t2 - t1)

    # Training Evolutionary DQN Population Based on Evolutionary Algorithms and DRL Technology #
    def train(self):
        self.gen_frames = 0
        ####################### EVOLUTION #####################
        t1 = time.time()

        # Obtain optimal fitness value
        best_train_fitness = max(self.all_fitness)
        # Evolutionary algorithms are used to evolve the network weights of the DDQN population and update the fitness values of the new population
        new_pop = self.evolver.epoch(self.pop, self.all_fitness)
        new_pop_fitness = []
        for net in new_pop:
            fitness, _ = self.evaluate(net)
            new_pop_fitness.append(fitness)
        self.pop, self.all_fitness = self.get_offspring(self.pop, self.all_fitness, new_pop, new_pop_fitness)
        t2 = time.time()
        print("epoch finished.    cost time:", t2 - t1)
        # Obtain the optimal fitness value of the current DDQN population
        fitness_best, _ = self.evaluate(self.pop[0], True)

        ####################### DRL Learning #####################
        # rl learning step
        t1 = time.time()
        # Based on the n-step Q-learning technique in DRL thinking, the optimal DDQN is reverse updated using empirical data from the cache pool, and its network weights are copied to the DQN with poor fitness values
        for _ in range(self.learningTimes):
            index = random.randint(len(self.pop) // 2, len(self.pop) - 1)
            self.rl_to_evo(self.pop[0], self.rl_agent)
            if len(self.replay_buffer) > self.batch_size * 2:
                beta = self.beta_start + self.num_frames * (1.0 - self.beta_start) / self.beta_frames
                beta = min(1.0, beta)
                sampled_data = self.replay_buffer.sample(self.batch_size, beta)
                samples = sampled_data["samples"]
                indices = sampled_data["indices"]
                weights = sampled_data["weights"]
                batch = Replay_Memory.Transition(*zip(*samples))
                self.update_parameters(batch,indices,weights)
                fitness, _ = self.evaluate(self.rl_agent, True)
                if fitness_best < fitness:
                    self.rl_to_evo(self.rl_agent, self.pop[index])
                    self.all_fitness[index] = fitness

        t2 = time.time()
        print("learning finished.    cost time:", t2 - t1)
        return best_train_fitness, sum(self.all_fitness) / len(self.all_fitness), self.rl_agent, self.pop[
                                                                                                 0:len(self.pop) // 10]

    # Based on the n-step Q-learning technique in DRL thinking, error values are calculated using empirical data of specific batch sizes in the cache pool, and the network weights of DQN are updated in reverse using error value gradients using random gradient descent technique #
    def update_parameters(self, batch, indices, weights):
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = None
        if self.use_done_mask: done_batch = torch.cat(batch.done)

        state_batch.requires_grad = False
        next_state_batch.requires_grad = True
        action_batch.requires_grad = False

        # Load everything to GPU if not already
        if self.is_cuda:
            self.rl_agent.cuda()
            self.target_agent.cuda()
            state_batch = state_batch.cuda();
            next_state_batch = next_state_batch.cuda();
            action_batch = action_batch.cuda();
            reward_batch = reward_batch.cuda()
            if self.use_done_mask: done_batch = done_batch.cuda()

        action_batch = action_batch.unsqueeze(1)
        q_values_all = self.rl_agent(state_batch).squeeze(-1)
        q_values = q_values_all.gather(1, action_batch).squeeze(1)


        with torch.no_grad():
            next_q_values_all = self.rl_agent(next_state_batch).squeeze(-1)
            next_actions = next_q_values_all.max(1)[1].unsqueeze(1)
            next_target_q_values = self.target_agent(next_state_batch).squeeze(-1)
            next_q_values = next_target_q_values.gather(1, next_actions).squeeze(1)
            gamma_n = self.discount_gamma ** self.replay_buffer.n_step
            target_q_values = reward_batch + (1 - done_batch.float()) * gamma_n * next_q_values

        td_errors = (q_values-target_q_values).detach().cpu().numpy().squeeze()
        self.replay_buffer.update_priorities(indices, abs(td_errors))

        weights = torch.tensor(weights).cuda() if self.is_cuda else torch.tensor(weights)
        loss = (weights * (q_values - target_q_values) ** 2).mean()

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.rl_agent.parameters(), 20000)
        self.optim.step()

        self.soft_update_target_netmork(tau=0.01)

        # Nets back to CPU if using memory_cuda
        if self.is_memory_cuda and not self.is_cuda:
            self.rl_agent.cpu()
            self.target_agent.cpu()

        # return td_errors

    # In the DDQN population that has evolved beyond the population size limit, first sort all DDQN populations, then select the top 50 DQN populations to be retained, and randomly select 50 populations from the remaining populations to be retained #
    def get_offspring(self, pop, fitness_evals, new_pop, new_fitness_evals):
        all_pop = []
        fitness = []
        offspring = []
        offspring_fitness = []
        for i in range(len(pop)):
            all_pop.append(pop[i])
            fitness.append(fitness_evals[i])
        for i in range(len(new_pop)):
            all_pop.append(new_pop[i])
            fitness.append(new_fitness_evals[i])

        index_rank = sorted(range(len(fitness)), key=fitness.__getitem__)
        index_rank.reverse()
        for i in range(len(pop) // 2):
            offspring.append(all_pop[index_rank[i]])
            offspring_fitness.append(fitness[index_rank[i]])

        randomNum = len(all_pop) - len(pop) // 2
        randomList = list(range(randomNum))
        random.shuffle(randomList)
        for i in range(len(pop) // 2, len(pop)):
            index = randomList[i - len(pop) // 2]
            offspring.append(all_pop[index])
            offspring_fitness.append(fitness[index])
            ...

        return offspring, offspring_fitness

    # Format output fitness score #
    def showScore(self, score):
        out = ""
        for i in range(len(score)):
            out = out + str(score[i])
            out = out + "\t"
        print(out)


def run(maxSeedsNum):
    # Create Env
    t1 = time.time()

    env = Env(maxSeedsNum)
    seed = 123
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create Agent
    agent = Agent(env)
    print("====================================New Start========================================")
    resultList = np.array([])
    timeList = np.array([])
    thresholdNum = 1
    avg_list = []
    best_list = []


    for thresholdIndex in range(thresholdNum):
        graphIndex = 0
        agent.initPop()
        print("**********************************************************************************")
        for i in range(generation):  # Generation
            if i == 0:
                agent.evaluate_all()
            print("=================================================================================")
            print(
                graphIndex, "th graph      Generation:", i, "    ",
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "Threshold Index:", thresholdIndex + 1)
            # Conduct training
            best_train_fitness, average, rl_agent, elitePop = agent.train()
            print('#Games:', agent.num_games, '#Frames:', agent.num_frames, ' Epoch_Max:', '%.2f' % best_train_fitness,
                  ' Avg:', average, "influence:", best_train_fitness)
            avg_list.append(average)
            best_list.append(best_train_fitness)

        fitness, seeds = agent.evaluate(agent.pop[0])
        print("#######################RESULT######################")
        print("best fitness:", fitness)
        print("seeds:", seeds)
        print("len of seeds", len(seeds))
        print("#######################RESULT######################")
        resultList = np.append(resultList, fitness)
        t2 = time.time()
        timeList = np.append(timeList, t2 - t1)
        t1 = t2

        if graphIndex < thresholdNum - 1:
            agent.env.nextThreshold()
        else:
            break
    print("time cost:")
    agent.showScore(timeList)
    print("influence:")
    agent.showScore(resultList)
    print("avg list:", avg_list)
    print('best_list:', best_list)


if __name__ == "__main__":
    for i in range(1, 5):
        seedNum = i * 10
        print("seedNum:", seedNum)
        run(seedNum)
