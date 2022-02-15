import numpy as np
import random
import copy
from collections import namedtuple, deque
from functools import reduce

from agent_model import Actor, Critic, RecurrentNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

ir = 0.01
WORKDAY = 250
dt = 1/WORKDAY
BUFFER_SIZE = int(2000)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = np.exp(-ir*dt)            # discount factor
TAU = 0.1              # for soft update of target parameters
LR_ACTOR = 0.0005        # learning rate of the actor
LR_CRITIC = 0.0005        # learning rate of the critic
LR_RECURRENT = 0.001
WEIGHT_DECAY = 0.0001   # L2 weight decay

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, encoding_size, action_size, random_seed, cuda_id=0,
                 recurrent=True, add_supervise=True, supervise_method='delta', bc=True, lambda_a=5, per_alpha=0.5,
                 is_per=True):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self._device = torch.device("cuda:"+str(cuda_id) if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.encoding_size = encoding_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self._count = 0

        self.recurrent = recurrent

        self.state_encoding = RecurrentNetwork(input_size=state_size, encoding_size=encoding_size,
                                               add_supervise=add_supervise).to(self._device)
        self.state_encoding_optimizer = optim.Adam(self.state_encoding.parameters(), lr=LR_RECURRENT)

        self.supervise_method = supervise_method
        self.bc = bc
        self.lambda_a = lambda_a
        self.per_alpha = per_alpha
        self.is_per = is_per

        # Actor Network (w/ Target Network)
        if recurrent is True:
            self.actor_local = Actor(encoding_size, action_size, random_seed).to(self._device)
            self.actor_target = Actor(encoding_size, action_size, random_seed).to(self._device)

        else:
            self.actor_local = Actor(state_size, action_size, random_seed).to(self._device)
            self.actor_target = Actor(state_size, action_size, random_seed).to(self._device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        #self.actor_optimizer = optim.Adam([{'params': list(self.actor_local.parameters()), 'lr': LR_ACTOR},
        #                                   {'params': list(self.state_encoding.parameters()), 'lr': LR_RECURRENT}])


        # Critic Network (w/ Target Network)
        if recurrent is True:
            self.critic_local = Critic(encoding_size, action_size, random_seed).to(self._device)
            self.critic_target = Critic(encoding_size, action_size, random_seed).to(self._device)

        else:
            self.critic_local = Critic(state_size, action_size, random_seed).to(self._device)
            self.critic_target = Critic(state_size, action_size, random_seed).to(self._device)

        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        #self.critic_optimizer = optim.Adam([{'params': list(self.critic_local.parameters()), 'lr': LR_CRITIC},
        #                                    {'params': list(self.state_encoding.parameters()), 'lr': LR_RECURRENT}])

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        if self.recurrent:
            self.memory = TrajectoryReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed, self._device)
            self.trajectory_list = []
            self.one_time_step = namedtuple('onetimestep', field_names=['state', 'action', 'reward'])
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed, self._device)

        # add supervise
        self.add_supervise = add_supervise

        self.per = PER(BUFFER_SIZE, BATCH_SIZE, lambda_a=self.lambda_a, alpha=self.per_alpha)

        self.supervise_loss_array = np.array([])


    def step(self, state, action, reward, next_state, done, delta_array=None, hi_holding_array=None, vol_array=None):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        if self.recurrent:
            self.trajectory_list.append(self.one_time_step(state, action, reward))

            if done:
                if self.add_supervise is False:
                    self.memory.add(self.trajectory_list)
                    critic_loss, actor_loss = self.get_actor_critic_error(self.trajectory_list, GAMMA,
                                                                          delta_array=delta_array, bc=self.bc)

                else:

                    if hi_holding_array is not None:
                        if vol_array is None:
                            self.memory.add([self.trajectory_list, (delta_array, hi_holding_array)])
                            critic_loss, actor_loss, actor_loss2 = self.get_actor_critic_error(self.trajectory_list, GAMMA,
                                                                                               delta_array=(delta_array, hi_holding_array),
                                                                                               bc=self.bc)
                            # print(f'ac = {actor_loss}, bc={actor_loss2}')
                            actor_loss = self.get_actor_loss_with_bc(actor_loss, actor_loss2, in_step=False)

                        else:
                            self.memory.add([self.trajectory_list, (delta_array, hi_holding_array, vol_array)])
                            critic_loss, actor_loss, actor_loss2 = self.get_actor_critic_error(self.trajectory_list, GAMMA,
                                                                                               delta_array=(delta_array, hi_holding_array),
                                                                                               bc=self.bc)
                            # print(f'ac = {actor_loss}, bc={actor_loss2}')
                            actor_loss = self.get_actor_loss_with_bc(actor_loss, actor_loss2, in_step=False)


                    else:
                        if vol_array is None:
                            self.memory.add([self.trajectory_list, delta_array])
                            critic_loss, actor_loss, actor_loss2 = self.get_actor_critic_error(self.trajectory_list, GAMMA,
                                                                                               delta_array=delta_array, bc=self.bc)
                            # print(f'ac = {actor_loss}, bc={actor_loss2}')
                            actor_loss = self.get_actor_loss_with_bc(actor_loss, actor_loss2, in_step=False)
                            # print(f'trajectory_list : {[self.trajectory_list[i].action for i in range(len(self.trajectory_list))]}',
                            #       f'delta_array : {delta_array}')
                        else:
                            self.memory.add([self.trajectory_list, (delta_array, vol_array)])
                            critic_loss, actor_loss, actor_loss2 = self.get_actor_critic_error(self.trajectory_list,
                                                                                               GAMMA,
                                                                                               delta_array=delta_array,
                                                                                               bc=self.bc)
                            # print(f'ac = {actor_loss}, bc={actor_loss2}')
                            actor_loss = self.get_actor_loss_with_bc(actor_loss, actor_loss2, in_step=False)
                            # print(f'trajectory_list : {[self.trajectory_list[i].action for i in range(len(self.trajectory_list))]}',
                            #       f'delta_array : {delta_array}')

                per_error = self.per.get_per_error(critic_loss, actor_loss)
                # print(f'critic : {critic_loss}, actor : {actor_loss}, per : {per_error}')
                self.per.add(per_error)

                self.trajectory_list = []

                if len(self.memory) > BATCH_SIZE:
                    if self.is_per is True:
                        experiences = self.memory.sample(is_per=self.is_per, per=self.per)
                    else:
                        experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

        else:
            # Save experience / reward
            self.memory.add(state, action, reward, next_state, done)

            # Learn, if enough samples are available in memory
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self._device)
        if self.recurrent:
            state = state.reshape(1, 1, -1)
            if len(self.trajectory_list) > 0:
                state_history = list(map(lambda x: x.state.tolist(), self.trajectory_list.copy()))
                state_history = torch.tensor(state_history, dtype=torch.float, device=self._device).unsqueeze(0)
                state = torch.cat((state_history, state), dim=1)

            self.state_encoding.eval()
            encoded_state = self.state_encoding(state).reshape(-1)
            # print(f"STATE : {state} ENCODED STATE :{encoded_state}")
        self.actor_local.eval()
        with torch.no_grad():
            if self.recurrent:
                action = self.actor_local(encoded_state).cpu().data.numpy()
                self.state_encoding.train()
            else:
                action = self.actor_local(state).cpu().data.numpy()

        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return action

    def test_step(self, state, action, reward, done):
        if self.recurrent:
            self.trajectory_list.append(self.one_time_step(state, action, reward))
        if done:
            self.trajectory_list = []

    def reset(self):
        self.noise.reset()

    def get_actor_critic_error(self, trajectory, gamma, delta_array=None, bc=False):

        num_trajectory = len(trajectory)
        num_state_dim = len(trajectory[0].state)

        state_tensor = torch.tensor(reduce(lambda x, y: np.vstack([x, y.state]), trajectory,
                                           np.empty((0, num_state_dim))), device=self._device,
                                    requires_grad=True).float()
        state_tensor = state_tensor.reshape(1, num_trajectory, -1)
        encoded_state_tensor = self.sequential_state_encode(state_tensor).reshape(num_trajectory, self.encoding_size)
        if len(trajectory[0].action) == 1:
            action_tensor = torch.tensor(list(map(
                lambda x: [torch.tensor([x.action], requires_grad=True)], trajectory)), requires_grad=True).to(
                self._device)

        else:
            action_tensor = torch.tensor(list(map(
                lambda x: x.action, trajectory)), requires_grad=True).to(
                self._device)

        reward_tensor = torch.tensor(list(map(
            lambda x: [torch.tensor([x.reward])], trajectory))).to(self._device)
        if type(delta_array) == tuple:
            underlying_holding_num, hi_holding_num = delta_array
            underlying_num_tensor = torch.tensor(underlying_holding_num, dtype=torch.float32, requires_grad=False).to(self._device)
            hi_num_tensor = torch.tensor(hi_holding_num, dtype=torch.float32, requires_grad=False).to(self._device)
        else:
            delta_tensor = torch.tensor(delta_array, dtype=torch.float32, requires_grad=False).to(self._device)

        actions_next = self.actor_target(encoded_state_tensor[1:])

        Q_expected = self.critic_local(encoded_state_tensor[:], action_tensor[:], recurrent=self.recurrent)
        Q_targets_next = self.critic_target(encoded_state_tensor[1:], actions_next, recurrent=self.recurrent)
        Q_targets_next = torch.cat(
            (Q_targets_next, torch.tensor([[0]], dtype=torch.float64, requires_grad=False).to(self._device)))
        if torch.isnan(Q_expected).any() or torch.isnan(Q_targets_next).any():
            print(f"state : {state_tensor} / encoded_state_tensor : {encoded_state_tensor}")
            print(f"actions : {action_tensor} / actions_next : {actions_next}")
            # print(f"Q_expected : {Q_expected} / Q_ target : {Q_targets_next}")

        Q_targets = reward_tensor + gamma * Q_targets_next

        critic_loss = F.mse_loss(Q_expected, Q_targets).mean()

        actions_pred = self.actor_local(encoded_state_tensor)
        actor_loss = - self.critic_local(encoded_state_tensor, actions_pred, recurrent=self.recurrent).mean()

        if bc is True:
            ####### BEHAVIOR CLONING #######
            if type(delta_array) == tuple:
                action_tensor = torch.cat((underlying_num_tensor.unsqueeze(1), hi_num_tensor.unsqueeze(1)), 1)
                Q_delta = self.critic_target(encoded_state_tensor[:], action_tensor.reshape(-1, 2),
                                             recurrent=self.recurrent)
            else:
                Q_delta = self.critic_target(encoded_state_tensor[:], delta_tensor.reshape(-1, 1), recurrent=self.recurrent)
            Q_diff = Q_delta - Q_expected
            if sum(Q_diff > 0) == 0:
                actor_loss2 = 0
            else:
                if type(delta_array) == tuple:

                    actor_loss2 = F.mse_loss(actions_pred[(Q_diff > 0).squeeze()], action_tensor.reshape(-1, 2)[(Q_diff > 0).squeeze()],0)
                else:
                    actor_loss2 = F.mse_loss(actions_pred[Q_diff > 0], delta_tensor.reshape(-1, 1)[Q_diff > 0]).mean()
            return critic_loss, actor_loss, actor_loss2

        else:
            return critic_loss, actor_loss

    def learn(self, experiences, gamma):
        if self.recurrent:
            num_episode = len(experiences)

            sum_critic_loss = 0
            sum_actor_loss = 0
            sum_supervise_loss = 0
            sum_actor_loss2 = 0

            # supervise_loss
            if self.add_supervise is True:
                for i, trajectory in enumerate(experiences):
                    importance_sampling_weight = float(self.per.importance_sampling_weight[i])
                    trajectory, delta_array = trajectory
                    num_trajectory = len(trajectory)
                    num_state_dim = len(trajectory[0].state)
                    delta_tensor = torch.tensor(delta_array, dtype=torch.float32, requires_grad=False).to(self._device)
                    state_tensor = torch.tensor(reduce(lambda x, y: np.vstack([x, y.state]), trajectory,
                                                       np.empty((0, num_state_dim))), device=self._device,
                                                requires_grad=True).float()
                    state_tensor = state_tensor.reshape(1, num_trajectory, -1)
                    reward_tensor = torch.tensor(list(map(lambda x: [torch.tensor([x.reward])],
                                                          trajectory))).to(dtype=torch.float32, device=self._device).reshape(-1)
                    if self.supervise_method == 'delta':
                        delta_predict = self.state_encoding(state_tensor, supervise_method='delta').reshape(-1)
                        # if i == 1:
                        #     print(f'predict : {delta_predict}')
                        #     print(f'target : {delta_tensor}')

                        delta_loss = F.mse_loss(delta_predict, delta_tensor).mean()

                        if self.is_per is True:
                            sum_supervise_loss += importance_sampling_weight * delta_loss

                        else:
                            sum_supervise_loss += delta_loss

                    elif self.supervise_method == 'deltagamma':
                        delta_predict, gamma_predict = torch.transpose(self.state_encoding(
                            state_tensor, supervise_method='deltagamma').reshape(-1, 2), 0, 1)
                        delta_tensor, gamma_tensor = delta_tensor
                        delta_loss = F.mse_loss(delta_predict, delta_tensor).mean()

                        gamma_loss = F.mse_loss(gamma_predict, gamma_tensor).mean()

                        if self.is_per is True:
                            sum_supervise_loss += importance_sampling_weight * delta_loss
                            sum_supervise_loss += importance_sampling_weight * gamma_loss

                        else:
                            sum_supervise_loss += delta_loss
                            sum_supervise_loss += gamma_loss

                        # if torch.isnan(delta_loss):
                        #     print(delta_predict)

                    elif self.supervise_method == 'dual':
                        delta_predict, reward_predict = torch.transpose(self.state_encoding(
                            state_tensor, supervise_method='dual').reshape(-1, 2), 0, 1)
                        delta_loss = F.mse_loss(delta_predict, delta_tensor).mean()

                        reward_loss = F.mse_loss(reward_predict, reward_tensor).mean()

                        if self.is_per is True:
                            sum_supervise_loss += importance_sampling_weight * delta_loss
                            sum_supervise_loss += importance_sampling_weight * reward_loss

                        else:
                            sum_supervise_loss += delta_loss
                            sum_supervise_loss += reward_loss

                    elif self.supervise_method == 'vol':
                        vol_predict = self.state_encoding(state_tensor, supervise_method='delta').reshape(-1)
                        # if i == 1:
                        #     print(f'predict : {delta_predict}')
                        #     print(f'target : {delta_tensor}')
                        vol_tensor = delta_tensor[-1]
                        vol_loss = F.mse_loss(vol_predict, vol_tensor).mean()

                        if self.is_per is True:
                            sum_supervise_loss += importance_sampling_weight * vol_loss

                        else:
                            sum_supervise_loss += vol_loss

                supervise_loss = sum_supervise_loss / num_episode
                print(supervise_loss)
                self.state_encoding_optimizer.zero_grad()
                supervise_loss.backward()
                self.state_encoding_optimizer.step()
                self.state_encoding.lr_control(self.state_encoding_optimizer)
                self.supervise_loss_array = np.append(self.supervise_loss_array, supervise_loss.detach().cpu().numpy())

                #print(f'\n delta_predict : {delta_predict} | delta_tensor : {delta_tensor}')
                #print(f'\n reward_predict : {reward_predict} | reward_tensor : {reward_tensor}')


            # RL method
            for i, trajectory in enumerate(experiences):
                if self.add_supervise is True:
                    trajectory, delta_array = trajectory
                if self.supervise_method is 'vol':
                    delta_array = delta_array[:-1]
                    if type(delta_array) == tuple and len(delta_array) == 1:
                        delta_array = delta_array[0]
                critic_loss, actor_loss, actor_loss2 = self.get_actor_critic_error(
                    trajectory, GAMMA, delta_array, bc=self.bc)

                new_per_error = self.per.get_per_error(critic_loss, actor_loss)
                # print(f'critic_loss : {critic_loss}')
                # print(f'actor_loss : {actor_loss}')
                # print(f'per_error : {new_per_error}')
                self.per.error_update(new_per_error, i)

                importance_sampling_weight = float(self.per.importance_sampling_weight[i])

                if self.is_per is True:
                    sum_critic_loss += importance_sampling_weight * critic_loss
                    sum_actor_loss += importance_sampling_weight * actor_loss

                else:
                    sum_critic_loss += critic_loss
                    sum_actor_loss += actor_loss

                if self.bc is True:
                    if self.is_per is True:
                        sum_actor_loss2 += importance_sampling_weight * actor_loss2
                    else:
                        sum_actor_loss2 += actor_loss2

            # critic loss
            expected_critic_loss = sum_critic_loss/num_episode

            # actor loss
            if self.bc is True:
                expected_actor_loss = self.get_actor_loss_with_bc(
                    sum_actor_loss, sum_actor_loss2, in_step=True) / num_episode

            else :
                expected_actor_loss = sum_actor_loss/num_episode

            # print(f'actor_loss : {expected_actor_loss}, critic_loss : {expected_critic_loss}')

            # optimizer update
            self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            expected_critic_loss.backward(retain_graph=True)
            expected_actor_loss.backward()
            # print(f"critic_gradient1 : {self.critic_local.fc5.weight.grad}")
            # print(f"actor_gradient1 : {self.actor_local.fc5.weight.grad}")
            self.critic_optimizer.step()
            self.actor_optimizer.step()

            self.soft_update(self.critic_local, self.critic_target, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)

        else:
            states, actions, rewards, next_states, dones = experiences

            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            actions_next = self.actor_target(next_states)
            Q_targets_next = self.critic_target(next_states, actions_next)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            # Compute critic loss
            Q_expected = self.critic_local(states, actions)
            critic_loss = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss = - self.critic_local(states, actions_pred).mean()  # to minimize we think
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local, self.critic_target, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)

    def get_actor_loss_with_bc(self, actor_loss, bc_loss, in_step=False):
        lambda1 = max(0.5, 0.9 - self._count * 0.000005)  # 0.9부터 0.4까지
        if in_step is True:
            self._count += 1
        total_actor_loss = ((1 - lambda1) * actor_loss + lambda1 * bc_loss)
        return total_actor_loss

    def sequential_state_encode(self, state_tensor):
        output = torch.empty(0, dtype=torch.float, device=self._device)
        truncated_batch_dim_state = state_tensor.squeeze()
        state_num = len(truncated_batch_dim_state)
        for i in range(state_num):
            sequential_encoded_state = self.state_encoding(truncated_batch_dim_state[:i+1].unsqueeze(0))
            output = torch.cat((output, sequential_encoded_state))

        output = output.reshape(1, state_num, -1)
        return output

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.2, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        self.changed_sigma = copy.copy(self.sigma)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.changed_sigma * np.array([random.gauss(0,1) for i in range(len(x))])
        self.state = x + dx
        self.changed_sigma = self.changed_sigma * 0.99999
        return self.state


class ReplayBuffer(object):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, is_per=False, per=None):
        """Randomly sample a batch of experiences from memory."""
        if is_per is False:
            experiences = random.sample(self.memory, k=self.batch_size)
        else:
            experiences = per.sample(self.memory)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class TrajectoryReplayBuffer(ReplayBuffer):

    def __init__(self, *args):
        super(TrajectoryReplayBuffer, self).__init__(*args)

    def add(self, trajectory_list):
        trajectory = trajectory_list.copy()
        self.memory.append(trajectory)

    def sample(self, is_per=False, per=None):
        if is_per is False:
            experiences = random.sample(self.memory, k=self.batch_size)
        else:
            experiences = per.sample(self.memory)

        return experiences


class PER(object):

    def __init__(self, buffer_size, batch_size, lambda_a, alpha):
        self.errors = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.epsilon = 0.05
        self.lambda_a = lambda_a
        self.alpha = alpha
        self.beta = 0
        self.probs_tensor = torch.tensor(0)
        self.importance_sampling_weight = torch.tensor(0)
        self.output_deque_index = []

    def add(self, error):
        p_i = abs(error) + self.epsilon
        self.errors.append(p_i)

    def get_per_error(self, critic_loss, actor_loss):
        per_error = critic_loss + self.lambda_a * actor_loss
        return per_error

    def error_update(self, error, index):
        self.errors[self.output_deque_index[index]] = abs(error) + self.epsilon

    def get_importance_sampling_weight(self, replay_buffer):
        n = len(replay_buffer)
        IS_weight = (self.probs_tensor * n) ** (-self.beta)
        IS_weight = IS_weight / IS_weight.max()
        self.beta_update()
        return IS_weight

    def beta_update(self):
        if self.beta <1:
            self.beta += 0.00067  # 1/1500
        elif self.beta > 1:
            self.beta == 1

    def sample(self, replay_buffer):
        errors = torch.FloatTensor(self.errors.copy())
        errors = errors**self.alpha
        sorted_index = torch.sort(errors).indices
        sorted_errors_array = torch.sort(errors).values
        cumsum_err = torch.cumsum(sorted_errors_array, dim=0)
        random_number_col_vector = torch.rand(self.batch_size, 1) * torch.sum(errors)
        output_index = len(errors) - torch.sum(random_number_col_vector < cumsum_err, axis=1)
        # print(f'sorted_index : {sorted_errors_array}')
        # print(f'output_index : {cumsum_err}')
        output_deque_index = sorted_index[output_index].tolist()

        output = [replay_buffer[index] for index in output_deque_index]

        sum_err = cumsum_err[-1]
        self.probs_tensor = errors[sorted_index[output_index]] / sum_err
        self.output_deque_index = output_deque_index
        self.importance_sampling_weight = self.get_importance_sampling_weight(replay_buffer)

        return output