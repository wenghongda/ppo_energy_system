from model import Actor,Critic
import torch.optim as optim
from parameters import *
import torch
import numpy as np
import torch.nn as nn

device = torch.device('cuda:0')
torch.cuda.empty_cache()
################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = N_A
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self,N_S,N_A):
        self.actor_net = Actor(N_S,N_A)
        self.actor_old_net = Actor(N_S,N_A)
        self.critic_net = Critic(N_S)
        self.actor_optim = optim.Adam(self.actor_net.parameters(),lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic_net.parameters(),lr=lr_critic,weight_decay=lr_rate)
        self.critic_loss_func = torch.nn.MSELoss()
        self.actor_old_net.load_state_dict(self.actor_net.state_dict())

    def train(self,memory):
        memory = np.array(memory,dtype="object")
        states = torch.tensor(np.vstack(memory[:,0]),dtype=torch.float32)

        # the shape of states are (length of memory,1)
        actions = torch.tensor(np.vstack(memory[:,1]),dtype=torch.float32)
        rewards = torch.tensor(np.vstack(memory[:,2]),dtype=torch.float32).squeeze(1)
        masks = torch.tensor(np.vstack(memory[:,3]),dtype=torch.float32).squeeze(1)
        values = self.critic_net(states)
        # Normalizing the rewards
        #rewards = torch.tensor(rewards,dtype=torch.float32)
        #rewards = (rewards-rewards.mean())/(rewards.std() + 1e-7)

        returns,advants = self.get_gae(rewards,masks,values)
        self.actor_old_net.load_state_dict(self.actor_net.state_dict())
        old_mu,old_std = self.actor_old_net(states)
        pi = self.actor_old_net.distribution(old_mu,old_std)
        old_log_prob = pi.log_prob(actions).sum(1,keepdims=True)

        n = len(states)
        arr = np.arange(n)
        for epoch in range(5):
            #np.random.shuffle(arr)
            for i in range(n//batch_size):
                b_index = arr[batch_size*i:batch_size*(i+1)]
                b_states = states[b_index] # the shape of b_states (batch_size,1)
                b_advants = advants[b_index].unsqueeze(1) # b_advants.shape:(batch_size,1)
                b_actions = actions[b_index]              # b_actions.shape:(batch_size,)
                b_returns = returns[b_index].unsqueeze(1) # b_returns.shape:(batch_size,1)
                mu,std = self.actor_net(b_states)
                pi = self.actor_net.distribution(mu,std)
                new_prob = pi.log_prob(b_actions).sum(1,keepdims=True)
                old_prob = old_log_prob[b_index].detach()

                ratio =torch.exp(new_prob-old_prob)

                surrogate_loss = ratio*b_advants

                values = self.critic_net(b_states)
                critic_loss = self.critic_loss_func(values,b_returns)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                ratio = torch.clamp(ratio,1.0-epsilon,1.0+epsilon)
                clipped_loss = ratio*b_advants
                actor_loss = -torch.min(surrogate_loss,clipped_loss).mean()
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
            self.actor_old_net.load_state_dict(self.actor_net.state_dict())

    #calculate GAE
    def get_gae(self,rewards,masks,values):
        rewards = torch.as_tensor(rewards)
        masks = torch.as_tensor(masks)
        returns = torch.zeros_like(rewards)
        advants = torch.zeros_like(rewards)
        running_returns = 0
        previous_value = 0
        running_advants = 0

        for t in reversed(range(0,len(rewards))):
            running_returns = rewards[t] + (gamma * running_returns)*masks[t]
            running_tderror = rewards[t] + gamma*previous_value*masks[t] - \
                              values.data[t]
            running_advants = running_tderror + gamma * lambd * \
                              running_advants * masks[t]

            returns[t] = running_returns
            previous_value = values.data[t]
            advants[t] = running_advants
        #advants = (advants - advants.mean()) / (advants.std() + 1e-7)
        return returns, advants
    def save(self,checkpoint_path):
        torch.save(self.actor_net.state_dict(),checkpoint_path)

    def load(self,checkpoint_path):
        self.critic_net.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.actor_net.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))