#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_bodies import *
from utils.torch_utils import *


class VanillaNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body):
        super(VanillaNet, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        y = self.fc_head(phi)
        return y


class DuelingNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body):
        super(DuelingNet, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        return q


class CategoricalNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body):
        super(CategoricalNet, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.feature_dim, action_dim * num_atoms))
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        log_prob = F.log_softmax(pre_prob, dim=-1)
        return prob, log_prob


class QuantileNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_quantiles, body):
        super(QuantileNet, self).__init__()
        self.fc_quantiles = layer_init(nn.Linear(body.feature_dim, action_dim * num_quantiles))
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        quantiles = self.fc_quantiles(phi)
        quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
        return quantiles


class OptionCriticNet(nn.Module, BaseNet):
    def __init__(self, body, action_dim, num_options):
        super(OptionCriticNet, self).__init__()
        self.fc_q = layer_init(nn.Linear(body.feature_dim, num_options))
        self.fc_pi = layer_init(nn.Linear(body.feature_dim, num_options * action_dim))
        self.fc_beta = layer_init(nn.Linear(body.feature_dim, num_options))
        self.num_options = num_options
        self.action_dim = action_dim
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        q = self.fc_q(phi)
        beta = F.sigmoid(self.fc_beta(phi))
        pi = self.fc_pi(phi)
        pi = pi.view(-1, self.num_options, self.action_dim)
        log_pi = F.log_softmax(pi, dim=-1)
        pi = F.softmax(pi, dim=-1)
        return {'q': q,
                'beta': beta,
                'log_pi': log_pi,
                'pi': pi}


class DeterministicActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_opt_fn,
                 critic_opt_fn,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(DeterministicActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        
        self.actor_opt = actor_opt_fn(self.actor_params + self.phi_params)
        self.critic_opt = critic_opt_fn(self.critic_params + self.phi_params)
        self.to(Config.DEVICE)

    def forward(self, obs):
        phi = self.feature(obs)
        action = self.actor(phi)
        return action

    def feature(self, obs):
        obs = tensor(obs)
        return self.phi_body(obs)

    def actor(self, phi):
        return torch.tanh(self.fc_action(self.actor_body(phi)))

    def critic(self, phi, a):
        return self.fc_critic(self.critic_body(phi, a))


class GaussianActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(GaussianActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)
        # self.std = nn.Parameter(torch.zeros(action_dim))
        self.fc_std = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.phi_params = list(self.phi_body.parameters())

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters()) + self.phi_params
        # self.actor_params.append(self.std)
        self.actor_params = self.actor_params + list(self.fc_std.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters()) + self.phi_params

        self.to(Config.DEVICE)

    def forward(self, obs, action=None, is_eval=False):
        obs = tensor(obs)
        phi = self.phi_body(obs)
        phi_a = self.actor_body(phi)
        # phi_v = self.critic_body(phi)
        # v = self.fc_critic(phi_v)
        mean = torch.sigmoid(self.fc_action(phi_a))
        self.std = torch.relu(self.fc_std(phi_a))
        self.std = torch.clamp(self.std, min=0.001, max=0.2)
        # dist = torch.distributions.Normal(mean, F.softplus(self.std))
        dist = torch.distributions.Normal(mean, self.std)
        if is_eval is True:
            action = mean

        if action is None:
            action = dist.sample()

        phi_v = self.critic_body(phi, action)
        v = self.fc_critic(phi_v)
        # else: print(action, mean, self.std)
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'mean': mean,
                'v': v,
                'dist': dist}


class CategoricalActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(CategoricalActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        
        self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi = self.phi_body(obs)
        phi_a = self.actor_body(phi)
        phi_v = self.critic_body(phi)
        logits = self.fc_action(phi_a)
        v = self.fc_critic(phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'v': v}


class TD3Net(nn.Module, BaseNet):
    def __init__(self,
                 action_dim,
                 actor_body_fn,
                 critic_body_fn,
                 actor_opt_fn,
                 critic_opt_fn
                 ):
        super(TD3Net, self).__init__()
        self.actor_body = actor_body_fn()
        self.critic_body_1 = critic_body_fn()
        self.critic_body_2 = critic_body_fn()

        self.fc_action = layer_init(nn.Linear(self.actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic_1 = layer_init(nn.Linear(self.critic_body_1.feature_dim, 1), 1e-3)
        self.fc_critic_2 = layer_init(nn.Linear(self.critic_body_2.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body_1.parameters()) + list(self.fc_critic_1.parameters()) +\
                             list(self.critic_body_2.parameters()) + list(self.fc_critic_2.parameters())

        self.actor_opt = actor_opt_fn(self.actor_params)
        self.critic_opt = critic_opt_fn(self.critic_params)
        self.to(Config.DEVICE)

    def forward(self, obs):
        obs = tensor(obs)
        return torch.sigmoid(self.fc_action(self.actor_body(obs)))

    def q(self, obs, a):
        obs = tensor(obs)
        a = tensor(a)
        x = torch.cat([obs, a], dim=1)
        q_1 = self.fc_critic_1(self.critic_body_1(x))
        q_2 = self.fc_critic_2(self.critic_body_2(x))
        return q_1, q_2


class RTD3Net(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_encoding_size,
                 critic_encoding_size,
                 actor_body_fn,
                 critic_body_fn,
                 actor_opt_fn,
                 critic_opt_fn,
                 config
                 ):
        super(RTD3Net, self).__init__()

        self.actor_body = actor_body_fn()
        self.critic_body_1 = critic_body_fn()
        self.critic_body_2 = critic_body_fn()

        self.fc_current_obs = layer_init(nn.Linear(state_dim, actor_encoding_size), 1e-3)
        if config.task != 'DeltaHedgingMulti':
            self.fc_action = layer_init(nn.Linear(actor_encoding_size + self.actor_body.feature_dim, action_dim), 1e-3)
        else:
            self.fc_action = layer_init(nn.Linear(actor_encoding_size + self.actor_body.feature_dim, int(action_dim - 1)), 1e-3)

        self.fc_current_ac_obs1 = layer_init(nn.Linear(state_dim + action_dim, critic_encoding_size), 1e-3)
        self.fc_current_ac_obs2 = layer_init(nn.Linear(state_dim + action_dim, critic_encoding_size), 1e-3)
        self.fc_critic_1 = layer_init(nn.Linear(critic_encoding_size + self.critic_body_1.feature_dim, 1), 1e-3)
        self.fc_critic_2 = layer_init(nn.Linear(critic_encoding_size + self.critic_body_2.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters()) + \
                            list(self.fc_current_obs.parameters())
        self.critic_params = list(self.critic_body_1.parameters()) + list(self.fc_critic_1.parameters()) +\
                             list(self.critic_body_2.parameters()) + list(self.fc_critic_2.parameters()) + \
                             list(self.fc_current_ac_obs1.parameters()) + list(self.fc_current_ac_obs2.parameters())

        self.actor_opt_fn = actor_opt_fn
        self.actor_opt = actor_opt_fn(self.actor_params)
        self.critic_opt = critic_opt_fn(self.critic_params)
        self.to(config.DEVICE)

    def forward(self, obs, history):
        obs = tensor(obs)
        history = tensor(history)
        last_hidden = self.actor_body(history)
        encoded_obs = torch.relu(self.fc_current_obs(obs))
        action_input = torch.cat([last_hidden, encoded_obs], dim=1) # TODO: dimension check
        return torch.sigmoid(self.fc_action(action_input))

    def q(self, obs, a, history):
        obs = tensor(obs)
        a = tensor(a)
        history = tensor(history)
        x = torch.cat([obs, a], dim=1) # TODO: dimemsion check

        last_hidden1 = self.critic_body_1(history)
        last_hidden2 = self.critic_body_2(history)

        encoded_ac_obs1 = self.fc_current_ac_obs1(x)
        encoded_ac_obs2 = self.fc_current_ac_obs2(x)

        action_input1 = torch.cat([last_hidden1, encoded_ac_obs1], dim=1)  # TODO: dimension check
        action_input2 = torch.cat([last_hidden2, encoded_ac_obs2], dim=1)  # TODO: dimension check
        q_1 = self.fc_critic_1(action_input1)
        q_2 = self.fc_critic_2(action_input2)
        return q_1, q_2


class multiRTD3Net(nn.Module):

    def __init__(self, network1, network2):
        super(multiRTD3Net, self).__init__()
        self.network1 = network1
        self.network2 = network2
        self.critic_opt = network1.critic_opt
        self.actor_opt = network1.actor_opt_fn(self.network1.actor_params + self.network2.actor_params)

    def forward(self, obs, history):
        action_input1 = self.network1(obs, history)
        action_input2 = self.network2(obs, history)
        return torch.cat([action_input1, action_input2], dim=1)

    def q(self, obs, a, history):
        return self.network1.q(obs, a, history)


class MetaGaussianActorCriticNet(MetaModule):
    def __init__(self,
                 state_dim,
                 action_dim
                 ):
        super(MetaGaussianActorCriticNet, self).__init__()
        hidden_size = 64
        self.actor_net = MetaSequential(
            layer_init(MetaLinear(state_dim, hidden_size)),
            nn.Tanh(),
            layer_init(MetaLinear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(MetaLinear(hidden_size, action_dim))
        )
        self.std = MetaTensor(torch.zeros(action_dim, requires_grad=True))

        self.critic_net = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1))
        )

        self.critic_mix_net = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1))
        )

        self.lam = torch.zeros(1, requires_grad=True)

        self.meta_params = [self.lam]
        self.critic_params = self.critic_net.parameters()

        self.to(Config.DEVICE)


    def forward(self, obs, params=None, action=None):
        obs = tensor(obs)
        phi_a = self.actor_net(obs, params=get_subdict(params, 'actor_net'))
        mean = torch.tanh(phi_a)
        v = self.critic_net(obs)
        v_mix = self.critic_net(obs)
        std = F.softplus(self.std(params=get_subdict(params, 'std')))
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'mean': mean,
                'v': v,
                'v_mix': v_mix}

    def penalty(self):
        return torch.sigmoid(self.lam)


class MVPNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(MVPNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)
        self.std = nn.Parameter(torch.zeros(action_dim))
        self.y = nn.Parameter(torch.zeros(1))

        self.to(Config.DEVICE)

    def forward(self, obs, action=None):
        obs = tensor(obs)
        phi = self.phi_body(obs)
        phi_a = self.actor_body(phi)
        phi_v = self.critic_body(phi)
        mean = torch.tanh(self.fc_action(phi_a))
        v = self.fc_critic(phi_v)
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        return {'a': action,
                'log_pi_a': log_prob,
                'ent': entropy,
                'mean': mean,
                'v': v}