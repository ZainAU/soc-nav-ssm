from crowd_sim.envs.policy.policy_factory import policy_factory
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.policy.lstm_rl import LstmRL
from crowd_nav.policy.sarl import SARL
from crowd_nav.policy.lstm_mamba_rl import LstmMamba
from crowd_nav.policy.naive_mamba_rl import Naive_MambaRL
from crowd_nav.policy.mambaRL import MambaRL


policy_factory['cadrl'] = CADRL
policy_factory['lstm_rl'] = LstmRL
policy_factory['sarl'] = SARL
policy_factory['lstm_mamba'] = LstmMamba
policy_factory['Naive-MambaRL'] = Naive_MambaRL
policy_factory['MambaRL'] = MambaRL