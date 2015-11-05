import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import *

import numpy as np
import numpy.random as npr
from pyrl.tasks.pacman.game_mdp import *
from pyrl.tasks.pacman.ghostAgents  import *
import pyrl.tasks.pacman.graphicsDisplay as graphicsDisplay
import pyrl.tasks.pacman.textDisplay as textDisplay
from pyrl.utils import Timer
from pyrl.prob import choice
import pyrl.agents.arch as arch
from pyrl.algorithms.valueiter import DeepQlearn
from pyrl.agents.agent import DQN
from pyrl.evaluate import reward_stochastic
from pyrl.layers import Conv, FullyConnected
import theano
import theano.tensor as T
import os
import dill as pickle
