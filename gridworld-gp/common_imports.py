import sys
sys.path.append('../../')

import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import *

import numpy as np
import numpy.random as npr
from pyrl.utils import Timer
from pyrl.prob import choice
import pyrl.agents.arch as arch
from pyrl.algorithms.valueiter import DeepQlearn
from pyrl.algorithms.drift_expert import *
from pyrl.algorithms.teleport import *
from pyrl.agents.agent import DQN
from pyrl.evaluate import *
from pyrl.layers import Conv, FullyConnected
from pyrl.tasks.gridworld import *
import theano
import theano.tensor as T
import os
import dill as pickle
import time
from pprint import pprint

from IPython.display import *

if not os.path.exists('result'):
    os.mkdir('result')
