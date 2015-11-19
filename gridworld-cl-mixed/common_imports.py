import sys
sys.path.append('../../')

import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import *
import matplotlib.animation as animation

import numpy as np
import numpy.random as npr
from pyrl.utils import Timer
from pyrl.prob import choice
import pyrl.agents.arch as arch
from pyrl.algorithms.valueiter import DeepQlearn
from pyrl.algorithms.multitask import DeepQlearnMT
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

# render json.
import uuid
from IPython.display import display_javascript, display_html, display
import json

class RenderJSON(object):
    def __init__(self, json_data):
        if isinstance(json_data, dict):
            self.json_str = json.dumps(json_data)
        else:
            self.json_str = json
        self.uuid = str(uuid.uuid4())

    def _ipython_display_(self):
        display_html('<div id="{}" style="height: 600px; width:100%;"></div>'.format(self.uuid),
            raw=True
        )
        display_javascript("""
        require(["https://rawgit.com/caldwell/renderjson/master/renderjson.js"], function() {
          document.getElementById('%s').appendChild(renderjson(%s))
        });
        """ % (self.uuid, self.json_str), raw=True)

