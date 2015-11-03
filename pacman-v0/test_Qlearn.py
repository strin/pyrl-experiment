import numpy as np
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
import cPickle as pickle

# load game settings.
_layout = layout.getLayout(os.environ['layout'])
ghostType = DirectionalGhost
agents = [ghostType( i+1 ) for i in range(2)]
display = textDisplay.NullGraphics()

# create the task.
task = PacmanTask(_layout, agents, display, state_repr=os.environ['state_repr'])
state_shape = task.state_shape

# define potential neural network architectures.
def two_layer(states):
    '''
    two layer neural network with same number of hidden units.
    '''
    state_vecs = states.flatten(2)
    return arch.two_layer(state_vecs, np.prod(task.state_shape), 128, task.num_actions)

def conv_net(states):
    '''
    conv+max-pool layer first followed by a fully-connected layer.
    '''
    conv1 = Conv(state_shape[0], output_dim=32, filter_size=(2,2), pool_size=(2,2), activation='relu')
    h1 = conv1(states)
    h1_dim = int((state_shape[1]-2+1)/2) * int((state_shape[2]-2+1)/2) * 32
    h1 = h1.flatten(2)
    fc1 = FullyConnected(input_dim=h1_dim, output_dim=128, activation='relu')
    h2 = fc1(h1)
    linear_layer = FullyConnected(input_dim=128, output_dim=task.num_actions, activation=None)
    output = linear_layer(h2)
    return (output, {
        'conv1': conv1,
        'fc1': fc1
    })

arch_func = globals()[os.environ['arch']]

# define learning agent.
dqn = DQN(task, arch_func=arch_func, state_type=T.tensor4)
learner = DeepQlearn(dqn, gamma=0.95, lr=1e-3, memory_size = 100, epsilon=0.5)

# learn.
scores = []
for it in range(1000):
    learner.run(task, num_episodes = 100)
    with Timer('iteration %d' % it):
        score = reward_stochastic(dqn, task, gamma=0.95, num_trials=10, tol=1e-4)
        scores.append(score)
        print 'score', scores[-1]

# save results.
with open(os.environ['output'], 'w') as result:
    pickle.dump({
        'scores': scores
    }, result)
