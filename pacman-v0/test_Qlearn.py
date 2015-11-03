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
import theano
import theano.tensor as T

_layout = layout.getLayout('pyrl/tasks/pacman/layouts/smallGrid.lay')
ghostType = DirectionalGhost
agents = [ghostType( i+1 ) for i in range(2)]
#display = graphicsDisplay.PacmanGraphics(zoom=1.0, frameTime = 0.1)
# display = textDisplay.PacmanGraphics()
display = textDisplay.NullGraphics()
task = PacmanTask(_layout, agents, display)

with Timer("random-play"):
    scores = []
    for it in range(1000):
        overall_reward = 0.
        if not task.is_end():
            actions = task.valid_actions
            action = choice(actions, 1)[0]
            reward = task.step(action)
            overall_reward += reward
        else:
            sys.stdout.flush()
            _layout = layout.getLayout('pyrl/tasks/pacman/layouts/smallGrid.lay')
            ghostType = DirectionalGhost
            agents = [ghostType( i+1 ) for i in range(2)]
            #display = graphicsDisplay.PacmanGraphics(zoom=1.0, frameTime = 0.1)
            # display = textDisplay.PacmanGraphics()
            display = textDisplay.NullGraphics()
            task = PacmanTask(_layout, agents, display)


input_dim = task.state_shape

def two_layer(states):
    state_vecs = states.flatten(2)
    return arch.two_layer(state_vecs, np.prod(task.state_shape), 128, task.num_actions)

dqn = DQN(task, arch_func=two_layer, state_type=T.tensor4)
learner = DeepQlearn(dqn, gamma=0.95, lr=1e-3, memory_size = 100, epsilon=0.5)
scores = []
for it in range(1000):
    learner.run(task, num_episodes = 100)
    score = reward_stochastic(dqn, task, gamma=0.95, num_trials=10, tol=1e-4)
    scores.append(score)
    print 'it', it, 'score', scores[-1]
