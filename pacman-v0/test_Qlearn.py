from common_imports import *

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
learner = DeepQlearn(dqn, gamma=0.95, lr=1e-3, memory_size = 1000, epsilon=0.1)

# learn.
scores = []
for it in range(100):
    with Timer('iteration %d train' % it):
        learner.run(task, num_episodes = 10, tol=1e-6)
    with Timer('iteration %d test' % it):
        score = reward_stochastic(dqn, task, gamma=0.95, num_trials=10, tol=1e-4)
        scores.append(score)
        print 'score', scores[-1]

plot(scores)
title('Game Score')
xlabel('iteration')
ylabel('score')
savefig(os.environ['output'] + '-score.png')

# save results.
with open(os.environ['output'], 'w') as result:
    pickle.dump({
        'scores': scores
    }, result)

with open(os.environ['output'] + '.model', 'w') as result:
    pickle.dump(dqn, result)

