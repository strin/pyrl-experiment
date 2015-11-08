from common_imports import *

# load game settings.
_layout1 = layout.getLayout(os.environ['layout1'])
_layout2 = layout.getLayout(os.environ['layout2'])

ghostType = DirectionalGhost
agents = [ghostType( i+1 ) for i in range(2)]
display = textDisplay.NullGraphics()

# create the task.
task1 = PacmanTask(_layout1, agents, display, state_repr=os.environ['state_repr'])
task2 = PacmanTask(_layout2, agents, display, state_repr=os.environ['state_repr'])

state_shape = task1.state_shape

# define potential neural network architectures.
def two_layer(states):
    '''
    two layer neural network with same number of hidden units.
    '''
    state_vecs = states.flatten(2)
    return arch.two_layer(state_vecs, np.prod(state_shape), 128, task1.num_actions)

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
    linear_layer = FullyConnected(input_dim=128, output_dim=task1.num_actions, activation=None)
    output = linear_layer(h2)
    return (output, {
        'conv1': conv1,
        'fc1': fc1
    })

arch_func = globals()[os.environ['arch']]

# define learning agent.
dqn = DQN(task1, arch_func=arch_func, state_type=T.tensor4)
learner = DeepQlearn(dqn, gamma=0.95, lr=1e-3, memory_size = 1000, epsilon=0.1)

# learn.
scores = []

print 'start by training task1'
for it in range(50):
    learner.run(task1, num_episodes = 100, tol=1e-6)
    with Timer('iteration %d' % it):
        score = reward_stochastic(dqn, task1, gamma=0.95, num_trials=10, tol=1e-4)
        scores.append(score)
        print 'score', scores[-1]

print 'switching to training task2'
for it in range(50):
    learner.run(task2, num_episodes = 100, tol=1e-6)
    with Timer('iteration %d' % it):
        score = reward_stochastic(dqn, task2, gamma=0.95, num_trials=10, tol=1e-4)
        scores.append(score)
        print 'score', scores[-1]

# plot scores.
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

# save the model.
with open(os.environ['output'] + '.model', 'w') as result:
    pickle.dump(dqn, result)
