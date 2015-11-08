from common_imports import *

# load game settings.
H = W = 10
rewards = {}
for hi in range(H):
    for wi in range(W):
        rewards[(wi, hi)] = -0.03
rewards[(H-1, W-1)] = 1.
grid = np.zeros((H, W))
task = GridWorld(grid=grid, action_stoch=0.2, goal={(H-1, W-1): 1.}, rewards=rewards, wall_penalty=0., gamma=0.9)

# create the task.
state_shape = task.state_shape

# define potential neural network architectures.
def two_layer(states):
    '''
    two layer neural network with same number of hidden units.
    '''
    state_vecs = states.flatten(2)
    return arch.two_layer(state_vecs, np.prod(task.state_shape), 32, task.num_actions)

def conv_net(states):
    '''
    conv+max-pool layer first followed by a fully-connected layer.
    '''
    conv1 = Conv(1, output_dim=32, filter_size=(2,2), pool_size=(2,2), activation='relu')
    h1 = conv1(states)
    h1_dim = int((state_shape[0]-2+1)/2) * int((state_shape[1]-2+1)/2) * 32
    h1 = h1.flatten(2)
    fc1 = FullyConnected(input_dim=h1_dim, output_dim=32, activation='relu')
    h2 = fc1(h1)
    linear_layer = FullyConnected(input_dim=32, output_dim=task.num_actions, activation=None)
    output = linear_layer(h2)
    return (output, {
        'conv1': conv1,
        'fc1': fc1
    })

arch_func = globals()[os.environ['arch']]

# define learning agent.
dqn = DQN(task, arch_func=arch_func, state_type=T.tensor4)
learner = DeepQlearn(dqn, gamma=0.95, lr=1e-3, memory_size = 128, epsilon=0.1)

# learn.
scores = []
exps = []
stds = []
for it in range(100):
    with Timer('iteration %d train' % it):
        learner.run(task, num_episodes = 1, tol=1e-6)
    with Timer('iteration %d test' % it):
        score, std = reward_stochastic_mean_std(dqn, task, gamma=0.95, num_trials=100, tol=1e-4)
        exps.append(learner.total_exp)
        scores.append(score)
        stds.append(std)
        print 'score', score, 'std', std

# plot scores.
plot(scores)
errorbar(exps, scores, stds)
xlabel('iteration')
ylabel('score')
savefig('result/' + os.environ['output'] + '(ql)-score.png')

# save results.
with open('result/' + os.environ['output'] + '(ql)', 'w') as result:
    pickle.dump({
        'scores': scores
    }, result)

# save the model.
with open('result/' + os.environ['output'] + '(ql).model', 'w') as result:
    pickle.dump(dqn, result)

