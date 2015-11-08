# command:
from common_imports import *

H = W = 10
grid = np.zeros((H, W))
task = GridWorldFixedStart(start_pos=(0, 0), grid=grid, action_stoch=0.2, goal={(H-1, W-1): 1.}, rewards={(H-1, W-1): 1.,}, wall_penalty=0., gamma=0.9)

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

# save the model.
with open('result/' + os.environ['output'] + '.model', 'w') as result:
    pickle.dump(dqn, result)
