from common_imports import *

###################################################################
## load task.
def load_task(path):
    _layout = layout.getLayout(path)
    ghostType = DirectionalGhost
    agents = [ghostType( i+1 ) for i in range(2)]
    display = textDisplay.NullGraphics()
    task = PacmanTask(_layout, agents, display, state_repr='stack')
    return task
task = load_task('data/layouts/smallGrid-1.lay')
tasks = [None]
for task_i in range(1, 7):
    tasks.append(load_task('data/layouts/smallGrid-%d.lay' % task_i))

###################################################################
## define neural network architecture.
state_shape = task.state_shape

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

def conv_net2(states):
    '''
    2 conv + max-pool layers followed by a fully-connected layer.
    '''
    conv1 = Conv(state_shape[0], output_dim=16, filter_size=(2,2), pool_size=(2,2), activation='relu')
    h1 = conv1(states)
    h1_shape = (16, int((state_shape[1]-2+1)/2), int((state_shape[2]-2+1)/2))
    print 'h1_shape', h1_shape
    conv2 = Conv(h1_shape[0], output_dim=32, filter_size=(2,2), pool_size=(2,2), activation='relu')
    h2_shape = (int((h1_shape[1]-2+1)/2), int((h1_shape[2]-2+1)/2), 32)
    print 'h2_shape', h2_shape
    h1 = conv1(states)
    h2 = conv2(h1)
    h2 = h2.flatten(2)
    fc1 = FullyConnected(input_dim=np.prod(h2_shape), output_dim=32, activation='relu')
    h2 = fc1(h2)
    linear_layer = FullyConnected(input_dim=32, output_dim=task.num_actions, activation=None)
    output = linear_layer(h2)
    return (output, {
        'conv1': conv1,
        'conv2': conv2,
        'fc1': fc1
    })

arch_func = globals()[os.environ['arch']]
proc = json.loads(os.environ['proc'])

print 'using arch', arch_func.__name__
print 'proc', str(proc)

result_path = 'result/' + arch_func.__name__ + '/' + str(proc) + '/'
if not os.path.exists(result_path):
    os.makedirs(result_path)
print 'result path', result_path

dqn = DQN(task, arch_func=arch_func, state_type=T.tensor4)

learner = DeepQlearnMT(dqn, gamma=0.9, lr=1e-2, memory_size = 100, epsilon=0.3)

scores = []
total_exps = []
its = []

def run_task(task, num_iter):
    print 'run with', num_iter, 'iterations'
    print task
    for it in range(num_iter):
        learner.run(task, num_episodes = 10)
        score = reward_stochastic(dqn, task, gamma=0.9, num_trials=10, tol=1e-4)
        scores.append(score)
        total_exps.append(learner.total_exp)
        its.append(it)
        print 'it', it, 'total exp', learner.total_exp, 'score', scores[-1]
        with open(result_path + 'dqn-%d.model' % learner.total_exp, 'w') as f:
            pickle.dump(dqn, f)

offset_exps = []
for epoch in proc:
    task = tasks[epoch[0]]
    num_iter = epoch[1]
    run_task(task, num_iter)
    offset_exps.append(total_exps[-1])

with open(result_path + 'score', 'w') as f:
    pickle.dump({
        'its': its,
        'total_exps': total_exps,
        'scores': scores,
        'offset_exps': offset_exps
    }, f)

