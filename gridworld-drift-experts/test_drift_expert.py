from common_imports import *

H = W = 10
grid = np.zeros((H, W))
pos_list = []
for h in range(H):
    for w in range(W):
        if h % 2 == 0 or w % 2 == 0:
            pos_list.append((h, w))

tasks = [GridWorldFixedStart(pos, grid=grid, action_stoch=0.2, goal={(H-1, W-1): 1.}, rewards={(H-1, W-1): 1.,}, wall_penalty=0., gamma=0.9)
         for pos in pos_list]
task0 = tasks[0]

state_shape = task0.state_shape
task = task0
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

def feat_goal_dist(task):
    return np.abs(task.start_pos[0] - W + 1) + np.abs(task.start_pos[1] - H + 1)

def mf_goal_dist_interval(im_mem, curr_task, interval, win_size):
    curr_feat = feat_goal_dist(curr_task)
    if curr_feat < interval[0] or curr_feat >= interval[1]:
        return 0.
    avg_im = 0.
    avg_count = 0
    for (task, im) in im_mem[:-win_size-1:-1]:
        feat = feat_goal_dist(task)
        if feat >= interval[0] and feat < interval[1]:
            avg_im += im
            avg_count += 1
    if avg_count == 0:
        return 0.
    else:
        return avg_im / avg_count

def mf_goal_dist_gen(interval, win_size):
    class MFGoalDist(object):
        def __call__(self, im_mem, curr_task):
            return mf_goal_dist_interval(im_mem, curr_task, interval, win_size)

        def __repr__(self):
            return str(interval) + '-' + str(win_size)
    return MFGoalDist()

intervals = [(0, 5), (3, 8), (5, 10), (8, 13), (10, 15), (13, 18), (15, 20)]
mf_goal_dist_exs = [mf_goal_dist_gen(interval, win_size=3) for interval in intervals]


dqn = DQN(task0, arch_func=conv_net, state_type=T.tensor4)
learner = DeepQlearn(dqn, gamma=0.95, lr=1e-3, memory_size = 128, epsilon=0.1)
budget = 20
tol=1e-6
train_func = lambda task:  learner.run(task, num_episodes = 10, tol=1e-6, budget=budget)
eval_func = lambda policy, task: reward_stochastic(policy, task, gamma=tol**(1. / budget), num_trials=100, tol=tol)

meta = DQCL.DriftExpert(dqn, mf_goal_dist_exs, train_func, eval_func, eta=1/50.)

for it in range(100):
    meta.run(tasks, num_epochs=1)
    print 'iteration', it
    print 'count'
    pprint(meta.task_count)
    print 'pred'
    pprint(meta.im_pred)
    print 'ucb'
    pprint(meta.im_ucb)
    print 'meta features'
    pprint(meta.im_feat)
    print 'im_mem'
    pprint(meta.im_mem)
    pprint(meta.diagnostics)
    print '---------------------------------------'
