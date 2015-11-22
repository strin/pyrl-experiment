from common_imports import *

arch_func = os.environ['arch']
num_episodes = int(os.environ['num_episodes'])
num_iter = int(os.environ['num_iter'])
memory_size = int(os.environ['memory_size'])

# generating tasks.
H = W = 10
grid = np.zeros((H, W))
# use diagnoal.
# pos_list = [(0,0), (1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8)]
# use all.
pos_list = []
for h in range(0, H, 2):
    for w in range(0, W, 2):
        if (h, w) == (H-1, W-1):
            continue
        pos_list.append((w, h))

tasks = [GridWorldFixedStart(pos, grid=grid, action_stoch=0.2, goal={(H-1, W-1): 1.}, rewards={(H-1, W-1): 1.,}, wall_penalty=0., gamma=0.9)
         for pos in pos_list]
test_pos_list = [(0,0), (3,3), (6,6), (8,8)]
test_tasks = [GridWorldFixedStart(pos, grid=grid, action_stoch=0.2, goal={(H-1, W-1): 1.}, rewards={(H-1, W-1): 1.,}, wall_penalty=0., gamma=0.9)
         for pos in test_pos_list]
task0 = tasks[0]

################################################
# define neural network architectures.
# create the task.
state_shape = task0.state_shape
task = task0
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
    conv1 = Conv(input_dim=state_shape[0], output_dim=32, filter_size=(2,2), pool_size=(2,2), activation='relu')
    h1 = conv1(states)
    h1_dim = int((state_shape[1]-2+1)/2) * int((state_shape[2]-2+1)/2) * 32
    h1 = h1.flatten(2)
    fc1 = FullyConnected(input_dim=h1_dim, output_dim=32, activation='relu')
    h2 = fc1(h1)
    linear_layer = FullyConnected(input_dim=32, output_dim=task.num_actions, activation=None)
    output = linear_layer(h2)
    return (output, {
        'conv1': conv1,
        'fc1': fc1,
        'linear_layer': linear_layer
    })

################################################
# define meta features
def feat_goal_dist(task):
    return np.abs(task.start_pos[0] - W + 1) + np.abs(task.start_pos[1] - H + 1)

def feat_dist(task_a, task_b):
    return np.abs(task_a.start_pos[0] - task_b.start_pos[0]) + np.abs(task_a.start_pos[1] - task_b.start_pos[1])

def mf_goal_dist_interval(im_mem, curr_task, interval, win_size):
    curr_feat = feat_goal_dist(curr_task)
    if curr_feat < interval[0] or curr_feat >= interval[1]:
        return 0.
    avg_im = 0.
    avg_count = 0
    win_count = 0
    for (task, im) in im_mem[::-1]:
        feat = feat_goal_dist(task)
        if feat >= interval[0] and feat < interval[1]:
            avg_im += im
            avg_count += 1
            win_count += 1
            if win_count >= win_size:
                break
    if avg_count == 0:
        return 0.
    else:
        return avg_im / avg_count

def mf_nb(im_mem, curr_task):
    avg_im = 0.
    for (task, im) in im_mem[::-1]:
        dist = feat_dist(task, curr_task)
        avg_im += np.exp(1.-dist) * im
        if task == curr_task:
            break
    return avg_im


def mf_goal(im_mem, curr_task):
    win_size = 3
    avg_im = 0.
    avg_count = 0
    win_count = 0
    for (task, im) in im_mem[::-1]:
        if task == curr_task:
            avg_im += im
            avg_count += 1
            win_count += 1
            if win_count >= win_size:
                break
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

# intervals = [(0, 5), (3, 8), (5, 10), (8, 13), (10, 15), (13, 18), (15, 20)]
# dists = [feat_goal_dist(task) for task in tasks]
# intervals = [(dist, dist+1) for dist in dists]
# mf_goal_dist_exs = [mf_goal_dist_gen(interval, win_size=3) for interval in intervals]
# mf_goal_exs = [mf_goal]
mf_exs = [mf_nb]

#######################################
## training
dqn = DQN(task0, arch_func=globals()[arch_func], state_type=T.tensor4)

learner = DeepQlearn(dqn, gamma=0.9, lr=1e-3, memory_size = memory_size, epsilon=0.1)
budget = 5
tol=1e-4
train_func = lambda task:  learner.run(task, num_episodes = num_episodes, tol=tol, budget=budget)
# eval_func = lambda policy, task: reward_stochastic(policy, task, gamma=tol**(1. / budget), num_trials=100, tol=tol)
eval_func = lambda policy, task: qval_stochastic(policy, task, gamma=0.9, budget=budget, num_trials=100, tol=tol)

meta = DQCL.DriftExpert(dqn, mf_exs, train_func, eval_func, eta=1/20.)


task = test_tasks[0]
dqn = DQN(task, arch_func=globals()[arch_func], state_type=T.tensor4)
learner = DeepQlearn(dqn, gamma=0.9, lr=1e-3, memory_size = memory_size, epsilon=0.1)
scores = []
exs = []
tol = 1e-4
for it in range(num_iter):
    learner.run(task, num_episodes=num_episodes, tol=tol)
    if (it + 1) % 1 == 0:
        test_score = {}
        for test_task in test_tasks:
            test_score[str(test_task)] = reward_stochastic(dqn, test_task, gamma=0.9,num_trials=100, tol=tol, method='eps-greedy', epsilon=0.)
        scores.append(test_score)
        exs.append(learner.total_exp)
    print 'it', it


with open('result-baseline', 'w') as f:
    to_save = ['exs', 'scores']
    json.dump({
        key: globals()[key]
        for key in to_save
    }, f)


