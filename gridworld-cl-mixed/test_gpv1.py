from common_imports import *

## parameters.
budget = int(os.environ['budget'])

## create a maze.
H = W = 10
grid = np.zeros((H, W))
goal = {(W-1, H-1): 1.}

pos_list = []
for h in range(0, H):
    for w in range(0, W):
        if (w, h) in goal:
            continue
        pos_list.append((w, h))

tasks = [GridWorldFixedStart(pos, grid=grid, action_stoch=0.2, goal=goal, rewards=goal, wall_penalty=0.)
         for pos in pos_list]
tasks_by_pos = {task.start_pos: task for task in tasks}
random.shuffle(tasks)

test_pos_list = [(0,0), (3,3), (6,6), (8,8)]
test_tasks = [GridWorldFixedStart(pos, grid=grid, action_stoch=0.2, goal=goal, rewards=goal, wall_penalty=0.)
         for pos in test_pos_list]
task0 = tasks[0]

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

# kernels.
def mf_xy(task):
    return np.array([task.start_pos[0], task.start_pos[1]])

def kernel_func(task_i, task_j):
    feat_i = mf_xy(task_i)
    feat_j = mf_xy(task_j)
    return np.exp(1 - 1 * np.sum(np.abs(feat_i - feat_j)))

def expand_func(task):
    (x,y) = task.start_pos
    new_tasks = []
    for xs in [-1, 0, 1]:
        for ys in [-1, 0, 1]:
            pos = (x + xs, y + ys)
            if pos in tasks_by_pos:
                new_tasks.append(tasks_by_pos[pos])
    return new_tasks

expand_func(tasks_by_pos[(8,9)])

dqn = DQN(task0, arch_func=two_layer, state_type=T.tensor4)
tol = 1e-4
gamma = 0.9
memory_size = 1000
lr = 1e-3
K = 1
K1 = 1
K0 = 1
eta = 1e-4

learner = DeepQlearn(dqn, gamma=gamma, lr=lr, memory_size=memory_size, epsilon=0.1)
train_func = lambda task:  learner.run(task, num_episodes=3, tol=tol, budget=budget)
eval_func = lambda task: qval_stochastic(dqn, task, gamma=gamma, budget=budget, num_trials=100, tol=tol)

meta = GPv1(dqn, kernel_func, expand_func, train_func, eval_func,
            eta=eta, sigma_n=0.1, K=K, K1=K1, K0=K0)

start_locs = []
ims = []
scores = []
exs = []
for it in range(20):
    meta.run(tasks, num_epochs=1)
    to_show = ['new-tasks-selected', 'im', 'pred', 'ucb']
    pprint({key: meta.diagnostics[key] for key in to_show})
    print '---------------------------------------'
    if (it + 1) % 1 == 0:
        test_score = {}
        for test_task in test_tasks:
            test_score[str(test_task)] = reward_stochastic(dqn, test_task, gamma=0.9,num_trials=100, tol=tol, method='eps-greedy', epsilon=0.)
        scores.append(test_score)
        exs.append(learner.total_exp)


with open('result-gpv1', 'w') as f:
    json.dump({
        'ims': ims,
        'scores': scores,
        'exs': exs,
        'budget': budget,
        'memory_size': memory_size,
        'K': K,
        'K1': K1,
        'K0': K0,
        'lr': lr,
        'eta': eta
    }, f)

