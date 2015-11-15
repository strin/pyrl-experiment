from common_imports import *

filename = 'result/' + __file__[:__file__.rfind('.')] + '-output.txt'
sys.stdout = open(filename, 'w')


H = W = 10
grid = np.zeros((H, W))
# use diagnoal.
pos_list = [(0,0), (1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8)]
# use all.
# pos_list = []
# for h in range(0, H, 2):
#     for w in range(0, W, 2):
#         if (h, w) == (H-1, W-1):
#             continue
#         pos_list.append((w, h))

tasks = [GridWorldFixedStart(pos, grid=grid, action_stoch=0.2, goal={(H-1, W-1): 1.}, rewards={(H-1, W-1): 1.,}, wall_penalty=0., gamma=0.9)
         for pos in pos_list]
task0 = tasks[0]

################################################
# define neural network architectures.
state_shape = task0.state_shape
task = task0
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


#######################################
## training
dqn = DQN(task0, arch_func=conv_net, state_type=T.tensor4)
learner = DeepQlearn(dqn, gamma=0.95, lr=1e-3, memory_size = 128, epsilon=0.1)
budget = 40
tol=1e-6
train_func = lambda task:  learner.run(task, num_episodes = 10, tol=1e-6, budget=budget)
eval_func = lambda policy, task: reward_stochastic(policy, task, gamma=tol**(1. / budget), num_trials=100, tol=tol)

def eval_all():
    avg = 0.
    for task in tasks:
        avg += eval_func(dqn, task)
    avg /= len(tasks)
    return avg

es = []
avgs = []

for it in range(300):
    task = prob.choice(tasks, size=1, replace=True)[0]
    train_func(task)
    avg = eval_all()
    es.append(learner.total_exp)
    avgs.append(avg)
    print 'iteration', it
    print 'avg score', avg
    print '---------------------------------------'

plot(es, avgs)
xlabel('#experience')
ylabel('average performance')
title('random selection')
savefig('result/random-selection.png')

sys.stdout.flush()

with open('result/random-selection-stats.txt', 'w') as f:
    pickle.dump({
        'es': es,
        'avgs': avgs
    }, f)

