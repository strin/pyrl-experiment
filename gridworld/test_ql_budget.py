# command:
from common_imports import *

# load game settings.
H = W = 10
grid = np.zeros((H, W))
x = int(os.environ['x'])
y = int(os.environ['y'])

rewards = {}
for hi in range(H):
    for wi in range(W):
        rewards[(wi, hi)] = 0.
rewards[(H-1, W-1)] = 1.

task = GridWorldFixedStart(start_pos=(x, y), grid=grid, action_stoch=0.2, goal={(H-1, W-1): 1.}, rewards=rewards, wall_penalty=0., gamma=0.9)
with open('result/' + os.environ['output'] + '.model', 'r') as f:
    dqn = pickle.load(f)

# define learning agent.
learner = DeepQlearn(dqn, gamma=0.95, lr=1e-3, memory_size = 128, epsilon=0.1)

# learn.
scores = []
stds = []
exps = []
gamma = 0.9
tol = 1e-4
for it in range(300):
    with Timer('iteration %d train' % it):
        learner.run(task, num_episodes = 1, tol=tol)
    with Timer('iteration %d test' % it):
        (score, std) = reward_stochastic_mean_std(dqn, task, gamma=gamma, num_trials=100, tol=tol)
        exps.append(learner.total_exp)
        scores.append(score)
        stds.append(std)
        print 'score', score, 'std', std

    # plot scores.
    errorbar(exps, scores, stds)
    title('Game Score')
    xlabel('iteration')
    ylabel('score')
    savefig('result/' + os.environ['output'] + '(%d,%d)-score.png' % (x, y))

    # save results.
    with open('result/' + os.environ['output'] + '(%d,%d)' % (x, y), 'w') as result:
        pickle.dump({
            'scores': scores
        }, result)

# save the model.
#with open('result/' + os.environ['output'] + '(%d,%d).model' % (x, y), 'w') as result:
#    pickle.dump(dqn, result)

