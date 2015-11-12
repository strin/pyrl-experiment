from common_imports import *

dqn_path = sys.argv[1]
with open(dqn_path, 'r') as f:
    dqn = pickle.load(f)

task_path = sys.argv[2]

def load_task(path):
    _layout = layout.getLayout(path)
    ghostType = DirectionalGhost
    agents = [ghostType( i+1 ) for i in range(2)]
    display = graphicsDisplay.PacmanGraphics(zoom=1.0, frameTime = 0.1)
    task = PacmanTask(_layout, agents, display, state_repr='stack')
    return task

task = load_task(task_path)

temp = raw_input('enter to start')
task.reset()
while not task.is_end():
    action = dqn.get_action(task.curr_state, method='eps-greedy', epsilon=0., valid_actions=task.valid_actions)
    task.step(action)

