import os
import matplotlib.pyplot as plt


episode_counts = []
losses = []

coord_x = []
coord_y = []

random_probs = []
episode_steps = []
episode_rewards = []
eat_counts = []

clear_step_x = []
clear_step_y = []

clear_x = []
clear_y = []


def loss_graph(episode_count, loss):
    episode_counts.append(episode_count)
    losses.append(loss.item())

    if episode_count % 100 == 0:
        x = episode_counts
        y = losses

        plt.figure()
        plt.plot(x, y, color='red')
        plt.title('Loss per each Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Loss')

        path = os.path.dirname(os.path.realpath(__file__)) + '/loss_graph'
        if not os.path.isdir(path):
            os.mkdir(path)
        plt.savefig(path + '/A1_Loss_per_each_Episode_ep{}.png'.format(episode_count))

        plt.close()


def heatmap(episode_count, loc):
    # path = os.path.dirname(os.path.realpath(__file__)) + 'heatmap.png'
    # img = cv2.imread(path)
    # if img is None:
    #     img = np.zeros((210, 160, 3), np.uint8)
    #
    # for i in loc:
    #     img = np.loc[i]
    #
    # cv2.imwrite(path, img)

    for i in loc:
        coord_x.append(i[0])
        coord_y.append(i[1] * -1)

    if episode_count % 100 == 0:
        plt.figure(figsize=(4.5, 5))
        plt.grid(True)
        plt.scatter(coord_x, coord_y, alpha=0.0001)
        plt.xlabel('x-coordinate', fontsize=9)
        plt.ylabel('y-coordinate', fontsize=9)
        plt.xlim(0, 160)
        plt.ylim(-172, 0)
        plt.title('Agent\'s Visited Locations over {} Episodes'.format(episode_count), fontsize=12)
        plt.tight_layout()

        path = os.path.dirname(os.path.realpath(__file__)) + '/heatmap'
        if not os.path.isdir(path):
            os.mkdir(path)
        plt.savefig(path + '/A2_Agent_Heatmap_Episode_ep{}.png'.format(episode_count))


def random_action_graph(episode_count, random_prob):
    random_prob = round(random_prob * 100, ndigits=4)
    random_probs.append(random_prob)

    if episode_count % 100 == 0:
        x = episode_counts
        y = random_probs

        plt.figure()
        plt.plot(x, y, color='cyan')
        plt.title('Random Action Selection per each Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Probability of Random Action Selection (%)')
        plt.tight_layout()

        path = os.path.dirname(os.path.realpath(__file__)) + '/random_action_prob_graph'
        if not os.path.isdir(path):
            os.mkdir(path)
        plt.savefig(path + '/A3_Random_Action_Selection_per_each_Episode_ep{}.png'.format(episode_count))

        plt.close()


def step_graph(episode_count, episode_step):
    episode_steps.append(episode_step)

    if episode_count % 100 == 0:
        x = episode_counts
        y = episode_steps

        plt.figure()
        plt.plot(x, y, color='orange')
        plt.title('Number of Steps per each Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Number of Steps')

        path = os.path.dirname(os.path.realpath(__file__)) + '/step_graph'
        if not os.path.isdir(path):
            os.mkdir(path)
        plt.savefig(path + '/B1_Number_of_Steps_per_each_Episode_ep{}.png'.format(episode_count))

        plt.close()


def reward_graph(episode_count, episode_reward):
    episode_rewards.append(episode_reward)

    if episode_count % 100 == 0:
        x = episode_counts
        y = episode_rewards

        plt.figure()
        plt.plot(x, y)
        plt.title('Sum of Rewards per each Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Sum of Rewards')

        path = os.path.dirname(os.path.realpath(__file__)) + '/reward_graph'
        if not os.path.isdir(path):
            os.mkdir(path)
        plt.savefig(path + '/B2_Sum_of_Rewards_per_each_Episode_ep{}.png'.format(episode_count))

        plt.close()


def progress_graph(episode_count, eat_count):
    eat_counts.append(eat_count)

    if episode_count % 100 == 0:
        x = episode_counts
        y = eat_counts
        
        plt.figure()
        plt.plot(x, y, color='green')
        plt.title('Number of Obtained Dots per each Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Number of Obtained Dots')

        path = os.path.dirname(os.path.realpath(__file__)) + '/progress_graph'
        if not os.path.isdir(path):
            os.mkdir(path)
        plt.savefig(path + '/B3_Number_of_Obtained_Dots_per_each_Episode_ep{}.png'.format(episode_count))

        plt.close()


def stage_clear_step_graph(episode_count, clear_steps):
    clear_step_x.append(episode_count)
    clear_step_y.append(clear_steps)

    plt.figure()
    plt.plot(clear_step_x, clear_step_y, color='purple', marker='o', markersize='3')
    plt.title('Number of Steps when Stages are Cleared')
    plt.xlabel('Episodes')
    plt.ylabel('Number of Steps')

    path = os.path.dirname(os.path.realpath(__file__)) + '/stage_clear_step_graph'
    if not os.path.isdir(path):
        os.mkdir(path)
    plt.savefig(path + '/C1_Number_of_Steps_when_Stages_are_Cleared_{}ep.png'.format(episode_count))

    plt.close()


def stage_clear_graph(episode_count, done_stage):
    clear_x.append(episode_count)
    clear_y.append(done_stage)

    plt.figure()
    plt.plot(clear_x, clear_y, color='magenta', marker='o', markersize='3')
    plt.title('Number of Episodes when Stages are Cleared')
    plt.xlabel('Episodes')
    plt.ylabel('Clearance')
    plt.yticks([0, 1], labels=['Fail', 'Clear'])

    path = os.path.dirname(os.path.realpath(__file__)) + '/stage_clear_graph'
    if not os.path.isdir(path):
        os.mkdir(path)
    plt.savefig(path + '/C2_Number_of_Episodes_when_Stages_are_Cleared_{}_ep.png'.format(episode_count))

    plt.close()

