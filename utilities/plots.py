import sys             # SKIPPED
import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import utilities.constants as ct

from pathlib import Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

'''
    Plot rewards, entropies, actor and critic losses, total losses, KL divergences
'''
def plot_rewards_losses(rewards, actor_losses, critic_losses, entropies, stds, losses, kl_divs, thread_rewards, best_reward,
                    best_ep, ep):
    plt.figure()
    fig, ax = plt.subplots(3, 2, figsize=(12, 15))

    actor_losses = [t.cpu().detach().numpy() for t in actor_losses]
    critic_losses = [t.cpu().detach().numpy() for t in critic_losses]
    entropies = [t.cpu().detach().numpy() for t in entropies]
    losses = [t.cpu().detach().numpy() for t in losses]
    kl_divs = [t.cpu().detach().numpy() for t in kl_divs]
    stds = [t.cpu().detach().numpy() for t in stds]
    std_rewards = np.std(rewards, axis=1)
    mean_rewards = np.mean(rewards, axis=1)
    std_losses = np.std(losses, axis=1)
    mean_losses = np.mean(losses, axis=1)
    std_kl = np.std(kl_divs, axis=1)
    mean_kl = np.mean(kl_divs, axis=1)
    mean_entropies = np.mean(entropies, axis=1)
    std_actor_losses = np.std(actor_losses, axis=1)
    mean_actor_losses = np.mean(actor_losses, axis=1)
    std_critic_losses = np.std(critic_losses, axis=1)
    mean_critic_losses = np.mean(critic_losses, axis=1)
    mean_stds = np.mean(stds, axis=1)

    range_ep = range(ep, len(mean_rewards)+ep)

    ax[0][0].fill_between(range_ep, mean_rewards-std_rewards, mean_rewards+std_rewards,
                        facecolor='blue', alpha=0.5, label='std interval')
    ax[0][1].fill_between(range_ep, mean_critic_losses-std_critic_losses, mean_critic_losses+std_critic_losses,
                        facecolor='blue', alpha=0.5, label='std interval')
    ax[1][0].fill_between(range_ep, mean_actor_losses-std_actor_losses, mean_actor_losses+std_actor_losses,
                        facecolor='blue', alpha=0.5, label='std interval')
    ax[2][0].fill_between(range_ep, mean_losses-std_losses, mean_losses+std_losses,
                        facecolor='blue', alpha=0.5, label='std interval')
    ax[2][1].fill_between(range_ep, mean_kl-std_kl, mean_kl+std_kl,
                        facecolor='blue', alpha=0.5, label='std interval')

    x = np.vstack(rewards)
    x = np.insert(x, 1, thread_rewards)
    y = np.vstack(critic_losses)

    ax[0][0].plot(range_ep, mean_rewards, c='blue', label="Mean Rewards")
    ax[0][0].plot(range_ep, thread_rewards, c='red', label="Best Thread Rewards")
    ax[0][0].legend(loc='lower right', framealpha=0.5)
    ax[0][0].set_ylabel('Rewards')
    ax[0][0].set_xlabel('Current Episode')
    ax[0][0].set_title('Mean Reward of all Threads, Highest Reward {} at Ep #{}'.format(best_reward, best_ep))
    ax[0][0].set_ylim([x.min(), 0])
    ax[0][0].grid(axis='y', color='0.95')

    ax[0][1].plot(range_ep, mean_critic_losses, c='b', label="Mean")
    ax[0][1].legend(loc='upper right', framealpha=0.5)
    ax[0][1].set_ylabel('Critic Loss')
    ax[0][1].set_xlabel('Current Episode')
    ax[0][1].set_title('Mean Critic Loss of all Threads per Episode')
    ax[0][1].set_ylim([y.min(), y.max()])

    y = np.vstack(actor_losses)
    ax[1][0].plot(range_ep, mean_actor_losses, c='b', label="Mean Actor Loss")
    ax[1][0].legend(loc='upper right', framealpha=0.5)
    ax[1][0].set_ylabel('Actor Loss')
    ax[1][0].set_xlabel('Current Episode')
    ax[1][0].set_title('Mean Actor Loss of all Threads per Episode')
    ax[1][0].set_ylim([y.min(), y.max()])

    ax[1][1].plot(range_ep, mean_entropies, c='blue', label="Mean Entropies")
    ax[1][1].plot(range_ep, mean_stds, c='red', label="Mean Std")
    ax[1][1].legend(loc='upper right', framealpha=0.5)
    ax[1][1].set_ylabel('Entropies')
    ax[1][1].set_xlabel('Current Episode')
    ax[1][1].set_title('Mean Entropies of all Threads per Episode')

    y = np.vstack(losses)
    ax[2][0].plot(range_ep, mean_losses, c='blue', label="Mean Losses")
    ax[2][0].legend(loc='upper right', framealpha=0.5)
    ax[2][0].set_ylabel('Losses')
    ax[2][0].set_xlabel('Current Episode')
    ax[2][0].set_title('Mean Loss of all Threads per Episode')
    ax[2][0].set_ylim([y.min(), y.max()])

    y = np.vstack(kl_divs)
    ax[2][1].plot(range_ep, mean_kl, c='blue', label="Mean Kl")
    ax[2][1].legend(loc='upper right', framealpha=0.5)
    ax[2][1].set_ylabel('KL(old policy logprob, new policy logprob)')
    ax[2][1].set_xlabel('Current Episode')
    ax[2][1].set_title('Mean KL divergences of all Threads per Episode')
    ax[2][1].set_ylim([y.min(), y.max()])
    plt.savefig(ct.FIGURE_PATH)
    plt.close('all')

'''
    Plot joints angular positions of target trajectory and best reward trajectory
'''
def plot_pos(best_ep_pos, kind, env):
    plt.figure()
    fig, ax = plt.subplots(3, 2, figsize=(12, 15))

    hip_r1 = [best_ep_pos[i][0][0] for i in range(len(best_ep_pos))]
    hip_l1 = [best_ep_pos[i][0][1] for i in range(len(best_ep_pos))]
    hip_r2 = [env.joint_activities[i][0] for i in range(len(env.joint_activities))]
    hip_l2 = [env.joint_activities[i][1] for i in range(len(env.joint_activities))]

    knee_r1 = [best_ep_pos[i][1][0] for i in range(len(best_ep_pos))]
    knee_l1 = [best_ep_pos[i][1][1] for i in range(len(best_ep_pos))]
    knee_r2 = [env.knee_activities[i][0] for i in range(len(env.knee_activities))]
    knee_l2 = [env.knee_activities[i][1] for i in range(len(env.knee_activities))]

    ankle_r1 = [best_ep_pos[i][2][0] for i in range(len(best_ep_pos))]
    ankle_l1 = [best_ep_pos[i][2][1] for i in range(len(best_ep_pos))]
    ankle_r2 = [env.ankle_activities[i][0] for i in range(len(env.ankle_activities))]
    ankle_l2 = [env.ankle_activities[i][1] for i in range(len(env.ankle_activities))]

    ax[0][0].plot(range(len(best_ep_pos)), hip_r1, c='blue', label="Best Model Positions")
    ax[0][0].plot(range(len(best_ep_pos)), hip_r2, c='red', label="Target Positions")
    ax[0][1].plot(range(len(best_ep_pos)), hip_l1, c='blue', label="Best Model Positions")
    ax[0][1].plot(range(len(best_ep_pos)), hip_l2, c='red', label="Target Positions")
    ax[0][0].legend(loc='upper left', framealpha=0.5)                     
    ax[0][0].set_ylabel('Hip Angular Position [rad]')
    ax[0][0].set_xlabel('Current Iteration')
    ax[0][0].set_title('Right Hip Positions')
    ax[0][1].legend(loc='lower right', framealpha=0.5)                     
    ax[0][1].set_ylabel('Hip Angular Position [rad]')
    ax[0][1].set_xlabel('Current Iteration')
    ax[0][1].set_title('Left Hip Positions')

    ax[1][0].plot(range(len(best_ep_pos)), knee_r1, c='blue', label="Best Model Positions")
    ax[1][0].plot(range(len(best_ep_pos)), knee_r2, c='red', label="Target Positions")
    ax[1][1].plot(range(len(best_ep_pos)), knee_l1, c='blue', label="Best Model Positions")
    ax[1][1].plot(range(len(best_ep_pos)), knee_l2, c='red', label="Target Positions")
    ax[1][0].legend(loc='lower left', framealpha=0.5)                     
    ax[1][0].set_ylabel('Knee Angular Position [rad]')
    ax[1][0].set_xlabel('Current Iteration')
    ax[1][0].set_title('Right Knee Positions')
    ax[1][1].legend(loc='upper left', framealpha=0.5)                     
    ax[1][1].set_ylabel('Knee Angular Position [rad]')
    ax[1][1].set_xlabel('Current Iteration')
    ax[1][1].set_title('Left Knee Positions')

    ax[2][0].plot(range(len(best_ep_pos)), ankle_r1, c='blue', label="Best Model Positions")
    ax[2][0].plot(range(len(best_ep_pos)), ankle_r2, c='red', label="Target Positions")
    ax[2][1].plot(range(len(best_ep_pos)), ankle_l1, c='blue', label="Best Model Positions")
    ax[2][1].plot(range(len(best_ep_pos)), ankle_l2, c='red', label="Target Positions")
    ax[2][0].legend(loc='upper right', framealpha=0.5)                     
    ax[2][0].set_ylabel('Ankle Angular Position [rad]')
    ax[2][0].set_xlabel('Current Iteration')
    ax[2][0].set_title('Right Ankle Positions')
    ax[2][1].legend(loc='upper right', framealpha=0.5)                     
    ax[2][1].set_ylabel('Ankle Angular Position [rad]')
    ax[2][1].set_xlabel('Current Iteration')
    ax[2][1].set_title('Left Ankle Positions')

    plt.savefig(f"{ct.FIGURE_PATH_POS}_{kind}.png")
    plt.close("all")

'''
    Plot Actions that got the best overall reward, and the best Actions from the policy that got
    the best average reward
'''
def plot_actions(best_all, best_mean):

    plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    hip_r1 = [best_all[i][0] for i in range(len(best_all))]
    hip_l1 = [best_all[i][1] for i in range(len(best_all))]
    hip_r2 = [best_mean[i][0] for i in range(len(best_mean))]
    hip_l2 =[best_mean[i][1] for i in range(len(best_mean))]

    ax[0].plot(range(len(hip_r1)), hip_r1, c='blue', label="Right Hip Actuator")
    ax[0].plot(range(len(hip_l1)), hip_l1, c='red', label="Left Hip Actuator")
    ax[1].plot(range(len(hip_r2)), hip_r2, c='blue', label="Right Hip Actuator")
    ax[1].plot(range(len(hip_l2)), hip_l2, c='red', label="Left Hip Actuator")
    ax[0].legend(loc='upper left', framealpha=0.5)                     
    ax[0].set_ylabel('Torque [N/m]')
    ax[0].set_xlabel('Current Iteration')
    ax[0].set_title('Highest Reward Actions')
    ax[1].legend(loc='upper right', framealpha=0.5)                     
    ax[1].set_ylabel('Torque [N/m]')
    ax[1].set_xlabel('Current Iteration')
    ax[1].set_title('Best Mean Reward Actions')

    plt.savefig(ct.FIGURE_PATH_TORQUES)
    plt.close('all')