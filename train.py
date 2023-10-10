import os.path
import datetime
import random

import torch
import matplotlib.pyplot as plt
import numpy as np
from parameters import *
from PPO import PPO
from collections import deque
from energy_system import EnergySystem

env = EnergySystem(0.5,0.5,20,150,10,'train')
env_name = 'energy_system'
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(2021)

PPO = PPO(N_S,N_A)
################################### logging ##########################
### log files for multiple runs are NOT overwritten
log_dir = r"D:\PPO_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_dir = log_dir + '/' + env_name + '/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

#### get number of log files in log directory
current_num_files = next(os.walk(log_dir))[2]
run_num = len(current_num_files)

#### create new log file for each run ####
log_f_name = log_dir + "\PPO_" +env_name+'_log_' + str(run_num)+".csv"
print("current logging run number for "+env_name+" : ",run_num)
print("logging at : "+log_f_name)
###################################################

##################### checkpointing ###################
run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder
directory = r"D:\\PPO_preTrained"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = directory + '/' + env_name + '/'
if not os.path.exists(directory):
    os.makedirs(directory)


checkpoint_path = directory + "PPO_{}_{}.pth".format(env_name, run_num_pretrained)
print("save checkpoint path : " + checkpoint_path)
#####################################################

########## print all hyperparameters#################
print("--------------------------------------------------------------------------------------------")
print("max training timesteps : ", max_training_iteration)
print("max timesteps per episode : ", max_ep_len)
print("model saving frequency : " + str(save_model_freq) + " timesteps")
print("log frequency : " + str(log_freq) + " timesteps")
print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
print("--------------------------------------------------------------------------------------------")
print("state space dimension : ", N_S)
print("action space dimension : ", N_A)
print("--------------------------------------------------------------------------------------------")

#track total training time
start_time = datetime.datetime.now().replace(microsecond = 0)
print("Training started at (GMT) :",start_time)
print("=======================================================")

episodes = 0
eva_episodes = 0
print(torch.cuda.is_available())
log_dir = r"D:/ppo_energy_system"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
### create new log file for each run
directory = 'D:/ppo_energy_system/'
log_f_name = 'D:/ppo_energy_system/' + "_log_" + str(run_num) + ".csv"
model_path = os.path.join(directory,"ppo_model_train"+".pth")

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)


# logging file
log_f = open(log_f_name, "w+")
log_f.write('episode,timestep,reward\n')

# printing and logging variables
print_running_reward = 0
print_running_episodes = 0

log_running_reward = 0
log_running_episodes = 0

time_step = 0
i_episode = 0

# debug list, we want to confirm whether our algorithm is getting better
# we need to plot a graph whose x-axis is iterations and y-axis is reward
iter_debug = []
reward_debug = []
#PPO.load(model_path)
for iter in range(max_training_iteration):
    memory = deque() # for every iter, we initialize a new and empty deque
    s = env.reset()
    reward_per_episode = 0
    print("====================This is No.{} iteration".format(iter))
    for h_time in range(24):
        print("time_step is :{}".format(h_time))
        a = PPO.actor_net.choose_action(torch.from_numpy(np.array(s).astype(np.float32)).unsqueeze(0))[0]
        print(a)
        s_,r,done = env.step(a)

        print("r：{}".format(r))
        print(s_)
        mask = (1-done)*1
        memory.append([s,a,r,mask])
        reward_per_episode += r
        print_running_reward += r
        log_running_reward += r
        s = s_

        if done == 1:
            env.reset()
            break
    print("====================================================")
    print("reward_per_episode:{}".format(reward_per_episode))
    print("====================================================")
    print("print_running_reward:{}".format(print_running_reward))

    print_running_episodes += 1
    log_running_episodes += 1

    # log in logging file
    if (iter+1) % log_freq == 0:
        # log average reward till last episode
        log_avg_reward = log_running_reward / log_running_episodes
        log_avg_reward = round(log_avg_reward,4)

        log_f.write('')
        log_f.flush()
        log_running_reward = 0
        log_running_episodes = 0

    if (iter+1) % print_freq == 0:
        #  print average reward
        print_avg_reward = print_running_reward /print_running_episodes
        print_avg_reward = round(print_avg_reward,2)
        iter_debug.append(iter+1)
        reward_debug.append(print_avg_reward)
        print("iter:  {} \t\t Average Reward : {}".format(iter,print_avg_reward))

        print_running_reward = 0
        print_running_episodes = 0

    # save model weights
    if (iter+1) % save_model_freq == 0:
        print("-------------------------------------------------------------")
        print("saving model at :" + model_path)
        PPO.save(model_path)
        print("model saved")
        print("Elapsed Time :",datetime.datetime.now().replace(microsecond=0) - start_time)

    # 每隔一定的timesteps 进行参数更新
    if (iter+1) % 3000 ==0:
        print(1)
    PPO.train(memory)


plt.plot(iter_debug, reward_debug)
plt.xlim((0,max_training_iteration))
plt.ylim((-10,100))
plt.show()
