from parameters import *
from PPO import PPO
import os
from energy_system import EnergySystem
import torch
import matplotlib.pyplot as plt
import numpy as np

env = EnergySystem(0.5,0.5,20,150,10,'test')
directory = 'D:/ppo_energy_system/'
model_path = os.path.join(directory,"ppo_model_train"+".pth")

print("loading network from : " + model_path)
test_ppo_agent = PPO(N_S,N_A)
test_ppo_agent.load(model_path)
s = env.reset()
soc_bat_history = []
soc_h2_history = []
bat_power_history = []
fc_power_history = []
el_power_history = []
for h_time in range(24):
    soc_bat,soc_h2 = s[0],s[1]
    soc_bat_history.append(soc_bat)
    soc_h2_history.append(soc_h2)
    a = test_ppo_agent.actor_net.choose_action(torch.from_numpy(np.array(s).astype(np.float32)).unsqueeze(0))[0]
    power_fc,power_el,power_bat = a[0],a[1],a[2]
    bat_power_history.append(power_bat)
    el_power_history.append(power_el)
    fc_power_history.append(power_fc)
    s_,r,done = env.step(a)
    mask = (1 - done) * 1
    s = s_
time_h = np.arange(0,24)
print(len(time_h))
print(soc_h2_history)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('time (h)')
ax1.set_ylabel('h2 soc (%)')
ax1.plot(time_h,soc_h2_history,color=color)
ax1.legend(['h2 soc'])
ax1.tick_params(axis='y',labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'

ax2.plot(time_h, fc_power_history, color=color)
plt.legend(['fuel cell power'])
color = 'tab:orange'
ax2.plot(time_h, el_power_history,color = color)
plt.legend(['electrolyser power'])
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylabel('output power (kw)')
plt.show()