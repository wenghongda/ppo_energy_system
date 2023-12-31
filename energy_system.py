from parameters import *
from reward_function import calculate_cost
import torch

class EnergySystem():
    def __init__(self,initial_battery_soc:float,initial_h2_soc:float,initial_fc_power:float,initial_el_power:float,initial_bat_power:float,flag:str):
        #basic initialize
        self.freq = freq
        self.r = r
        self.life = life
        self.lambda_inv = lambda_inv
        self.P_price = P_price
        self.time_step = time_step
        self.cost_history = []
        self.flag = flag

        #battery initialize
        self.c_bat = c_bat
        self.battery_charge_power = battery_charge_power
        self.battery_discharge_power = battery_discharge_power
        self.eta_charge = eta_charge
        self.eta_discharge = eta_discharge
        self.C_inv_bat = C_inv_bat
        self.C_bat = C_bat
        self.bat_soc_history = [initial_battery_soc]
        self.bat_power_history = [initial_bat_power]

        #electricity initialize
        self.c_G_buy = c_G_buy
        self.p_load_prediction_history = p_load_prediction_history
        self.p_load_reality_history = p_load_reality_history
        self.EL_power_history = [initial_el_power]

        #pv initialize
        self.n_pv = n_pv
        self.G_STC = G_STC
        self.co2_pv = co2_pv
        self.C_inv_pv = C_inv_pv
        self.solar_prediction_history = solar_prediction_history
        self.solar_realty_history = solar_realty_history

        #wt initialize
        self.n_wt = n_wt
        self.V_cut_in = V_cut_in
        self.V_r = V_r
        self.V_cut_out = V_cut_out
        self.co2_wt = co2_wt
        self.C_wt = C_wt
        self.windspeed_prediction_history = windspeed_prediction_history
        self.windspeed_reality_history = windspeed_reality_history

        #fuel cell initialize
        self.n_FC = n_FC
        self.C_inv_FC = C_inv_FC
        self.C_FC = C_FC
        self.FC_power_history = [initial_fc_power]
        self.pre_p_FC = initial_fc_power

        #electrolyser initialize
        self.eta_EL = eta_EL
        self.n_FC = n_EL
        self.C_inv_EL = C_inv_EL
        self.C_EL = C_EL
        self.pre_p_EL = initial_el_power

        #h2 storage tank initialize
        self.V_h2 = V_h2
        self.R = R
        self.T = T
        self.b = b
        self.h2_pressure_max = h2_pressure_max
        self.H2_soc_history = [initial_h2_soc]
        """
        Notice: if we need to train our model, our model could not know the realistic conditions but to make 
            judgements accroding to the results of Informer's prediction
        """
        if self.flag == 'train':
            self.pv_power_history = pv_power_prediction_history
            self.p_load_history = p_load_prediction_history
            self.wt_power_history = wt_prediction_power_history
            self.fcv_h2_history = np.array(fcv_h2_prediction_m_history)/2

        elif self.flag == 'test':
            self.pv_power_history = pv_power_reality_history
            self.p_load_history = p_load_reality_history
            self.wt_power_history= wt_reality_power_history
            self.fcv_h2_history = fcv_h2_reality_m_history

        self.reset()
    def step(self,action):
        soc_bat,soc_h2,p_load, pv_power,wt_power,m_fcvs,time_step = self.state
        soc_bat = round(soc_bat,4)
        soc_h2 = round(soc_h2,4)
        action = np.array(action)


        p_bat, p_FC,p_EL = action[0],action[1],action[2]
        #p_EL = p_EL.clip(0,n_EL)
        #p_FC = p_h2 if p_h2 > 0 else 0
        #p_EL = abs(p_h2) if p_h2 < 0 else 0

        e_buy =-(pv_power + wt_power + p_FC + p_bat - p_EL - p_load)
        e_buy = 0 if e_buy < 0 else e_buy
        print("e_buy:{}".format(e_buy))
        soc_bat = soc_bat - (eta_charge * p_bat * freq)/self.c_bat
        soc_h2,delta_m= self.calculate_new_h2_soc(p_FC,p_EL,m_fcvs)
        new_p_load = self.p_load_history[self.time_step+1] if self.time_step!= 23 else self.p_load_history[0]
        new_pv_power = self.pv_power_history[self.time_step+1] if self.time_step!= 23 else self.pv_power_history[0]
        new_wt_power = self.wt_power_history[self.time_step+1] if self.time_step!=23 else self.wt_power_history[0]
        new_m_fcvs = self.fcv_h2_history[self.time_step+1] if self.time_step != 23 else self.fcv_h2_history[0]
        costs = calculate_cost(self.time_step,pv_power,wt_power,p_FC,self.pre_p_FC,p_EL,self.pre_p_EL,p_bat,e_buy,soc_bat,soc_h2,delta_m,m_fcvs)
        self.state = [soc_bat,soc_h2,new_p_load,new_pv_power,new_wt_power,new_m_fcvs,self.time_step+1]
        self.time_step += 1
        done = 1 if self.time_step == 24 else 0
        self.pre_p_FC = p_FC
        self.pre_p_EL = p_EL
        return self.state,costs,done

    def reset(self):
        self.time_step = 0
        self.bat_soc_history = self.bat_soc_history[0:1]
        self.H2_soc_history = self.H2_soc_history[0:1]
        self.FC_power_history = self.FC_power_history[0:1]
        self.EL_power_history = self.EL_power_history[0:1]
        self.bat_power_history = self.bat_power_history[0:1]
        self.cost_history = []
        self.m = self.V_h2 / (self.b + self.R * self.T /( self.h2_pressure_max * (self.H2_soc_history[0])))
        self.state = [self.bat_soc_history[0], self.H2_soc_history[0], self.p_load_history[0],
                      self.pv_power_history[0], self.wt_power_history[0],
                      self.fcv_h2_history[0],self.time_step]
        return self.state

    def calculate_new_h2_soc(self,p_fc,p_el,m_fcvs):
        # p_h2 donates the fuel cell power - electrolyser power
        b = self.b
        V = self.V_h2
        eta_EL = self.eta_EL

        R = self.R
        T = self.T
        freq = self.freq
        previous_soc_h2 = self.H2_soc_history[-1]
        # 3600 represent 1 Wh = 3600 J
        delta_m = 3600*0.002*(freq * eta_EL * p_el / LHV_h2-freq*p_fc/(eta_FC*LHV_h2)) - m_fcvs
        print("delta_m:{}".format(delta_m))
        self.m += delta_m
        self.soc_h2 = R*T/((V/self.m)-b)/h2_pressure_max

        return self.soc_h2, delta_m
