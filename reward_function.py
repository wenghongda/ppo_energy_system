from parameters import *
def calculate_cost(pv_power,wt_power,p_fc,pre_p_fc,p_el,pre_p_el,p_bat,e_buy,p_load):
    carbon_cost = calculate_carbon(pv_power,wt_power,e_buy)
    carbon_cost = 0 if carbon_cost < 0 else carbon_cost
    degrade_cost = calculate_degrade(p_fc,pre_p_fc,p_el,pre_p_el,p_bat)
    degrade_cost = 0 if degrade_cost < 0 else degrade_cost
    unbalance_penalty = calculate_unbalance_cost(pv_power,wt_power,p_fc,p_bat,e_buy,p_el,p_load)
    print("carbon_cost:{}".format(carbon_cost))
    print("degrade_cost:{}".format(degrade_cost))
    print("unbalance_penalty:{}".format(unbalance_penalty))
    total_cost = carbon_cost + degrade_cost-unbalance_penalty
    reward = calculate_extra_reward(pv_power,wt_power,p_fc,p_el,p_bat,e_buy,p_load)
    print("reward:{}".format(reward))
    total_cost = -total_cost+reward
    print(total_cost)
    return total_cost

def calculate_carbon(power_PV,power_WT,e_buy):
    c_co2 = e_buy # electricity bought from main grid; kwh
    P_CO2 = P_price # RMB/Kg
    c_PV = co2_pv
    c_WT = co2_wt
    carbon_cost = P_CO2 * (c_G_buy * c_co2 + c_PV * power_PV*freq + c_WT * power_WT*freq)
    return carbon_cost

def calculate_degrade(power_FC,FC_pre_power,power_EL,EL_pre_power,power_bat):

    degrade_FC_cost = calculate_FC_degrade(power_FC, FC_pre_power)
    degrade_EL_cost = calculate_EL_degrade(power_EL, EL_pre_power)
    degrade_bat_cost = calculate_bat_degrade(power_bat)
    total_degrade_cost = degrade_FC_cost + degrade_EL_cost + degrade_bat_cost
    return total_degrade_cost

def calculate_FC_degrade(power_FC, pre_power):
    pre_delta = 1 if pre_power > 0 else 0
    delta_FC = 1 if power_FC > 0 else 0

    delta_FC_low = 1 if power_FC < 0.2 * n_FC else 0
    delta_FC_high = 1 if power_FC > 0.8 * n_FC else 0
    C_FC_low = delta_FC_low * zeta_FC_low * C_FC / V_FC_eol
    C_FC_high = delta_FC_high * zeta_FC_high * C_FC / V_FC_eol
    C_FC_chg = zeta_FC_chg * abs(power_FC - pre_power) * C_FC / (n_FC * V_FC_eol)
    C_FC_s = zeta_FC_s * abs(delta_FC - pre_delta) * C_FC / V_FC_eol
    C_FC_deg = C_FC_low + C_FC_high + C_FC_chg + C_FC_s

    return C_FC_deg

def calculate_EL_degrade(power_EL,EL_pre_power):
    delta_EL = 1 if power_EL > 0 else 0
    pre_delta_EL = 1 if EL_pre_power > 0 else 0
    C_EL_op = delta_EL * zeta_EL_op * C_EL / V_EL_eol
    C_EL_s = zeta_EL_s * C_EL * abs(delta_EL - pre_delta_EL) / V_EL_eol
    C_EL_degrade = C_EL_op + C_EL_s

    return C_EL_degrade

def calculate_bat_degrade(power_bat):
    Q_0 = C_bat * 1000
    N_cycles = calculate_N_cycles(power_bat)
    degrade_bat_cost = C_bat*freq*power_bat/(2*N_cycles*Q_0)
    return degrade_bat_cost

def calculate_N_cycles(power):
    N_cycles = p1*power**6 + p2*power**5+p3*power**4+p4*power**3 + \
                p5 * power**2+p6*power+p7

    return N_cycles

def calculate_unbalance_cost(p_pv,p_wt,p_fc,p_bat,p_buy,p_el,p_load):

    penalty = 100 if abs(p_pv+p_wt+p_fc+p_bat+p_buy - (p_el+p_load)) > 1 else 0
    return penalty
def calculate_extra_reward(pv_power,wt_power,p_fc,p_el,p_bat,e_buy,p_load):
    if 0 <= p_fc <= n_FC and -25 <= p_bat <= 25 and 0 <= p_el <= n_EL and abs(e_buy)<2000:
        reward = 50
    else:
        reward = 0

    return reward