import numpy as np
import gym
from gym.envs.classic_control.pendulum import PendulumEnv
import math
from matplotlib import pyplot as plt

class Controller():

    def __init__(self):

        self.parameters = {
            "swing_up_force": 4.0,
            "max_theta_dot_setpoint": 7.0,
            "gain_theta_dot_setpoint": 4.0,
            "gain_torque":2.0,
        }

        self.env = PendulumEnv()

        self.step_count = 0
        self.log_dict = {}

    def run(self):
        observation = self.env.reset()
        try:
            while True:
                action = self.controller(observation)
                observation = self.env.step(action)[0]
                self.env.render()
                self.step_count += 1
        except KeyboardInterrupt:
            self.display_logs()

    def controller(self,observation):

        x,y,theta_dot = observation

        swing_up_action = self.swing_up_controller(observation)
        balance_action = self.balance_controller(observation)
        if x < 0:
            action = swing_up_action
        else:
            action = balance_action
        
        # print(observation,action)
        return action

    def swing_up_controller(self,observation):
        x,y,theta_dot = observation
        action = math.copysign(self.parameters["swing_up_force"],theta_dot)
        return np.array([action])

    def balance_controller(self,observation):
        p = self.parameters
        x,y,theta_dot = observation
        theta = math.atan2(y,x)
        self.log("theta_dot",theta_dot)
        self.log("theta",theta)

        
        theta_dot_setpoint = np.clip(-theta*p["gain_theta_dot_setpoint"],-p["max_theta_dot_setpoint"],p["max_theta_dot_setpoint"])
        self.log("theta_dot_setpoint",theta_dot_setpoint)

        theta_dot_error = theta_dot_setpoint - theta_dot
        self.log("theta_dot_error",theta_dot_error)

        torque = theta_dot_error*p["gain_torque"]
        self.log("torque",torque)

        return np.array([torque])

    def log(self,key,value):
        if key in self.log_dict:
            step_list, value_list = self.log_dict[key]
        else:
            step_list,value_list = list(),list()
            self.log_dict[key] = (step_list,value_list)

        step_list.append(self.step_count)
        value_list.append(value)

    def display_logs(self):
        figure, axis = plt.subplots(figsize=(16,9))

        for key,(step_list,value_list) in self.log_dict.items():
            axis.plot(step_list,value_list,label=key)
        figure.legend()
        figure.savefig("plot.png",dpi=120)
        # plt.show()




if __name__ == "__main__":
    c = Controller()
    c.run()