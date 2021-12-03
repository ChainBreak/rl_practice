from matplotlib import pyplot as plt
import numpy as np

class PendulumPhasePlot():

    def __init__(self):

        self.figure, self.axis_1 = plt.subplots()

        self.set_axis_1_layout_settings()

    def set_axis_1_layout_settings(self):
        self.axis_1.set_xlim(-np.pi,np.pi)
        self.axis_1.set_xlabel("Angle (radians), 0 rads is facing down")
        self.axis_1.set_ylabel("Anglular Speed (radians)")

    def plot_transition_dataset(self,transition_dataset):
        states = transition_dataset.state_tensor.numpy()
        next_states = transition_dataset.next_state_tensor.numpy()

        theta = self.extract_theta_from_states(states)
        theta_dot = self.extract_theta_dot_from_states(states)

        theta_2 = self.extract_theta_from_states(next_states)
        theta_dot_2 = self.extract_theta_dot_from_states(next_states)

        theta_diff = theta_2 - theta
        theta_dot_diff = theta_dot_2 - theta_dot

        mask = theta_diff > 3.141
        theta_diff[mask] -= 3.141*2

        mask = theta_diff < -3.141
        theta_diff[mask] += 3.141*2

        self.axis_1.quiver(theta,theta_dot,theta_diff,theta_dot_diff, angles="xy",scale_units='xy')

        plt.show()

    
    def extract_theta_from_states(self,states):
        theta = np.arctan2(states[:,1],-states[:,0])
        return theta

    def extract_theta_dot_from_states(self,states):
        theta_dot = states[:,2]
        return theta_dot





