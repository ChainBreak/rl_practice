from matplotlib import pyplot as plt
import numpy as np
import torch
from batching import generate_batch_indicies

class PendulumPhasePlot():

    def __init__(self):

        self.figure, self.axis_1 = plt.subplots()

        self.set_axis_1_layout_settings()

    def set_axis_1_layout_settings(self):
        self.axis_1.set_xlim(-np.pi,np.pi)
        self.axis_1.set_xlabel("Angle (radians), 0 rads is facing down")
        self.axis_1.set_ylabel("Anglular Speed (radians)")

    def plot_transitions_on_phase_plot(self,state_tensor,next_state_tensor):
        states = state_tensor.numpy()
        next_states = next_state_tensor.numpy()

        theta = self.extract_theta_from_states(states)
        theta_dot = self.extract_theta_dot_from_states(states)

        theta_2 = self.extract_theta_from_states(next_states)
        theta_dot_2 = self.extract_theta_dot_from_states(next_states)

        theta_diff = theta_2 - theta
        theta_dot_diff = theta_dot_2 - theta_dot

        mask = theta_diff > np.pi
        theta_diff[mask] -= np.pi*2

        mask = theta_diff < -np.pi
        theta_diff[mask] += np.pi*2

        self.axis_1.quiver(theta,theta_dot,theta_diff,theta_dot_diff, angles="xy",scale_units='xy',scale=1,width=0.0005)

        plt.show()

    
    def extract_theta_from_states(self,states):
        theta = np.arctan2(states[:,1],states[:,0])
        return theta

    def extract_theta_dot_from_states(self,states):
        theta_dot = states[:,2]
        return theta_dot


def generate_pendulum_state_space_mesh(mesh_width,mesh_height):
    
    max_theta_dot = 8

    theta = torch.linspace(-np.pi,np.pi,mesh_width)
    theta_dot = torch.linspace(max_theta_dot,-max_theta_dot,mesh_height)

    theta_dot_mesh, theta_mesh  = torch.meshgrid(theta_dot,theta)

    x_mesh = torch.cos(theta_mesh)
    y_mesh = torch.sin(theta_mesh)

    state_space_mesh = torch.stack([x_mesh,y_mesh,theta_dot_mesh],dim=2)

    return state_space_mesh


def render_value_image(value_model, width, height):
    batch_size=1024
    with torch.no_grad():
        mesh = generate_pendulum_state_space_mesh(width,height)

        state_tensor = mesh.reshape(-1,3)
        num_states = len(state_tensor) 

        value_tensor = torch.zeros(num_states,1)

        for batch_indicies in generate_batch_indicies(batch_size, num_states):

            batch_state_tensor = state_tensor[batch_indicies]

            batch_value_tensor = value_model(batch_state_tensor)

            value_tensor[batch_indicies] = batch_value_tensor

        
        value_tensor = value_tensor.reshape(height,width)

        return value_tensor.numpy()


def plot_value_on_phase_plot(value_model,output_file):
    width = 192
    height = 108
    
    value_image = render_value_image(value_model,width,height)

    figure, axes_1 = plt.subplots()


    axes_1.set_xticks( np.linspace(0,width-1, 5) )
    axes_1.set_xticklabels( ["-pi" ,"-pi/2", 0, "pi/2", "pi"] )

    axes_1.set_yticks( np.linspace(0,height-1, 5) )
    axes_1.set_yticklabels( np.linspace(8,-8, 5) )

    axes_1.set_xlabel("Angle (Radians), 0 is balancing up")
    axes_1.set_ylabel("Anglular Speed (radians)")

  
    axes_1.imshow(value_image)
    plt.show()
