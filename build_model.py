import torch
import torch.nn as nn

# H_r = nn.Parameter(torch.randn(256, 2), requires_grad=True)
# print(H_r)
# x_state = phase2bf(state)
# x_state_r, x_state_i = x_state[:, :256], x_state[:, 256:]
# print("x_state_r",x_state_r,x_state_i) 
class LearningModel(nn.Module):

    def __init__(self, in_size, ou_size): #input size, output size
        super(LearningModel, self).__init__()

        self.M = int(in_size / 2) #calculates half of the input size 
        self.output_size = ou_size

        # just for debugging
        # self.ch_r = torch.from_numpy(ch[:, :self.M].transpose()).float()
        # self.ch_i = torch.from_numpy(ch[:, self.M:].transpose()).float()

        self.H_r = nn.Parameter(torch.randn(self.M, 2), requires_grad=True)
        #print(self.H_r)
#torch.randn : initializes a tensor of shape (self.M, 2) with random values drawn from a normal distribution with mean 0 and standard deviation 1. Here, self.M represents half of the input size, 
#and 2 represents the number of output features (real and imaginary parts).
#requires_grad=True:Indicates that gradients with respect to these parameters should be computed during backpropagation, 
#allowing the optimizer to update their values
        self.H_i = nn.Parameter(torch.randn(self.M, 2), requires_grad=True)

    def forward(self, state_action_pair):

        # x = torch.cat((state, action), 1)
#state_action_pair: input tensor containing both the state and action information.
        state = state_action_pair[:, :self.M] 
        #print('state',state)
        action = state_action_pair[:, self.M:]
        #print('action',action)
        x_state = phase2bf(state)
#phase2bf: This function converts the phase matrix into a beamforming matrix.        
        x_state_r, x_state_i = x_state[:, :self.M], x_state[:, self.M:]
        #print("x_state_r",x_state_r,x_state_i) 
        z_r = x_state_r @ self.H_r + x_state_i @ self.H_i 
#State Beamforming Matrix: self.H_r & self.H_i
#Real & Imaginary Parts of the Output: z_r, z_i after applying the state beamforming matrix to the input 

#x_state is obtained by applying a function called phase2bf() to the input state_action_pair  
#These variables compute the real and imaginary parts of the output after applying the state beamforming matrix.        
        z_i = x_state_i @ self.H_r - x_state_r @ self.H_i

        z = z_r ** 2 + z_i ** 2
#computes the magnitude squared of the output after applying the state beamforming matrix.
        z_min = torch.mean(z, dim=1).reshape(-1, 1)
        # calculates the minimum value across all dimensions of z.


        x_action = phase2bf(action)
        x_action_r, x_action_i = x_action[:, :self.M], x_action[:, self.M:]

        u_r = x_action_r @ self.H_r + x_action_i @ self.H_i
        u_i = x_action_i @ self.H_r - x_action_r @ self.H_i

        u = u_r ** 2 + u_i ** 2 
#computes the magnitude squared of the output after applying the action beamforming matrix.
        u_min = torch.mean(u, dim=1).reshape(-1, 1)
#calculates the minimum value across all dimensions of u
        out = 10 * torch.log10(u_min) - 10 * torch.log10(z_min)
#computes difference between the logarithm of u_min and z_min.
        return out


def phase2bf(ph_mat):
#function converts the phase matrix (ph_mat) into a beamforming matrix.
    # ph_mat: (i) a tensor, (ii) B x M
    # bf_mat: (i) a tensor, (ii) B x 2M
    # B stands for batch size and M is the number of antenna

    M = torch.tensor(ph_mat.shape[1]).to(ph_mat.device)
    bf_mat = torch.exp(1j * ph_mat) # computes the complex exponential of the phase matrix.
    bf_mat_r = torch.real(bf_mat) #real and imaginary parts of the beamforming matrix.
    bf_mat_i = torch.imag(bf_mat)

    bf_mat_ = (1 / torch.sqrt(M)) * torch.cat((bf_mat_r, bf_mat_i), dim=1)
#concatenates the real and imaginary parts of the beamforming matrix and normalizes it.
    return bf_mat_
