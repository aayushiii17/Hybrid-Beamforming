import os
import random
import torch
import numpy as np
import scipy.io as scio

from train_ddpg import train
from util_lib import load_ch, bf_gain_calc

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

if __name__ == '__main__':

    options = {
        'gpu_idx': 0,
        'num_ant': 256,
        'num_beams': 1,
        'num_bits': 3,
        'num_NNs': 1,
        'num_loop': 1,  # outer loop
        'target_update': 3,
        'pf_print': 10,

        'beam_idx': 0,

        'path': 'analog_beam_learning\H_fc.mat',
        

        'save_freq': 50000
    }

    train_opt = {
        'state': 0,
        'best_state': 0,
        'num_iter': 100,
        'tau': 1e-2,
        'overall_iter': 1,
        'replay_memory': [],
        'replay_memory_size': 256,
        'minibatch_size': 256,
        'gamma': 0
    }

    if not os.path.exists('beams/'):
        os.mkdir('beams/')

    if not os.path.exists('pfs/'):
        os.mkdir('pfs/')

    ch = load_ch(options['path'])  # numpy.ndarray: (# of user, 2 * # of ant)

    
#Pre-trained model from critic main file
    options['H_r'] = scio.loadmat('critic_net_training\critic_params_trsize_2000_epoch_500_3bit.mat')['H_r']
    options['H_i'] = scio.loadmat('critic_net_training\critic_params_trsize_2000_epoch_500_3bit.mat')['H_i']

# 'H_r' and 'H_i' from a .mat file and store them in the options dictionary.


    options['exp_name'] = 'trsize_2000_epoch_500_3bit'

    # -----------------------------------------------------------------------------

    X = ch

    target = np.array([1.])  # statistics interested
    options['target'] = target

    # ------------------------------- Quantization settings ---------------------------------------------- #

    options['num_ph'] = 2 ** options['num_bits'] 
    #calculates the number of phase levels (num_ph) based on the number of bits 
    options['multi_step'] = torch.from_numpy(
        np.linspace(int(-(options['num_ph'] - 2) / 2),
                    int(options['num_ph'] / 2),
                    num=options['num_ph'],
                    endpoint=True)).float().reshape(1, -1) # Reshapes the array into a 2D tensor with one row and as many columns as necessary to accommodate all elements.
    # np.linspace: returns an array of evenly spaced numbers over the interval
    options['pi'] = torch.tensor(np.pi)
    options['ph_table'] = (2 * options['pi']) / options['num_ph'] * options['multi_step']
    #(2 * options['pi']): Calculates the total phase range (from 0 to 2π).
    #/ options['num_ph']: Divides the total phase range by the number of phase levels to determine the size of each interval
    #Multiplies each interval size by the corresponding multi-step value to determine the phase angle represented by each quantization level.
    options['ph_table_rep'] = options['ph_table'].repeat(options['num_ant'], 1)

# line repeats the phase table for each antenna to create a table (ph_table_rep) where each row represents the quantization levels for one antenna
# It uses the repeat function to replicate the phase table num_ant times along the first dimension (rows), keeping the second dimension (columns) unchanged.
# This ensures that each antenna has access to the same set of quantization levels
    for beam_id in range(options['num_NNs']): #Number of neural networks.
        train(X, options, train_opt, beam_id) #beam id :  current value of the loop variable, representing the index of the beam

