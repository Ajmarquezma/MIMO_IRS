# This script contains the generator code for producing wireless network layouts and path-losses

import numpy as np
from settings_MIMO import *
import scipy.io

# Generate i.i.d Rayleigh fading channels
def generate_real_channels(flug_path):

    matlab_file= scipy.io.loadmat(flug_path, squeeze_me=True,simplify_cells= True)
    nested_data = matlab_file['chanData_EKF']
    sub_variable_data_1 =[]
    sub_variable_data_2 = []

    for x in range(len(nested_data)):
        sub_variable_data_1.append(nested_data[x]['chan1'])
    H_1 = np.array(sub_variable_data_1).transpose()


    for x in range(len(nested_data)):
        sub_variable_data_2.append(nested_data[x]['chan2'])
    H_2 = np.array(sub_variable_data_2).transpose()


    # Check the shapes of the arrays
    # print(f"Shape of H_1: {np.shape(H_1)}")
    # print(f"Shape of H_2: {np.shape(H_2)}")
    # Combine the arrays into a 3D matrix
    Channel = np.stack((H_1, H_2), axis=0).transpose()
    # print(f"Shape of Channel: {np.shape(Channel)}")
    return Channel




def estimate_channels(flug_path):

    matlab_file= scipy.io.loadmat(flug_path, squeeze_me=True,simplify_cells= True)
    nested_data = matlab_file['chanData_VIC']
    sub_variable_data_1 =[]
    sub_variable_data_2 = []

    for x in range(len(nested_data)):
        sub_variable_data_1.append(nested_data[x]['chan1'])
    H_1 = np.array(sub_variable_data_1).transpose()


    for x in range(len(nested_data)):
        sub_variable_data_2.append(nested_data[x]['chan2'])
    H_2 = np.array(sub_variable_data_2).transpose()


    # Check the shapes of the arrays
    # print(f"Shape of H_1_est: {np.shape(H_1)}")
    # print(f"Shape of H_2_est: {np.shape(H_2)}")
    # Combine the arrays into a 3D matrix
    H_est = np.stack((H_1, H_2), axis=0).transpose()
    # print(f"Shape of H_est: {np.shape(H_est)}")
    return H_est


def validate_channels(H):
    n_networks = np.shape(H)[0]
    n_BS_reals = np.random.normal(loc=0.0, scale=np.sqrt(NOISE_POWER/2), size=(n_networks, M, K))
    n_BS_imags = np.random.normal(loc=0.0, scale=np.sqrt(NOISE_POWER/2), size=(n_networks, M, K))
    n_BS = n_BS_reals + 1j*n_BS_imags
    pilots_received_BS = np.sqrt(ESTIMATION_PILOT_POWER)*np.conjugate(H) + n_BS
    H_conj_est = np.sqrt(ESTIMATION_PILOT_POWER)/(ESTIMATION_PILOT_POWER+NOISE_POWER) * pilots_received_BS
    H_est = np.conjugate(H_conj_est)
    return H_est

# Utilize only channel estimates
def get_beamformers(H_est):
    # zero-forcing beamforming
    B = np.matmul(H_est, np.linalg.inv(np.matmul(np.transpose(np.conjugate(H_est), (0,2,1)), H_est)+RCI_BF_ALPHA*np.eye(K)))
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return B

if(__name__=="__main__"):
    # Due to training and validation data being generated online
    # only generate testing data here (to be read by matlab code for computing geometric programming solutions)
    print(f'[MIMO {SETTING_STRING}] Generating testing data...')
    generate_real_channels('Data_MIMO/data/Flug1/Chans_EKF_1.mat')
    H_est = estimate_channels('Data_MIMO/data/Flug1/Chans_VIC_1.mat')
    B = get_beamformers(H_est)

    # compute effective channels to be read by geometric programming
    effectiveChannels = np.matmul(np.transpose(np.conjugate(H_est), (0, 2, 1)), B)
    # compute channel gains, with (i,j)th component being jth beamformer to ith user
    effectiveChannelGains = np.power(np.absolute(effectiveChannels),2)
    
    # save files
    np.save(f'Data_MIMO/channelEstimates_test_{SETTING_STRING}.npy', H_est)
    np.save(f'Data_MIMO/RCI_beamformers_test_{SETTING_STRING}.npy', B)
    scipy.io.savemat(f'Data_MIMO/effectiveChannelGains_test_{SETTING_STRING}.mat', {'effectiveChannelGains':effectiveChannelGains})
    print('Script finished successfully!')
