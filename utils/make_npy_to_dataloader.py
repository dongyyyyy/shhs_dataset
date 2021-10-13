from utils.function import *

import os
import numpy as np

# Multi-processing
import multiprocessing
from multiprocessing import Process, Manager, Pool, Lock

def func_make_dataloader_dataset(arg_list):

    filename = arg_list[0]
    preprocessing = arg_list[1]
    min_value = arg_list[2]
    max_value = arg_list[3]
    
    signals_path = '/mnt/ssd2/dataset/SHHS/5channel_bandpassfilter_each/'
    annotations_path = '/mnt/ssd2/dataset/SHHS/5channel_bandpassfilter_each/annotations/'
    save_path = '/mnt/ssd2/dataset/SHHS/5channels_standardScaler_dataloader_each/'

    current_c4a1_save_path = save_path + 'C4A1/' +filename.split('.npy')[0]+'/'
    current_c3a2_save_path = save_path + 'C3A2/' +filename.split('.npy')[0]+'/'
    current_eogL_save_path = save_path + 'EOGL/' +filename.split('.npy')[0]+'/'
    current_eogR_save_path = save_path + 'EOGR/' +filename.split('.npy')[0]+'/'
    current_emg_save_path = save_path + 'EMG/' +filename.split('.npy')[0]+'/'

    # os.makedirs(current_c4a1_save_path,exist_ok=True)
    # os.makedirs(current_c3a2_save_path,exist_ok=True)
    # os.makedirs(current_eogL_save_path,exist_ok=True)
    # os.makedirs(current_eogR_save_path,exist_ok=True)
    # os.makedirs(current_emg_save_path,exist_ok=True)

    # print(current_save_path)
    channel_list = ['C4A1','C3A2','EOGL','EOGR','EMG']
    for channel in channel_list:
        signals = np.load(signals_path+channel+'/'+filename)
        signals_std = np.std(signals)
        signals_mean = np.mean(signals)
        signals = data_preprocessing_numpy_mean_std(signals,signals_mean=signals_mean,signals_std=signals_std)
        
        annotations = np.load(annotations_path+filename)

        current_save_path = save_path + channel + '/' + filename.split('.npy')[0]+'/'
        os.makedirs(current_save_path,exist_ok=True)

        width = 125 * 30
        signals_len = len(signals)// width
        if signals_len == len(annotations):
            for index in range(signals_len):
                save_signals = signals[index*width : (index+1)*width]
                if index < 10:
                    save_index = '000%d'%(index)
                elif index < 100:
                    save_index = '00%d'%index
                elif index < 1000:
                    save_index = '0%d'%index
                else:
                    save_index = '%d'%index
                save_filename = current_save_path+'%s_%d.npy'%(save_index,annotations[index])
                # print(save_filename, annotations[index])
                # exit(1)
                np.save(save_filename,save_signals.reshape(1,-1))
            print(f'finish : {current_save_path}')
            # exit(1)


def make_dataloader_dataset():
    signals_path = '/mnt/ssd2/dataset/SHHS/5channel_bandpassfilter_each/C4A1/'
    annotations_path = '/mnt/ssd2/dataset/SHHS/5channel_bandpassfilter_each/annotations/'
    save_path = '/mnt/ssd2/dataset/SHHS/5channels_standardScaler_dataloader_each/'
    
    os.makedirs(save_path,exist_ok=True)

    file_list = os.listdir(signals_path)
    # print(file_list)
    cpu_num = multiprocessing.cpu_count()
    print('cpu_num : ',cpu_num)
    arg_list = []
    for i in range(len(file_list)):
        arg_list.append([file_list[i],'Standard',-1,1])

    # start = time.time()
    pool = Pool(cpu_num//4)

    pool.map(func_make_dataloader_dataset,arg_list)
    pool.close()
    pool.join()