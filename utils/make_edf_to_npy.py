import matplotlib.pyplot as plt
import mne
import os
import numpy as np
import re
# Multi-processing
import multiprocessing
from multiprocessing import Process, Manager, Pool, Lock

#pyedf Lib
from pyedflib import highlevel
from sklearn import preprocessing

from utils.function import *


def my_thread_SHHS(arg):
    signals_path = arg[0]
    annotations_path = arg[1]
    signals_pyedf, signals_info_pyedf, info = highlevel.read_edf(signals_path)
    root_save_path = '/mnt/ssd2/dataset/SHHS/5channel_bandpassfilter_each/'

    annotation_savepath = '/mnt/ssd2/dataset/SHHS/5channel_bandpassfilter_each/annotations/'
    os.makedirs(root_save_path+'C4A1/',exist_ok=True)
    os.makedirs(root_save_path+'C3A2/',exist_ok=True)
    os.makedirs(root_save_path+'EOGL/',exist_ok=True)
    os.makedirs(root_save_path+'EOGR/',exist_ok=True)
    os.makedirs(root_save_path+'EMG/',exist_ok=True)
    os.makedirs(annotation_savepath,exist_ok=True)

    c4a1_index = -1
    c3a2_index = -1
    eogL_index = -1
    eogR_index = -1
    emg_index = -1

    for i in range(len(signals_info_pyedf)):
        if signals_info_pyedf[i]['label'] == 'EEG(sec)':
            c3a2_index = i
        if signals_info_pyedf[i]['label'] == 'EEG':
            c4a1_index = i
        if signals_info_pyedf[i]['label'] == 'EMG':
            emg_index = i
        if signals_info_pyedf[i]['label'] == 'EOG(L)':
            eogL_index = i
        if signals_info_pyedf[i]['label'] == 'EOG(R)':
            eogR_index = i
    if c4a1_index != -1 and c3a2_index != -1 and eogL_index != -1 and eogR_index != -1 and emg_index != -1:

        c4a1_signals = np.array(signals_pyedf[c4a1_index])
        c3a2_signals = np.array(signals_pyedf[c3a2_index])
        eogL_signals = np.array(signals_pyedf[eogL_index])
        eogR_signals = np.array(signals_pyedf[eogR_index])
        emg_signals = np.array(signals_pyedf[emg_index])

        eeg_lowcut = 0.5
        eeg_highcut = 35

        eog_lowcut = 0.3
        eog_highcut = 14 #

        emg_lowcut = 10
        emg_highcut = 40

        eeg_sample_rate = 125
        eog_sample_rate = 50
        emg_sample_rate = 125

        order = 4

        resampling = True
        resampling_size = 125


        c4a1_signals = butter_bandpass_filter(c4a1_signals, eeg_lowcut, eeg_highcut, fs=eeg_sample_rate , order = order)
        c3a2_signals = butter_bandpass_filter(c3a2_signals, eeg_lowcut, eeg_highcut, fs=eeg_sample_rate, order=order)

        emg_signals = butter_bandpass_filter(emg_signals, emg_lowcut, emg_highcut, fs=emg_sample_rate, order= order)
        eogR_signals = butter_bandpass_filter(eogR_signals, eog_lowcut, eog_highcut, fs=eog_sample_rate, order=order)
        eogL_signals = butter_bandpass_filter(eogL_signals, eog_lowcut, eog_highcut, fs=eog_sample_rate, order=order)


        if resampling:
            eogR_signals = signal.resample(eogR_signals, resampling_size*(len(c4a1_signals)//eeg_sample_rate))
            # print('origin shape : ',eogL_signals.shape)
            eogL_signals = signal.resample(eogL_signals, resampling_size*(len(c4a1_signals)//eeg_sample_rate))
            # print('resample shape : ',eogL_signals.shape)


        stages = read_annot_regex(annotations_path)
        stages = np.array(stages)

        # Sleep stage 4 merge in Sleep stage 3
        if np.max(stages) > 5:
            print(f'{signals_path} file is fault(Annotations) !!!')
        else:
            indices = np.where(stages == 4)
            stages[indices] = 3

            # [0,1,2,3,4,5] => [0,1,2,3,4] 3 = Sleep 3 + 4, 4 = REM
            le = preprocessing.LabelEncoder()
            le.fit([0, 1, 2, 3, 5])
            stages = le.transform(stages)
            if len(stages) != int(len(c4a1_signals)/30/eeg_sample_rate):
                print(f'{signals_path} file is fault(Different length between signals and labels !!!')
            else:
                # os.makedirs(root_save_path+'C4A1/',exist_ok=True)
                # os.makedirs(root_save_path+'C3A2/',exist_ok=True)
                # os.makedirs(root_save_path+'EOGL/',exist_ok=True)
                # os.makedirs(root_save_path+'EOGR/',exist_ok=True)
                # os.makedirs(root_save_path+'EMG/',exist_ok=True)
                patient_name = signals_path.split('/')[-1].split('.edf')[0]
                np.save(root_save_path+'C4A1/'+patient_name+'.npy',c4a1_signals)
                np.save(root_save_path+'C3A2/'+patient_name+'.npy',c3a2_signals)
                np.save(root_save_path+'EOGL/'+patient_name+'.npy',eogL_signals)
                np.save(root_save_path+'EOGR/'+patient_name+'.npy',eogR_signals)
                np.save(root_save_path+'EMG/'+patient_name+'.npy',emg_signals)

                np.save(annotation_savepath + patient_name+'.npy',stages)

    else:
        print(f'{signals_path} file is fault!!!')


def make_edf_to_npy_shhs():
    data_path = '/mnt/hdd3/shhs/polysomnography/edfs/shhs1/'
    annotation_path = '/mnt/hdd3/shhs/polysomnography/annotations-events-nsrr/shhs1/'
    cpu_num = multiprocessing.cpu_count()
    data_list = os.listdir(data_path)
    data_list.sort()
    path_list = []
    for i in range(len(data_list)):
        path_list.append([data_path+data_list[i],annotation_path+data_list[i].split('.edf')[0]+'-nsrr.xml'])

    for i in range(len(path_list)):
        if path_list[i][0].split('/')[-1].split('.edf') != path_list[i][1].split('/')[-1].split('-nsrr.xml'):
            print('This is fault!!!')
        
    pool = Pool(cpu_num//4)

    pool.map(my_thread_SHHS,path_list)
    pool.close()
    pool.join()