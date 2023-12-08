import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from src.folderconstants import *
from shutil import copyfile
from sklearn import preprocessing
import matplotlib.pyplot as plt
from settings import *

datasets = ['synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR', 'MBA', 'NAB','addr1394']

wadi_drop = ['2_LS_001_AL', '2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS']

def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape

def load_and_save2(category, filename, dataset, dataset_folder, shape):
    temp = np.zeros(shape)
    with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
        ls = f.readlines()
    for line in ls:
        pos, values = line.split(':')[0], line.split(':')[1].split(',')
        start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i)-1 for i in values]
        temp[start-1:end-1, indx] = 1
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)

def normalize(a):
    a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
    return (a / 2 + 0.5)

def normalize2(a, min_a = None, max_a = None):
    if min_a is None: min_a, max_a = min(a), max(a)
    return (a - min_a) / (max_a - min_a), min_a, max_a

def normalize3(a, min_a = None, max_a = None):
    if min_a is None: min_a, max_a = np.min(a, axis = 0), np.max(a, axis = 0)
    return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a

def convertNumpy(df):
    x = df[df.columns[3:]].values[::10, :]
    return (x - x.min(0)) / (x.ptp(0) + 1e-4)

def load_data(dataset):
    folder = os.path.join(output_folder, dataset)
    os.makedirs(folder, exist_ok=True)
    if dataset == 'synthetic':
        train_file = os.path.join(data_folder, dataset, 'synthetic_data_with_anomaly-s-1.csv')
        test_labels = os.path.join(data_folder, dataset, 'test_anomaly.csv')
        dat = pd.read_csv(train_file, header=None)
        split = 10000
        train = normalize(dat.values[:, :split].reshape(split, -1))
        test = normalize(dat.values[:, split:].reshape(split, -1))
        lab = pd.read_csv(test_labels, header=None)
        lab[0] -= split
        labels = np.zeros(test.shape)
        for i in range(lab.shape[0]):
            point = lab.values[i][0]
            labels[point-30:point+30, lab.values[i][1:]] = 1
        test += labels * np.random.normal(0.75, 0.1, test.shape)
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
    elif dataset == 'SMD':
        dataset_folder = 'data/SMD'
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith('.txt'):
                load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
                s = load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
                load_and_save2('labels', filename, filename.strip('.txt'), dataset_folder, s)
    elif dataset == 'UCR':
        dataset_folder = 'data/UCR'
        file_list = os.listdir(dataset_folder)
        for filename in file_list:
            if not filename.endswith('.txt'): continue
            vals = filename.split('.')[0].split('_')
            dnum, vals = int(vals[0]), vals[-3:]
            vals = [int(i) for i in vals]
            temp = np.genfromtxt(os.path.join(dataset_folder, filename),
                                dtype=np.float64,
                                delimiter=',')
            min_temp, max_temp = np.min(temp), np.max(temp)
            temp = (temp - min_temp) / (max_temp - min_temp)
            train, test = temp[:vals[0]], temp[vals[0]:]
            labels = np.zeros_like(test)
            labels[vals[1]-vals[0]:vals[2]-vals[0]] = 1
            train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
            for file in ['train', 'test', 'labels']:
                np.save(os.path.join(folder, f'{dnum}_{file}.npy'), eval(file))
    elif dataset == 'NAB':
        dataset_folder = 'data/NAB'
        file_list = os.listdir(dataset_folder)
        with open(dataset_folder + '/labels.json') as f:
            labeldict = json.load(f)
        for filename in file_list:
            if not filename.endswith('.csv'): continue
            df = pd.read_csv(dataset_folder+'/'+filename)
            vals = df.values[:,1]
            labels = np.zeros_like(vals, dtype=np.float64)
            for timestamp in labeldict['realKnownCause/'+filename]:
                tstamp = timestamp.replace('.000000', '')
                index = np.where(((df['timestamp'] == tstamp).values + 0) == 1)[0][0]
                labels[index-4:index+4] = 1
            min_temp, max_temp = np.min(vals), np.max(vals)
            vals = (vals - min_temp) / (max_temp - min_temp)
            train, test = vals.astype(float), vals.astype(float)
            train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
            fn = filename.replace('.csv', '')
            for file in ['train', 'test', 'labels']:
                np.save(os.path.join(folder, f'{fn}_{file}.npy'), eval(file))
    elif dataset == 'MSDS':
        dataset_folder = 'data/MSDS'
        df_train = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
        df_test  = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))
        df_train, df_test = df_train.values[::5, 1:], df_test.values[::5, 1:]
        _, min_a, max_a = normalize3(np.concatenate((df_train, df_test), axis=0))
        train, _, _ = normalize3(df_train, min_a, max_a)
        test, _, _ = normalize3(df_test, min_a, max_a)
        labels = pd.read_csv(os.path.join(dataset_folder, 'labels.csv'))
        labels = labels.values[::1, 1:]
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
        # print(labels.shape)
        # print(labels)
    elif dataset == 'SWaT':
        dataset_folder = 'data/SWaT'
        file = os.path.join(dataset_folder, 'series.json')
        df_train = pd.read_json(file, lines=True)[['val']][3000:6000]
        df_test  = pd.read_json(file, lines=True)[['val']][7000:12000]
        train, min_a, max_a = normalize2(df_train.values)
        test, _, _ = normalize2(df_test.values, min_a, max_a)
        labels = pd.read_json(file, lines=True)[['noti']][7000:12000] + 0
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))

        # plt.plot(labels[:,0],c='b')
        # plt.plot(labels[:,1],alpha=0.7,c='r')
        # plt.show()
    elif dataset in ['SMAP', 'MSL']:
        dataset_folder = 'data/SMAP_MSL'
        file = os.path.join(dataset_folder, 'labeled_anomalies.csv')
        values = pd.read_csv(file)
        values = values[values['spacecraft'] == dataset]
        filenames = values['chan_id'].values.tolist()
        for fn in filenames:
            train = np.load(f'{dataset_folder}/train/{fn}.npy')
            test = np.load(f'{dataset_folder}/test/{fn}.npy')
            train, min_a, max_a = normalize3(train)
            test, _, _ = normalize3(test, min_a, max_a)
            np.save(f'{folder}/{fn}_train.npy', train)
            np.save(f'{folder}/{fn}_test.npy', test)
            labels = np.zeros(test.shape)
            indices = values[values['chan_id'] == fn]['anomaly_sequences'].values[0]
            indices = indices.replace(']', '').replace('[', '').split(', ')
            indices = [int(i) for i in indices]
            for i in range(0, len(indices), 2):
                labels[indices[i]:indices[i+1], :] = 1
            np.save(f'{folder}/{fn}_labels.npy', labels)
    elif dataset == 'WADI':
        dataset_folder = 'data/WADI'
        ls = pd.read_csv(os.path.join(dataset_folder, 'WADI_attacklabels.csv'))
        train = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days.csv'), skiprows=1000, nrows=2e5)
        test = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdata.csv'))
        train.dropna(how='all', inplace=True); test.dropna(how='all', inplace=True)
        train.fillna(0, inplace=True); test.fillna(0, inplace=True)
        test['Time'] = test['Time'].astype(str)
        test['Time'] = pd.to_datetime(test['Date'] + ' ' + test['Time'])
        labels = test.copy(deep = True)
        for i in test.columns.tolist()[3:]: labels[i] = 0
        for i in ['Start Time', 'End Time']: 
            ls[i] = ls[i].astype(str)
            ls[i] = pd.to_datetime(ls['Date'] + ' ' + ls[i])
        for index, row in ls.iterrows():
            to_match = row['Affected'].split(', ')
            matched = []
            for i in test.columns.tolist()[3:]:
                for tm in to_match:
                    if tm in i: 
                        matched.append(i); break            
            st, et = str(row['Start Time']), str(row['End Time'])
            labels.loc[(labels['Time'] >= st) & (labels['Time'] <= et), matched] = 1
        train, test, labels = convertNumpy(train), convertNumpy(test), convertNumpy(labels)
        print(train.shape, test.shape, labels.shape)
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
    elif dataset == 'MBA':
        dataset_folder = 'data/MBA'
        ls = pd.read_excel(os.path.join(dataset_folder, 'labels.xlsx'))
        train = pd.read_excel(os.path.join(dataset_folder, 'train.xlsx'))
        test = pd.read_excel(os.path.join(dataset_folder, 'test.xlsx'))
        train, test = train.values[1:,1:].astype(float), test.values[1:,1:].astype(float)
        train, min_a, max_a = normalize3(train)
        test, _, _ = normalize3(test, min_a, max_a)
        ls = ls.values[:,1].astype(int)
        labels = np.zeros_like(test)
        for i in range(-20, 20):
            labels[ls + i, :] = 1
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))

    elif dataset == 'addr1394':
        dataset_folder = 'data/addr1394'
        features_num = FEATURES_NUM #<arg>
        #~Read channel data (address)
        # 1394 protocal
        df_dst = pd.read_csv(os.path.join(dataset_folder,"channels_1394_DST.csv"))#目的地址（*）
        df_id = pd.read_csv(os.path.join(dataset_folder,"channels_1394_ID.csv"))   
        df_comb = pd.concat([df_dst, df_id], axis=1)
        channel = df_comb.apply(lambda x: x.astype(str).map(lambda x: int(x, base=16))).astype(float)
        
        # channel = pd.read_csv(os.path.join(dataset_folder,'channel.csv'), header=None)
        # channel = channel.apply(lambda x: x.astype(str).map(lambda x: int(x, base=16)))
        # channel = channel.astype(float)
        
        #~Normalization
        range_n = RANGE_N #<arg>
        scaler = preprocessing.MinMaxScaler()  #preprocessing.StandardScaler()##!!

        xc = channel.values[:range_n] #[0:8000] #[0:5000] #[0:3000] #[0:2000]  # [0:1500] #cut values
        
        #TODO: test linear/non-linear normalization
        xc_scaled = scaler.fit_transform(xc) #!!
        xc_log2 = np.log2(xc + 1) #!!
        xc_sin = np.sin(xc) #!!

        for name, xc_scaled in zip(['', '_log2', '_sin'], [xc_scaled, xc_log2, xc_sin]): #!!
            if DEBUG:
                print(xc_scaled.shape)
                plt.plot(xc[:range_n], label='xc')
                plt.plot(xc_scaled[:range_n], label='xc_s')
                plt.legend()
                plt.show()
            ##
            test_num = TEST_NUM#<arg>
            # split_ratio = SPLIT_RATIO #0.7 #0.5 #<arg>
            # xc_scaled = channel.values# !!
            # tc = xc_scaled[int(len(xc) * split_ratio):]
            # xc = xc_scaled[:int(len(xc) * split_ratio)]
            tc = xc_scaled[-test_num:]
            xc = xc_scaled[:-test_num]
            if DEBUG:
                # plt.plot(xc[0:500])
                # plt.plot(tc[0:500])
                # plt.show()
                print("train shape:", xc.shape)
                print("test shape:", tc.shape)
                plt.plot(tc[:1000], label='tc')
                # plt.plot(tc_scaled[:1000], label='tc_s')
                plt.legend()
                plt.show()
            ##

            #Label generatoin for tc
            disturb_scale = DISTURB_SCALE #255 #0.25 #<arg>
            disturb_probability = DISTURB_PROBABILITY #0.05 0.02 #<arg> 0.01
            disturb_n_threshold_min = DISTURB_N_THRESHOLD_MIN #0.2 <arg>
            disturb_n_threshold_max = DISTURB_N_THRESHOLD_MAX #0.8 #<arg>
            error_split_probablity = ERROR_SPLIT_PROBABLITY #<arg>
            # dd = 1
            disturbc = []
            labelsc = np.zeros_like(tc)
            ttc = tc.copy()
            np.random.seed(RANDOM_SEED) #<arg>

            for i,t in enumerate(tc):
                if np.random.rand(1)[0] < disturb_probability:
                # if np.random.randn(1)[0] > 2.1: # disturb_probability:
                    # if np.random.rand(1)[0] < error_split_probablity or i == 0 or i == len(tc)-1:
                    # add disturb
                    d = disturb_scale * np.random.random(1)[0]
                    while abs(d) < disturb_n_threshold_min or abs(d) > disturb_n_threshold_max or abs(d) == ttc[:,0][i]:
                        d = disturb_scale * np.random.random(1)[0] #disturb_scale * np.abs(np.random.randn(1)[0]) + 0.3 #(np.random.rand(1)[0] + 0.5) * 2 * disturb_scale
                    ttc[:,0][i] += d  #TODO: disturb_scale
                    ttc[:,0][i] = abs(ttc[:,0][i])

                    # d = disturb_scale * np.random.randn(1)[0]
                    # while d < disturb_n_threshold_min or d > disturb_n_threshold_max or d == ttc[i]:
                    #     d = disturb_scale * np.random.randn(1)[0]
                    # ttc[i] = d
                    # labelsc.append(np.array([1.0, 0.0])) #1 # abnormal
                    # print(ttc[i])
                    labelsc[i,0] = 1.0 #False #1 # abnormal [-TODO](+DONE:shift USAD and TrainAD output)：comment for test
                    disturbc.append(d)
                    # else:
                    #     # swap error
                    #     temp = tc[i + 1]
                    #     tc[i + 1] = tc[i]
                    #     tc[i] = temp
                    #     labelsc.append(np.array([1.0]))
                else:
                    #labelsc.append(np.array([0.0])) #0#[-TODO](+DONE:shift USAD and TrainAD output)：comment for test
                    disturbc.append(0.0)#[-TODO](+DONE:shift USAD and TrainAD output)：comment for test

            # print(labelsc[:200])
            # channel = pd.DataFrame(xc)
            # channel_ano = pd.DataFrame(tc)

            # print(channel.shape, channel_ano.shape)
            # channel.head(20)
            # channel_ano.head(50)
            if DEBUG:
                plt.figure(figsize=(12, 8))

                # Plot disturb
                plt.subplot(411)
                plt.plot(disturbc, label='disturb', linewidth=2, alpha=0.7)
                plt.legend()

                # Plot ttc
                plt.subplot(412)
                plt.plot(np.array(ttc), label='ttc', linewidth=2, alpha=0.7)
                plt.legend()

                # Plot vari
                plt.subplot(413)
                plt.plot(np.array(ttc) - np.array(tc), label='vari', linewidth=2, alpha=0.7)
                plt.legend()

                plt.subplot(414)
                plt.plot(labelsc, label='labels')
                plt.legend()
                plt.show()
            ###

            #################################################
            # train, min_a, max_a = normalize2(xc)
            # test, _, _ = normalize2(tc, min_a, max_a)
            train = np.array(xc).reshape((-1,features_num))#TODO：comment for test
            test = np.array(ttc).reshape((-1,features_num))#TODO：comment for test
            labels = np.array(labelsc, dtype=float).reshape((-1,features_num)) #pd.read_json(file, lines=True)[['noti']][7000:12000] + 0
            for file in ['train', 'test', 'labels']:
                np.save(os.path.join(folder, f'{file}.npy'), eval(file+name))

    else:
        raise Exception(f'Not Implemented. Check one of {datasets}')

if __name__ == '__main__':
    commands = sys.argv[1:]
    load = []
    if len(commands) > 0:
        for d in commands:
            load_data(d)
    else:
        print("Usage: python preprocess.py <datasets>")
        print(f"where <datasets> is space separated list of {datasets}")