import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import mne
import csv
from scipy.signal import butter, lfilter

#import plotly.plotly as py


def File_read(folder_sel):
    #default set A folder
    filepath = r"C:\Users\Abhi\Desktop\2018\pHd\EEG Database\Epilipcy\Time Series\Z"

    if (folder_sel == "Set A"):
        ##Set A - Z folder Healthy voulenteer eyes open
        print ("Set A")
        filepath = r"C:\Users\Abhi\Desktop\2018\pHd\EEG Database\Epilipcy\Time Series\Z"

    if (folder_sel == "Set B"):
        ##Set B - O folder Healthy voulenteer eyes closed
        print ("Set B")
        filepath = r"C:\Users\Abhi\Desktop\2018\pHd\EEG Database\Epilipcy\Time Series\O"

    if (folder_sel == "Set C"):
        ##Set C - N folder hippocampal formated patients in seizure free interval
        print ("Set C")
        filepath = r"C:\Users\Abhi\Desktop\2018\pHd\EEG Database\Epilipcy\Time Series\N"

    if (folder_sel == "Set D"):
        ##Set D - F folder Epileptic patients in seizure free interval
        print ("Set D")
        filepath = r"C:\Users\Abhi\Desktop\2018\pHd\EEG Database\Epilipcy\Time Series\F"

    if (folder_sel == "Set E"):
        ##Set E - S folder Epileptic patients in seizure interval
        print ("Set E")
        filepath = r"C:\Users\Abhi\Desktop\2018\pHd\EEG Database\Epilipcy\Time Series\S"




    all_files = glob.glob(filepath + "/*.txt")
    return all_files

def data_fetch(all_files):
    #This code will load files into respective sensor array
    updatedvalues = np.empty([100, 4097],dtype =int)
    j = 0
    indexes = np.arange(4097)
    channelname = np.arange(100)

    ch_names = channelname.tolist()
    ch_names = ([str(x) for x in ch_names])

    ch_types = 'eeg'
    sfreq = 173.61

    info = mne.create_info(ch_names, sfreq, 'eeg')


    for filename in all_files :
        updatedvalues[j] = np.loadtxt(filename)
        j = j+1
        #print ("Processing file :", filename)
        


    print ((updatedvalues).shape, "length updated values")
    print (updatedvalues)
    return updatedvalues, info

def raw_plot(updatedvalues, info):
    #This code will show matplotlib timeseries representation

    ##plt.plot(indexes, updatedvalues[0])
    ##plt.xlabel('time (s)')
    ##plt.ylabel('MEG data (T)')
    ##plt.show()

    scalings = {'eeg': 500}

    raw = mne.io.RawArray(updatedvalues, info)
    raw.plot(n_channels = 15, scalings=scalings)


def power_spectral(fmin, fmax, n_fft, updatedvalues, info):
    raw = mne.io.RawArray(updatedvalues, info)
    
    raw.plot_psd(fmin = fmin, fmax = fmax, n_fft=n_fft) 



def time_domain_features(updatedvalues):
    mean = np.mean(updatedvalues)
    std_deviation = np.std(updatedvalues)
    print ("mean", mean)
    print ("Standard deviation",std_deviation) 
    print("Q1 quantile  : ", np.quantile(updatedvalues, .25))
    print("Q2 quantile  : ", np.quantile(updatedvalues, .50))
    print("Q3 quantile  : ", np.quantile(updatedvalues, .75))
    print("Max : ", np.amax(updatedvalues))
    print("Min : ", np.amin(updatedvalues))


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    ##print ("low cut", lowcut, "high cut", highcut)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y




def filtereddata(updatedvalues, band):
    if (band == "delta"):
        #print ("delta wave")
        filtered_data = butter_bandpass_filter(updatedvalues, lowcut = 0.5, highcut=4 , fs=173.61, order=5)
        #time_domain_features(filtered_data)

    elif (band == "theta"):
        #print ("theta wave")
        filtered_data = butter_bandpass_filter(updatedvalues, lowcut = 4, highcut=7 , fs=173.61, order=5)
        #time_domain_features(filtered_data)

    elif (band == "alpha"):
        #print ("alpha wave")
        filtered_data = butter_bandpass_filter(updatedvalues, lowcut = 7, highcut=13 , fs=173.61, order=5)
        #time_domain_features(filtered_data)

    elif (band == "beta"):
        #print ("beta wave")
        filtered_data = butter_bandpass_filter(updatedvalues, lowcut = 13, highcut=39 , fs=173.61, order=5)
        #time_domain_features(filtered_data)

    elif (band == "gama"):
        #print ("gama wave")
        filtered_data = butter_bandpass_filter(updatedvalues, lowcut = 40, highcut=85 , fs=173.61, order=5)
        #time_domain_features(filtered_data)
    return filtered_data

def prediction_timeseries(setval):    

    all_files = File_read(setval)
    updatedvalues, info = data_fetch(all_files)

    #print (updatedvalues.shape)
    start,end = 0, 100
    segments_start = []
    segment_end = []
    std_deviation_array = []
    std_deviation_max_array = []
    std_deviation_min_array = []
    classifer = []
    for i in range (0,40):
        for j in range (0,99):
            filtered_data = filtereddata(updatedvalues[j][start:end],"gama")
            std_deviation = np.std(filtered_data)
            std_deviation_array.append(std_deviation)
        if  (max(std_deviation_array) > 220):
            classifer.append(3)
        elif(max(std_deviation_array) > 60 and max(std_deviation_array) < 220):
            classifer.append(2)
        else:
            classifer.append(1)
        std_deviation_max_array.append(max(std_deviation_array))
        std_deviation_min_array.append(min(std_deviation_array))
        start = start + 100
        end = end +100
        std_deviation_array = []
    #print ("std_deviation_array min max", min(std_deviation_array),max(std_deviation_array) )
    print (len(std_deviation_array))
    print ("Max of std_deviation_max_array",max(std_deviation_max_array),std_deviation_max_array.index(max(std_deviation_max_array)))
##    print ("Min of std_deviation_max_array",min(std_deviation_max_array))
##    print ("Max of std_deviation_min_array",max(std_deviation_min_array))
##    print ("Min of std_deviation_min_array",min(std_deviation_min_array))


    print ("Epileptic", classifer.count(3))
    print ("Early Epileptic", classifer.count(2))
    print ("Non Epileptic", classifer.count(1))


def box_plot(frequen):
    all_files = File_read("Set A")
    updatedvalues, info = data_fetch(all_files)    
    filtered_data = filtereddata(updatedvalues,frequen)
    all_files = File_read("Set B")
    updatedvalues, info = data_fetch(all_files)    
    filtered_data1 = filtereddata(updatedvalues,frequen)
    all_files = File_read("Set C")
    updatedvalues, info = data_fetch(all_files)    
    filtered_data2 = filtereddata(updatedvalues,frequen)
    all_files = File_read("Set D")
    updatedvalues, info = data_fetch(all_files)    
    filtered_data3 = filtereddata(updatedvalues,frequen)
    all_files = File_read("Set E")
    updatedvalues, info = data_fetch(all_files)    
    filtered_data4 = filtereddata(updatedvalues,frequen)


    data = [filtered_data[18], filtered_data1[3], filtered_data2[36], filtered_data3[21], filtered_data4[4]]
    fig7, ax7 = plt.subplots()
    ax7.set_title(frequen,': wave comparison')
    ax7.boxplot(data)

    plt.show()


all_files = File_read("Set D")
updatedvalues, info = data_fetch(all_files)
filtered_data = filtereddata(updatedvalues,"delta")
raw = mne.io.RawArray(filtered_data, info)
events = mne.find_events(raw,stim_channel = '7',min_duration=0.01, shortest_event=2)
epochs = mne.Epochs(raw, events, event_id=1, tmin=-0.2, tmax=0.5,
                    baseline=(-0.2, 0.0), decim=3,  # we'll decimate for speed
                    verbose='error')
print (len(epochs.events))
noise_cov_baseline = mne.compute_covariance(epochs, tmax=0)
noise_cov_baseline.plot(epochs.info, proj=True)
noise_cov_reg = mne.compute_covariance(epochs, tmax=0., method='auto',
                                       rank=None)
evoked = epochs.average()
evoked.plot_white(noise_cov_reg, time_unit='s')
##raw_empty_room.info['bads'] = [bb for bb in raw.info['bads'] if 'EEG' not in bb]
##raw_empty_room.add_proj([pp.copy() for pp in raw.info['projs'] if 'EEG' not in pp['desc']])
##noise_cov = mne.compute_raw_covariance(raw_empty_room, tmin=0, tmax=None)

##power_spectral(0.5, 85, 2048, filtered_data, info)
##with open('SetE delta.csv', 'w', newline='') as file:
##    writer = csv.writer(file)
##    for i in filtered_data:
##        writer.writerow(i)



#raw_plot(updatedvalues, info)
##power_spectral(0.5, 85, 2048, updatedvalues, info)
##time_domain_features(updatedvalues)
