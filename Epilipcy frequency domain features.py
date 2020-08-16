import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import mne
from scipy.signal import butter, lfilter
from scipy.stats import kurtosis
import pylab as p
##import keras

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
    #print ('info',info)
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
    print ("low cut", lowcut, "high cut", highcut)
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def filtereddata(updatedvalues, band):
    if (band == "delta"):
        print ("delta wave")
        filtered_data = butter_bandpass_filter(updatedvalues, lowcut = 0.5, highcut=4 , fs=173.61, order=5)
        time_domain_features(filtered_data)

    elif (band == "theta"):
        print ("theta wave")
        filtered_data = butter_bandpass_filter(updatedvalues, lowcut = 4, highcut=7 , fs=173.61, order=5)
        time_domain_features(filtered_data)

    elif (band == "alpha"):
        print ("alpha wave")
        filtered_data = butter_bandpass_filter(updatedvalues, lowcut = 7, highcut=13 , fs=173.61, order=5)
        time_domain_features(filtered_data)

    elif (band == "beta"):
        print ("beta wave")
        filtered_data = butter_bandpass_filter(updatedvalues, lowcut = 13, highcut=39 , fs=173.61, order=5)
        time_domain_features(filtered_data)

    elif (band == "gama"):
        print ("gama wave")
        filtered_data = butter_bandpass_filter(updatedvalues, lowcut = 40, highcut=85 , fs=173.61, order=5)
        time_domain_features(filtered_data)
    return filtered_data


def hjorth(X, D=None):
    """ Compute Hjorth mobility and complexity of a time series from either two
    cases below:
        1. X, the time series of type list (default)
        2. D, a first order differential sequence of X (if D is provided,
           recommended to speed up)
    In case 1, D is computed using Numpy's Difference function.
    Notes
    -----
    To speed up, it is recommended to compute D before calling this function
    because D may also be used by other functions whereas computing it here
    again will slow down.
    Parameters
    ----------
    X
        list
        a time series
    D
        list
        first order differential sequence of a time series
    Returns
    -------
    As indicated in return line
    Hjorth mobility and complexity
    """

    if D is None:
        D = np.diff(X)
        D = D.tolist()

    D.insert(0, X[0])  # pad the first difference
    D = np.array(D)

    n = len(X)

    M2 = float(sum(D ** 2)) / n
    TP = sum(np.array(X) ** 2)
    M4 = 0
    for i in range(1, len(D)):
        M4 += (D[i] - D[i - 1]) ** 2
    M4 = M4 / n

    return np.sqrt(M2 / TP), np.sqrt(
        float(M4) * TP / M2 / M2
    )  # Hjorth Mobility and Complexity
    
def Hjorthparam(updatedvalues):
    Hjorthmobility_arr = []
    Hjorthcompleity_arr = []

    filtered_data = filtereddata(updatedvalues,"beta")

    for i in range (0,100):
        Hjorthmobility, Hjorthcompleity = hjorth(filtered_data[i])
        Hjorthmobility_arr.append(Hjorthmobility)
        Hjorthcompleity_arr.append(Hjorthcompleity)

        
    print ("Hjorthmobility max ", max(Hjorthmobility_arr),"Hjorthcompleity max", max(Hjorthcompleity_arr))
    print ("Hjorthmobility min ", min(Hjorthmobility_arr),"Hjorthcompleity min", min(Hjorthcompleity_arr))


def kurtosysplot():


    indexes = np.arange(4097)

    kurtosyys = []
    for i in range (0,100):
        kurtosyys.append(kurtosis(updatedvalues[i]))

    max_kurtosys = max(kurtosyys)
    max_kurtosys_index = kurtosyys.index(max_kurtosys)

    min_kurtosys = min(kurtosyys)
    min_kurtosys_index = kurtosyys.index(min_kurtosys)


    print ("max kurtosys",max_kurtosys ,"index",max_kurtosys_index )
    print ("min kurtosys", min_kurtosys,"index", min_kurtosys_index )
    print ("average kurtosys",sum(kurtosyys)/len(kurtosyys))
    p.hist(updatedvalues[max_kurtosys_index],bins = 80)
    p.hist(updatedvalues[min_kurtosys_index],bins = 80)
    p.show()


all_files = File_read("Set E")

updatedvalues, info = data_fetch(all_files)
raw_plot(updatedvalues, info)

#raw_plot(updatedvalues, info)
#raw = mne.io.RawArray(updatedvalues, info)

##mne.find_events(raw,stim_channel = '8',  min_duration=0.01, shortest_event=5,initial_event=True,uint_cast=True)
##
##filtered_data = filtereddata(updatedvalues,"delta")
##power_spectral(0.5, 85, 2048, filtered_data, info)
##
##
##raw_plot(updatedvalues, info)
##power_spectral(0.5, 85, 2048, updatedvalues, info)
##time_domain_features(updatedvalues)
