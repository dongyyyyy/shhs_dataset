from scipy import signal
from scipy.signal import butter,filtfilt, lfilter
import re

# lowpass filter
def butter_lowpass_filter(data, cutoff, order=4,nyq=100):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(N=order, Wn=normal_cutoff, btype='low', analog=False,output='ba')
    y = filtfilt(b, a, data)
    return y

# highpass filter
def butter_highpass_filter(data, cutoff, order=4,fs=200):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(N=order, Wn=normal_cutoff, btype='high', analog=False,output='ba')

    y = filtfilt(b, a, data)
    # b = The numerator coefficient vector of the filter (분자)
    # a = The denominator coefficient vector of the filter (분모)

    return y

def butter_bandpass(lowcut, highcut, fs=200 , order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b,a = butter(N=order,Wn=[low,high],btype='bandpass', analog=False,output='ba')
    return b,a

# bandpass filter
def butter_bandpass_filter(signals, lowcut, highcut, fs , order = 4):
    b,a = butter_bandpass(lowcut,highcut,fs,order=order)

    y = lfilter(b,a,signals)
    return y

def butter_filter_sos(signals, lowcut=None, highcut=None, fs=200 , order =4):
    if lowcut != None and highcut != None: # bandpass filter
        sos = signal.butter(N=order,Wn=[lowcut,highcut],btype='bandpass',analog=False,output='sos',fs=fs)
        filtered = signal.sosfilt(sos,signals)
    elif lowcut != None and highcut == None: # highpass filter
        sos = signal.butter(N=order,Wn=lowcut,btype='highpass',analog=False,output='sos',fs=fs)
    elif lowcut == None and highcut != None:
        sos = signal.butter(N=order,Wn=highcut,btype='lowpass',analog=False,output='sos',fs=fs)
    else: # None filtering
        return signals
    filtered = signal.sosfilt(sos,signals)
    return filtered

def ellip_filter_sos(signals,rp=6,rs=53, lowcut=None, highcut=None, fs = 200 , order = 4):
    if lowcut != None and highcut != None: # bandpass filter
        sos = signal.ellip(N=order,rp=rp,rs=rs,Wn=[lowcut,highcut],btype='bandpass',analog=False,output='sos',fs=fs)
    elif lowcut != None and highcut == None: # highpass filter
        sos = signal.ellip(N=order,rp=rp,rs=rs,Wn=lowcut,btype='highpass',analog=False,output='sos',fs=fs)
    elif lowcut == None and highcut != None:
        sos = signal.ellip(N=order,rp=rp,rs=rs,Wn=highcut,btype='lowpass',analog=False,output='sos',fs=fs)
    else: # None filtering
        return signals
    filtered = signal.sosfilt(sos,signals)
    return filtered

def read_annot_regex(filename):
    with open(filename, 'r') as f:
        content = f.read()
    # Check that there is only one 'Start time' and that it is 0
    patterns_start = re.findall(
        r'<EventConcept>Recording Start Time</EventConcept>\n<Start>0</Start>',
        content)
    assert len(patterns_start) == 1
    # Now decode file: find all occurences of EventTypes marking sleep stage annotations
    patterns_stages = re.findall(
        r'<EventType>Stages.Stages</EventType>\n' +
        r'<EventConcept>.+</EventConcept>\n' +
        r'<Start>[0-9\.]+</Start>\n' +
        r'<Duration>[0-9\.]+</Duration>',
        content)
    # print(patterns_stages[-1])
    stages = []
    starts = []
    durations = []
    for pattern in patterns_stages:
        lines = pattern.splitlines()
        stageline = lines[1]
        stage = int(stageline[-16])
        startline = lines[2]
        start = float(startline[7:-8])
        durationline = lines[3]
        duration = float(durationline[10:-11])
        assert duration % 30 == 0.
        epochs_duration = int(duration) // 30

        stages += [stage]*epochs_duration
        starts += [start]
        durations += [duration]
    # last 'start' and 'duration' are still in mem
    # verify that we have not missed stuff..
    assert int((start + duration)/30) == len(stages)
    return stages