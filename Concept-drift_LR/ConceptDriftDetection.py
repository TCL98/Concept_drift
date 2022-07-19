from Utils import loadData
from skmultiflow.drift_detection import PageHinkley, KSWIN
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.data.file_stream import FileStream
import os

def executeDriftDetection():
        
    # Use FileStream method to create a Data Stream throught our csv wind energy data.
    stream = FileStream(os.getcwd() + "/eol.csv")

    aw = ADWIN(delta=0.2)
    # Adding stream elements to the ADWIN drift detector and verifying if drift occurred
    indices = []
    i = 0
    while stream.n_remaining_samples()>0:
        nextSample = stream.next_sample()[1][0]
        aw.add_element(nextSample)
        
        if aw.detected_change():
            indices.append(i)
        i = i + 1
    print()
    print("ADWIN RESULTS:")
    print()
    print("Number of Concept Drifts detected:", len(indices))
    print(indices)

    stream = FileStream(os.getcwd() + "/eol.csv")
    kswin = KSWIN(alpha=0.001, window_size=72,
                    stat_size=40)
    indices = []
    i = 0
    while stream.n_remaining_samples()>0:
        nextSample = stream.next_sample()[1][0]
        kswin.add_element(nextSample)
        
        if kswin.detected_change():
            indices.append(i)
        i = i + 1      

    print()
    print("KSWIN RESULTS:")
    print()
    print("Number of Concept Drifts detected:", len(indices))
    print(indices)

    stream = FileStream(os.getcwd() + "/eol.csv")
    ph = PageHinkley(delta=0.1, threshold=25, alpha=0.7)
    indices = []
    i = 0
    while stream.n_remaining_samples()>0:
        nextSample = stream.next_sample()[1][0]
        ph.add_element(nextSample)
        
        if ph.detected_change():
            indices.append(i)
        i = i + 1      
            
    print()
    print("Page Hinkley RESULTS:")
    print()
    print("Number of Concept Drifts detected:", len(indices))
    print(indices)
    print()