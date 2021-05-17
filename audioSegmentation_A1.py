import os
import csv
from datetime import datetime

import librosa
import librosa.display

import numpy as np

from supportClasses import Timer
from supportClasses import Logfile



def loadCSV(path):
    dictionary = {}
    
    with open(path, "r") as file:
        reader = csv.reader(file, delimiter=";")
        for row in reader:
            key = row[0]
            
            if row[1] == "":
                start = None
            else:
                start = int(row[1])
            
            if row[2] == "":
                end = None
            else:
                end = int(row[2])

            dictionary.update({key: [start, end]})
    
    file.close()
    
    return dictionary


def relativeOverlap(a_start, a_end, b_start, b_end):
    if b_start == None:
        return 0.00, None
    
    if b_start > a_end or a_start > b_end:
        intersection = 0.00
    else:
        intersection = min(a_end, b_end) - max(a_start, b_start)
    
    union = (a_end - a_start) + (b_end - b_start) - intersection
        
    return intersection / union, intersection



#________________________________________________________________________
#
#DIRECTORY SETUP
#________________________________________________________________________
#
t0 = Timer(); t1 = Timer(); t2 = Timer(); timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#text files (results)
log = Logfile(r"results/" + timeStamp + "_audioSegmentation_log.csv")
log.writeLine("scenario;File;loadDur;chromaDur;dtwDur;totalDur", False)
results = Logfile(r"results/" + timeStamp + "_audioSegmentation_results.csv")
results.writeLine("scenario;Y;y;y_start;y_end;y'_start;y'_end;overlap;relOverlap", False)

#reference recording X
X_filePath = r"assets/WAM-79__Track2_Channel1.wav"
X_segmPath = r"assets/WAM-79__Track2_Channel1.csv"

#unknown recordings Y
Y_dirPath = r"assets"


#________________________________________________________________________
#
#PARAMETER SETUP
#________________________________________________________________________
#
#feature extraction
function = "cens"
sampleRate = 22050
hopLength = 8192
hz = sampleRate / hopLength # ~2.69Hz

#dynamic time warping
stepSizesSigma=np.array([[1, 1], [1, 2], [2, 1]])

#other
scenario = "load(%i), chroma_%s(%.2fHz), dtw(%s)" % (sampleRate, function, hz, np.array2string(stepSizesSigma, separator=',').replace('\n',''))
print("scenario: " + scenario)



#________________________________________________________________________
#
#LOAD REFERENCE RECORDING (X)
#________________________________________________________________________
#
#load reference recording
t0.start()

X_segments = {}
X_csv = loadCSV(X_segmPath)

print("start X:  " + X_filePath)
for key, x in X_csv.items():
    x_data, sr = librosa.load(path=X_filePath, sr=sampleRate, offset=x[0], duration= (x[1]-x[0]))
    if function == "stft": x_chroma = librosa.feature.chroma_stft(y= x_data, sr= sampleRate, hop_length= hopLength)
    if function == "cens": x_chroma = librosa.feature.chroma_cens(y= x_data, sr= sampleRate, hop_length= hopLength)
    x_data = None
    X_segments.update({key: x_chroma})
    x_chroma = None

log.writeLine("%s;%s;;;;%f" % (scenario, X_filePath, t0.getDuration()), False)
print("total:   %.3fs\n" % (t0.getDuration()))



#________________________________________________________________________
#
#ITERATE OVER UNKNOWN RECORDINGS (Y)
#________________________________________________________________________
#
fileType = ".wav"
for Y_name in os.listdir(Y_dirPath):
    if Y_name.lower().endswith(fileType):
        t1.start();
        print("start Y:  " + Y_name)
        
        t2.start()
        Y_data, Sr = librosa.load(path= Y_dirPath + "/" + Y_name, sr= sampleRate) 
        loadDur = t2.getDuration()
        print("load:     %.3fs" % (loadDur))
        
        t2.start()
        if function == "stft": Y_chroma = librosa.feature.chroma_stft(y= Y_data, sr= sampleRate, hop_length= hopLength)
        if function == "cens": Y_chroma = librosa.feature.chroma_cens(y= Y_data, sr= sampleRate, hop_length= hopLength)
        Y_data = None
        chromaDur = t2.getDuration()
        print("chroma:   %.3fs" % (chromaDur))

        testData = loadCSV(Y_dirPath + "/" + Y_name.replace(fileType, ".csv"))
        
        t2.start()
        print("segments: y\t|y_st\t|y_end\t|y'_st\t|y'_end\t|ovLap\t|relOvlap")
        for key, x_chroma in X_segments.items():
            #dynamic time warping
            costMatrix, warpingPath = librosa.sequence.dtw(x_chroma, Y_chroma, step_sizes_sigma= stepSizesSigma, weights_add= np.array([0, 0, 0]), weights_mul= np.array([1, 1, 1]), subseq= True)
            
            #results
            y_start = int(round(warpingPath[-1][1] / hz))
            y_end = int(round(warpingPath[0][1] / hz))
            ys_start = testData[key][0]
            ys_end = testData[key][1]
            quality, intersection = relativeOverlap(y_start, y_end, ys_start, ys_end)

            #save data
            results.writeLine("%s;%s;%s;%s;%s;%s;%s;%s;%f" % (scenario, Y_name, key, y_start, y_end, ys_start, ys_end, intersection, quality), False)
            print("          %s\t|%s\t|%s\t|%s\t|%s\t|%s\t|%.2f" % (key, y_start, y_end, ys_start, ys_end, intersection, quality))
            
            #free memory
            costMatrix = None
            warpingPath = None
            
        Y_chroma = None
        
        dtwDur = t2.getDuration()
        print("dtw:      %.3fs" % (dtwDur))
        
        totDur = t1.getDuration()
        log.writeLine("%s;%s;%f;%f;%f;%f" % (scenario, Y_name, loadDur, chromaDur, dtwDur, totDur), False)
        print("total:    " + str(int(round(totDur))) + "s\n")

print("FINISHED: total duration = " + str(int(round(t0.getDuration()))) + "s")