import os
from datetime import datetime

import librosa
import numpy as np

from supportClasses import MyTimer
from supportClasses import MyLogfile
from supportClasses import MyFunctions



#________________________________________________________________________
#
#PARAMETER SETUP
#________________________________________________________________________
#

#result (.csv files)
resultsPath = r"results/"
logfilePath = r"results/"

#reference recording X
X_filePath = r"assets/WAM-79__Track2_Channel1.wav"
X_segmPath = r"assets/WAM-79__Track2_Channel1.csv"

#unknown recordings Y
Y_dirPath = r"assets"


#feature extraction
function = "cens"
sampleRate = 22050
hopLength = 8192
hz = sampleRate / hopLength # ~2.69Hz
tuning = 3 # +- 1 semitone

#dynamic time warping
stepSizesSigma=np.array([[1, 1], [1, 2], [2, 1]])

#other
scenario = "A3"



#________________________________________________________________________
#
#INITIALIZE VARIABLES
#________________________________________________________________________
#
t0 = MyTimer(); t1 = MyTimer(); t2 = MyTimer(); timeStamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log = MyLogfile(resultsPath + "/" + timeStamp + "_audioSegmentation_log.csv")
res = MyLogfile(logfilePath + "/" + timeStamp + "_audioSegmentation_results.csv")

log.writeLine("scenario;parameter;File;loadDur;chromaDur;dtwDur;totalDur", False)
res.writeLine("scenario;parameter;Y;y;tuning;cost;y_start;y_end;y'_start;y'_end;overlap;relOverlap", False)

parameter = "load(%i), chroma_%s(%.2fHz, %i Tuning), dtw(%s)" % (sampleRate, function, hz, tuning, np.array2string(stepSizesSigma, separator=',').replace('\n',''))
print("start scenario: '" + scenario + "' with parameter: " + parameter)

t0.start()

#________________________________________________________________________
#
#LOAD REFERENCE RECORDING (X)
#________________________________________________________________________
#
X_segments = {}
X_csv = MyFunctions.loadCSV(X_segmPath)

print("start X:  " + X_filePath)
for key, x in X_csv.items():
    x_data, sr = librosa.load(path=X_filePath, sr=sampleRate, offset=x[0], duration= (x[1]-x[0]))
    
    x_chroma_tuned = {}
    for x_tuning in range(-tuning, tuning + 1):
        x_chroma = librosa.feature.chroma_cens(x_data, sr=sampleRate, hop_length= hopLength, tuning= x_tuning)
        x_chroma_tuned.update({x_tuning: x_chroma})
        x_chroma = None
    
    x_data = None
    X_segments.update({key: x_chroma_tuned})
    x_chroma_tuned = None

log.writeLine("%s;%s;%s;;;;%f" % (scenario, parameter, X_filePath, t0.getDuration()), False)
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
        Y_chroma = librosa.feature.chroma_cens(y= Y_data, sr= sampleRate, hop_length= hopLength)
        Y_data = None
        chromaDur = t2.getDuration()
        print("chroma:   %.3fs" % (chromaDur))

        testData = MyFunctions.loadCSV(Y_dirPath + "/" + Y_name.replace(fileType, ".csv"))
        
        t2.start()
        for key, x_chroma_tuned in X_segments.items():
            
            #dynamic time warping
            for x_tuning, x_chroma in x_chroma_tuned.items():
                D, wp = librosa.sequence.dtw(x_chroma, Y_chroma, step_sizes_sigma= stepSizesSigma, weights_add= np.array([0, 0, 0]), weights_mul= np.array([1, 1, 1]), subseq= True)
                
                #results
                y_start = wp[-1][1] / hz 
                y_end = wp[0][1] / hz
                
                y_cost = 0
                for y, x in wp:
                    y_cost += D[y][x]
                y_cost = y_cost/len(wp) #cost in relation to wp size (ref segment size)
                
                D = None
                wp = None

                ys_start = testData[key][0]
                ys_end = testData[key][1]
                quality, intersection = MyFunctions.relativeOverlap(y_start, y_end, ys_start, ys_end)

                #save data
                res.writeLine("%s;%s;%s;%s;%s;%f;%f;%f;%s;%s;%f;%s" % (scenario, parameter, Y_name, key, x_tuning, y_cost, y_start, y_end, ys_start, ys_end, intersection, str(quality)), False)
        Y_chroma = None
        
        dtwDur = t2.getDuration()
        print("dtw:      %.3fs" % (dtwDur))
        
        totDur = t1.getDuration()
        log.writeLine("%s;%s;%s;%f;%f;%f;%f" % (scenario, parameter, Y_name, loadDur, chromaDur, dtwDur, totDur), False)
        print("total:    " + str(int(round(totDur))) + "s\n")

print("FINISHED: total duration = " + str(int(round(t0.getDuration()))) + "s")