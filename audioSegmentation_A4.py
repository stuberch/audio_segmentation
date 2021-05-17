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
X_filePath = r"assets/WAM-100__Track2_Channel2.wav"
X_segmPath = r"assets/WAM-100__Track2_Channel2.csv"

#unknown recordings Y
Y_dirPath = r"assets"
Y_testData = True

#feature extraction
function = "cens"
sampleRate = 48000
hopLength = 9600
hz = sampleRate / hopLength # = 5.00Hz
tuning = 3 # +- 2 semitones

#dynamic time warping
stepSizesSigma=np.array([[1, 1], [1, 2], [2, 1]])

#best match
first_segment_start = 0
first_segment_end = 36000 #10h
max_cost = 9999 #filter, dtw cost must be smaller
x_factor = 2 #increases search window (duration of ref segment * this factor)

#other
scenario = "A4.7"
fileType = ".wav"



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

parameter = "load(%i), chroma_%s(%.2fHz, +-%i Tuning), dtw(%s), bestMatchSearch(start=y_end, end=x_dur*%f)" % (sampleRate, function, hz, tuning, np.array2string(stepSizesSigma, separator=',').replace('\n',''), x_factor)
print("start scenario: '" + scenario + "' with parameter: " + parameter)

t0.start()



#________________________________________________________________________
#
#LOAD REFERENCE RECORDING (X)
#________________________________________________________________________
#
t1.start()
print("start X:  " + X_filePath)

X_segments = {}
X_csv = MyFunctions.loadCSV(X_segmPath)
x_cntr = 0

for key, x in X_csv.items():
    x_cntr += 1
    x_data, sr = librosa.load(path=X_filePath, sr=sampleRate, offset=x[0], duration= (x[1]-x[0]))
    x_chroma_tuned = {}
    for x_tuning in range(-tuning, tuning + 1):
        x_chroma = librosa.feature.chroma_cens(x_data, sr=sampleRate, hop_length= hopLength, tuning= x_tuning)
        x_chroma_tuned.update({x_tuning: x_chroma})
        x_chroma = None
    
    x_data = None
    X_segments.update({key: [x_chroma_tuned, x[0], x[1]]})
    x_chroma_tuned = None
    
totDur = t1.getDuration()
log.writeLine("%s;%s;%s;;;;%f" % (scenario, parameter, X_filePath, totDur), False)
print("total:    " + str(int(round(totDur))) + "s\n")



#________________________________________________________________________
#
#ITERATE OVER UNKNOWN RECORDINGS (Y)
#________________________________________________________________________
#
Y_number = 0
for file in os.listdir(Y_dirPath):
    if file.lower().endswith(fileType):
        Y_number += 1

Y_cntr = 0
for Y_name in os.listdir(Y_dirPath):
    if Y_name.lower().endswith(fileType):
        t1.start()
        Y_cntr += 1
        Y_sum = 0
        print("start Y:  " + Y_name + " (" + str(Y_cntr) + "/" + str(Y_number) + ")")
        
        t2.start()
        Y_data, Sr = librosa.load(path= Y_dirPath + "/" + Y_name, sr= sampleRate) 
        loadDur = t2.getDuration()
        print("load:     %.3fs" % (loadDur))
        
        t2.start()
        Y_chroma = librosa.feature.chroma_cens(y= Y_data, sr= sampleRate, hop_length= hopLength)
        Y_data = None
        chromaDur = t2.getDuration()
        print("chroma:   %.3fs" % (chromaDur))

        if Y_testData: testData = MyFunctions.loadCSV(Y_dirPath + "/" + Y_name.replace(fileType, ".csv"))
        
        t2.start()
        
        #initial search window
        search_start = first_segment_start
        search_end = 0
        skip_s = 0
        
        #iterate over reference segments (x)
        x_cntr = 0
        for key, values in X_segments.items():
            x_cntr += 1
            x_chroma_tuned = values[0]
            x_start = values[1]
            x_end = values[2]
            x_dur = x_end - x_start
            
            #dynamic time warping (over whole unknown recording Y)
            bestMatch = [None,max_cost,None,None]
            segmfound = False
            
            
            #adjust search end
            if x_cntr == 1:
                search_end = first_segment_end
            else:
                search_end = search_start + (x_dur + skip_s) * x_factor
            
            
            #iterate over different tunings
            for x_tuning, x_chroma in x_chroma_tuned.items():
                D, wp = librosa.sequence.dtw(x_chroma, Y_chroma, step_sizes_sigma= stepSizesSigma, weights_add= np.array([0, 0, 0]), weights_mul= np.array([1, 1, 1]), subseq= True)
                
                y_start = wp[-1][1] / hz 
                y_end = wp[0][1] / hz
                
                y_cost = 0
                for y, x in wp:
                    y_cost += D[y][x]
                y_cost = y_cost/len(wp) 
                D = None
                wp = None
            
                #best match
                if(y_start > search_start and y_end < search_end and y_cost < bestMatch[1]):
                    segmfound = True
                    bestMatch = [x_tuning, y_cost, y_start, y_end]
            
            
            #adjust search window
            if(segmfound):
                search_start = bestMatch[3] #y_end
                skip_s = 0
            else:
                skip_s += x_dur
            

            #test data
            if Y_testData:
                ys_start = testData[key][0]
                ys_end = testData[key][1]
                
                if (ys_start == None and not segmfound):
                    quality = 1.00
                    intersection = 0.00
                elif (ys_start == None and segmfound):
                    quality = 0.00
                    intersection = 0.00
                elif (not segmfound):
                    quality = 0.00
                    intersection = 0.00
                else:
                    quality, intersection = MyFunctions.relativeOverlap(bestMatch[2], bestMatch[3], ys_start, ys_end)
            
            else:
                ys_start = None
                ys_end = None
                quality = None
                intersection = 0.00
            

            #save data
            res.writeLine("%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s;%s" % (scenario, parameter, Y_name, key, bestMatch[0], bestMatch[1], bestMatch[2], bestMatch[3], ys_start, ys_end, intersection, str(quality)), False)
            Y_sum += quality
            
        Y_chroma = None
        
        dtwDur = t2.getDuration()
        print("dtw:      %.3fs" % (dtwDur))
        
        totDur = t1.getDuration()
        log.writeLine("%s;%s;%s;%f;%f;%f;%f" % (scenario, parameter, Y_name, loadDur, chromaDur, dtwDur, totDur), False)
        print("total:    " + str(int(round(totDur))) + "s")
        print("quality:  " + str(round(Y_sum/x_cntr, 2)) + "\n")

print("FINISHED: total duration = " + str(int(round(t0.getDuration()))) + "s")