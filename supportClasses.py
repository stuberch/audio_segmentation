# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 10:52:23 2021

@author: Christian Stuber
"""
from datetime import datetime
import time
import matplotlib.pyplot as plt


class Logfile:   
    def __init__(self, path):
        self.__path = path.replace("\\", "/")

    def writeLine(self, text, timeStamp = True):
        if timeStamp:
            dt = datetime.now().strftime("%d.%m.%Y %H:%M:%S.%f") + ";"
        else:
            dt = ""
        line = dt + text
        file = open(self.__path, "a")
        file.write(line + "\n")
            

class Timer:
    def start(self):
        self.startTime = time.time()
    
    def getDuration(self):
        self.endTime = time.time()
        return self.endTime - self.startTime
    
    def reset(self):
        self.startTime = 0
        self.endTime = 0
        

class Plot:
    def waveform(t, recording, title = None):
        
        """Plots Waveform of a Signal
        
        Args:
            t: Time axis (in seconds)
            recording: Input signal
            title: Title of the plot
        """
        
        plt.figure(figsize=(10, 2))
        plt.plot(t, recording, color='gray')
        plt.title(title)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.tick_params(direction='in')
        plt.tight_layout()