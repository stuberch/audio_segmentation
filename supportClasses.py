# -*- coding: utf-8 -*-
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
