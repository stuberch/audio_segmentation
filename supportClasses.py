from datetime import datetime
import time
import csv

class MyLogfile:   
    def __init__(self, path):
        self.path = path

    def writeLine(self, text, timeStamp = True):
        if timeStamp:
            dt = datetime.now().strftime("%d.%m.%Y %H:%M:%S.%f") + ";"
        else:
            dt = ""
        line = dt + text
        
        with open(self.path, "a+") as f:
            f.write(line + "\n")
            f.close()

        
class MyTimer:
    def start(self):
        self.startTime = time.time()
    
    def getDuration(self):
        self.endTime = time.time()
        return self.endTime - self.startTime
    
    def reset(self):
        self.startTime = 0
        self.endTime = 0


class MyFunctions:
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
        if a_start == None or a_end == None or b_start == None or b_end == None:
            return None, 0.00
        
        if b_start > a_end or a_start > b_end:
            intersection = 0.00
        else:
            intersection = min(a_end, b_end) - max(a_start, b_start)
        
        union = (a_end - a_start) + (b_end - b_start) - intersection
            
        return intersection / union, intersection