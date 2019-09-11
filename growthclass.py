#!/usr/bin/env python3

import numpy as np
import argparse
import pandas as pd
import sys,math


class GrowthData():
    def __init__(self,**kwargs):
        self.__infiles = kwargs.get('infiles',[])
        self.__verbose = kwargs.get('verbose',False)
        
        self.__data = dict()
        
        for filename in self.__infiles:
            p = self.ParametersFromFilename(filename)
            divfile = self.DivFilenameFromPopFilename(filename)
            if self.__verbose: print("reading files '{}' and '{}'".format(filename,divfile))
            
            try:
                data1 = np.genfromtxt(filename)
                data2 = np.array([np.genfromtxt(divfile)]).T
            
                df = pd.DataFrame(np.concatenate([data1,data2],axis=1), columns = ['time', 'popsize', 'divtime'])
            
                self.__data[filename] = df
            except:
                if self.__verbose: print("ERROR in reading files '{}' and '{}'".format(filename,divfile))
                continue
            
    
    def ParametersFromFilename(self,filename):
        values = filename.split('.')
        n0     = int(values[1][1:])
        runID  = int(values[2][2:])
        return n0,runID
    
    
    def DivFilenameFromPopFilename(self,filename):
        return filename.replace('popdyn','divtime')
    
    
    def getDataInitialSize(self,popsize):
        for key in self.__data.keys():
            p = self.ParametersFromFilename(key)
            if p[0] == popsize:
                yield key,self.__data[key]
    
    
    def __iter__(self):
        for fn in self.__data.keys():
            yield fn, self.__data[fn]
        
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--infiles",nargs="*",default=[])
    parser.add_argument("-v","--verbose",default=False,action="store_true")
    args = parser.parse_args()
    
    
    data = GrowthData(**vars(args))
    
    for x,y in data:
        print(x)
        print(y)


if __name__ == "__main__":
    main()
    
