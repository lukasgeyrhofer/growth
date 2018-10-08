#!/usr/bin/env python3

import numpy as np
import argparse

import networkx as nx
import matplotlib.pyplot as plt

import pygraphviz
from networkx.drawing.nx_agraph import graphviz_layout

class EventLine(object):
    def __init__(self,**kwargs):
        self.__eventtime = list()
        self.__eventdata = list()

        self.__current_time = 0.
        self.__current_eventID = 0
        self.__store_for_reset = kwargs
        self.__largest_time = 0.
        self.__verbose = kwargs.get("verbose",False)

    def addevent(self,time,**kwargs):
        eventID = len(self.__eventtime)
        self.__eventtime.append(time)
        self.__eventdata.append(kwargs)

        if time > self.__largest_time:
            self.__largest_time = time

        return self[eventID]

    def nextevent(self):
        # find index of next event
        timediff = np.array([t - self.__current_time if t > self.__current_time else self.__largest_time for t in self.__eventtime ])
        idx = timediff.argmin()
        
        # set the current status 
        self.__current_time = self.__eventtime[idx]
        self.__current_eventID = idx
        
        return self[idx]
    
    def reset(self):
        self.__init__(**self.__store_for_reset)
    
    def __getattr__(self,key):
        if key == "times":
            return self.__eventtime
        elif key == "curtime":
            return self.__current_time

    def __getitem__(self,key):
        if len(self.__eventdata) > key:
            return key, self.__eventtime[key], self.__eventdata[key]
        else:
            raise KeyError
    
    
class DivisionTimes(object):
    def __init__(self,**kwargs):
        self.__mindivtime = kwargs.get("mindivtime",1.)
        self.__avgdivtime = kwargs.get("avgdivtime",2.)
    
    def DrawDivisionTime(self,size = 2):
        dt = np.random.normal(loc = self.__avgdivtime,size = size)
        dt[dt < self.__mindivtime] = self.__mindivtime
        return dt
    

class Population(object):
    def __init__(self,**kwargs):
        self.__verbose               = kwargs.get("verbose",False)
        self.__initialpopulationsize = kwargs.get("initialpopulationsize",5)
        self.__populationsize        = self.__initialpopulationsize
        self.events                  = EventLine(verbose = self.__verbose)
        self.divtimes                = DivisionTimes()
        
        self.graph                   = nx.Graph()
        
        dt = self.divtimes.DrawDivisionTime(size = 5)
        for i in range(self.__initialpopulationsize):
            self.events.addevent(time = dt[i], parentID = 0)
            self.graph.add_nodes_from([i])
        
        
    def growth(self):
        curID, curtime, curdata = self.events.nextevent()
        growthtime              = self.divtimes.DrawDivisionTime()
        
        if self.__verbose:
            print("# population growth (N = {:4d}) at time {:.4f}, (parentID {:4d}, cellID {:4d}, divisiontimes {:.6f} {:.6f})".format(self.__populationsize,curtime,curdata['parentID'],curID,growthtime[0],growthtime[1]))
        print("{:.6f} {:4d}".format(curtime,self.__populationsize))
        
        newtime,newID,newdata = self.events.addevent(time = curtime + growthtime[0], parentID = curID)
        self.graph.add_nodes_from([newID])
        self.graph.add_edge(newID,curID,length = growthtime[0])
        
        newtime,newID,newdata = self.events.addevent(time = curtime + growthtime[1], parentID = curID)
        self.graph.add_nodes_from([newID])
        self.graph.add_edge(newID,curID,length = growthtime[1])

        self.__populationsize += 1
        
    
    def plotGraph(self,filename):
        layout = graphviz_layout(self.graph,prog='twopi')
        nx.draw(self.graph,pos=layout,node_size = 0)
        plt.savefig(filename)
    
    def timeline(self):
        return np.array(self.__events._EventLine__eventtime)
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--initialpopulationsize",type=int,default=5)
    parser.add_argument("-N","--maxSize",type=int,default=100)
    parser.add_argument("-v","--verbose",default=False,action="store_true")
    parser.add_argument("-o","--outputfile",default="test.png")
    args = parser.parse_args()
    
    pop = Population(**vars(args))
    for i in range(args.maxSize):
        pop.growth()
    
    pop.plotGraph(args.outputfile)
    
if __name__ == "__main__":
    
    main()
    
