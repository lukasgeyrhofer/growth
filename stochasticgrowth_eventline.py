#!/usr/bin/env python3

import numpy as np
import argparse
import math,sys

import networkx as nx
import matplotlib.pyplot as plt

import pygraphviz
from networkx.drawing.nx_agraph import graphviz_layout

class EventLine(object):
    def __init__(self,**kwargs):
        # storing data and times in different lists
        self.__eventtime = list()
        self.__eventdata = list()

        self.__current_time = 0.
        self.__current_eventID = 0
        self.__store_for_reset = kwargs
        self.__largest_time = 0.
        self.__verbose = kwargs.get("verbose",False)

    # delete everything and reset
    def reset(self):
        self.__init__(**self.__store_for_reset)

    def addevent(self,time,**kwargs):
        eventID = len(self.__eventtime)
        self.__eventtime.append(time)
        self.__eventdata.append(kwargs)

        if time > self.__largest_time:
            self.__largest_time = time

        return self[eventID]

    def nextevent(self):
        # find index of next event
        timediff = np.array([t - self.__current_time if t > self.__current_time else self.__largest_time for t in self.__eventtime ]) # set all negative values of the calculation to the maximal time, self.__largest_time
        idx = timediff.argmin()
        
        # set the current status 
        self.__current_time = self.__eventtime[idx]
        self.__current_eventID = idx
        
        return self[idx]
    
    # return internal variables
    def __getattr__(self,key):
        if key == "times":
            return self.__eventtime
        elif key == "curtime":
            return self.__current_time

    # output should be generated over adressing an index in the list
    def __getitem__(self,key):
        if len(self.__eventdata) > key:
            return key, self.__eventtime[key], self.__eventdata[key]
        else:
            raise KeyError


class DivisionTimes_flat(object):
    def __init__(self,**kwargs):
        self.__mindivtime = kwargs.get("mindivtime",1.)
        self.__avgdivtime = kwargs.get("avgdivtime",2.)
    
    def DrawDivisionTimes(self,size = 2):
        dt = np.random.normal(loc = self.__avgdivtime,size = size)
        dt[dt < self.__mindivtime] = self.__mindivtime
        return dt

    
class DivisionTimes_2dARP(object):
    def __init__(self,**kwargs):
        self.__mindivtime        = kwargs.get('mindivtime',.1)
        self.__meandivtime       = kwargs.get('meandivtime',1)
        self.__lambda            = np.array(kwargs.get('lambda',[.3,.9]),dtype=np.float)
        self.__beta              = kwargs.get('beta',.3)
        self.__alpha             = kwargs.get('alpha',.6)
        self.__noiseamplidute    = kwargs.get('noiseamplidute',.6)
        self.__divtime_deviation = kwargs.get('divtimedev',.2)
        
        self.A                   = np.array([[self.__lambda[1],0],[(self.__lambda[0] - self.__lambda[1])/np.tan(2*np.pi*self.__beta),self.__lambda[0]]],dtype=np.float)
        self.projection          = self.__divtime_deviation * np.array([-np.sin(self.__alpha),np.cos(self.__alpha)],dtype=np.float)
        self.__stationaryCov     = self.ComputeStationaryCovariance()
        
        self.recorded_DivTimes   = list()

        self.__ignore_parents    = kwargs.get('ignoreParents',False) # debug mode
        
        
    def ComputeStationaryCovariance(self):
        il1l1 = 1./(1-self.__lambda[0]**2)
        il2l2 = 1./(1-self.__lambda[1]**2)
        il1l2 = 1./(1-self.__lambda[0]*self.__lambda[1])
        itan  = 1./np.tan(2 * np.pi * self.__beta)
        return self.__noiseamplidute**2 * np.array([[ il2l2,                  (il1l2 - il2l2) * itan],
                                                    [ (il1l2 - il2l2) * itan, il1l1 + (il1l1 - 2*il1l2 + il2l2)*itan*itan]], dtype = np.float)

    def TimeFromState(self,state):
        dt = np.max([self.__meandivtime + np.dot(self.projection,state),self.__mindivtime])
        return dt

        
    def DrawDivisionTimes(self, parentstate = None, size = 2):
        # get division time for two offspring cells
        daugtherstates = list()
        divtime        = list()
        
        # debug more: no inheritance should lead to stationary distribution
        if self.__ignore_parents:
            if not parentstate is None:
                self.recorded_DivTimes.append(self.TimeFromState(parentstate))                
            parentstate = None
        
        
        if parentstate is None:
            for i in range(size):
                daugtherstates.append(np.random.multivariate_normal(mean = np.zeros(2), cov = self.__stationaryCov))
        else:
            self.recorded_DivTimes.append(self.TimeFromState(parentstate))
            for i in range(size):
                daugtherstates.append(np.dot(self.A,parentstate) + self.__noiseamplidute * np.random.normal(size = 2))

        for state in daugtherstates:
            divtime.append(self.TimeFromState(state))
        
        return divtime,daugtherstates

    
    def WriteDivTimes(self,filename = 'divtimes.txt'):
        fp = open(filename,'w')
        for dt in self.recorded_DivTimes:
            fp.write('{:.6f}\n'.format(dt))
        fp.close()


    def __getattr__(self,key):
        if   key == 'divtimevariance':
            return np.dot(self.projection,np.dot(self.__stationaryCov,self.projection))
        elif key == 'average':
            return self.__meandivtime




class Population(object):
    def __init__(self,**kwargs):
        self.__verbose               = kwargs.get("verbose",False)
        self.__initialpopulationsize = kwargs.get("initialpopulationsize",5)
        self.__populationsize        = self.__initialpopulationsize
        self.events                  = EventLine(verbose = self.__verbose)
        self.divtimes                = DivisionTimes_2dARP(**kwargs)
        self.graphoutput             = kwargs.get("graphoutput",False)
        
        self.graph                   = nx.Graph()
        
        growthtimes,states = self.divtimes.DrawDivisionTimes(size = 5)
        for i in range(self.__initialpopulationsize):
            self.events.addevent(time = growthtimes[i], parentID = 0, parentstate = states[i])
            if self.graphoutput:
                self.graph.add_nodes_from([i])

        
    def growth(self):
        # go to the next event in the eventline, initialize random variables
        curID, curtime, curdata = self.events.nextevent()
        growthtimes,states      = self.divtimes.DrawDivisionTimes(parentstate = curdata['parentstate'])
        
        # add two new daugther cells to the eventline when they will divide in the future
        newID,newtime,newdata = self.events.addevent(time = curtime + growthtimes[0], parentID = curID, parentstate = states[0])
        if self.graphoutput:
            self.graph.add_nodes_from([newID])
            self.graph.add_edge(newID,curID,length = growthtimes[0])
        
        newID,newtime,newdata = self.events.addevent(time = curtime + growthtimes[1], parentID = curID, parentstate = states[1])
        if self.graphoutput:
            self.graph.add_nodes_from([newID])
            self.graph.add_edge(newID,curID,length = growthtimes[1])

        # store to keep track
        self.__populationsize += 1
        
        if self.__verbose:
            print("# population growth (N = {:4d}) at time {:.4f}, ({})-->({})-->({})&({}), with new division times ({:f}, {:f})".format(self.__populationsize,curtime,curdata['parentID'],curID,newID-1,newID,growthtime[0],growthtime[1]))

    
    def plotGraph(self,filename):
        if self.graphoutput:
            layout = graphviz_layout(self.graph,prog='twopi')
            nx.draw(self.graph,pos=layout,node_size = 0)
            plt.savefig(filename)
        else:
            raise IOError('graph output not recorded during simulation')
    
    
    # return internal variables
    def __getattr__(self,key):
        if key == "timeline":
            return self.events.curtime, self.events.times


    # output current state
    def __str__(self):
        return "{:.6f} {:4d}".format(self.events.curtime,self.__populationsize)
    
    def __int__(self):
        return self.__populationsize
    
    def __getattr__(self,key):
        if key == 'size':
            return int(self)
    
    
def main():
    parser = argparse.ArgumentParser()
    parser_IO = parser.add_argument_group(description = "==== I/O parameters ====")
    parser_IO.add_argument("-d","--divtimefile",   default = None,  type = str)
    parser_IO.add_argument("-o","--outputfile",    default = None,  type = str)
    parser_IO.add_argument("-G","--graphoutput",   default = False, action = "store_true")
    parser_IO.add_argument("-I","--ignoreParents", default = False, action = "store_true")
    parser_IO.add_argument("-v","--verbose",       default = False, action = "store_true")
    
    parser_alg = parser.add_argument_group(description = "==== algorithm parameters ===="
    parser_alg.add_argument("-n", "--initialpopulationsize", type = int, default = 5)
    parser_alg.add_argument("-N", "--maxSize",               type = int, default = 100)
    args = parser.parse_args()

    if args.outputfile is None: out = sys.stdout
    else:                       out = open(args.outputfile,'w')
    
    
    pop = Population(**vars(args))
    
    out.write('# stationary values: {} {}\n'.format(pop.divtimes.average, pop.divtimes.divtimevariance))
    while pop.size < args.maxSize:
        pop.growth()
        out.write("{:s}\n".format(str(pop)))
    
    out.close()
    
    if not args.divtimefile is None:
        pop.divtimes.WriteDivTimes(args.divtimefile)
    
if __name__ == "__main__":
    
    main()
    
