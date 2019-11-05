#!/usr/bin/env python3

import numpy as np
import pandas as pd
import argparse
import math,sys

import networkx as nx
import matplotlib.pyplot as plt

import pygraphviz
from networkx.drawing.nx_agraph import graphviz_layout

import inheritanceclass as ic
import eventclass as ec


class Population(object):
    def __init__(self,**kwargs):
        self.__verbose               = kwargs.get("verbose",False)
        self.__initialpopulationsize = kwargs.get("initialpopulationsize",5)
        self.__populationsize        = self.__initialpopulationsize
        
        self.events                  = ec.EventLineLL(verbose = self.__verbose) # default behavior is linked list, data extraction not implemented in old eventline
        self.divtimes                = ic.DivisionTimes_matrix(**kwargs)
        self.graphoutput             = kwargs.get("graphoutput",False)
        
        self.graph                   = nx.Graph()
        
        self.PopulationData          = None
        
        growthtimes,states = self.divtimes.DrawDivisionTimes(size = self.__initialpopulationsize)
        for i in range(self.__initialpopulationsize):
            if not kwargs.get("SyncInitialDivision",False):
                remaining_growthtime = np.random.uniform(high = growthtimes[i])
            else:
                remaining_growthtime = growthtimes[i]
            self.events.AddEvent(time = remaining_growthtime, parentID = -1, parentstate = states[i])
            if self.graphoutput:
                self.graph.add_nodes_from([i])
        
        self.PopulationData          = self.events.FounderPopulationData().copy()
        
    def division(self):
        # go to the next event in the eventline, initialize random variables
        curID, curtime, curdata = self.events.NextEvent()
        self.PopulationData     = self.PopulationData.append(self.events.CurrentEventDict(force_list_output = False), ignore_index = True)
        growthtimes,states      = self.divtimes.DrawDivisionTimes(parentstate = curdata['parentstate'])
        
        # add two new daugther cells to the eventline when they will divide in the future
        newID,newtime,newdata = self.events.AddEvent(time = curtime + growthtimes[0], parentID = curID, parentstate = states[0])
        if self.graphoutput:
            self.graph.add_nodes_from([newID])
            self.graph.add_edge(newID,curID,length = growthtimes[0])
        
        newID,newtime,newdata = self.events.AddEvent(time = curtime + growthtimes[1], parentID = curID, parentstate = states[1])
        if self.graphoutput:
            self.graph.add_nodes_from([newID])
            self.graph.add_edge(newID,curID,length = growthtimes[1])

        # store to keep track
        self.__populationsize += 1
        
        if self.__verbose:
            print("# population growth (N = {:4d}) at time {:.4f}, ({})-->({})-->({})&({}), with new division times ({:f}, {:f})".format(self.__populationsize,curtime,curdata['parentID'],curID,newID-1,newID,growthtime[0],growthtime[1]))


    def growth(self,divisionevents = None, maxpopsize = None, maxtime = None, time = None):
        if divisionevents is None and maxpopsize is None and maxtime is None and time is None:
            self.division()
        elif not divisionevents is None:
            for i in range(divisionevents):
                self.division()
        elif not maxpopsize is None:
            while self.size <= maxpopsize:
                self.division()
        elif not maxtime is None:
            while self.events.curtime <= maxtime:
                self.division()
        elif not time is None:
            maxtime = self.events.curtime + time
            while self.events.curtime <= maxtime:
                self.division()
        else:
            raise NotImplementedError

    
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
        if   key == 'size':
            return self.__populationsize
        elif key == 'time':
            return self.events.curtime
        elif key == 'divisiontimes':
            return self.divtimes.divisiontimes
        elif key == 'founderdivisiontimes':
            return self.divtimes.divisiontimes[:self.__initialpopulationsize]
        elif key == 'data':
            divtimes  = self.divisiontimes
            popsize   = np.arange(len(divtimes)) + self.__initialpopulationsize + 1
            divtimedf = pd.DataFrame({'#populationsize' : popsize, 'divisiontime' : divtimes})
            return pd.concat([divtimedf, self.PopulationData.reindex(divtimedf.index)], axis=1)
            
    

    

def MakeDictFromParameterList(params):

    def AddEntry(d,key,val):
        tmp = dict()
        if not key is None:
            if len(val) == 1:
                tmp[key] = val[0]
            elif len(val) > 1:
                tmp[key] = np.array(val)
        tmp.update(d)
        return tmp

    p = dict()
    curkey = None
    curvalue = list()
    for entry in params:
        try:
            v = float(entry)
            curvalue.append(v)
        except:
            p = AddEntry(p,curkey,curvalue)
            curvalue = list()
            curkey = entry

    p = AddEntry(p,curkey,curvalue)
    return p

    


def main():
    parser = argparse.ArgumentParser()
    parser_IO = parser.add_argument_group(description = "==== I/O parameters ====")
    parser_IO.add_argument("-o","--outputfile",    default = 'popdata.txt',  type = str)
    parser_IO.add_argument("-G","--graphoutput",   default = False, action = "store_true")
    parser_IO.add_argument("-I","--ignoreParents", default = False, action = "store_true")
    parser_IO.add_argument("-v","--verbose",       default = False, action = "store_true")
    
    parser_alg = parser.add_argument_group(description = "==== algorithm parameters ====")
    parser_alg.add_argument("-n", "--initialpopulationsize", type = int, default = 5)
    parser_alg.add_argument("-N", "--maxSize",               type = int, default = 100)
    parser_alg.add_argument("-P", "--parameters",            nargs = "*", default = None)
    parser_alg.add_argument("-S", "--SyncInitialDivision",   default = False, action = "store_true")
    args = parser.parse_args()

    argument_dict = vars(args)
    if not args.parameters is None:
        # add all entries in 'args.parameters' to the argument list itself, then delete its entry from the original dict
        # all parameters for division times can now be processed
        argument_dict.update(MakeDictFromParameterList(args.parameters))
        argument_dict.pop('parameters')
    
    # generate population object
    pop = Population(**argument_dict)
    
    # write standard parameters of divition time distribution
    if args.verbose:
        print('# stationary values division time distribution: {} {}\n'.format(pop.divtimes.mean, pop.divtimes.variance))
    
    # growth
    while pop.size < args.maxSize:
        pop.growth()
        print("{:.3f} {:5d}".format(pop.time, pop.size))

    # save output
    pop.data.to_csv(args.outputfile, sep = ' ', index = False)



if __name__ == "__main__":
    main()
    
