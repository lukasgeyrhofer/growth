#!/usr/bin/env python3

import numpy as np
import pandas as pd
import argparse
import math,sys

import networkx as nx
import matplotlib.pyplot as plt

import pygraphviz
from networkx.drawing.nx_agraph import graphviz_layout


class EventNode(object):
    def __init__(self,**kwargs):
        # pointers in time
        self.next_ref     = kwargs.get('next_ref', None)
        self.last_ref     = kwargs.get('last_ref', None)
        # pointers in tree
        self.parent_ref   = kwargs.get('parent_ref', None)
        self.daugther_ref = kwargs.get('daugther_ref', [None, None])
        # actual relevant data
        self.ID           = kwargs.get('ID', None)
        self.time         = kwargs.get('time', 0)
        self.data         = kwargs.get('data', None)



class NewEventLine(object):
    def __init__(self,**kwargs):
        self.__store_for_reset = kwargs
        self.__verbose         = kwargs.get('verbose',False)
        
        self.__start_ref       = None
        self.__end_ref         = None
        self.__current_ref     = None
        
        self.__count_events    = 0
        self.__nextID          = 0

    
    def Reset(self):
        self.__init__(**self.__store_for_reset)

        
    def InsertEvent_Between(self, time, data, last_ref, next_ref):
        n = EventNode(ID = self.__nextID,time = time, data = data, last_ref = last_ref, next_ref = next_ref)
        last_ref.next_ref = n
        next_ref.last_ref = n
        return n

    def InsertEvent_Start(self, time, data, next_ref):
        n = EventNode(ID = self.__nextID, time = time, data = data, last_ref = None, next_ref = next_ref)
        self.__start_ref = n
        next_ref.last_ref = n
        return n

    def InsertEvent_End(self,time,data, last_ref):
        n = EventNode(ID = self.__nextID, time = time, data = data, last_ref = last_ref, next_ref = None)
        self.__end_ref = n
        last_ref.next_ref = n
        return n

    def InsertEvent_First(self,time,data):
        n = EventNode(ID = self.__nextID, time = time, data = data, last_ref = None, next_ref = None)
        self.__end_ref = n
        self.__start_ref = n
        return n

    def getTime(self,e):
        if not e is None:
            return e.time
        else:
            return 0



    def AddEvent(self, time = None, **kwargs):
        n = EventNode(time = time, data = kwargs)
        
        if self.__start_ref is None:
            self.__start_ref = n
        else:
            e = self.__start_ref
            if time < e.time:
                n.next_ref = e
                self.__start_ref = n
            else:
                while time < e.time:
                    if not e.next_ref is None:
                        if e.next_ref.time < time:
                            n.next_ref = e.next_ref
                            e.next_ref = n
                            break
                        else:
                            e = e.next_ref
                    else:
                        e.next_ref = n
                        break

        return self.EventData(n)
                


    def AddEvent3(self, time = None, **kwargs):
        n = None
        if self.__start_ref is None:
            n = EventNode(ID = self.__nextID,time = time, data = kwargs);
        else:
            e = self.__start_ref
            if time < e.time:
                n = EventNode(ID = self.__nextID, time = time, data = kwargs, next_ref = e)
                self.__start_ref = n
                e.last_ref = n
            else:
                while e.time < time:
                    if e.next_ref is None:
                        # reached end of linked list
                        n = EventNode(ID = self.__nextID, time = time, data = kwargs, last_ref = e)
                        self.__end_ref = n
                        e.last_ref = n
                        break
                    else:
                        # continue iterating
                        e = e.next_ref
                if n is None:
                    n = EventNode(ID = self.__nextID, time = time, data = kwargs, next_ref = e.next_ref, last_ref = e)
                    e.last_ref.next_ref = n
                    e.last_ref = n


        
        print(self.times)
        #print('{} {:.6f} {:.6f} {:.6f}'.format(it,self.getTime(n.last_ref),self.getTime(n),self.getTime(n.next_ref)))
        #print('{} {} {}'.format(n.last_ref,n,n.next_ref))
        return self.EventData(n)




    def AddEvent2(self, time = None, start_insert_from = None, **kwargs):
        insert_type = 0
        if start_insert_from is None:
            if self.__start_ref is None:
                # generate first event
                n = self.InsertEvent_First(time = time, data = kwargs); insert_type = 0
            else:
                e = self.__start_ref
                if e.time > time:
                    n = self.InsertEvent_Start(time = time, data = kwargs, next_ref = e); insert_type = 1
                else:
                    while e.time < time:
                        if e.next_ref is None:  break
                        else:                   e = e.next_ref
                    if e.next_ref is None:
                        n = self.InsertEvent_End(time = time, data = kwargs, last_ref = e); insert_type = 2
                    else:
                        n = self.InsertEvent_Between(time = time, data = kwargs, last_ref = e, next_ref = e.next_ref); insert_type = 3
                
        else:
            # get direction
            if time < start_insert_from.time:
                e = start_insert_from.last_ref
                while e.time > time:
                    if not e.last_ref is None:  e = e.last_ref
                    else:                       break
                if e.last_ref is None:          n = self.InsertEvent_Start  (time = time, data = kwargs,                        next_ref = e); insert_type = 4
                else:                           n = self.InsertEvent_Between(time = time, data = kwargs, last_ref = e.last_ref, next_ref = e); insert_type = 5
            else:
                e = start_insert_from.next_ref
                while e.time < time:
                    if not e.next_ref is None:  e = e.next_ref
                    else:                       break
                if e.next_ref is None:          n = self.InsertEvent_End    (time = time, data = kwargs, last_ref = e); insert_type = 6
                else:                           n = self.InsertEvent_Between(time = time, data = kwargs, last_ref = e, next_ref = e.next_ref); insert_type = 7

        self.__nextID += 1
        
        print('{} {:.6f} {:.6f} {:.6f}'.format(insert_type,self.getTime(n.last_ref),self.getTime(n),self.getTime(n.next_ref)))
        return self.EventData(n)
    
    
    
    def NextEvent(self):
        if self.__current_ref is None:
            self.__current_ref = self.__start_ref
            #elif not self.__current_ref.next_ref is None:
        else:
            self.__current_ref = self.__current_ref.next_ref
        return self.EventData(self.__current_ref)


    def EventData(self,e = None):
        if not e is None:
            return e.ID, e.time, e.data
        else:
            return None,None,None


    def EventTimes(self,first = None):
        return self.GenerateListOfTimes()
    

    def GenerateListOfTimes(self):
        lot = list()
        n = self.__start_ref
        while not n is None:
            lot.append(n.time)
            n=n.next_ref
        return np.array(lot,dtype=np.float)


    def __getattr__(self,key):
        if key  == 'times':
            return self.GenerateListOfTimes()
        elif key == 'curtime':
            if not self.__current_ref is None:
                return self.__current_ref.time
            else:
                return 0


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
        
        print('old event line')

    # delete everything and reset
    def reset(self):
        self.__init__(**self.__store_for_reset)


    def AddEvent(self,time,**kwargs):
        eventID = len(self.__eventtime)
        self.__eventtime.append(time)
        self.__eventdata.append(kwargs)

        if time > self.__largest_time:
            self.__largest_time = time

        return self[eventID]


    def NextEvent(self):
        # find index of next event
        timediff = np.array([t - self.__current_time if t > self.__current_time else self.__largest_time for t in self.__eventtime ]) # set all negative values of the calculation to the maximal time, self.__largest_time
        idx = timediff.argmin()
        
        # set the current status 
        self.__current_time = self.__eventtime[idx]
        self.__current_eventID = idx
        
        return self[idx]

    
    def EventTimes(self,first = None, uptocurrent = False):
        d = np.sort(self.times)

        if uptocurrent:
            d = d[d<self.curtime]

        if not first is None:
            if len(d) < first:
                d = d[:first]

        return d
    
    
    # return internal variables
    def __getattr__(self,key):
        if  key == "times":
            return np.sort(np.array(self.__eventtime))
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
        self.__vardivtime = kwargs.get("vardivtime",1.)
        self.__stddev_divtime = np.sqrt(self.__vardivtime)
    
    def DrawDivisionTimes(self,size = 2, **kwargs):
        dt = np.random.normal(loc = self.__avgdivtime,scale = self.__stddev_divtime,size = size)
        dt[dt < self.__mindivtime] = self.__mindivtime
        return dt

    def __getattr__(self,key):
        if   key == 'mean':
            return self.__avgdivtime
        elif key == 'variance':
            return self.__vardivtime

    
class DivisionTimes_2dARP(object):
    def __init__(self,**kwargs):
        # distribution of division times
        self.__divtime_min     = kwargs.get('divtime_min',.1)
        self.__divtime_mean    = kwargs.get('divtime_mean',1)
        self.__divtime_var     = kwargs.get('divtime_var',.01)
        self.recorded_DivTimes = list()
        
        # dynamics of internal state
        self.__eigenvalues     = np.array(kwargs.get('eigenvalues',[.3,.9]),dtype=np.float)
        self.__beta            = kwargs.get('beta',.3)
        self.__alpha           = kwargs.get('alpha',.6)
        self.__noiseamplitude  = kwargs.get('noiseamplitude',.6)
        
        self.A                 = np.array([[self.__eigenvalues[1],0],[(self.__eigenvalues[0] - self.__eigenvalues[1])/np.tan(2*np.pi*self.__beta),self.__eigenvalues[0]]],dtype=np.float)
        self.__stationaryCov   = self.ComputeStationaryCovariance()
        self.projection        = np.array([-np.sin(self.__alpha),np.cos(self.__alpha)],dtype=np.float)
        self.projection       *= np.sqrt(self.__divtime_var/self.variance)
        
        # debug mode
        self.__ignore_parents = kwargs.get('ignoreParents',False)
        
        
    def ComputeStationaryCovariance(self):
        il1l1 = 1./(1-self.__eigenvalues[0]**2)
        il2l2 = 1./(1-self.__eigenvalues[1]**2)
        il1l2 = 1./(1-self.__eigenvalues[0]*self.__eigenvalues[1])
        itan  = 1./np.tan(2 * np.pi * self.__beta)
        return self.__noiseamplitude**2 * np.array([[ il2l2,                  (il1l2 - il2l2) * itan],
                                                    [ (il1l2 - il2l2) * itan, il1l1 + (il1l1 - 2*il1l2 + il2l2)*itan*itan]], dtype = np.float)

    def TimeFromState(self,state):
        dt = np.max([self.__divtime_mean + np.dot(self.projection,state),self.__divtime_min])
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
                daugtherstates.append(np.dot(self.A,parentstate) + self.__noiseamplitude * np.random.normal(size = 2))

        for state in daugtherstates:
            divtime.append(self.TimeFromState(state))
        
        return divtime,daugtherstates

    
    def WriteDivTimesToFile(self,filename = 'divtimes.txt'):
        fp = open(filename,'w')
        for dt in self.recorded_DivTimes:
            fp.write('{:.6f}\n'.format(dt))
        fp.close()


    def __getattr__(self,key):
        if   key == 'variance':
            return np.dot(self.projection, np.dot(self.__stationaryCov, self.projection))
        elif key == 'mean':
            return self.__divtime_mean
        elif key == 'divisiontimes':
            return np.array(self.recorded_DivTimes, dtype = np.float)




class Population(object):
    def __init__(self,**kwargs):
        self.__verbose               = kwargs.get("verbose",False)
        self.__initialpopulationsize = kwargs.get("initialpopulationsize",5)
        self.__populationsize        = self.__initialpopulationsize
        
        if int(kwargs.get('EventLine',0)) == 0:  self.events = EventLine(verbose = self.__verbose)
        else:                                    self.events = NewEventLine(verbose = self.__verbose)

        self.divtimes                = DivisionTimes_2dARP(**kwargs)
        self.graphoutput             = kwargs.get("graphoutput",False)
        
        self.graph                   = nx.Graph()
        
        growthtimes,states = self.divtimes.DrawDivisionTimes(size = self.__initialpopulationsize)
        for i in range(self.__initialpopulationsize):
            self.events.AddEvent(time = growthtimes[i], parentID = -1, parentstate = states[i])
            if self.graphoutput:
                self.graph.add_nodes_from([i])

        
    def division(self):
        print(self.events.NextEvent())
        # go to the next event in the eventline, initialize random variables
        curID, curtime, curdata = self.events.NextEvent()
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
            while int(self) <= maxpopsize:
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
            return int(self)
        elif key == 'divisiontimes':
            return self.divtimes.divisiontimes
        elif key == 'founderdivisiontimes':
            return self.divtimes.divisiontimes[:self.__initialpopulationsize]
        elif key == 'divisiondata':
            d = self.divisiontimes
            t = self.events.times[:len(d)]
            s = np.arange(len(d))
            return pd.DataFrame({'times':t, 'divisiontimes': d, 'populationsize': s})
            
    

    

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
    parser_IO.add_argument("-d","--divtimefile",   default = None,  type = str)
    parser_IO.add_argument("-o","--outputfile",    default = None,  type = str)
    parser_IO.add_argument("-G","--graphoutput",   default = False, action = "store_true")
    parser_IO.add_argument("-I","--ignoreParents", default = False, action = "store_true")
    parser_IO.add_argument("-v","--verbose",       default = False, action = "store_true")
    
    parser_alg = parser.add_argument_group(description = "==== algorithm parameters ====")
    parser_alg.add_argument("-n", "--initialpopulationsize", type = int, default = 5)
    parser_alg.add_argument("-N", "--maxSize",               type = int, default = 100)
    parser_alg.add_argument("-P", "--parameters",            nargs = "*", default = None)
    args = parser.parse_args()

    if args.outputfile is None: out = sys.stdout
    else:                       out = open(args.outputfile,'w')
    
    
    argument_dict = vars(args)
    if not args.parameters is None:
        # add all entries in 'args.parameters' to the argument list itself, then delete its entry from the original dict
        # all parameters for division times can now be processed
        argument_dict.update(MakeDictFromParameterList(args.parameters))
        argument_dict.pop('parameters')
    
    # generate population object
    pop = Population(**argument_dict)
    
    # write standard parameters of divition time distribution
    if args.verbose:    out.write('# stationary values: {} {}\n'.format(pop.divtimes.mean, pop.divtimes.variance))
    
    # growth
    while pop.size < args.maxSize:
        pop.growth()
        out.write("{:s}\n".format(str(pop)))
    
    # output
    out.close()
    if not args.divtimefile is None:
        pop.divtimes.WriteDivTimesToFile(args.divtimefile)



if __name__ == "__main__":
    main()
    
