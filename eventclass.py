#!/usr/bin/env python3


import numpy as np
import pandas as pd

class EventNode(object):
    def __init__(self,**kwargs):
        # pointers in time
        self.next_ref     = kwargs.get('next_ref', None)
        self.last_ref     = kwargs.get('last_ref', None)
        # pointers in tree
        self.parent_ref   = kwargs.get('parent_ref', None)
        self.daughter_ref = kwargs.get('daughter_ref', [None, None])
        # actual relevant data
        self.ID           = kwargs.get('ID', None)
        self.time         = kwargs.get('time', 0)
        self.data         = kwargs.get('data', None)


    def to_dict(self, force_list_output = True):
        curdata = {'ID':self.ID,'time':self.time}

        # add all values in data to dictionary as well
        for key,value in self.data.items():
            if not isinstance(value,(list,tuple,np.ndarray)):
                curdata.update({key:value})
            else:
                # if entry in data has more dimensions, then flatten it and enumerate keys
                for i,vi in enumerate(value):
                    curdata.update({key + str(i):vi})

        # wrap each output element into list, such that the whole dictionary can be appended to pandas dataframe as new entry
        if force_list_output:
            for key,value in curdata.items():
                curdata[key] = [value]

        return curdata
        


class EventLineLL(object):
    def __init__(self,**kwargs):
        self.__store_for_reset        = kwargs
        self.__verbose                = kwargs.get('verbose',False)
        
        # various pointers to different events in the linked list
        self.__start_ref              = None # start of linked list
        self.__end_ref                = None # end of linked list
        self.__current_ref            = None # current event
        
        # keep track of current ID, needed for data structures outside of this linked list
        self.__nextID                 = 0
        
        self.__last_added_event       = None # last added event
        self.__insert_from_last_event = kwargs.get('InsertFromLast',True)

        self.__have_return_dataframe  = False
    
    def Reset(self):
        self.__init__(**self.__store_for_reset)


    def EventData(self, e = None):
        # format output for 'Population' object
        if not e is None:
            return e.ID, e.time, e.data
        else:
            return None,None,None


    def AddEvent(self, time = None, **kwargs):
        self.__have_return_dataframe = False # event list changed, need to compute dataframe from 'self.data' again
        updated                      = False
        
        
        # create event ... assume parent is the event that 'current_ref' points to
        n = EventNode(ID = self.__nextID, time = time, data = kwargs, parent_ref = self.__current_ref)
        
        # set parent
        if not self.__current_ref is None:
            if self.__current_ref.daughter_ref[0] is None:
                self.__current_ref.daughter_ref[0] = n
            else:
                self.__current_ref.daughter_ref[1] = n
            
        # ... then sort it into the linked list
        if self.__start_ref is None:
            # first option is that nothing in the linked list exists sofar
            self.__start_ref = n
            self.__end_ref   = n
            updated = True
        elif not self.__insert_from_last_event:
            # set pointer to first element and go through options
            e = self.__start_ref
            if time < e.time:
                # new time is smaller than the time of first element, thus add new event at beginning of linked list
                e.last_ref       = n
                n.next_ref       = e
                self.__start_ref = n
                updated          = True
            else:
                while e.time < time:
                    if e.next_ref is None:
                        # reached end of linked list
                        e.next_ref     = n
                        n.last_ref     = e
                        self.__end_ref = n
                        updated        = True
                        break
                    else:
                        # continue iterating
                        e              = e.next_ref
                if not updated:
                    # just jumped to the item 'e' that has a time one step ahead of 'n', then the while loop stopped
                    n.last_ref          = e.last_ref
                    n.next_ref          = e
                    e.last_ref.next_ref = n
                    e.last_ref          = n
        else:
            # start at last added event ...
            e = self.__last_added_event
            if e.time < time:
                # ... then check for direction of going though linked list. either forward ...
                while e.time < time:
                    if e.next_ref is None:
                        e.next_ref      = n
                        n.last_ref      = e
                        self.__end_ref  = n
                        updated         = True
                        break
                    else:
                        e               = e.next_ref
                if not updated:
                    n.last_ref          = e.last_ref
                    n.next_ref          = e
                    e.last_ref.next_ref = n
                    e.last_ref          = n
            else:
                # ... or go backward 
                while e.time > time:
                    if e.last_ref is None:
                        n.next_ref       = e
                        e.last_ref       = n
                        self.__start_ref = n
                        updated          = True
                        break
                    else:
                        e                = e.last_ref
                if not updated:
                    e.next_ref.last_ref  = n
                    n.next_ref           = e.next_ref
                    e.next_ref           = n
                    n.last_ref           = e



        self.__nextID           += 1
        self.__last_added_event  = n
        return self.EventData(n)


    def NextEvent(self):
        self.__have_return_dataframe = False # pointer in time changes, need to compute dataframe 'self.data' again
        if self.__current_ref is None:
            self.__current_ref = self.__start_ref
        else:
            self.__current_ref = self.__current_ref.next_ref
        return self.EventData(self.__current_ref)


    def EventTimes(self, first = None):
        # backward compatibility
        return self.GenerateListOfTimes()
    
    
    def CurrentEventDict(self):
        self.__current_ref.to_dict()
    

    def GenerateListOfTimes(self):
        lot = list()
        n = self.__start_ref
        while not n is None:
            lot.append(n.time)
            n = n.next_ref
        return np.array(lot,dtype=np.float)

    
    # wrapper for pandas output
    def DataFrameAppend(self, dataframe, event):
        if dataframe is None:   dataframe = pd.DataFrame(event.to_dict())
        else:                   dataframe = dataframe.append(pd.DataFrame(event.to_dict()), ignore_index = True)
        return dataframe
    
    
    # output various slices of the data
    def CurrentPopulationData(self):
        df = None
        if not self.__start_ref is None:
            n = self.__start_ref
            while not n is None:
                if not n.daughter_ref[0] is None:
                    if n.time < self.curtime and n.daughter_ref[0].time > self.curtime and n.daughter_ref[1].time > self.curtime:
                        df = self.DataFrameAppend(df,n)
                n = n.next_ref
        return df


    def FounderPopulationData(self):
        df = None
        if not self.__start_ref is None:
            n = self.__start_ref
            while not n is None:
                if n.parent_ref is None:
                    df = self.DataFrameAppend(df,n)
                n = n.next_ref
        return df


    def LineageData(self, ID):
        n = self[ID]
        df = self.DataFrameAppend(None,n)
        while not n.parent_ref is None:
            n = n.parent_ref
            df = self.DataFrameAppend(df,n)
        return df
        

    def __getattr__(self, key):
        if key  == 'times':
            return self.GenerateListOfTimes()
        elif key == 'curtime':
            if not self.__current_ref is None:
                return self.__current_ref.time
            else:
                return 0
        elif key == 'data':
            if not self.__have_return_dataframe:
                n = self.__start_ref
                if not n is None:
                    df = self.DataFrameAppend(None, n)
                    while n != self.__current_ref:
                        n = n.next_ref
                        df = self.DataFrameAppend(df,n)
                self.__return_dataframe      = df
                self.__have_return_dataframe = True
            return self.__return_dataframe


    def __getitem__(self, key):
        # also backward compatibility, accessing a particular event via its ID is probably not the best idea
        e = self.__start_ref
        while e.ID != key:
            if not e.next_ref is None: e = e.next_ref
            else: break
        if e.ID == key:
            return e
        else:
            return None


    def __str__(self):
        max_time = None
        if not self.__end_ref is None:
            max_time = self.__end_ref.time
        cur_time = None
        if not self.__current_ref is None:
            cur_time = self.__current_ref.time
        return '# EventLine Linked Lists, collected {} events, maximum time: {}, current time: {}'.format(self.__nextID,max_time, cur_time)



