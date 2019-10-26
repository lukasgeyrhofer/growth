#!/usr/bin/env python3


import numpy as np
import pandas as pd


class DivisionTimes(object):
    '''
    Object to provide basic functionality
    
    Provides access to 'DrawDivisionTimes' and the internal variables 'mean' and 'variance'
    
    In this simplest implementation does not rely on inheritance of information, division times are normally distributed
    
    '''
    
    def __init__(self,**kwargs):
        self.__min_divtime     = kwargs.get("mindivtime",None)
        self.__avg_divtime     = kwargs.get("avgdivtime",2.)
        self.__var_divtime     = kwargs.get("vardivtime",1.)
        self.__stddev_divtime  = np.sqrt(self.__vardivtime)

        self.recorded_DivTimes = list()


    def DrawDivisionTimes(self,size = 2, **kwargs):
        dt = np.random.normal(loc = self.__avg_divtime,scale = self.__stddev_divtime,size = size)
        if not self.__min_divtime is None:
            dt[dt < self.__min_divtime] = self.__min_divtime
        return dt


    def __getattr__(self,key):
        if   key == 'variance':
            return self.__var_divtime
        elif key == 'mean':
            return self.__avg_divtime
        elif key == 'divisiontimes':
            return np.array(self.recorded_DivTimes, dtype = np.float)




def ConstructMatrix(eigenvalues = np.ones(2),theta = np.array([0,.25])):
    # construct matrix from its 2 eigenvalues and the two angles determining the eigendirections
    eigenvalues = np.array(eigenvalues,dtype=np.float)
    theta       = np.array(theta,dtype = np.float) # assume theta in interval [0 ... 1]
    ct          = np.cos(2*np.pi*theta)
    st          = np.sin(2*np.pi*theta)
    isdt        = 1./np.sin(2*np.pi*(theta[0] - theta[1]))
    return isdt * np.array([[
        (eigenvalues[0]*ct[1]*st[0] - eigenvalues[1]*ct[0]*st[1]),
        (eigenvalues[0] - eigenvalues[1])*st[0]*st[1] ],[
        (eigenvalues[1] - eigenvalues[0])*ct[0]*ct[1],
        (eigenvalues[1]*ct[1]*st[0] - eigenvalues[0]*ct[0]*st[1])
    ]])


def ConstructRotationMatrix(angles = [0]):
    def RotM(angle):
        return np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]],dtype = np.float)
    
    for i,angle in enumerate(angles):
        if i == 0:
            r = RotM(angle)
        else:
            


class DivisionTimes_matrix(DivisionTimes):
    def __init__(self,**kwargs):
        super(DivisionTimes_matrix,self).__init__(kwargs)
        
        if not kwargs.get('inheritancematrix',None) is None:
            self.inheritancematrix     = np.array(kwargs.get('inheritancematrix'),dtype=np.float)
            self.dimension_shape       = np.shape(self.inheritancematrix)
        
            # check if square 2d matrix, otherwise convert
            if len(self.dimension_shape) != 2 or self.dimension_shape[0] != self.dimension_shape[1]:
                self.inheritancematrix = np.flatten(self.inheritancematrix)
                self.dimensions        = np.sqrt(len(self.inheritancematrix))
                self.inheritancematrix = np.reshape(self.inheritancematrix, (self.dimensions,self.dimensions))
        else:
            self.eigenvalues = np.array(kwargs.get('eigenvalues',[.3,.9]),dtype=np.float)
            self.dimensions  = len(self.eigenvalues)
            
            self.eigenvectors = list()
            
            if not kwargs.get('matrixangles',None) is None:
                self.matrixangles = kwargs.get('matrixangles')
                if not self.dimensions == 2:
                    raise NotImplementedError('matrix construction via eigenvector angles only implemented for 2d')
                for angle in self.matrixangles:
                    self.eigenvectors.append(np.array([np.cos(angle),-np.sin(angle)],dtype=np.float))
            else:
                self.eigenvectors = np.array(kwargs.get('eigenvectors',np.eye(self.dimensions)),dtype=np.float).reshape((self.dimensions,self.dimensions))
            
            for i in range(self.dimensions):
                self.eigenvectors[:,i] /= np.linalg.norm(self.eigenvectors[:,i])
            
            self.inheritancematrix = np.matmul(self.eigenvectors,np.matmul(np.diag(self.eigenvalues),np.linalg.inv(self.eigenvectors)))
        
        
        self.projection = np.array(kwargs.get('projection',[1,0]),dtype=np.float)
        

    
class DivisionTimes_2dARP(DivisionTimes):
    def __init__(self,**kwargs):
        # distribution of division times
        self.__divtime_min            = kwargs.get('divtime_min',.1)
        self.__divtime_mean           = kwargs.get('divtime_mean',1)
        self.__divtime_var            = kwargs.get('divtime_var',.01)
        self.recorded_DivTimes        = list()
        
        self.__noiseamplitude         = kwargs.get('noiseamplitude',.6)

        
        if kwargs.get("inheritancematrix",None) is None:
            # inheritance of internal state
            self.__iterateInheritance = False # don't iterate, as expressions are available analytically
            self.__eigenvalues        = np.array(kwargs.get('eigenvalues',[.3,.9]),dtype=np.float)
            self.__beta               = kwargs.get('beta',.3)
            self.A                    = ConstructMatrix(eigenvalues = self.__eigenvalues, theta = [0,self.__beta])

        else:
            # inheritance matrix directly provided
            self.A                    = np.array(kwargs.get("inheritancematrix"),dtype=np.float)
            self.__iterateInheritance = True
            
        self.__stationaryCov          = self.ComputeStationaryCovariance()
        self.__alpha                  = kwargs.get('alpha',.6)
        self.projection               = np.array([-np.sin(self.__alpha),np.cos(self.__alpha)],dtype=np.float)
        self.projection              *= np.sqrt(self.__divtime_var/self.variance)
        
        # debug mode
        self.__ignore_parents         = kwargs.get('ignoreParents',False)
        
        
    def ComputeStationaryCovariance(self):
        if self.__iterateInheritance:
            # iteratively compute sum_m A^m (A.T)^m
            maxiterations = 100
            r   = np.eye(2)
            cur = np.eye(2)
            for i in range(1,maxiterations):
                cur = np.matmul(self.A,np.matmul(cur,self.A.T))
                r  += cur
            return self.__noiseamplitude**2 * r
        else:
            # analytic solution based on geometric series
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


    def __getattr__(self,key):
        if key == 'variance':
            return np.dot(self.projection, np.dot(self.__stationaryCov, self.projection))
        elif key == 'divisiontimes':
            return np.array(self.recorded_DivTimes, dtype = np.float)
        else:
            super(DivisionTimes_2dARP,self).__getattr__(key)


