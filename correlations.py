#!/usr/bin/env python3

import numpy as np
import argparse
import pandas as pd
import sys,math

import stochasticgrowth_eventline as sge

class CorrelationStructure(object):
    def __init__(self,**kwargs):
        self.__eigenvalues = kwargs.get('eigenvalues',[.3,.9])
        self.__theta       = kwargs.get('theta',[0,.3])
        
        if 'InheritanceMatrix' in kwargs.keys():
            self.A          = np.array(kwargs.get('InheritanceMatrix'), dtype = np.float)
            self.iterateInheritance = True
        else:
            self.A          = self.ConstructMatrix2D(eigenvalues = self.__eigenvalues, theta = self.__theta)
            self.iterateInheritance = False
        
        self.dimensions = np.shape(self.A)[0]
        
        self.__noisecorrelation = np.array(kwargs.get('noisecorrelation',1),dtype = np.float)
        print(type(self.__noisecorrelation))
        if isinstance(self.__noisecorrelation,(float,np.float)):
            
            self.__noisecorrelation = self.__noisecorrelation * np.eye(self.dimensions)
        elif isinstance(self.__noisecorrelation,(np.ndarray)):
            if len(np.shape(self.__noisecorrelation)) == 1:
                self.__noisecorrelation = np.diag(self.__noisecorrelation)
            elif len(np.shape(self.__noisecorrelation)) == 0:
                self.__noisecorrelation = self.__noisecorrelation * np.eye(self.dimensions)

        print(self.__noisecorrelation)

        self.StationaryCov = self.ComputeStationaryCovariance(maxiterations = kwargs.get('maxiterations',100), iterate = self.iterateInheritance)

    def Projection(self,projection_angle = None):
        if projection_angle is None:
            projection_angle = 0
        return np.array([-np.sin(2*np.pi*projection_angle),np.cos(2*np.pi*projection_angle)],dtype = np.float)

        
    def Correlation(self,lineage = [0,0], projection_angle = None, projection = None):
        if projection_angle is None:
            if projection is None:
                projection = self.Projection()
        else:
            projection = self.Projection(projection_angle = projection_angle)
        
        
        c     = self.StationaryCov.copy()
        for l in range(lineage[0]): c = np.matmul(self.A,c)
        for r in range(lineage[1]): c = np.matmul(c,self.A.T)
        c     = np.dot(projection,np.dot(c,projection))
        c    /= np.dot(projection,np.dot(self.StationaryCov,projection))
        
        return c
        
        
    def ComputeStationaryCovariance(self,maxiterations = 100, iterate = False):
        # stationary covariance given by
        # <x x.T> = sum_m A^m <xi xi.T> (A.T)^m
        if not iterate and self.dimensions == 2:
            # analytic solution based on geometric series
            
            il1l1   = 1./(1-self.__eigenvalues[0]**2)
            il2l2   = 1./(1-self.__eigenvalues[1]**2)
            il1l2   = 1./(1-self.__eigenvalues[0]*self.__eigenvalues[1])
                    
            c1      = np.cos(2*np.pi*self.__theta[0])
            s1      = np.sin(2*np.pi*self.__theta[0])
            c2      = np.cos(2*np.pi*self.__theta[1])
            s2      = np.sin(2*np.pi*self.__theta[1])

            n1      = self.__noisecorrelation[0,0]**2
            n2      = self.__noisecorrelation[1,1]**2
                    
            dd      = 1./((c2*s1 + c1*s2)**2)
                    
            sc      = np.zeros((2,2))
            sc[0,0] =   il1l1 * s1**2*(c2**2*n1+n2*s2**2) - 2 * il1l2 * s1*s2*(c1*c2*n1+n2*s1*s2)         + il2l2 * (c1**2*n1+n2*s1**2)*s2**2
            sc[0,1] = - il1l1 * c1*s1*(c2**2*n1+n2*s2**2) +     il1l2 * (c2*s1+c1*s2)*(c1*c2*n1+n2*s1*s2) - il2l2 * c2*(c1**2*n1+n2*s1**2)*s2
            sc[1,0] =   sc[0,1]
            sc[1,1] =   il1l1 * c1**2*(c2**2*n1+n2*s2**2) - 2 * il1l2 * c1*c2*(c1*c2*n1+n2*s1*s2)         + il2l2 * c2**2*(c1**2*n1+n2*s1**2)
            sc     *= dd
        else:
            # iteratively compute sum_m A^m (A.T)^m
            sc  = self.__noisecorrelation
            cur = self.__noisecorrelation
            
            for i in range(maxiterations):
                cur = np.matmul(self.A,np.matmul(cur,self.A.T))
                sc += cur
            
        return sc
        
    
    
    def ConstructMatrix2D(self,eigenvalues = np.ones(2),theta = np.array([0,.25])):
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


    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l","--lineage",nargs=2,default=[0,0],type=int)
    parser.add_argument("-P","--parameters",nargs="*",default=None)
    args = parser.parse_args()
    
    argument_dict = vars(args)
    if not args.parameters is None:
        argument_dict.update(sge.MakeDictFromParameterList(args.parameters))
        argument_dict.pop('parameters')
    
    cs = CorrelationStructure(**argument_dict)
    
    
    print(cs.Correlation(args.lineage))
    
if __name__ == "__main__":
    main()

        
