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
            self.__iterateInheritance = True
        else:
            self.A          = self.ConstructMatrix2D(eigenvalues = self.__eigenvalues, theta = self.__theta)
            self.__iterateInheritance = True
        
        self.dimensions = np.shape(self.A)[0]
        
        self.__noisecorrelation = np.array(kwargs.get('noisecorrelation',1))
        if isinstance(self.__noisecorrelation,(float,np.float)):
            self.__noisecorrelation *= np.eye(self.dimensions)
        elif isinstance(self.__noisecorrelation,(np.ndarray)):
            if len(np.shape(self.__noisecorrelation)) == 1:
                self.__noisecorrelation = np.diag(self.__noisecorrelation)

        self.StationaryCov = self.ComputeStationaryCovariance(maxiterations = kwargs.get('maxiterations',100))

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
        
        
    def ComputeStationaryCovariance(self,maxiterations = 100):
        if self.__iterateInheritance:
            # iteratively compute sum_m A^m (A.T)^m
            r   = self.__noisecorrelation
            cur = self.__noisecorrelation
            
            for i in range(maxiterations):
                cur = np.matmul(self.A,np.matmul(cur,self.A.T))
                r  += cur
            
            return r

        #else:
            ## analytic solution based on geometric series
            #il1l1 = 1./(1-self.__eigenvalues[0]**2)
            #il2l2 = 1./(1-self.__eigenvalues[1]**2)
            #il1l2 = 1./(1-self.__eigenvalues[0]*self.__eigenvalues[1])
            #itan  = 1./np.tan(2 * np.pi * self.__beta)
            #return self.__noiseamplitude**2 * np.array([[ il2l2,                  (il1l2 - il2l2) * itan],
                                                        #[ (il1l2 - il2l2) * itan, il1l1 + (il1l1 - 2*il1l2 + il2l2)*itan*itan]], dtype = np.float)
        
    
    
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

        
