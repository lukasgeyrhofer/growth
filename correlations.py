#!/usr/bin/env python3

import numpy as np
import argparse
import pandas as pd
import sys,math

import stochasticgrowth_eventline as sge

import matplotlib.pyplot as plt

def LMSQ(x,y):
    n   = len(x)
    sx  = np.sum(x)
    sy  = np.sum(y)
    sxx = np.dot(x,x)
    sxy = np.dot(x,y)
    syy = np.dot(y,y)
    
    denom  = (n*sxx-sx*sx)
    b      = (n*sxy - sx*sy)/denom
    a      = (sy-b*sx)/n
    estim  = np.array([a,b],dtype=np.float)

    sigma2 = syy + n*a*a + b*b*sxx + 2*a*b*sx - 2*a*sy - 2*b*sxy
    cov    = sigma2 / denom * np.array([[sxx,-sx],[-sx,n]],dtype=np.float)

    return estim,cov


def GetCorrelation( alpha = .6, eigenvalue0 = .3, eigenvalue1=.9, theta0=0., theta1=.3, noiseamplitude0 = .6, noiseamplitude1 = .6 , fit = True, plot = True, logscale = True, maxtreedepth = 30):
    
    def doubledecayInterpNaturalParams(x,d0,d1,a):
        return a * np.exp(-d0*x) + (1-a)*np.exp(-d1*x)    
    
    m       = np.arange(maxtreedepth)
    cs      = CorrelationStructure(eigenvalues = [eigenvalue0,eigenvalue1], theta = [theta0,theta1], noiseamplitude = [noiseamplitude0,noiseamplitude1])
    mdcorr  = cs.MDCorrelation    (size = maxtreedepth, projection_angle = alpha)
    siscorr = cs.SisterCorrelation(size = maxtreedepth, projection_angle = alpha)

    if plot:
        plt.plot(m,mdcorr,  marker = 'o', lw = 1.5, c = 'orange')
        plt.plot(m,siscorr, marker = 'o', lw = 1.5, c = 'blue')
        if logscale:
            plt.ylim((1e-5,1))
            plt.yscale('log')
        else:
            plt.ylim((-.1,1))
    
    if fit:
        secondtimescale = 15
        mdcorr1  = mdcorr[secondtimescale:]
        if mdcorr1[-1] < 0:
            mdcorr1 *= -1
            fitsignMD = -1
        else:
            fitsignMD = +1
        mmd1 = (m[secondtimescale:])[mdcorr1 > 0]
        mdcorr1  = mdcorr1[mdcorr1 > 0]
        
        siscorr1  = siscorr[secondtimescale:]
        if siscorr1[-1] < 0:
            siscorr1 *= -1
            fitsignSIS  = -1
        else:
            fitsignSIS  = +1
        ms1 = (m[secondtimescale:])[siscorr1 > 0]
        siscorr1  = siscorr1[siscorr1 > 0]
        
        decay1_sis,cov1_sis = LMSQ(ms1, np.log(siscorr1))
        decay1_md, cov1_md  = LMSQ(mmd1,np.log(mdcorr1))
        
        mdcorr2 = (mdcorr - fitsignMD * np.exp(decay1_md[0] + m * decay1_md[1]))[:secondtimescale]
        mmd2    = (m[:secondtimescale])[mdcorr2 > 0]
        mdcorr2  = mdcorr2[mdcorr2 > 0]
        
        siscorr2 = (siscorr     - fitsignSIS * np.exp(decay1_sis[0] + m * decay1_sis[1]))[:secondtimescale]
        ms2      = (m[:secondtimescale])[siscorr2 > 0]
        siscorr2  = siscorr2[siscorr2 > 0]
        
        decay2_sis,cov2_sis = LMSQ(ms2, np.log(siscorr2))
        decay2_md, cov2_md  = LMSQ(mmd2,np.log(mdcorr2))

        if plot:
            x = np.linspace(0,maxtreedepth,200)
            plt.plot(x,doubledecayInterpNaturalParams(x,-decay2_sis[1],-decay1_sis[1],1 - fitsignSIS * np.exp(decay1_sis[0])), c = 'darkorange')
            plt.plot(x,doubledecayInterpNaturalParams(x,-decay2_md[1], -decay1_md[1], 1 - fitsignMD  * np.exp(decay1_MD[0])),  c = 'darkblue')
    
    
        return np.array([-decay2_sis[1],-decay1_sis[1],1 - fitsignSIS * np.exp(decay1_sis[0]),-decay2_md[1],-decay1_md[1],1 - fitsignMD * np.exp(decay1_md[0])])
    else:
        return None 



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
        if isinstance(self.__noisecorrelation,(float,np.float)):
            self.__noisecorrelation = self.__noisecorrelation * np.eye(self.dimensions)
        elif isinstance(self.__noisecorrelation,(np.ndarray)):
            if len(np.shape(self.__noisecorrelation)) == 1:
                self.__noisecorrelation = np.diag(self.__noisecorrelation)
            elif len(np.shape(self.__noisecorrelation)) == 0:
                self.__noisecorrelation = self.__noisecorrelation * np.eye(self.dimensions)

        self.StationaryCov = self.ComputeStationaryCovariance(maxiterations = kwargs.get('maxiterations',100), iterate = self.iterateInheritance)

    def Projection(self,projection_angle = None):
        if projection_angle is None:
            projection_angle = 0
        return np.array([-np.sin(2*np.pi*projection_angle),np.cos(2*np.pi*projection_angle)],dtype = np.float)

        
    def Correlation(self,lineage = [0,0], projection_angle = None):
        projection = self.Projection(projection_angle = projection_angle)
        
        c     = self.StationaryCov.copy()
        for l in range(lineage[0]): c = np.matmul(self.A,c)
        for r in range(lineage[1]): c = np.matmul(c,self.A.T)
        c     = np.dot(projection,np.dot(c,projection))
        c    /= np.dot(projection,np.dot(self.StationaryCov,projection))
        
        return c

    def MDCorrelation(self, size = 5, projection_angle = None):
        return np.array([self.Correlation(lineage = (0,m), projection_angle = projection_angle) for m in range(size)],dtype = np.float)
    
    def SisterCorrelation(self, size = 5, projection_angle = None):
        return np.array([self.Correlation(lineage = (m,m), projection_angle = projection_angle) for m in range(size)],dtype = np.float)
        
        
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


    def MD_CC_Inequality(self,projection_angle = None):
        return int(self.Correlation(lineage = (0,1),projection_angle = projection_angle) < self.Correlation(lineage = (2,2), projection_angle = projection_angle))
    

    
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

        
