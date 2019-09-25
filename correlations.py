#!/usr/bin/env python3

import numpy as np
import argparse
import pandas as pd
import sys,math



class CorrelationStructure(object):
    def __init__(self,**kwargs):
        self.A = np.eye(2)
    
    
    def Correlation(self,lineage = [0,0]):
        return 1.
    

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l","--lineage",nargs=2,default=[0,0],type=int)
    args = parser.parse_args()
    
    cs = CorrelationStructure(**vars(args))
    
    
    print(cs.Correlation(args.lineage))
    
if __name__ == "__main__":
    main()

        
