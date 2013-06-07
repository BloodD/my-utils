# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import os 
import sys
import time as time
import numpy as np
import scipy as sp

from scipy.sparse import csc_matrix
from mvpa2.datasets.mri import map2nifti,fmri_dataset

def load_dot():
    """
    load dot file
    """
    filename = raw_input("dot>>>")
    maskfile = raw_input("mask>>>")
    
    print "load data:"    
    data = np.loadtxt(filename)
    print data
    
    print "load mask:"
    seed_set = fmri_dataset(samples=maskfile,mask=maskfile)
    seed = seed_set.copy(sa=[])

    print seed

    sparse_set = csc_matrix((data[:,2],(data[:,0]-1,data[:,1]-1)))
    seed.samples = sparse_set.T.todense()
    
    print seed.samples.shape
    print seed.a
    print seed.sa
    print seed.fa
    
    seed.save(filename.replace('.dot','.T.hdf5'))
    return 0

def main():
    load_dot()

if __name__== '__main__':
    main()


