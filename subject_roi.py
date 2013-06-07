#!/usr/bin/env python
#Make subject ROI
"""
Make ROI for single subject.
"""
import os 
import sys
import csv
import nibabel as nib
import numpy as np
import argparse
import subprocess as subp
import time 
import scipy.ndimage.morphology  as morph
import operator 
def main():
    """
    Make ROI 
    """
    parser = argparse.ArgumentParser(description = 'Make ROIs mask')
    parser.add_argument('-s',
                        dest = 'sess',
                        required = True,
                        help = 'Input sess list file.'
                        )
    parser.add_argument('-r',
                        dest = 'roi',
                        required = True,
                        help = 'Input ROI file.'
                        )
    parser.add_argument('-m',
                        dest = 'mask',
                        required = True,
                        help = 'Input Harvard mask file.'
                        )
    parser.add_argument('-t',
                        dest = 'thresh',
                        required = True,
                        type = float,
                        default = 100,
                        help = 'Voxels of roi threshold.'
                        )
    args = parser.parse_args()
   
    sessfile = args.sess
    sess = read_sess_list(sessfile)

    roi = args.roi
    roi_mask = args.mask
    size_th = args.thresh

    data_dir = os.getcwd()
    xfm = 'xfm/standard2brain.mat'
    
    mp = make_pair(roi,roi_mask,0.5)
    for sid in sess:
    
        sess_dir = os.path.join(data_dir, sid)
        xfm_file = os.path.join(sess_dir, xfm)
        mask_dir = os.path.join(sess_dir, 'mask')
        brain_s  = os.path.join(sess_dir,'anat/brain.nii.gz')
        brain_wm = os.path.join(sess_dir,'anat/brain_pve_2.nii.gz')
        
        if not os.path.exists(mask_dir):
            os.mkdir(mask_dir)
        
        if not os.path.exists(brain_wm):
            print "fast %s"%sid
            continue

        st = time.time()        
        #trans_to_subject(brain_s,xfm_file,roi,roi_mask,mask_dir)
        
        roi_img = nib.load(os.path.join(mask_dir,'facenet.nii.gz'))
        pve_img = nib.load(brain_wm)
        mask_harv = nib.load(os.path.join(mask_dir,'harvard.nii.gz'))

        roi_generator(roi_img, mask_harv,pve_img,mp,size_th)

        print 'single time: %s'%(time.time()-st)

def make_pair(roi,ht,th):
    """
    make roi mask id pairs.
    """
    img = nib.load(roi)
    img_h = nib.load(ht)

    data = img.get_data()
    data_h = img_h.get_data()

    roi_id = np.unique(data)
    roi_id = roi_id.tolist()
    roi_id = [int(i) for i in roi_id]
    roi_id.pop(0)
    print roi_id

    dict = {}
    log = open('mskroi.log','wb')
    log.write("threshold : %s \n"%th)

    for id in  roi_id:
        tmp = data_h[data==id]
        lables = np.unique(tmp)
        lables = np.trim_zeros(lables)
        lables = lables.tolist()
       
    #    print lables
        
        bin_all = data==id
        size = bin_all.sum()
        
        buff = []
        for lab in lables:
          bin  = tmp==lab
          ct = bin.sum()
          fac = float(ct)/size
          buff.append((lab,fac))
        
        dtype = [('lab',int),('fac',float)]
        buffarr = np.array(buff,dtype=dtype)
        sorted = np.sort(buffarr,order='fac')
        sorted = sorted.tolist()
        sorted.reverse()
        
        buff = []
        tmpf = 0
     
        for p in sorted:
            tmpf += p[1]
            if tmpf <= th:
                buff.append(p[0])
            else:
                buff.append(p[0])
                break
        print buff
        dict.update({id:buff})
    
    log.close()
    print dict
    return dict

def read_sess_list(sess):
    """docstring for fname"""
    sf = open(sess,'r')
    sess = sf.readlines()
    sess = [line.rstrip('\n') for line in sess]
    print sess
    return sess

def read_pair(mskpair):
    reader = csv.reader(open(mskpair, 'rb'))
    mydict = dict(x for x in reader)
    #sort a dictionary by key itemgetter(0) /by value itemgetter(1)
    #sortp = sorted(mydict.iteritems(),key=operator.itemgetter(0))
    return mydict

def trans_to_subject(brain, xfm, roi, roi_mask,outdir):
    """
    Transform to subject space.
    """
    
    outf = os.path.join(outdir,'facenet.nii.gz')
    cmdf = "flirt -in %s  -ref %s -interp nearestneighbour\
    -applyxfm -init %s -out %s"%(roi,brain,xfm,outf)
    print cmdf
    os.system(cmdf)

    harv= os.path.join(outdir, 'harvard.nii.gz')
    cmdh = "flirt -in %s  -ref %s -interp nearestneighbour\
    -applyxfm -init %s -out %s"%(roi_mask,brain,xfm,harv)
    print cmdh
    os.system(cmdh)


def roi_generator(img,mask,brain_wm,pair_list,th):
    """
    #3.get ids    id->roi->morph->mask->size->morph   ++bin++id 
    #4.save
    """
    data = img.get_data()
    maskdata = mask.get_data()
    wmdata = brain_wm.get_data()
    print data.shape
    print maskdata.shape
    print wmdata.shape

    roi_ids = np.unique(data)
    id = roi_ids[1:]
    print "\nroi:"
    print  id
    print '\n'
    
    imgbuff = np.zeros(data.shape)
    binbuff = np.zeros(data.shape)
    print imgbuff.shape
    print binbuff.shape

    #make single roi
    for i in id:
        tmproi = data
        tmproi = data== i
        label = pair_list.get(int(i))
        print 'do mask:'
        print i ,label

        tmpmask = np.zeros(data.shape,dtype=np.int8)
        for lab in label:
            tmpmask += maskdata==lab
        #    print tmpmask.sum()

       #roi dilation
        size = 0
        while size < th: 
            dila_roi = morph.binary_dilation(tmproi)
            print dila_roi.sum()
            #mask roi
            dila_roi = np.logical_and(dila_roi,wmdata)
            print dila_roi.sum()
#            dila_roi = np.logical_and(dila_roi,tmpmask)
#           print dila_roi.sum()
            size = (dila_roi==1).sum()
#            print size
        '''
        resultimg = dila_roi    
        result = img
        result._data = resultimg
        nib.save(result,'tm_%s.nii.gz'%i)
        '''

        binbuff = binbuff + dila_roi
        dila_roi = dila_roi*i
        print np.unique(dila_roi)
        resultimg = dila_roi    
        result = img
        result._data = resultimg
        nib.save(result,'tm_%s.nii.gz'%i)
        imgbuff = imgbuff + dila_roi

    tmpbin = binbuff==1
    resultimg = tmpbin*imgbuff
    result = img
    result._data = resultimg
    nib.save(result,'tm_bin_mask.nii.gz')

if __name__=='__main__':
    main()
