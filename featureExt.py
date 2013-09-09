#!/usr/local/env python
"""
Feature extraction form csv file.
"""
__author__ = 'BloodD'

import os
import sys
import numpy as np
import string as str
import csv
import time 
    
author_dict = {}
paper_dict = {}


def main():
    print "input file name"
    st = time.time()
    load_paper_author('data/PaperAuthor.csv')
    load_train('data/Train.csv')
    print "total time: %s"%(time.time()-st)

def load_train(filename):
    log = open('log','w')
    with open(filename,'rb') as myCsvFile:
        TruePaperIds = []
        FalsePaperIds = []
        AuthorPaper = []
        Aline = []
        content = csv.reader(myCsvFile)
        content.next()
        for AuthorId,ConfirmedPaperIds,DeletedPaperIds in content:
            print AuthorId
            log.write(AuthorId)
            coauthor = []
            papers = author_dict.get(int(AuthorId))
            #print papers
            paper = np.unique(papers)
            for p in paper:
                coa = paper_dict.get(int(p))
               # print p,'--->',coa
                coa = np.unique(coa)
                coauthor.extend(coa.tolist())
            #coauthor = np.unique(coauthor)
            #print coauthor
            
            cpaper = ConfirmedPaperIds.split()
            dpaper = DeletedPaperIds.split()
            cc = []
            dc = []
            all = 0
            for cp in cpaper:
           #     print int(cp)
                cpco = paper_dict.get(int(cp))
                cpco = np.unique(cpco)
            #    print cpco
                num = 0
                for p in cpco:
                    if p!=int(AuthorId):
                        num += coauthor.count(p)
           #     print num
                cc.append(num)
                all += num
            log.write( 'total true  coauthor number:%s\n'%all)

            alld = 0
            for dp in dpaper:
            #    print int(dp)
                dpco = paper_dict.get(int(dp))
             #   print dpco
                dpco = np.unique(dpco)
                num = 0
                for p in dpco:
                    if p!=int(AuthorId):
                        num += coauthor.count(p)
              #  print num
                dc.append(num)
                alld += num
            log.write( 'total coauthor number:%s\n'%alld)
            

            '''
            IdC = ConfirmedPaperIds.split(" ")
            IdD = DeletedPaperIds.split(" ")
            TruePaperIds.append(IdC)
            FalsePaperIds.append(IdD)
            Aline.append(AuthorId)
            Aline.append(TruePaperIds)
            Aline.append(FalsePaperIds)
            AuthorPaper.append(Aline)
            '''
            #print AuthorPaper[0]

def load_paper_author(filename):
    '''
    paper : 179681
    author:2293830
    '''
    #co_author = np.zeros((2293830,2293830),dtype=np.int8)
    with open(filename,'rb') as paperAuthor:
        content =  csv.reader(paperAuthor)
        content.next()
        for PaperId,AuthorId,Name,Affiliation in content:
               # print int(PaperId),int(AuthorId)
                author_dict.setdefault(int(AuthorId), []).append(int(PaperId))
                paper_dict.setdefault(int(PaperId), []).append(int(AuthorId))

if __name__== '__main__':
    main()
