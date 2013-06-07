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
    with open(filename,'rb') as myCsvFile:
        TruePaperIds = []
        FalsePaperIds = []
        AuthorPaper = []
        Aline = []
        content = csv.reader(myCsvFile)
        content.next()
        for AuthorId,ConfirmedPaperIds,DeletedPaperIds in content:
            print AuthorId
            coauthor = []
            papers = author_dict.get(int(AuthorId))
            #print papers
            paper = np.unique(papers)
            for p in paper:
                coa = paper_dict.get(int(p))
               # print p,'--->',coa
                coauthor.extend(coa)
            coauthor = np.unique(coauthor)
            print coauthor
            
            cpaper = ConfirmedPaperIds.split()
            dpaper = DeletedPaperIds.split()
            for cp in cpaper:
                print int(cp)
                cpco = paper_dict.get(int(cp))
                print cpco
                cpco = np.unique(cpco)
                co_au = coauthor[cpco!=int(AuthorId)]
                print co_au
                """
                get co_author number as one feature.
                """
                break
            break

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
