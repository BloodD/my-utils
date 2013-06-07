#!/usr/bin/env python
# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#-*- coding: utf-8 -*-
############################################################
#auto generate docstr   Date:2013.03.12  
############################################################
"""
usage: docstr_generator [-h] [-r dir] -s source

Extract .py file docstring and insert into the .py file.

optional arguments:
  -h, --help  show this help message and exit
  -r dir      choose dir mode
  -s source   source file or dir.

"""

import os
import sys
import re
import argparse


parser = argparse.ArgumentParser(description = '''Extract .py file 
                            docstring and insert into the .py file.''', 
                                prog = 'docstr_generator')
parser.add_argument('-r',
                        dest = 'dir',
                        required = False,
                        metavar = 'dir',
                        help = 'choose dir mode')
parser.add_argument('-s',
                        dest = 'source',
                        required = True,
                        metavar = 'source',
                        help = 'source file or dir.')
args = parser.parse_args()

if args.dir != None:
    print 'dir mode  test.'
    pass
else:
    print '--------------------------------------------------------------'
    print 'Target command file is: %s'%args.source
    count = 0
    num = 0
    lines=[]
    f = open(args.source,"r+")
    content = f.readlines()
    for line in content:  
        if line.find('"""' )!=-1 and count<2:
            count=count+1
            num=num+1
            lines.append(num)
        elif count==2:
            break
        else:
             num=num+1 

    docstring = content[lines[0]:lines[1]-1]
    
    for dstr in docstring:
        content.remove(dstr)
    cmd = "python %s -h>tmp"%args.source
    print '-----------------------------------------------------------------'
    print "Run>>>", cmd
    print '-----------------------------------------------------------------'
    os.system(cmd)
    tmp = open('tmp',"r+")
    newdocstr = tmp.readlines()
    
    print "New docstring is:"
    print ' '.join(newdocstr)
    print '-----------------------------------------------------------------'
    
    ln = lines[0]
    for newstr in newdocstr:
        content.insert(ln,newstr)
        ln = ln+1 
    content.insert(ln,'\n')
    print content
    content = [l.replace('\r\n','\n') for l in content]
    print content
    f.close()
    wf = open(args.source,"w")
    wf.writelines(content)
    wf.close()
    tmp.close()
    os.remove('tmp')
    print "Insert it into .py file is done!"
    print '-----------------------------------------------------------------'
