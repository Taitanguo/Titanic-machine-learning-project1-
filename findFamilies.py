#!/usr/bin/python

import xlrd
import sys
import pickle

workbook = xlrd.open_workbook('/Users/Uvaguy/Desktop/ml/titannic/titanic3.xls')
data=workbook.sheet_by_name('titanic3')

sibsp=[]
parch=[]

#generating an array of passenger last names
col=data.col(5)
for i in range(len(col)):
    sibsp.append(col[i].value)
col=data.col(6)
for i in range(len(col)):
    parch.append(col[i].value)

fam=[]
nonfam=[]

rowNum=1
while rowNum < len(sibsp):
    if (sibsp[rowNum]+parch[rowNum]) == 0:
        nonfam.append(rowNum)
    else:
        fam.append(rowNum)
    rowNum=rowNum+1

print fam
print nonfam

final_fam=[]
final_nonfam=[]

#removing families that have members of unknown age
for i in range(len(fam)):
    addFamily=True
    if data.cell(fam[i],4).value == xlrd.empty_cell.value:
        print "Reject indivual at row "+str(fam[i]+1)+' entry!\n'
        addFamily=False
    if addFamily:
        final_fam.append(fam[i])

for i in range(len(nonfam)):
    addFamily=True
    if data.cell(nonfam[i],4).value == xlrd.empty_cell.value:
        print "Reject indivual at row "+str(nonfam[i]+1)+' entry!\n'
        addFamily=False
    if addFamily:
        final_nonfam.append(nonfam[i])


print final_fam
print final_nonfam

pickle.dump(final_fam,open('fam.pickle','wt'))
pickle.dump(final_nonfam,open('nonfam.pickle','wt'))








