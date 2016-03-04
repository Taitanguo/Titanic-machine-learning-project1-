#!/usr/bin/python

import xlrd
import sys
import pickle
import numpy
import xlwt

workbook = xlrd.open_workbook('/Users/Uvaguy/Desktop/ml/titannic/titanic3.xls')
data=workbook.sheet_by_name('titanic3')

# print(data);

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

# print fam
# print nonfam

final_fam=[]
final_nonfam=[]

#replacing missing value with mean families that have members of unknown age
average = []
col = data.col(4)
for i in range(1 , len(col)):
    if col[i].value == xlrd.empty_cell.value:
        col[i].value = 0
    average.append(col[i].value)
ave = numpy.mean(average)

#print ave

# print col
for i in range(len(col)):
    if col[i].value == xlrd.empty_cell.value:
        data.col(4)[i].value = ave


def writetocsv(self, filename):
    self.data.to_csv(filename,sep=',')

# from xlwt import Workbook
# wb = Workbook()
# Sheet2 = wb.add_sheet(data)
# wb.save('ex.xls')


# for i in range(len(nonfam)):
#     addFamily=True
#     if data.cell(nonfam[i],4).value == xlrd.empty_cell.value:
#         # data.cell(nonfam[i],4),value = average
#         addFamily=True
#     if addFamily:
#         final_nonfam.append(nonfam[i])


# print final_fam
# print final_nonfam

pickle.dump(final_fam,open('fam.pickle','wt'))
pickle.dump(final_nonfam,open('nonfam.pickle','wt'))








