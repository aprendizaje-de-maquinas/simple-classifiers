from __future__ import print_function

import scipy as sp
from numpy import log

def parse(X, Y , dataset , settype ):
    import csv
    with open('datasets/'+dataset+'-'+settype+'.txt' , 'rb') as csvfile:
        reader=csv.reader(csvfile , delimiter=' ',quotechar='|')
        for row in reader:
            flag = 0
            temp = list() 
            for el in row:
                if flag:
                    Y.append(int(el))
                else:
                    if ':' in el:
                        flag =1
                        temp.append(int(el[0]))
                    elif el !='':
                        temp.append(int(el))
            X.append(temp)


def MLE(X , Y , numFeatures , numExamples , probabilities):
    ## x1y1 x0y1 x1y0 x0y0
    numY1 = float(sum(Y))
    numY0 = float(len(Y)-numY1)
    for i in range(0,numFeatures):
        temp = [float(0)]*4
        for j in range(0,numExamples):
            if Y[j]==1 and X[j][i]==1:
                temp[0]+=1
            elif Y[j]==1:
                temp[1]+=1
            elif Y[j]==0 and X[j][i]==1:
                temp[2]+=1
            elif Y[j]==-1 and X[i][j]==1: ##added
                temp[2]+=1 #added
            else:
                temp[3]+=1
        temp[0] /= numY1
        temp[1] /= numY1
        temp[2] /= numY0
        temp[3] /= numY0

        probabilities.append(temp)

def MAP(X , Y , numFeatures , numExamples , probabilities):
    ## x1y1 x0y1 x1y0 x0y0
    numY1 = float(sum(Y))
    numY0 = float(len(Y)-numY1)
    
    for i in range(0,numFeatures):
        temp = [float(0)]*4
        for j in range(0,numExamples):
            if Y[j]==1 and X[j][i]==1:
                temp[0]+=1
            elif Y[j]==1:
                temp[1]+=1
            elif Y[j]==0 and X[j][i]==1:
                temp[2]+=1
            else:
                temp[3]+=1

        temp[0] +=1 ;
        temp[1] +=1 ;
        temp[2] +=1 ;
        temp[3] +=1 ;

        temp[0] /= numY1+2
        temp[1] /= numY1+2
        temp[2] /= numY0+2
        temp[3] /= numY0+2

        probabilities.append(temp)


def test(X , Y , probabilities):
    numCorrect=0
    for i in range(0 , len(Y)):
        pY1 = float(0)
        pY0 = float(0) 
        for j in range(0 , len(X[0])):
            if probabilities[j][1-X[i][j]] != 0:
                pY1+= log(probabilities[j][1-X[i][j]])
            else:
                pY1 -=100000000000000
            if probabilities[j][3-X[i][j]] != 0:
                pY0+= log( probabilities[j][3-X[i][j]])
            else:
                pY0 -=100000000000000
        prediction = 0
        if pY1 > pY0:
            prediction = 1
        if prediction==Y[i]:
            numCorrect+=1

    return 100*(numCorrect / float(len(Y)))
            

dataset = 'ancestry'
settype = 'train'


Y = list()
X = list()

parse(X,Y , dataset , settype)

numFeatures = X[0][0]
numExamples = X[1][0]


X.remove(X[0])
X.remove(X[0])


probabilitiesMLE = list()
probabilitiesMAP = list()
MLE(X, Y , numFeatures , numExamples ,  probabilitiesMLE)
MAP(X, Y , numFeatures , numExamples ,  probabilitiesMAP)

nX = list()
nY = list()

settype = 'test'
parse(nX , nY , dataset , settype)
nX.remove(nX[0])
nX.remove(nX[0])


MAP = test(nX , nY , probabilitiesMAP)
MLE = test(nX , nY , probabilitiesMLE)

print('MLE Percent correct =  '+str(MLE)+'%')
print('MAP Percent correct =  '+str(MAP)+'%')
