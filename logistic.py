from __future__ import print_function

import scipy as sp
from numpy import log
from math import exp

from accelerate.cuda.blas import *
import numpy as np
from timeit import default_timer as timer

from numba import vectorize

RATE=0.0001
NUM_EPOCHS=3000

## no acceleration
def parse(X, Y , dataset , settype ):
    import csv
    with open('datasets/'+dataset+'-'+settype+'.txt' , 'rb') as csvfile:
        reader=csv.reader(csvfile , delimiter=' ',quotechar='|')
        for row in reader:
            flag = 0

            temp = list() 
            temp.append(1)
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


def sigmoid(x):
    try:
        return 1/(1+exp(-x))
    except OverflowError:
        return float(0) 
    
def vecMult(x,y):
    res = float(0)
    for i in range(0 , len(x)):
        res+=x[i]*y[i]
    return res


def gradientAscent(X , Y , thetas):
    global RATE
    global NUM_EPOCHS

    #added
    rows , cols = X.shape
    #end added

    
    for x in range(0 , NUM_EPOCHS):
        print( x ) 
        update = thetas
        # changed from range( 0 , len(X[0])) 
        for j in range(0,cols):
            add = float(0)
            # changed from range( 0 , len(Y))
            for i in range(0 , rows):
                # commented
                #add += (Y[i] - sigmoid(    vecMult(thetas , X[i])   )     )*X[i][j]

                #added
                add += (Y[i] - sigmoid(    np.dot(thetas , X[i])   )     )*X[i][j]
                #end add
                
            update[j] += add*RATE
            
        thetas = update

    #print ( "*********************" )
    #print ( thetas)
    return thetas 
            
def test(X , Y , thetas):
    
    numCorrect=0
    for i in range(0 , len(Y)):
        pY1 = float(0)

        for j in range(0 , len(X[i])):
            pY1+= thetas[j]*X[i][j]

        pY1 = sigmoid(pY1)
        
        prediction = 0
        if pY1 >= 0.5:
            prediction = 1

        if prediction==Y[i]:
            numCorrect+=1

    return 100*(numCorrect / float(len(Y)))

           
dataset = 'netflix'
settype = 'train'


Y = list()
X = list()

parse(X,Y , dataset , settype)

numFeatures = X[0][1]+1
numExamples = X[1][1]

X.remove(X[0])
X.remove(X[0])

#added
X = np.asarray(X)
Y = np.asarray(Y)

# end add

# commented
#thetas = [0]*len(X[0])

#added
thetas = np.zeros(numFeatures) 
#end added

start = timer()
thetas = gradientAscent(X , Y , thetas)
elapsed = timer() - start

nX = list()
nY = list()

settype = 'test'
parse(nX , nY , dataset , settype)
nX.remove(nX[0])
nX.remove(nX[0])

accuracy = test(nX , nY , thetas)

print( 'Logisitc Regression  :    ' + str(accuracy) +'%')
print ( " Took    "+str(elapsed)+" seconds to train" ) 

