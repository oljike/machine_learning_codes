#Term project_problem 2
#Problem1:linear regression
#Author: Zhanat Makhataeva
#Date: 13th November
#Note: execute the program from the comand line in following form:
#F:\Fall_2016\ROBT407-Machine_Learning\project1>logistic_regression_zhanat.py data\Dtest.txt data\Dtrain.txt
import sys, numpy
import random
from random import choice
from numpy import linalg as LA
import random as np
import matplotlib.pyplot as plt
from numpy import exp 
from numpy.linalg import inv
globalerror=0
finalgradient=[]
listdtestx1=[]; listdtestx2=[];listdtesty=[]
listdtrainx1=[]; listdtrainx2=[];listdtrainy=[]
userdefineddata=True
namedtest=sys.argv[1]
namedtrain=sys.argv[2]
datatest=numpy.genfromtxt(namedtest, delimiter=' ', skip_header=0)
datatrain=numpy.genfromtxt(namedtrain, delimiter=' ', skip_header=0)
learningrate=0.001
iteration=0
learningratemat = numpy.mat(learningrate)
# weight= numpy.random.rand(3)*2-1
weight = [0.1,0.1,0.1]
g=0


def logisticregression(X,Y, w):
	xdtrainmat = numpy.mat(X)
	ydtrainmat = numpy.mat(Y)
	xdtrainmattranspose=numpy.transpose(xdtrainmat)
	wtranspose=numpy.transpose(w)
	wdotx=numpy.dot(xdtrainmat,wtranspose)
	s=numpy.dot(ydtrainmat,wdotx)
	logisticfunction=1/(1+exp(-s))
	gradient=numpy.dot(logisticfunction,ydtrainmat)
	gradient=numpy.dot(numpy.transpose(gradient),xdtrainmat)
	gradientnorm=LA.norm(gradient)
	gradient=gradient/gradientnorm
	return gradient	
	
def generatedata(N):
	inputs=[]
	xdata=[]
	ydata=[]
	result=[]
	x0=numpy.ones(N)
	xb=(numpy.random.rand(N)*2-1)/2-0.5
	yb=(numpy.random.rand(N)*2-1)/2+0.5
	xr=(numpy.random.rand(N)*2-1)/2+0.5
	yr=(numpy.random.rand(N)*2-1)/2-0.5
	a=len(xb)*0.9
	for i in range(0,int(a)):
		inputs.append([x0[i],xb[i],yb[i],1])
		inputs.append([x0[i],xr[i],yr[i],-1])
	for j in range(int(a), len(xb)):
		inputs.append([x0[j],xb[j],yb[j],-1])
		inputs.append([x0[j],xr[j],yr[j],1])
	return inputs 


def fromfiledatatest():
	inputsdtest=[]
	xdtest=[]
	ydtest=[]
	result=[]
	for i in datatest:
		listdtesty.append(i[0])
		listdtestx1.append(i[1])
		listdtestx2.append(i[2])
		dtesty=numpy.array(listdtesty)
		dtestx0=numpy.ones(len(datatest))
		dtestx1=numpy.array(listdtestx1)
		dtestx2=numpy.array(listdtestx2)
		numberoftestpoints=len(dtestx1)
	for j in range(0, len(dtestx1)):
		inputsdtest.append([dtestx0[j],dtestx1[j],dtestx2[j],dtesty[j]])
	return inputsdtest 

def fromfiledatatrain():
	inputsdtrain=[]
	xdtrain=[]
	ydtrain=[]
	result=[]
	for i in datatrain:
		listdtrainy.append(i[0])
		listdtrainx1.append(i[1])
		listdtrainx2.append(i[2])
		dtrainy=numpy.array(listdtrainy)
		dtrainx0=numpy.ones(len(datatrain))
		dtrainx1=numpy.array(listdtrainx1)
		dtrainx2=numpy.array(listdtrainx2)
		numberoftrainpoints=len(dtrainx1)
	for j in range(0, len(dtrainx1)):
		inputsdtrain.append([dtrainx0[j],dtrainx1[j],dtrainx2[j],dtrainy[j]])
	return inputsdtrain	


def train(traindata):
	global weight, g
	learned = False
	iteration = 0
	xdat=[]
	ydat=[]
	numberofiterations=2000
	for x in traindata:
		xdat.append([x[0], x[1], x[2]])
		ydat.append([x[3]])
	g=logisticregression(xdat,ydat,weight)
	g=numpy.array(g)
	while not learned:
		weight+=g[0]*learningrate
		g=logisticregression(xdat,ydat,weight)
		g=numpy.array(g)
		iteration+=1
		if iteration>=numberofiterations:
			learned=True
			

def response(x):
	global weight
	xdata=[]
	xdata.append([x[0], x[1], x[2]])
	xmat = numpy.mat(xdata)
	wmattranspose=numpy.transpose(weight)
	wdotx=numpy.dot(xmat,wmattranspose)
	logreg=1/(1+exp(-wdotx))
	if logreg >= 0.5:
		return 1
	else:
		return -1			
										
			
def validation_logreg(chunksx, chunksy, n):
	global weight
	chunksx = [chunksx[x:x+n] for x in range(0, len(chunksx), n)]
	chunksy =  [chunksy[x:x+n] for x in range(0, len(chunksy), n)]
	error_values = []
	outfile=open('logistic_regression_validation_train.txt',"w")
	outfile.write('logistic regression results from validation on train set of experiment data\n')
	for i in range(n):
		valsetx = chunksx[i]
		valsety = chunksy[i]
		valset = []
		new_train=[]
		new_label=[]
		newset_train=[]
		for w in range(len(valsetx)):
			s = valsetx[w]
			t = valsety[w]	
			z = s+t
			valset.append(z)	
		error =0.0
		for j in range(n):
			if j != i:
				new_train.extend(chunksx[j])
				new_label.extend(chunksy[j])
		for r in range(len(new_train)):
			p = new_train[r]
			y = new_label[r]	
			u = p+y
			newset_train.append(u)
		train(newset_train)		
		for x in valset:
			r = response(x)
			if r != x[3]:
				error+=1
			if x[3] == 1:
				plt.plot(x[1],x[2],'ob')
			else:
				plt.plot(x[1],x[2],'or')
		error=error/len(valset)	
		print len(valset)
		error_values.append(error)
	for i in range (0, n):
		c=error_values[i]
		outfile.write(str(i+1)+'  '+str(c)+'\n')
	outfile.close()
	return error_values		

	
def main():
	global weight
	error=0.00
	xdat=[]
	ydat=[]
	if userdefineddata:
		trainset=generatedata(100)
		testset=generatedata(1000)
	else:
		trainset= fromfiledatatrain()
		testset = fromfiledatatest()
	
	xdat = []
	ydat = []
	for x in testset:
		xdat.append([x[0], x[1], x[2]])
		ydat.append([x[3]])
	finalerror=validation_logreg(xdat, ydat,10)	
	print finalerror
	a=[]
	b=[]
	for i in range(int(2)):
		a.append(i)
		eq = (-weight[1]*a[i])/weight[2]-weight[0]/weight[2]
		b.append(eq)
	
	plt.plot(a,b,'-')	
	titleprint='Logistic Regression, validation on train set, Error: ' + str(error)
	plt.title(titleprint)
	plt.show()
		
main()	

