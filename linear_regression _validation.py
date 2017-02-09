#Term project_problem 2
#Problem1:linear regression
#Author: Zhanat Makhataeva/Alibi Jangeldin/ Olzhas Kabdolov
#Date: 8th November
#Note: execute the program from the comand line in following form:
#F:\Fall_2016\ROBT407-Machine_Learning\project1>pla_zhanat.py data\Dtest.txt data\Dtrain.txt
import sys, numpy
import random
from random import choice
from numpy import linalg as LA
import random as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
globalerror=0
pocketweight=[]
listdtestx1=[]; listdtestx2=[];listdtesty=[]
listdtrainx1=[]; listdtrainx2=[];listdtrainy=[]
userdefineddata=False
namedtest=sys.argv[1]
namedtrain=sys.argv[2]
datatest=numpy.genfromtxt(namedtest, delimiter=' ', skip_header=0)
datatrain=numpy.genfromtxt(namedtrain, delimiter=' ', skip_header=0)
errorlastchank=0
def responsevalid(x, pocket):
	global globalerror, pocketweight
	hperceptron = x[0]*pocket[0]+x[1]*pocket[1]+x[2]*pocket[2]
	if hperceptron >= 0:
		return 1
	else:
		return -1

def validation_linreg(chunksx, chunksy, n):
	global globalerror, pocketweight, errorlastchank
	chunksx = [chunksx[x:x+n] for x in range(0, len(chunksx), n)]
	chunksy =  [chunksy[x:x+n] for x in range(0, len(chunksy), n)]
	error_values = []
	outfile=open('linear_regression_validation_train.txt',"w")
	outfile.write('linear regression results from validation on train set of experiment data\n')
	for i in range(n):
		new_train=[]
		new_trainx = []
		new_label=[]
		valsetx = chunksx[i]
		valsety = chunksy[i]
		valset = []
		for w in range(len(valsetx)):
			s = valsetx[w]
			t = valsety[w]
			z = s+t
			valset.append(z)
		for j in range(n):
			if j != i:
				new_train.extend(chunksx[j])
				new_label.extend(chunksy[j])		
		pocket = linearregression(new_train, new_label)
		error =0.000
		for x in valset:
			r = responsevalid(x, pocket)
			if r != x[3]:
				error+=1
			if x[3] == 1:
				plt.plot(x[1],x[2],'ob')
			else:
				plt.plot(x[1],x[2],'or')
			errorlastchank=error/len(valsetx)
		error_values.append(error/len(valsetx))
	for i in range (0, n):
		c=error_values[i]
		outfile.write(str(i+1)+'  '+str(c)+'\n')
	outfile.close()
	return error_values

def linearregression(X,Y):
	global globalerror, pocketweight
	xdtrainmat = numpy.mat(X)
	ydtrainmat = numpy.mat(Y)
	xdtrainmattranspose=numpy.transpose(xdtrainmat)
	inverseofxandxt=inv(numpy.dot(xdtrainmattranspose,xdtrainmat))
	pseudoinversexdtrain=numpy.dot(inverseofxandxt,xdtrainmattranspose)
	wlin=numpy.dot(pseudoinversexdtrain,ydtrainmat)
	wlinarray=numpy.array(wlin)
	# print 'pocketweight values from training: \n', pocketweight
	return wlinarray

def generatedata(N):
	global globalerror, pocketweight
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
		xdata.append([x0[i],xb[i],yb[i]])
		ydata.append([1])
		xdata.append([x0[i],xr[i],yr[i]])
		ydata.append([-1])
	for j in range(int(a), len(xb)):
		inputs.append([x0[j],xb[j],yb[j],-1])
		inputs.append([x0[j],xr[j],yr[j],1])
		xdata.append([x0[j],xb[j],yb[j]])
		ydata.append([1])
		xdata.append([x0[j],xr[j],yr[j]])
		ydata.append([-1])
	pocketweight=linearregression(xdata,ydata)
	return inputs 


def generatedatatest():
	global globalerror, pocketweight, numberoftestpoints
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
		xdtest.append([dtestx0[j],dtestx1[j],dtestx2[j]])
		ydtest.append([dtesty[j]])
	return inputsdtest 

def generatedatatrain():
	global globalerror, pocketweight, numberoftrainpoints
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
		xdtrain.append([dtrainx0[j],dtrainx1[j],dtrainx2[j]])
		ydtrain.append([dtrainy[j]])
	pocketweight=linearregression(xdtrain,ydtrain)
	return inputsdtrain

w = numpy.random.rand(3)*2-1
learningrate = 0.1	
print 'pocketweight values from training: \n', pocketweight

def response(x):
	global globalerror, pocketweight
	hperceptron = x[0]*pocketweight[0]+x[1]*pocketweight[1]+x[2]*pocketweight[2]
	if hperceptron >= 0:
		return 1
	else:
		return -1
		
	
def main():
	global numberdatapoints, numberoftestpoints, globalerror, errorlastchank
	xdat=[]
	ydat=[]
	error=0.0
	if userdefineddata:
		trainset=generatedata(100)
		testset=generatedata(1000)
	else:
		trainset = generatedatatrain()
		testset = generatedatatest()
	# errorcalc(testset)
	for x in trainset:
		xdat.append([x[0], x[1], x[2]])
		ydat.append([x[3]])
	err=validation_linreg(xdat, ydat,10)	
	print 'err', err
	for x in testset:
		r = response(x)
		if r != x[3]:
			error+=1	
		# if x[3] == 1:
			# plt.plot(x[1],x[2],'ob')
		# else:
			# plt.plot(x[1],x[2],'or')
	# error=error/len(testset)
	# print 'globalerror is:\n', error
	a=[]
	b=[]
	# err = validation_linreg()
	# print err
	for i in range(int(2)):
		a.append(i)
		eq = (-pocketweight[1]*a[i])/pocketweight[2]-pocketweight[0]/pocketweight[2]
		b.append(eq)
	titleprint='Linear Regression, validation on the train set of data, Error: ' + str(errorlastchank)
	plt.title(titleprint)
	plt.plot(a,b,'-')	
	plt.show()
		
main()