import random
from random import randint
import matplotlib.pyplot as plt
from numpy import mean

in_errors1 = []
out_errors1=[]
in_errors2 = []
out_errors2=[]
from pylab import plot, ylim
for w in range(20):
    rate = 0.1;
    points_train = 100;
    points_test = 1000
    treshold = 0.0

    def step_func(treshold, weights, x, y):
        total = x*weights[0]+y * weights[1] + weights[2]
        if total >= treshold:
            return 1
        else:
            return 0

    x_train = []
    y_train = []
    x_test = []
    y_test=[]
    outputValues_train = []
    outputValues_test = []

    for i in range(int(0.5*points_train)):
        x_train.append(randint(120,220))
        y_train.append(randint(120,220))
        outputValues_train.append(1)
    # for i in range(int(0.05*points_train)):
    #     x_train.append(randint(120,220))
    #     y_train.append(randint(120,220))
    #     outputValues_train.append(0)


    for i in range(int(0.5*points_train)):
        x_train.append(randint(0,100))
        y_train.append(randint(0,110))
        outputValues_train.append(0)
    # for i in range(int(0.05*points_train)):
    #     x_train.append(randint(0,100))
    #     y_train.append(randint(0,110))
    #     outputValues_train.append(1)

    for i in range(int(points_test/2)):
        x_test.append(randint(120,220))
        y_test.append(randint(120,220))
        outputValues_test.append(1)


    for i in range(int(points_test/2)):
        x_test.append(randint(0,100))
        y_test.append(randint(0,110))
        outputValues_test.append(0)

    weights = []
    weights.append(random.uniform(0, 1))
    weights.append(random.uniform(0, 1))
    weights.append(1)

    globalError = 0
    it = 0
    best_weight = []
    best_weight.append(random.uniform(0, 1))
    best_weight.append(random.uniform(0, 1))
    best_weight.append(1)


    curr_error = 1
    while it<1000:
        errors_in1 = []
        errors_in2 = []
        errors_out1 = []
        errors_out2 = []
        it += 1
        globalError = 0
        for z in range(points_train):
            if outputValues_train[z] != step_func(treshold, weights,x_train[z], y_train[z]):
                errors_in1.append(1)
            else:
                errors_in1.append(0)
            localError = outputValues_train[z]-step_func(treshold, weights, x_train[z], y_train[z])
            weights[0] += localError*x_train[z]
            weights[1] += localError*y_train[z]
            weights[2] += localError
            globalError += localError*localError
        if globalError == 0:
            break
        if (sum(errors_in1)/points_train) < curr_error:
            curr_error = sum(errors_in1)/points_train
            best_weight.insert(0,weights[0])
            best_weight.insert(1,weights[1])
            best_weight.insert(2,weights[2])

        for q in range(points_train):
            y = step_func(treshold, best_weight, x_train[q], y_train[q])
            if y != outputValues_train[q]:
                errors_in2.append(1)
            else:
                errors_in2.append(0)

    for q in range(points_test):
        x = step_func(treshold, weights, x_test[q], y_test[q])
        if  x!= outputValues_test[q]:
            errors_out1.append(1)
        else:
            errors_out1.append(0)
    for q in range(points_test):
         t =step_func(treshold, best_weight, x_test[q], y_test[q])
         if t != outputValues_test[q]:
            errors_out2.append(1)
         else:
            errors_out2.append(0)

            # out_error = outputValues_test[q]-step_func(treshold, best_weight, x_test[q], y_test[q])
            # errors2.append(out_error)
    in_errors1.append(sum(errors_in1)/points_train)
    out_errors1.append(sum(errors_out1)/points_test)
    in_errors2.append(sum(errors_in2)/points_train)
    out_errors2.append(sum(errors_out2)/points_test)
print(in_errors1)
print(out_errors1)
print(in_errors2)
print(out_errors2)
print(len(out_errors1))
print(len(in_errors1))

plt.axis([0, 220, 0, 220])
for i in range(19):
   plt.plot(i, mean(in_errors1[0:i+1]), 'bo')

for i in range(19):
   plt.plot(i, mean(in_errors2[0:i+1]), 'ro')
plt.axis([-1, 21, -1, 1])
plt.xlabel('Number of iterations, red plots represent the best wight, blues represent the last chose weight')
plt.ylabel('In-sample Error')
plt.show()
for i in range(19):
   plt.plot(i, mean(out_errors1[0:i+1]), 'bo')

for i in range(19):
   plt.plot(i, mean(out_errors2[0:i+1]), 'ro')
plt.axis([-1, 21, -1, 1])
plt.xlabel('Number of iterations, red plots represent the best wight, blues represent the last chose weight')
plt.ylabel(' Out-of-sample Error')
plt.show()

a=[]
b=[]
for i in range(int(500)):
    a.append(i)
    equation = (-weights[0]*a[i])/weights[1]-weights[2]/weights[1]
    b.append(equation)
plt.scatter(x_train[:int(points_train/2)], y_train[:int(points_train/2)], color = 'red')
plt.scatter(x_train[int(points_train/2):], y_train[int(points_train/2):], color = 'blue')
plt.axis([0, 220, 0, 220])
plt.xlabel('X, red plots represent 1 class, blues represent 0 class')
plt.ylabel('Y')
plt.show()
print("hh")
plt.scatter(x_train[:int(points_train/2)], y_train[:int(points_train/2)], color = 'red')
plt.scatter(x_train[int(points_train/2):], y_train[int(points_train/2):], color = 'blue')
plt.plot(a,b, '-')
# for i in range(int(500)):
#     plt.plot(i, (-weights[0]*i)/weights[1]-weights[2]/weights[1],'ro')
plt.axis([0, 220, 0, 220])
plt.xlabel('X, red plots represent 1 class, blues represent 0 class')
plt.ylabel('Y')
plt.show()
print("hh")

print("The Number of iteration is "+str(it))
print("The equation of the final line is: "+ str(weights[0]) + "*x + " + str(weights[1]) + "*y " + str(weights[2])+' = 0')

