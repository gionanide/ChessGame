#Getting started with NumPy
import numpy as np
a = np.array([0,1,2,3,4,5])
print(a)
print(a.ndim)#dimensions of the array (in our case 1)
print(a.shape)#the elements (in our case 6 elements)
#handling the Data
#trnasform your array
b = a.reshape((3,2))#i make an array with 3 rows and 2 columns
#if we make an exchange in b array it will occur in a array too
b[1][0] = 77
#the third element in a array is going to take this value too
#if we want to make an original independent array
#we are going to do this like this
c = a.reshape((3,2)).copy


#Handle alla the data of an array in NumPy
#first make an array like this

#checking every element in the a array if it is bigger than 4 and return an array with boolean values only
print(a>4)
#another case
#return an array with the elements that are bigger than 4
print(a[a>4])


#Hndling nonexisting values
#EXAMPLE
c = np.array([1,2,np.NAN,3,4])# let's pretend we have read this and the third element is missing
print(np.isnan(c))#return an array with boolean values and true in the posisition of the missing element
#now we have the delete this position, delete the invalid values
c[~np.isnan(c)]#deleting the isNAN positions
#also i can the average of the array elements lie this
print(np.mean(c[~np.isnan(c)]))


#In this point in a larger scale we have clean our data
#And now we are going to compare the runtime behavior between NumPy and Python lists
#In the following code, we will calculate the sum of all squared numbers from 1 to
#1000 and see how much time it will take. We perform it 10,000 times and report the
#total time so that our measurement is accurate enough.
import timeit
normalPy = timeit.timeit('sum(x*x for x in range(1000))',number=10000)
naiveNumPy = timeit.timeit('sum(na*na)',setup='import numpy as np; na=np.arange(1000)',number=10000)
goodNumPy = timeit.timeit('na.dot(na)',setup='import numpy as np; na=np.arange(1000)',number=10000)
print("Normal Python: %f sec" % normalPy)
print("Naive NumPy: %f sec" % naiveNumPy)
print("Good NumPy: %f sec" % goodNumPy)
#OBSERVATIONS
#We have come to a conclusiont that naive NumPy is not faster than we expected because it is written as a C
#extension.One reason for this is that the access
#of individual elements from Python itself is rather costly. Only when we are able
#to apply algorithms inside the optimized extension code is when we get speed
#improvements.

#Other observation is tha if we se the dot() function of NumP ,which does exactl the sme, allows us
#to be ore than 25 times faster. But the speed comes at a price. We lose in flexibiity which means that
#we are able to use only an arra with elements of the same tpe for example integers.


#Getting started with SciPy
import numpy as np
import scipy as sp
data = sp.genfromtxt('yourfile.csv',delimeter='\t')
#reading a file and using tha data
#chechikng the data
print(data[:10])
#Preprocessing and Cleaning the data
x = data[:0]#we are going to split the columns, we are take the first one (in other cases we take alla tha variables and y take the target var)
y = data[:,1]#the last viariable or target
#Cleaning the data
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]#cleaning the data based on the target state, keep only the states where target state is valid
#And now we are ready  with our data and we can use them