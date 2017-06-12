#After the visualization of our data we have usefull conclusions
#Proccedure of finding the rifht model and learning algorithm
#We are going to name a function as the error between the models
import scipy as sp
import matplotlib as plt
import numpy as anp
def error(f,x,y):
    return sp.sum((f(x)-y)**2)
#this function is about to calculate the squared distance of the model's prediction to the real data


#With the polyfit() function of SciPy we make a straight line in the sample. This straight line making the results
#have the smallest approximation error.
fp1 , residuals , rank , sv , rcond = sp.polyfit(x,y,1,full=True)
#a straight line has order 1
#the polyfit function returns the parameters of the fitted model function,fp1.And by setting full  = Ttrue
#we also get additional backround information on the fitting process.Only residuals are of interest, which is the roor
#of the approximation

#using model parameters fp1 we initialize the line
print(fp1)#returning two numbers
#so the line is
f(x) = fp1(first variable) *x + fp1(second variable)

#then using poly1d() to create a model function from the model parameters
f1 = sp.poly1d(fp1)
#we call for error
print(error(f1,x,y))
#We are going to visualize this line in our sample with the following code
fx = sp.linspace(0,x[-1], 1000) # generate X-values for plotting
plt.plot(fx, f1(fx), linewidth=4)
plt.legend(["d=%i" % f1.order], loc="upper left")

#Now with the printed error and the visualization of our line in our sample we can compare the error if it is valid
#based on the plot
#Comparing two model based on their errors
#we continue with our first model until we find a better one
#Whatever model we come up with in the future, we will compare it againt the current straight line


#Continue with another model f2p
f2p = sp.polyfit(x,y,2)
f2 = sp.polyfit(f2p)
print(error(f2,x,y))
#because our line has order two our function is the following
f(x) = f2p(first variable)*x**2 - f2p(second variable)*x + f2p(thrid variable)
#now we have more complexity and we check if it can provide us smaller error


#in this point we have to come to a decision between
#Choosing one of the fitted polynomial models.
#Switching to another more complex model class.
#Thinking differently about the data and start again.

#These three steps constitutes three procedures that are independent one another
