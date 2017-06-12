#Getting started with your data visualization
import matplotlib.pyplot as plt
#plot the (x,y) points dots of size 10
#In this way we are going to work as MATLAB does
#the example is based in a file with 2 columns data
plt.scatter(x,y,s=10)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)],
['week %i' % w for w in range(10)])
plt.autoscale(tight=True)
# draw a slightly opaque, dashed grid
plt.grid(True, linestyle='-', color='0.75')
plt.show()
#MATLAB staff
#We can come in conclusition from the resulting chart , particularly in which algorithm to use
#Main case is that using this library is like working in MATLAB
