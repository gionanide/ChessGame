from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

# load King-Rook vs King  dataset
dataset = numpy.loadtxt("K-R-R.data.txt", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:6]
Y = dataset[:,6]


# create model
model = Sequential()
model.add(Dense(16, input_dim=6, activation='relu'))
#the first layer has 16(because of the chess board) neurons and expects 6 input variables
model.add(Dense(6, activation='relu'))
#the second hidden layer has 12 neuron to predict the class (Outcome)
#because outcome = (draw,zero,one,two,three,four,five,six,seven,eight,nine,ten)
model.add(Dense(12, activation='sigmoid'))



# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
#we are going to run it for 150 iteration just as an example
#and use a relatively small batch size of 10
model.fit(X, Y, epochs=150, batch_size=10)



# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#now you are able to make predictions
# calculate predictions
predictions = model.predict(X)
# round predictions (not necessary)
rounded = [round(x[0]) for x in predictions]
print(rounded)
#in prediction  we are using a sigmoid activation function on the output layer
#'target layer' so the predictions will be in the reange
# draw or between zero and ten.
