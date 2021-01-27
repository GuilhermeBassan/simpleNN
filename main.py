from tflearn import DNN
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression

## X = Input data, Y = Output data
X = [[0,0], [0,1], [1,0], [1,1]]
Y = [[0], [1], [1], [0]]

## One input layer, one hidden layer and one output layer
input_layer = input_data(shape=[None, 2])
hidden_layer = fully_connected(input_layer , 2, activation='tanh') 
output_layer = fully_connected(hidden_layer, 1, activation='tanh')

regression = regression(output_layer , optimizer='sgd', loss='binary_crossentropy', learning_rate=5)
model = DNN(regression)

model.fit(X, Y, n_epoch=5000, show_metric=True)

[i[0] > 0 for i in model.predict(X)]

print(model.get_weights(hidden_layer.W), model.get_weights(hidden_layer.b))
print(model.get_weights(output_layer.W), model.get_weights(output_layer.b))