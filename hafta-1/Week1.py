import math

def neuron(inputs, weights, bias, activation="sigmoid"):
    z = 0
    for i in range(len(inputs)):
        z = z + inputs[i] * weights[i]
    z = z + bias
    if activation == "sigmoid":
        return 1 / (1 + math.exp(-z))
    elif activation == "tanh":
        return math.tanh(z)
    else:
        return z

inputs = [0.7, 0.3, 0.9]
weights = [0.5, -0.2, 0.1]
bias = 0.05
tahmin = neuron(inputs, weights, bias, activation="sigmoid")
print("Tek nöron tahmini:", tahmin)

def layer(inputs, weights_list, biases, activation="tanh"):
    outputs = []
    for i in range(len(weights_list)):
        out = neuron(inputs, weights_list[i], biases[i], activation)
        outputs.append(out)
    return outputs

weights_list = [[0.5, -0.2, 0.1], [0.1, 0.4, -0.3], [-0.6, 0.2, 0.5]]
biases = [0.05, -0.1, 0.2]

layer_output = layer(inputs, weights_list, biases, activation="tanh")
print("Layer çıktısı:", layer_output)

def squared_error_loss(tahmin, gercek):
    return (tahmin - gercek) ** 2

tahmin = neuron(inputs, weights, bias, activation="sigmoid")
gercek = 1
loss = squared_error_loss(tahmin, gercek)
print("Tahmin:", tahmin)
print("Loss:", loss)

import matplotlib.pyplot as plt

w1_degerleri = []
loss_degerleri = []

w1 = -1.0
while w1 <= 1.0:
    test_weights = [w1, weights[1], weights[2]]
    tahmin = neuron(inputs, test_weights, bias, activation="sigmoid")
    loss = squared_error_loss(tahmin, gercek)

    w1_degerleri.append(w1)
    loss_degerleri.append(loss)

    w1 = w1 + 0.02

plt.plot(w1_degerleri, loss_degerleri)
plt.xlabel("w1 değeri")
plt.ylabel("loss")
plt.title("w1 değişirken loss nasıl değişiyor?")
plt.grid(True)
plt.show()

def sayisal_turev(index, weights, inputs, bias, gercek, h=0.0001):
    weights_arttirilmis = weights[:]
    weights_arttirilmis[index] = weights_arttirilmis[index] + h

    tahmin_arttirilmis = neuron(inputs, weights_arttirilmis, bias, activation="sigmoid")
    loss_arttirilmis = squared_error_loss(tahmin_arttirilmis, gercek)

    tahmin_orijinal = neuron(inputs, weights, bias, activation="sigmoid")
    loss_orijinal = squared_error_loss(tahmin_orijinal, gercek)

    turev = (loss_arttirilmis - loss_orijinal) / h
    return turev

weights = [0.5, -0.2, 0.1]
bias = 0.05
learning_rate = 0.5
gercek = 1

for adim in range(100):
    tahmin = neuron(inputs, weights, bias, activation="sigmoid")
    loss = squared_error_loss(tahmin, gercek)

    for i in range(len(weights)):
        grad = sayisal_turev(i, weights, inputs, bias, gercek)
        weights[i] = weights[i] - learning_rate * grad

    if adim % 10 == 0:
        print(f"Adım {adim}: loss = {loss:.5f}, weights = {[round(w,4) for w in weights]}")

          gecici_liste = []
  for w in weights:
      gecici_liste.append(round(w, 4))