import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, Activation, GlobalMaxPooling1D, Dense, Dropout
import matplotlib.pyplot as plt
nnuc = 4
motif_len = 5
number_of_iteration = 5
number_of_epoch = 60

sequence_train = []
sequence_test = []
y_train = []
y_test = []
with open('/Users/futo/Desktop/motif_data_train.txt', "r") as f:
    f.readline()
    while True:
        line = f.readline()
        if not line:
            break
        sequence_train.append(line[2:])
        y_train.append(line[0])

with open('/Users/futo/Desktop/motif_data_test.txt', "r") as f:
    f.readline()
    while True:
        line = f.readline()
        if not line:
            break
        sequence_test.append(line[2:])
        y_test.append(line[0])

seq_len = len(sequence_test[0])
x_train = np.zeros((len(sequence_train), seq_len, nnuc), dtype=int)
x_test = np.zeros((len(sequence_test), seq_len, nnuc), dtype=int)

def nuc_to_code(nuc):
    return {'A': [1, 0, 0, 0],
            'C': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'T': [0, 0, 0, 1],
    }.get(nuc, [0, 0, 0, 0])

for i in range(len(sequence_train)):
    for j in range(seq_len):
        x_train[i,j,:] = np.array(nuc_to_code(sequence_train[i][j]), dtype=int)

for i in range(len(sequence_test)):
    for j in range(seq_len):
        x_test[i,j,:] = np.array(nuc_to_code(sequence_test[i][j]), dtype=int)

y_train = np.array(y_train, dtype=int)
y_test = np.array(y_test, dtype=int)


#ここから
loss_list = np.zeros(number_of_epoch, dtype=float)
accuracy_list = np.zeros(number_of_epoch, dtype=float)
for iteration in range(number_of_iteration):
    for epoch in range(number_of_epoch):
        model = Sequential()
        model.add(Conv1D(1, motif_len, input_shape=(seq_len, nnuc)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(GlobalMaxPooling1D())
        model.add(Activation('sigmoid'))
        #ここまでを変更することでscoreの変化を見る

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        model.fit(x_train, y_train,
                  epochs=epoch,
                  batch_size=128)

        score = model.evaluate(x_test, y_test, batch_size=128)
        loss_list[epoch] += score[0]
        accuracy_list[epoch] += score[1]

loss_list = loss_list / number_of_iteration
accuracy_list = accuracy_list / number_of_iteration



number = np.arange(number_of_epoch, dtype=int) + 1
xaxis = np.arange(1,number_of_epoch, 5)


# グラフを書く
plt.figure()
plt.plot(number, loss_list, marker="x")
plt.title("epoch/loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.xticks(xaxis)
plt.grid(True)

plt.figure()
plt.plot(number, accuracy_list, marker="x")
plt.title("epoch/accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.xticks(xaxis)
plt.grid(True)
plt.show()