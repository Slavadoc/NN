import numpy as np
#библиотека содержаит сигмойду expit() - выезд
import scipy.special as plt

data_file = open("mnist_dataset/mnist_train_100.csv", 'r') # "открытие файла для чтения " r - только чтение без изменений
data_list = data_file.readlines() # readlines () считывает весь файл целиком доступ к записям по типу data_list[0] data_list[5]  ...
data_file.close() # закртытие файла ... что бы не жрал ресурсы

scaled_input = (np.afarray(all_values[1:]) / 255.0 * 0.99 ) + 0.01
print(scaled_input)

print(len(data_list))
print(data_list[5])
