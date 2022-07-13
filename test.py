import numpy as np
#библиотека содержаит сигмойду expit() - выезд
import scipy.special as plt

data_file = open("mnist_dataset/mnist_train_100.csv", 'r') # "открытие файла для чтения " r - только чтение без изменений
data_list = data_file.readlines() # readlines () считывает весь файл целиком доступ к записям по типу data_list[0] data_list[5]  ...
data_file.close() # закртытие файла ... что бы не жрал ресурсы

#scaled_input = (np.afarray(all_values[1:]) / 255.0 * 0.99 ) + 0.01
#print(scaled_input)

print(len(data_list))
print(data_list[5])

class NeuralNetwork:
    #инициализация нейроной сети
    def __init__( self, inputnodes, hiddennodes, outputnodes, learningrate):
        # задаем количсетво узлов в входном скрытом и выходном слое
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # Матрицы весовых коофицентов связей  wih (между входным и и скрытым слоями )  и who (между скрытым и выходным слоями)
        # Весовые коофиценты связей узлов i и узлов j  следующего слоя обозначены как w_i_j  w11 w21  w12 w22 и т.д.
        # self.wih = (np.random.rand(self.hnodes, self.innodes) - 0.5) elf.who = (np.random.rand(self.outnodes, self.hnodes) - 0.5)
        self.wih = np.random.normal( 0.0, pow( self.hnodes, - 0.5), (self.hnodes, self.inodes))
        #1) 0.0  центр нормального распределения 2) стандартное отклонение -  по следующей функции  - 0.5 3) конфигурация массива numpy
        self.who = np.random.normal( 0.0, pow( self.onodes, - 0.5), (self.onodes, self.hnodes))
        self.lr = learningrate #коофицент обучения
        self.activation_function = lambda x: plt.expit(x)  #использование сигнмойды в качестве функции активации   +   лямбда функцция
        pass

    #тренировка нейроной сети
    def train( self, inputs_list, targetslist): # входящий список , целнвой список
        # преобразование список входящих значений в двухменрый массив
        inputs = np.array(inputs_list, ndmin=2).T # ndmin:  Specifies the minimum number of dimensions that result array should have. Каоличество измерений
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.whi, inputs) # расчитать входящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs) # рассчитать исходящие сигналы для скрутого слоя

        final_inputs = np.dot(self.who, hidden_outputs)  # расситать входящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)  # расчитать исходящие сигналы для выходящего слоя
        # ошибки выходного слоя = ( целевое значение - фактическое значение)
        output_errors = targets - final_outputs
        #ошибки скрытого слоя - это ошибки output_errors
        # распределенные согласно весовым коофицентам связей и рекомбенированны на скрытых улах
        hidden_errors = np.dot(self.who.T, output_errors)
        # обновить весовые коофиценты для связей между скрытым и выходным слоями
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outpus))
        # обновить веовые коофиценты для связей между входным и скрытым слоями
        self.whi += self.lr * np.dot((hidden_errors * hidden_outputs * ( 1.0 - hidden_outputs)), np.transpose(inputs))
        pass

    #опрос нейроной сети
    def query(self):  #функция скалярного произведения ( мартица весов и входные сигналы)
        #приобразование списка входных значений в двухмерный массив
        inputs = np.array(inputs_list, ndmin = 2 ).T

        hidden_inputs = np.dot(self.whi, inputs) # расчитать входящие сигналы для скрsтого слоя
        hidden_outpus = self.activation_function(hidden_inputs) # расчитать исходящие сигналы для скрытого слоя
        final_inputs = np.dot(self.who, hidden_outpus)  # расчитать входящие сигналы выходного слоя
        final_outputs = self.activation_function(final_inputs) # расчитать исходящие сигналы выхрдного слоя

        return final_outputs

input_nodes = 3
hidden_nodes = 3
output_nodes = 3
learningrate = 0.3

n = NeuralNetwork( input_nodes, hidden_nodes, output_nodes, learningrate)
