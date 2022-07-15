import numpy as np
#библиотека содержаит сигмойду expit() - выезд
import scipy.special as plt

# загрузка тренировочного набора данных
traning_data_file = open("mnist_dataset/mnist_train.csv", 'r')
traning_data_list = traning_data_file.readlines()
traning_data_file.close()

#загрузка тестового набора данных
test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

all_values = test_data_list[1].split(',')
# print(all_values[0])

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
    def train( self, inputs_list, targets_list): # входящий список , целнвой список
        # преобразование список входящих значений в двухменрый массив
        inputs = np.array(inputs_list, ndmin=2).T # ndmin:  Specifies the minimum number of dimensions that result array should have. Каоличество измерений
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs) # расчитать входящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs) # рассчитать исходящие сигналы для скрутого слоя

        final_inputs = np.dot(self.who, hidden_outputs)  # расситать входящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)  # расчитать исходящие сигналы для выходящего слоя
        # ошибки выходного слоя = ( целевое значение - фактическое значение)
        output_errors = targets - final_outputs
        #ошибки скрытого слоя - это ошибки output_errors
        # распределенные согласно весовым коофицентам связей и рекомбенированны на скрытых улах
        hidden_errors = np.dot(self.who.T, output_errors)
        # обновить весовые коофиценты для связей между скрытым и выходным слоями
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        # обновить веовые коофиценты для связей между входным и скрытым слоями
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * ( 1.0 - hidden_outputs)), np.transpose(inputs))
        pass

    #опрос нейроной сети
    def query(self,inputs_list):  #функция скалярного произведения ( мартица весов и входные сигналы)
        #приобразование списка входных значений в двухмерный массив
        inputs = np.array(inputs_list, ndmin = 2 ).T
        hidden_inputs = np.dot(self.wih, inputs) # расчитать входящие сигналы для скрsтого слоя# получение значений с разделением ,
        hidden_outpus = self.activation_function(hidden_inputs) # расчитать исходящие сигналы для скрытого слоя
        final_inputs = np.dot(self.who, hidden_outpus)  # расчитать входящие сигналы выходного слоя
        final_outputs = self.activation_function(final_inputs) # расчитать исходящие сигналы выхрдного слоя

        return final_outputs

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learningrate =   0.1
print("learningrate" , learningrate)

n = NeuralNetwork( input_nodes, hidden_nodes, output_nodes, learningrate)

epochs = 5
print("Epochs = ", epochs)
for e in range(epochs):
    for record in traning_data_list: # перебор всех запесей в тренировочном наборе
        all_values = record.split(',') # получение значений с разделением ,
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 # создание целевого входных значений желаемое значение = 0.99
        targets = np.zeros(output_nodes) + 0.01 # all_values[0] - целевое маркерное значение для данной записи
        targets[int(all_values[0])] = 0.99
        n.train(inputs,targets)
        pass
    pass

#тестирование нейроной сети
scorecard = [] # журнал оценки работы сети пустой
for record in test_data_list:#прербрать все записи в тестовом наборе
    all_values = record.split(',') # получит список значений , как разделитель
    correct_label = int(all_values[0])# правильный ответ перове значение
    # print(correct_label, "истинный маркер ")
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01# маштабировать и сместить входные значения
    outputs = n.query(inputs) # опрос сместить
    label = np.argmax(outputs)#индекс наибольшего значения = маркерное значение
    # print(label, "ответ сети")
    if (label == correct_label):# присоеденить оценку ответа к концу списка
        scorecard.append(1)# если ответ правильный присоедяется знаяение 1
    else:
        scorecard.append(0)#в случае неправльного зеачения 0
        pass

    pass
# расчитать показатель эффективности в виде доли правильных ответов
scorecard_array = np.asarray(scorecard)
print("Эффективность = ", scorecard_array.sum() / scorecard_array.size )
