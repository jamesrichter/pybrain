from fann2 import libfann
import datetime
print datetime.datetime.now()
import numpy as np

array = np.loadtxt('abalone.txt', dtype = "U", delimiter = ',')
array[array == "M"] = "-1"
array[array == "F"] = "1"
array[array == "I"] = "0"
house_data = array[:,0:-1]
house_target = array[:,-1]

f = open('abaloneFann.txt', 'w+')
f.write("4177 8 40\n")
for index, line in enumerate(house_data):
    for element in line:
        f.write(str(element) + ' ')
    f.write('\n')
    target = int(house_target[index])
    target_array = [0] * target + [1] + [0] * (40-target-1)
    target_string = ""
    for x in target_array:
        target_string += str(x) + ' ' 
    if index > 4166:
        print ":)"
    f.write(target_string)
    f.write('\n')
f.close()
connection_rate = 1
learning_rate = 0.7
num_input = 8
num_hidden = 8
num_output = 40
num_layers = 5

desired_error = 0.0001
max_iterations = 1000#00
iterations_between_reports = 1000

ann = libfann.neural_net()
[num_hidden] * num_layers
ann.create_standard_array((num_input, num_hidden * num_layers, num_output))
ann.set_learning_rate(learning_rate)
ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)
ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC_STEPWISE)

ann.train_on_file("abaloneFann.txt", max_iterations, iterations_between_reports, desired_error)

ann.save("abaloneFann.net")

print datetime.datetime.now()