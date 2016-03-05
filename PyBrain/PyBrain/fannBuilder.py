from fann2 import libfann
import datetime
print datetime.datetime.now()

#include "fann.h"

#int main()
#{
#    const unsigned int num_input = 2;
#    const unsigned int num_output = 1;
#    const unsigned int num_layers = 3;
#    const unsigned int num_neurons_hidden = 3;
#    const float desired_error = (const float) 0.001;
#    const unsigned int max_epochs = 500000;
#    const unsigned int epochs_between_reports = 1000;

#    struct fann *ann = fann_create_standard(num_layers, num_input,
#        num_neurons_hidden, num_output);

#    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
#    fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

#    fann_train_on_file(ann, "xor.data", max_epochs,
#        epochs_between_reports, desired_error);

#    fann_save(ann, "xor_float.net");

#    fann_destroy(ann);

#    return 0;
#}


connection_rate = 1
learning_rate = 0.7
num_input = 2
num_hidden = 4
num_output = 1
num_layers = 5

desired_error = 0.0001
max_iterations = 100000
iterations_between_reports = 1000

ann = libfann.neural_net()
[num_hidden] * num_layers
ann.create_standard_array((num_input, num_hidden * num_layers, num_output))
#ann.create_standard((num_layers, num_input, num_hidden, num_output))
#ann.create_sparse_array(connection_rate, (num_input, num_hidden, num_output))
ann.set_learning_rate(learning_rate)
ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)
ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC_STEPWISE)

ann.train_on_file("xor.data", max_iterations, iterations_between_reports, desired_error)

ann.save("xor.net")

print datetime.datetime.now()