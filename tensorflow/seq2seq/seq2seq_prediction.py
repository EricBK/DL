import numpy as np
import tensorflow as tf
#from tensorflow.python.ops import seq2seq
#from tensorflow.python.ops import rnn_cell
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.contrib import rnn
seed = 7
np.random.seed(7)


def generate_sequences(sequence_num, sequence_length, batch_size):
    x_data = np.random.uniform(0, 1, size=(int(sequence_num / batch_size), sequence_length, batch_size, 26))
    x_data = np.array(x_data, dtype=np.float32)
    y_data = []
    for x in x_data:
        sequence = [x[0]]
        for index in range(1, len(x)):
            sequence.append(x[0] * x[index])
        # sequence.append([np.max(sequence, axis=0)])
        # candidates_for_min = sequence[1:]
        # sequence.append([np.min(candidates_for_min, axis=0)])
        y_data.append(sequence)
    y_data = np.random.uniform(0, 1, size=(int(sequence_num / batch_size), sequence_length, batch_size, 1))
    y_data = np.array(y_data, dtype=np.float32)

    return x_data, y_data


def convert_seq_of_seq(inputs):
    tensor_array = []
    for sequence in inputs:
        tensor_array.append([tf.constant(x) for x in sequence])

    return tensor_array


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


def main():
    datapoints_number = 70
    sequence_size = 50
    batch_size = 10
    data_point_dim = 26
    decoder_dim = 1
    use_lstm = False
    num_layers = 3
    if datapoints_number % float(batch_size) != 0:
        raise ValueError('Number of samples must be divisible with batch size')

    inputs, outputs = generate_sequences(sequence_num=datapoints_number, sequence_length=sequence_size,
                                         batch_size=batch_size)
    print(inputs.shape)
    print(outputs.shape)
    input_dim = len(inputs[0])
    print('input_dim:',input_dim)
    output_dim = len(outputs[0])
    print('output dim:',output_dim)
    encoder_inputs = [tf.placeholder(tf.float32, shape=[batch_size, data_point_dim]) for _ in range(input_dim)]
    decoder_inputs = [tf.placeholder(tf.float32, shape=[batch_size, decoder_dim]) for _ in range(output_dim)]

    # Create the internal multi-layer cell for our RNN.
    single_cell = rnn.GRUCell(decoder_dim)
    if use_lstm:
        single_cell = rnn.BasicLSTMCell(decoder_dim)
    cell = single_cell
    if num_layers > 1:
        cell = rnn.MultiRNNCell([single_cell] * num_layers)
    '''
    model_outputs, states = seq2seq.basic_rnn_seq2seq(encoder_inputs,
                                                      decoder_inputs,
                                                      rnn.BasicLSTMCell(1, state_is_tuple=True))
    '''
    model_outputs, states = seq2seq.basic_rnn_seq2seq(encoder_inputs,
                                                      decoder_inputs,
                                                      cell)

    reshaped_outputs = tf.reshape(model_outputs, [-1])
    reshaped_results = tf.reshape(decoder_inputs, [-1])

    cost = tf.reduce_sum(tf.squared_difference(reshaped_outputs, reshaped_results))
    variable_summaries(cost, 'cost')

    step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

    init = tf.initialize_all_variables()

    merged = tf.summary.merge_all()

    import matplotlib.pyplot as plt

    with tf.Session() as session:
        session.run(init)
        # writer = tf.train.SummaryWriter("/tmp/tensor/train", session.graph, )

        costs = []
        n_iterations = 1
        print('training...')
        for i in range(n_iterations):
            batch_costs = []
            summary = None

            for batch_inputs, batch_outputs in zip(inputs, outputs):
                x_list = {key: value for (key, value) in zip(encoder_inputs, batch_inputs)}
                y_list = {key: value for (key, value) in zip(decoder_inputs, batch_outputs)}

                #print(list(x_list.keys())[1],x_list[list(x_list.keys())[1]])
                print(list(y_list.keys())[1], y_list[list(y_list.keys())[1]])
                exit()
                input_list = x_list.copy()
                input_list.update(y_list)
                #print(dict(list(input_list.items())))
                #exit()
                summary, err, _ = session.run([merged, cost, step], feed_dict=dict(list(input_list.items())))
                # err, _ = session.run([ cost, step], feed_dict=dict(list(input_list.items())))

                batch_costs.append(err)
            # if summary is not None:
            #     writer.add_summary(summary, i)
            costs.append(np.average(batch_costs, axis=0))
            print(i)
        #plt.plot(costs)
        #plt.show()
        print('testing...')
        inputs, outputs = generate_sequences(sequence_num=datapoints_number, sequence_length=sequence_size,
                                             batch_size=batch_size,type_ = 'test')
        # 1个batch 的 输入
        for batch_inputs in inputs:
            x_list = {key: value for (key, value) in zip(encoder_inputs, batch_inputs)}
            y_list
            model_outputs_ = session.run([model_outputs], feed_dict=dict(list(x_list.items())))
        print(model_outputs_)
if __name__ == '__main__':
    main()
