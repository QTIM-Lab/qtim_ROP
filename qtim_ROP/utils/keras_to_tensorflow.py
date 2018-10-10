"""
Copyright (c) 2017, by the Authors: Amir H. Abdi
This software is freely available under the MIT Public License.
Please see the License file in the root for details.

The following code snippet will convert the keras model file,
which is saved using model.save('kerasmodel_weight_file'),
to the freezed .pb tensorflow weight file which holds both the
network architecture and its associated weights.
"""
import argparse
import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json
# from keras.applications.mobilenet import relu6, DepthwiseConv2D
from .common import make_sub_dir


def keras_to_tensorflow(model_json, weights, save_dir, num_output=1, quantize=False):

    # Load keras model and rename output
    K.set_learning_phase(0)

    try:
        with open(model_json, 'r') as f:
            net_model = model_from_json(f.read()) #, custom_objects={'relu6': relu6, 'DepthwiseConv2d': DepthwiseConv2D})
            net_model.load_weights(weights)
    except ValueError as err:
        print("Error loading model")
        raise err

    # Define output tensor
    pred = [None] * num_output
    pred_node_names = [None] * num_output
    for i in range(num_output):
        pred_node_names[i] = 'pred' + str(i)
        pred[i] = tf.identity(net_model.outputs[i], name=pred_node_names[i])

    sess = K.get_session()
    out_dir = make_sub_dir(save_dir, 'tf_model')

    # convert variables to constants and save
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io

    if quantize:
        from tensorflow.tools.graph_transforms import TransformGraph
        transforms = ["quantize_weights", "quantize_nodes"]
        transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [], pred_node_names, transforms)
        constant_graph = graph_util.convert_variables_to_constants(sess, transformed_graph_def, pred_node_names)
    else:
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, out_dir, 'tf_model.pb', as_text=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='set input arguments')
    parser.add_argument('-input_fld', action="store",
                        dest='input_fld', type=str, default='.')
    parser.add_argument('-output_fld', action="store",
                        dest='output_fld', type=str, default='')
    parser.add_argument('-input_model_file', action="store",
                        dest='input_model_file', type=str, default='model.h5')
    parser.add_argument('-output_model_file', action="store",
                        dest='output_model_file', type=str, default='')
    parser.add_argument('-output_graphdef_file', action="store",
                        dest='output_graphdef_file', type=str, default='model.ascii')
    parser.add_argument('-num_outputs', action="store",
                        dest='num_outputs', type=int, default=1)
    parser.add_argument('-graph_def', action="store",
                        dest='graph_def', type=bool, default=False)
    parser.add_argument('-output_node_prefix', action="store",
                        dest='output_node_prefix', type=str, default='output_node')
    parser.add_argument('-quantize', action="store",
                        dest='quantize', type=bool, default=False)
    parser.add_argument('-theano_backend', action="store",
                        dest='theano_backend', type=bool, default=False)
    parser.add_argument('-f')
    args = parser.parse_args()
    parser.print_help()
    print('input args: ', args)

    if args.theano_backend is True and args.quantize is True:
        raise ValueError("Quantize feature does not work with theano backend.")

    keras_to_tensorflow(args)