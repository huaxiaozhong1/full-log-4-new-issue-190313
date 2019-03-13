<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png"><br><br>
</div>

-----------------


| **`Documentation`** |
|-----------------|
| [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/api_docs/) |


------------------------

### System information
- **What is the top-level directory of the model you are using**:   https://github.com/tensorflow/models/tree/master/research/slim .
- **Have I written custom code (as opposed to using a stock example script provided in TensorFlow)**:   No, I just used commands provided in TensorFlow.
- **OS Platform and Distribution (e.g., Linux Ubuntu 16.04)**:   Linux Ubuntu 18.04.
- **TensorFlow installed from (source or binary)**:   I used a docker image provided by TensorFlow, which is “tensorflow/tensorflow:1.12.0-devel”. When call
```
#bazel build tensorflow/python/tools:freeze_graph
```
it fell into a long-time building processing. 

- **TensorFlow version (use command below)**:   
```
python -c "import tensorflow as tf; print(tf.GIT_VERSION, tf.VERSION)"
('unknown', '1.12.0')
```
- **Bazel version (if compiling from source)**: Build label: 0.15.0
- **CUDA/cuDNN version**:   Only run with CPU.
- **GPU model and memory**:   Only run with CPU.
- **Exact command to reproduce**:    
```
#cd $HOME/workspace
#git clone https://github.com/tensorflow/models/
#cd $HOME/workspace/models/research/slim

#python download_and_convert_data.py --dataset_name=cifar10 --dataset_dir=/tmp/data/cifar10

#python train_image_classifier.py --train_dir=/tmp/log-incep-v3 --dataset_dir=/tmp/data/cifar10 --model_name=inception_v3 --clone_on_cpu=true --dataset_name=cifar10 --dataset_split_name=train –max_number_of_steps=100

#python export_inference_graph.py   --alsologtostderr   --model_name=inception_v3   --output_file=/tmp/log-incep-v3/incep_v3.pb

#cd /tensorflow

#bazel build tensorflow/python/tools:freeze_graph

# bazel-bin/tensorflow/python/tools/freeze_graph  --input_graph=/tmp/log-incep-v3/incep_v3.pb --input_checkpoint=/tmp/log-incep-v3/model.ckpt-100 --input_binary=true --output_graph=/tmp/log-incep-v3/incep_v3_frozen.pb –output_node_names=InceptionV3/Predictions/Reshape_1
```
It exited running with reporting an error. The key part inside trackback is as below. And here is the full trackback. Why I put it over there is because I couldn't find where to upload it with raising the **New Issue**.  
>tensorflow.python.framework.errors_impl.**InvalidArgumentError**: Restoring from checkpoint failed. This is most likely due to a **mismatch** between the current graph and the graph from the checkpoint. Please ensure that you have not altered the graph expected based on the checkpoint. Original error:
Assign requires shapes of both tensors to match. **lhs shape= [1001] rhs shape= [10]**
	 [[node save/Assign_8 (defined at /tensorflow/bazel-bin/tensorflow/python/tools/freeze_graph.runfiles/org_tensorflow/tensorflow/python/tools/freeze_graph.py:491)  = Assign[T=DT_FLOAT, _class=["loc:@InceptionV3/AuxLogits/Conv2d_2b_1x1/biases"], use_locking=true, validate_shape=true, _device="/job:localhost/replica:0/task:0/device:CPU:0"](InceptionV3/AuxLogits/Conv2d_2b_1x1/biases, save/RestoreV2:8)]]

