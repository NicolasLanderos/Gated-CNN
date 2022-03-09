# Gated-CNN
Framework of Gated-CNN paper.

## Requirements:
The framework with the following packets:  
- python == 3.8  
- tensorflow == 2.5  
- tensorflow-datasets == 4.5.2  
- cudatoolkit == 11.0.221  
- cudnn == 8.2.1.32  
- numpy == 1.19.5  
- numba == 0.54.1  
- pandas == 1.4.1  

An example of installation under a conda environment is:  
1. conda create --name env_name python=3.8  
2. conda activate env_name  
3. conda install -c anaconda cudatoolkit  
4. conda install -c conda-forge cudnn  
5. pip install tensorflow==2.5  
6. pip install tensorflow-datasets  
7. conda install numba  
8. conda update numpy  
9. conda install pandas  
  
## Usage:

network_train.py: Generates trained weights for the selected network and a train file used in later stages.
 - parameters:
   - --batch: batch size to use during training
   - --dataset: dataset from https://www.tensorflow.org/datasets/catalog/overview
   - --distribution: percentage distribution of training/validation/testing
   - --input_shape: input dimensions (height,width,channels)
   - --output_shape: output dimensions
   - --network_name: network architecture to train (one of the eight architecures used in the analisis)
 
network_quantization.py: print network accuracy and loss for different combinations of integer/fractional part bits in activations and weights, 
varying one and leaving the others constant.
  - parameters:
   - --batch: batch size to use during inference
   - --dataset: dataset from https://www.tensorflow.org/datasets/catalog/overview
   - --network_name: network architecture to train (one of the eight architecures used in the analisis)
   - --base_bits: number of bits to be used as base for each part.

network_access.py: Elaboration of read/write stats on the buffer, under the following assumptions: activation size = address size = 16 bits. Generates two panda dataframes
the results with and without gated_cnn active.
  - parameters:
   - --dataset: dataset from https://www.tensorflow.org/datasets/catalog/overview
   - --network_name: network architecture to train (one of the eight architecures used in the analisis)
   - --addressing_space: number of address in the buffer
   - --samples: number of infered images

network_buffer_sim.py: simulation of buffer stats (number of high cycles, low cycles, off cycles and flips). Generates periodically a dictionary with the stats.
  - parameters:
   - --dataset: dataset from https://www.tensorflow.org/datasets/catalog/overview
   - --network_name: network architecture to train (one of the eight architecures used in the analisis)
   - --addressing_space: number of address in the buffer
   - --samples: number of infered images
   - --afb: activation fractional part number of bits
   - --aib: activation integer part number of bits
   - --wfb: weight fractional part number of bits
   - --wib: weight integer part number of bits
   - --gated_CNN: boolean to apply or not gated_CNN
    
network_aging.py: simulation of inference under memory faults, Generates a file with the results.
  - parameterss:
   - --dataset: dataset from https://www.tensorflow.org/datasets/catalog/overview
   - --network_name: network architecture to train (one of the eight architecures used in the analisis)
   - --addressing_space: number of address in the buffer
   - --samples: number of times the same portion of faults is tested (with different fault distribution)
   - --afb: activation fractional part number of bits
   - --aib: activation integer part number of bits
   - --wfb: weight fractional part number of bits
   - --wib: weight integer part number of bits
   - --wgt_faults: True for faults in weight buffer, False for faults in activation buffer
   - --batch: batch size to use during inference
   - --portion: portion of the buffer under faults
