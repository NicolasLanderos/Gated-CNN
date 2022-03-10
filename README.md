# Gated-CNN
This project is the simulation framework of Gated-CNN. Gated-CNN is a microarchitectural technique to overcome aging effects in on-chip memories of CNN accelerators. The technique consists of a cyclically bank assignment scheme and power-gating mechanism. 

## Requirements
The framework uses the following packets:  
- python == 3.8  
- tensorflow == 2.5  
- tensorflow-datasets == 4.5.2  
- cudatoolkit == 11.0.221  
- cudnn == 8.2.1.32  
- numpy == 1.19.5  
- numba == 0.54.1  
- pandas == 1.4.1  

Example of installation under a conda environment:  
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

network_train.py: generates trained weights for the selected network and a train file used in later stages.
 - Parameters:
   - --batch: batch size to use during training
   - --dataset: dataset from https://www.tensorflow.org/datasets/catalog/overview
   - --distribution: percentage distribution of training/validation/testing
   - --input_shape: input dimensions (height, width, channels)
   - --output_shape: output dimensions
   - --network_name: network architecture to train (one of the eight architecures used in the analysis)
 
network_quantization.py: prints network accuracy and loss for different combinations of integer/fractional parts (in bits) in both activations and weights, 
varying one and leaving the other part constant.
  - Parameters:
   - --batch: batch size to use during inference
   - --dataset: dataset from https://www.tensorflow.org/datasets/catalog/overview
   - --network_name: network architecture to train (one of the eight architecures used in the analysis)
   - --base_bits: number of bits to be used as base for each part

network_access.py: elaboration of read/write stats on the I/O buffer, under the following assumptions: activation size = address size = 16 bits. Generates two panda dataframes with the results (activating or deactivating the Gated-CNN approach).
  - Parameters:
   - --dataset: dataset from https://www.tensorflow.org/datasets/catalog/overview
   - --network_name: network architecture to train (one of the eight architecures used in the analysis)
   - --addressing_space: number of address in the buffer
   - --samples: number of infered images

network_buffer_sim.py: simulation of buffer stats (number of high cycles, low cycles, off cycles, and flips). Generates periodically a dictionary with the stats.
  - Parameters:
   - --dataset: dataset from https://www.tensorflow.org/datasets/catalog/overview
   - --network_name: network architecture to train (one of the eight architecures used in the analysis)
   - --addressing_space: number of addresses in the buffer
   - --samples: number of infered images
   - --afb: activation fractional part (number of bits)
   - --aib: activation integer part (number of bits)
   - --wfb: weight fractional part (number of bits)
   - --wib: weight integer part (number of bits)
   - --gated_CNN: boolean to apply or not Gated-CNN
    
network_aging.py: simulation of inference under memory faults. Generates a file with the results.
  - Parameters:
   - --dataset: dataset from https://www.tensorflow.org/datasets/catalog/overview
   - --network_name: network architecture to train (one of the eight architecures used in the analysis)
   - --addressing_space: number of address in the buffer
   - --samples: number of times the same portion of faults is tested (with a different fault distribution)
   - --afb: activation fractional part (number of bits)
   - --aib: activation integer part (number of bits)
   - --wfb: weight fractional part (number of bits)
   - --wib: weight integer part (number of bits)
   - --wgt_faults: true for faults in weight buffer, false for faults in activation buffer
   - --batch: batch size to use during inference
   - --portion: portion of the buffer under faults
