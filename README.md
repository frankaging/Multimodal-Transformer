# Attending to Emotional Narratives
2019 8th International Conference on Affective Computing and Intelligent Interaction (ACII)

**cite** contains a model implementation for our [paper](https://arxiv.org/abs/1907.04197).  If you find this code useful in your research, please consider citing:

    @inproceedings{wu2019attending,
	  title={Attending to Emotional Narratives},
	  author={Wu, Zhengxuan and Zhang, Xiyu and Zhi-Xuan, Tan and Zaki, Jamil and Ong, Desmond C.},
	  journal={IEEE Affective Computing and Intelligent Interaction (ACII)},
	  year={2019}
	}

## What it is?
This repo consist 5 different deep neural networks predicting emotion valence with multi-modal inputs. This repo contains evaluation scripts and pre-trained models.

## Description
Attention mechanisms in deep neural networks have achieved excellent performance on sequence-prediction tasks. Here, we show that these recently-proposed attention-based mechanisms---in particular, the *Transformer* with its parallelizable self-attention layers, and the *Memory Fusion Network* with attention across modalities and time---also generalize well to multimodal time-series emotion recognition. Using a recently-introduced dataset of emotional autobiographical narratives, 
we adapt and apply these two attention mechanisms to predict emotional valence over time.
Our models perform extremely well, in some cases reaching a performance comparable with human raters. We end with a discussion of the implications of attention mechanisms to affective computing.

## Models
### Memory Fusion Transformer

### Simple Fusion Transformer

### Baseline 1 - LSTM Model For Multi-modal Inputs

### Baseline 2 - Transformer With A Linear Header For Multi-modal Inputs

### Baseline 3 - Memory Fusion Network


## How To Run?
### Requirements:
VM with image, Data science for linux, on Microsoft Azure

### Command:
python train.py (with all default settings)
