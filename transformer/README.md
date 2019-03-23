Code for synchronous time-series models, which make predictions for every time-step at a constant rate.

Models are stored in `models.py`, training code in `train.py`, data loading code in `datasets.py`. Run `python train.py -h` for a list of training options.

## Models

### LSTM

Models the MLE of *p(y<sub>t</sub>|x<sub>1:t</sub>)* with a multimodal LSTM.

The LSTM computes hidden states from the input history, i.e. *h<sub>t</sub> = f(x<sub>t</sub>, h<sub>t-1</sub>)*, local attention over hidden states is optionally applied, then a multi-layer perceptron (MLP) decodes the hidden states to predictions, i.e. *y<sub>t</sub> = g(h<sub>t</sub>)* where *g* is an MLP.

### ED-LSTM

Models the MLE of *p(y<sub>t</sub>|x<sub>1:t</sub>, y<sub>1:t-1</sub>)* with an LSTM encoder-decoder architecture.

Encoding and attention is the same as above, but the decoder also uses an LSTM which takes both the previous prediction *y<sub>t-1</sub>* and *h<sub>t</sub>* as inputs, i.e., *y<sub>t</sub> = g(y<sub>t-1</sub>, h<sub>t</sub>)*.

### AR-LSTM

Models the MLE of *p(y<sub>t</sub>|x<sub>1:t</sub>, y<sub>t-1</sub>)* with an LSTM and auto-regressive decoding layer.

The autoregressive weights are learned as function of the LSTM hidden: *y<sub>t</sub> = &phi;(h<sub>t</sub>) y<sub>t-1</sub> + &psi;(x<sub>t</sub>)*, where &phi; and &psi; are parameterized by MLPs, and h<sub>t</sub> is computed from the LSTM as in the first model.