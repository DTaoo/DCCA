# DCCA: Deep Canonical Correlation Analysis

This repository provides a DCCA model, implemented in Keras with tensorflow backend. This repository is developed 
based on the Keras-theano . For more details, please visit the [theano version](https://github.com/VahidooX/DeepCCA).

## Tips:

After talking with Vahid, we both find that the DCCA loss function based on the automatic gradient is not very stable:

* Using ReLu instead of sigmoid often cause the gradient exploding. 

* The SGD with momentum works much worse than adam and rmsprop.

* For each sub-network, the activation function of the last layer should be linear, the second last one should be sigmoid, 
and all the former ones can be other activation function, e.g., ReLu.


## Acknowledgement:

Thanks Vahid for providing the DCCA implementation in Keras-theano and the constructive advices.

## Reference:

Galen, A. et. al., [Deep Canonical Correlation Analysis](http://www.jmlr.org/proceedings/papers/v28/andrew13.pdf), ICML 2013