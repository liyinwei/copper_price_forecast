# Copper price forecast

This is an open source project aims at making a prediction of copper price using machine learning / deep learning approach. The language of this project is Python and the ideas may extend to other time series prediction problems as you like.

# Motivation

The motivation of launching this project is that copper is a kind of important raw material of some midstream and downstream materials, whose price is of great relevance to copper price. 

Here is a comparison of the fluctuation of the price of cooper and a downstram material:

![image](https://github.com/liyinwei/res/raw/master/2017/copper_price_readme_1.png)

# Prerequisites

- [Python](https://www.python.org/) 3.5 + 
- [numpy](http://www.numpy.org/) 1.13.0 + 
- [pandas](http://pandas.pydata.org/) 0.20.2 + 
- [scikit-learn](http://scikit-learn.org/stable/) 0.18.1 + 
- [Keras](https://keras.io/) 2.0.4 + 
- [tensorflow](https://www.tensorflow.org/) 1.1.0 + 
- [matplotlib](http://matplotlib.org/) 2.0.2 +
- [mysql-connector-python-rf](https://pypi.python.org/pypi/mysql-connector-python-rf) 2.2.2 + 

# Structure
- [common](https://github.com/liyinwei/copper_price_forecast/tree/master/common): common method such as data loading, model evaluation & model visualization etc.
- [mlp](https://github.com/liyinwei/copper_price_forecast/tree/master/mlp): predict copper price using sklearn.neural_network.MLPRegressor
- [lstm](https://github.com/liyinwei/copper_price_forecast/tree/master/lstm): predict copper price using keras.layers.recurrent.LSTM
- [pcb](https://github.com/liyinwei/copper_price_forecast/tree/master/pcb): correlation analysis of copper and a downstream material price

# Running
There is a main method in each python file so you can run it easily and the following is a sample of run result of the [mlp](https://github.com/liyinwei/copper_price_forecast/tree/master/mlp) method:

![image](https://github.com/liyinwei/res/raw/master/2017/copper_price_readme_2.png)


# Authors
- [Yinwei Li](https://github.com/liyinwei)
  - **weichat**: coridc
  - **email**: 251469031@qq.com

*Don't hesitate to contact me on any topics about this project at your convenience.*


# Contributors
- [Yinwei Li](https://github.com/liyinwei)


# Contributing

When contributing to this repository, you can first discuss the change you wish to make via issue, email, or any other method with the owners of this repository.


# License

This project is licensed under the [GNU General Public License v3.0](http://www.gnu.org/licenses/gpl-3.0.html) License - see the [LICENSE](https://github.com/liyinwei/copper_price_forecast/blob/master/LICENSE) file for details.

# Acknowledgments

I'd like hat tip to anyone who use the codes or send me any proposals of the project.