# myTorch

Installation
------------

To install the requirements

````
python -m pip install requirements.txt
````

To install the main package in debug mode
````
python -m pip install -e .
````

To install the main package in release mode
````
python -m pip install .
````

Running Examples
----------------

Examples are stored in training examples. At the moment only the examples for predicting WSS are supplied. Internal field predictions will be added at a later date. To run an example, in the terminal,

````
cd Training_Examples/WSSPrediction/case_folder
python train.py
````