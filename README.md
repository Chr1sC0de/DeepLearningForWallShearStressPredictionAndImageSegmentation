# myTorch

Installation
------------

To install the requirements

First install pytorch
````
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
````

````
python -m pip install -r requirements.txt
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

Examples are stored in training examples. At the moment only the examples for predicting WSS are supplied. Internal field predictions will be added at a later date (accuracy may vary from machine to machine). To run an example, in the terminal,

````
cd Training_Examples/WSSPrediction/case_folder
python train.py
````