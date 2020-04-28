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

to view the vtk files download and install paraview https://www.paraview.org/.

Data
----
Data fpr training and testing is assumed to be contained in the `DATA` folder. The data is stored in two formats, .npz and .vtk. the vtk files can be easily viewed in paraview and show the meshes parameterized in in 2D, while the npz files contain the tensor data with dimensions (channels, height, width). The data files are quite large and are unable to fit on the github, however they can be requested directly from the author at cmamon@student.unimelb.edu.au.

Running Examples
----------------

Examples are stored in the `Training_Examples` directory. At the moment only the examples for predicting WSS are supplied. Internal field predictions will be added at a later date once the paper is completed(accuracy may vary from machine to machine). To run an example, in the terminal,

````
cd Training_Examples/WSSPrediction/<case_folder>
python train.py
````

A note on output meshes
-----------------------

The meshes output from the the training procedures are all from the pyvista `StructuredGrid` methods, the method produces a poor quality mesh and will be updated in future.