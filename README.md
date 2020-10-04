# myTorch
Code for unpublished work,

Deep Convolutional Neural Networks for Real-Time Patient-Specific Wall Shear Stress Estimation 
Automatic Coronary Artery Lumen Border Detection Using Deep Convolutional Neural Networks

Installation
------------

To install the requirements

First install pytorch

````powershell
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
````

````powershell
python -m pip install -r requirements.txt
````

To install the main package in debug mode

````powershell
python -m pip install -e .
````

To install the main package in release mode

````powershell
python -m pip install .
````

to view the vtk files download and install paraview https://www.paraview.org/.

Data
----

Data for training and testing is assumed to be contained in the `DATA` folder in the top directory. The data is stored in two formats, .npz and .vtk. the vtk files can be easily viewed in paraview and show the original meshes as wells as the meshes parameterized in 2D. The npz files contain the tensor data with dimensions (channels, height, width). The data files are quite large and are unable to fit on the free version of github, however they can be obtained directly from the following link,

https://unimelbcloud-my.sharepoint.com/:f:/g/personal/cmamon_student_unimelb_edu_au/Epx-e_s_Zj1HtlF88MOmXF4B1MJMpbBEbQ7Zu1gVJizOLA?e=VYqjIn.

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
