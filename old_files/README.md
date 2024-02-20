# imagnet
A python based tool to identify the different magnetic states in an atomic simulation such as helical, saturated, random and skyrmion lattice sates with PyTorch based image recognition. 
# usage
Install the required libraries

`pip install -r requirements.txt`

Ready to use: The trained model is saved as "newfile.h5", download this and run "ReadImages_and_Label.py" to detect the labels of each image. 

Train your own version: The training is done by the "train_the_model.py" file: place your training images in the ./images folder and give the  appropriate labels and names in the code. 
