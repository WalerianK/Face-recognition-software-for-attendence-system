# Face Recognition Pipeline 

This repository contains the official code for our Face Recognition pipeline as part of MCEME internship . It uses a one-click setup, where the pipeline automatically completes the setup for your list of registered users.

## Environment Setup 
If package manager [pip](https://pip.pypa.io/en/stable/) is preferred inside [conda](https://www.anaconda.com/) , then please use the following commands: 
Please note the exact version of Python version to avoid issues in environment setup . 

bash
conda create -n facerec python=3.10.11
conda activate facerec
pip install -r requirements.txt

## Setting up 
We are providing the codebase and the required structure for setting up your custom dataset. We assume A,B,C.. are unique users. We advise a minimum of 10 images with good variations in-terms of facial emotions , lightening conditions , head shift , etc for an optimal pipeline. 

### Input Directory 
├── Images/             # Data Folder 
        ├── A_1.jpg 
        ├── A_2.jpg
        ├── A_3.jpg
        ......
        ├── B_1.jpg 
        ├── B_2.jpg
        ├── B_3.jpg
        ......
    
Please place them in appropriate location and add the value of the parent directory inside the following bash file . Few values are present for the developer's reference , you will have to override every parameter for your setup .

bash
cd project/
bash pipeline.sh

## Inference 
Please refer the classification_code.py for testing out single image and a set of images. You also have an option to visual detected user name on image.

## Contact 
For any suggestions/contributions to the repository , please contact : <br />
Prasanth Vadlamudi - f20210780@hyderabad.bits-pilani.ac.in / prasanth060703@gmail.com
