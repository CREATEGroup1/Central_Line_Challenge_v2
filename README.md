# Central_Line_Challenge
## Create conda environment  
For those using the School of Computing GPU Server, you may skip this step. Anaconda has already been installed and the required environment has already been created for you.  

For those wishing to run on a local computer:  
1. Ensure Anaconda has been installed on your device: https://www.anaconda.com/products/distribution  
> - during installation, make sure to select the option to add anaconda to your search path  
2. Create a tensorflow environment
> - with GPU support
```
conda create -n createKerasEnv tensorflow-gpu
```
> - CPU only
```
conda create -n createKerasEnv tensorflow
```
3. Install other required packages
```
conda activate createKerasEnv  
pip install pandas, scikit-learn, matplotlib, opencv-python
```

## Download Data
Download links are password protected and will only be available until May 6th, 2022. Registered participants will receive the password via email on May 2nd, 2022.  
  
#### Training Data:
Training data can be downloaded in 4 parts using the following links: [Part 1](https://tinyurl.com/5drkkcrk), [Part 2](https://tinyurl.com/pthcvjfk), [Part 3](https://tinyurl.com/5n8mbbt4), [Part 4](https://tinyurl.com/4f7zwt6s)  
  
#### Test Data:
Test data can be downloaded using the following link on May 5th, 2022: [Test Data]()

## Prepare Dataset for Training
Once all parts of the dataset have been downloaded for training, donwload code or clone this repository. Navigate to the location where the code is located and use the prepareDataset.py script to unpack and format your dataset. The script can be run by entering the following lines into your command prompt (replace all instances of UserName with your real username):  
```
conda activate createKerasEnv  
cd <path_to_repository>  
python prepareDataset.py --compressed_location=C:/Users/UserName/Downloads --target_location=C:/Users/UserName/Documents/CreateChallenge --dataset_type=Train  
```  
To prepare the test set, follow the same steps, but change the --dataset_type flag to Test  
  
If the code is executed correctly, you should see a new directory in your target location called either Training_Data or Test_Data. These directories will contain a set of subdirectories (one for each video) that contain the images and labels. Within that folder you will also see a csv file that contains a compiled list of all images and labels within the dataset. (Note: there will not be any labels for the test images).  
