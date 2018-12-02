# JBCB2018
#1 JBCB2018 
 <p> <a href="https://www.worldscientific.com/doi/10.1142/S021972001850021X">Protein secondary structure prediction improved by recurrent neural networks integrated with 2-dimensional convolutional neural networks</a>. Journal of Bioinformatics and Computational Biology. In press.2018>
 <br>
The matrices of protein sequence features comprises the amino acid dimension (time-step dimension) and the feature
vector dimension. Common approaches to predict 8-state secondary structure only concentrate on the amino acid dimension. The
paper propose a hybrid deep learning framework, recurrent neural networks (RNNs) integrated with 2-dimensional (2D)
convolutional neural networks (CNNs), for protein secondary structure prediction.


#2 Dataset:
     
     
 For cb513+profile_split1.npy.gz, cullpdb+profile_6133_filtered.npy.gz, please download from this website
 "http://www.princeton.edu/~jzthree/datasets/ICML2014/".<br>
     
 For CASP10 and CASP11, please download from this website
"https://drive.google.com/drive/folders/1404cRlQmMuYWPWp5KwDtA7BPMpl-vF-d".<br>
      
Finally, Download data and put them in ./data folder.


#3 Settings:
      Install the requirements (you can use pip or Anaconda):
      
      
      conda install pip h5py cython numpy scipy
      
      conda install keras
      
      conda install tensorflow-gpu
    
  The version number of keras is 1.2
  The version number of tensorflow-gpu is 0.12
  
#4 Referable:

    1 https://github.com/icemansina/IJCAI2016
    2 https://github.com/wentaozhu/protein-cascade-cnn-lstm


