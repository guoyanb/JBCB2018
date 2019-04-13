# JBCB2018
#1 JBCB2018 
 <p> <a href="https://www.worldscientific.com/doi/10.1142/S021972001850021X">Protein secondary structure prediction improved by recurrent neural networks integrated with 2-dimensional convolutional neural networks</a>. Journal of Bioinformatics and Computational Biology.
 <br>
The matrices of protein sequence features comprises the amino acid dimension (time-step dimension) and the feature
vector dimension. Common approaches to predict 8-state secondary structure only concentrate on the amino acid dimension. The
paper propose a hybrid deep learning framework, recurrent neural networks (RNNs) integrated with 2-dimensional (2D)
convolutional neural networks (CNNs), for protein secondary structure prediction.The evaluation metric should be weighted accuracy. You can copy the function and paste it into the keras metric.py, which can be downloaded in 'https://github.com/wentaozhu/proteincascade-cnn-lstm'. Then compile keras, install. 


#2 Dataset:
         
 For cb513+profile_split1.npy.gz, cullpdb+profile_6133_filtered.npy.gz, please download from this website
 "http://www.princeton.edu/~jzthree/datasets/ICML2014/".<br>
     
 For CASP10 and CASP11, please download from this website
"https://drive.google.com/drive/folders/1404cRlQmMuYWPWp5KwDtA7BPMpl-vF-d".<br>
      
Finally, Download data and put them in ./data folder.


#3 Settings:
      Install the requirements (you can use pip or Anaconda):
      
      conda install keras
      
      conda install tensorflow-gpu
    
  The version of keras is 1.2
  The version of tensorflow-gpu is 0.12
  
#4 Reference:

    1 https://github.com/icemansina/IJCAI2016
    2 https://github.com/wentaozhu/protein-cascade-cnn-lstm

### Citation：
Please cite the following paper in your publication if it helps your research:

<a href="https://www.worldscientific.com/doi/10.1142/S021972001850021X">Protein secondary structure prediction improved by recurrent neural networks integrated with 2-dimensional convolutional neural networks</a>. Journal of Bioinformatics and Computational Biology Vol. 16, No. 05, 1850021 (2018).
