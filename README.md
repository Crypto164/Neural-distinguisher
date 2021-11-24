# Integral Neural-Distinguisher

1. The project implements the integral distingusher on different block ciphers using division prperty (using minizinc tool), and deep learning CNN algorithms namely ResNet, ResNext and DensNet. 

2. The implementaton of transfer learning in training different block ciphers are in the Transfer folder

3. The implementation of the key recovery attack on speck block cipher is in the key recovery file. In order to use this file, you should first save the trained model of speck and then load it in this file.

4. The implementation of the same-difference experiment on Speck32/64 is in the same-difference file.
5. The implementation of algorithm 2 on the sBOX of PRESENT cipher is in the algorithmm 2 file.


The division property is implemented using minizinc optmization tool which can be found at https://www.minizinc.org

Also to run the deep learning CNN algorithms,  a fast gpu-based enviorment can be found at https://colab.research.google.com
