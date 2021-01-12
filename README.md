# Neural-distinguisher

1. This project contains the supporting material of the paper entitled "An Improved Integral distinguisher scheme based on Deep Learning". The paper can be found at https://easychair.org/publications/preprint/9v46

2. For more information, feedback or questions, pleast contact at bzahednezhad@gmail.com

3. These algorithms are implemented by behnam zahednejad, Institute of Artifcial Intelligence and Blockchain, Guangzhou University.

4. The project implements the integral distingusher on different block ciphers using division prperty (using minizinc tool), and deep learning CNN algorithms namely ResNet, ResNext and DensNet. 

5. The implementaton of few-shot learning in training different block ciphers are in the Few-shot folder

6. The implementation of the key recovery attack on speck block cipher is in the key recovery file. In order to use this file, you should first save the trained model of speck and then load it in this file.


The division property is implemented using minizinc optmization tool which can be found at https://www.minizinc.org

Also to run the deep learning CNN algorithms,  a fast gpu-based enviorment can be found at https://colab.research.google.com
