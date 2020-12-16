# Generative-Models-for-Video-Prediction
Utilize multiple model architectures to improve sharpness in video prediction

## Model Structure

### Convolutional Autoencoder (C-AE) 
![CONV_AE](./PlotNeuralNetModels/pdf2png/CONV_AE/CONV_AE-1.png?raw=true "title")

### Convolutional Variational Autoencoder (C-VAE)
![CONV_VAE](./PlotNeuralNetModels/pdf2png/CONV_VAE/CONV_VAE-1.png?raw=true "title")

###  Fully-Connected Autoencoder (FC-AE)
![Dense_AE](./PlotNeuralNetModels/pdf2png/Dense_AE/Dense_AE-1.png?raw=true "title")

###  Fully-Connected Variational Autoencoder (FC-VAE)
![Dense_VAE](./PlotNeuralNetModels/pdf2png/Dense_VAE/Dense_VAE-1.png?raw=true "title")

###  Convolutional Generative-Adversarial Autoencoder (C-GA-AE)
![CONV_GAAE](./PlotNeuralNetModels/pdf2png/CONV_GAAE/CONV_GAAE-1.png?raw=true "title")

###  Fully-Connected Generative-Adversarial Autoencoder (FC-GA-AE)
![Dense_GAAE](./PlotNeuralNetModels/pdf2png/Dense_GAAE/Dense_GAAE-1.png?raw=true "title")

###  Convolutional Generative-Inference-Adversarial Autoencoder (C-GIA-AE)
![CONV_GIAAE](./PlotNeuralNetModels/pdf2png/CONV_GIAAE/CONV_GIAAE-1.png?raw=true "title")

###  Fully-Connected Generative-Inference-Adversarial Autoencoder (FC-GIA-AE)
![Dense_GIAAE](./PlotNeuralNetModels/pdf2png/Dense_GIAAE/Dense_GIAAE-1.png?raw=true "title")

###  Convolutional Inference-Adversarial Autoencoder (C-IA-AE) 
![CONV_IAAE](./PlotNeuralNetModels/pdf2png/CONV_IAAE/CONV_IAAE-1.png?raw=true "title")

### Full-Connected Inference-Adversarial Autoencoder (FC-IA-AE)
![Dense_IAAE](./PlotNeuralNetModels/pdf2png/Dense_IAAE/Dense_IAAE-1.png?raw=true "title")

### Convolutional LSTM Autoencoder (C-LSTM-AE)
![CONV_LSTM_AE](./PlotNeuralNetModels/pdf2png/CONV_LSTM_AE/CONV_LSTM_AE-1.png?raw=true "title")

### Convolutional LSTM Variational Autoencoder (C-LSTM-VAE)
![CONV_LSTM_VAE](./PlotNeuralNetModels/pdf2png/CONV_LSTM_VAE/CONV_LSTM_VAE-1.png?raw=true "title")

### Convolutional Time-Distributed Autoencoder (C-TD-AE)
![Time_CONV_AE](./PlotNeuralNetModels/pdf2png/Time_CONV_AE/Time_CONV_AE-1.png?raw=true "title")

### Convolutional Time-Distributed Variational Autoencoder (C-TD-VAE)
![Time_CONV_VAE](./PlotNeuralNetModels/pdf2png/Time_CONV_VAE/Time_CONV_VAE-1.png?raw=true "title")



## Datasets
Atari Datasets [11] originally intended for RNN(Recurrent Neural Network)usage, consists of recorded game play of 5 old Atari console games. This Thesis uses all 5 of them on different models as well as on 2 reference works as baselinefor the models. Below are some details and samples from all 5 games.
* Ms. Pacman : 1172401 training images 381757 validation images (a)
* Video Pinball : 901479 training images 295328 validation images (b)
* Q*bert : 1124726 training images 360062 validation images (c)
* Montezuma’s Revenge : 1783796 training images578075 validation images (d)
* Space Invaders : 1331762 training images 434316 validation images (e)

![Pacman](./PlotNeuralNetModels/dataset_samples/Pacman.png?raw=true "title") ![Pinball](./PlotNeuralNetModels/dataset_samples/Pinball.png?raw=true "title") ![Qbert](./PlotNeuralNetModels/dataset_samples/Qbert.png?raw=true "title") ![Revenge](./PlotNeuralNetModels/dataset_samples/Revenge.png?raw=true "title") ![Spaceinvaders](./PlotNeuralNetModels/dataset_samples/Spaceinvaders.png?raw=true "title")

## Results 
Notebooks Currently hosted on Google Colab for testing purposes. In order to retrain the best models on the datasets. Please click the "Open in colab" icon and copy the notebook to run the code used in the notebooks are open source 
- Pacman
     - C-GIA-AE
       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg
    )](https://colab.research.google.com/drive/1TTYoZ7IwvLIRNXMiyDQpRWsL_qd2ij8c#offline=true&sandboxMode=true)
     - C-LSTM-VAE
       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg
       )](https://colab.research.google.com/drive/1DqcxfpD4ya6eT_7XO69fVM9Gy6rhikj_#offline=true&sandboxMode=true)
       
- Pinball
     - C-GIA-AE
       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg
    )](https://colab.research.google.com/drive/1ZsFM18WAL_JgXPK3Gm0hKqY9E1k9rbTA#offline=true&sandboxMode=true)
     - C-LSTM-VAE
       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg
       )](https://colab.research.google.com/drive/1fGICc07-HJ4oQh9C2cjlwlKLpjPLlk8y#offline=true&sandboxMode=true)
       
- Q*bert
     - C-GIA-AE
       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg
    )](https://colab.research.google.com/drive/1E0RzzvBlG5uh2G6ADJkNNamx-_sk9LG3#offline=true&sandboxMode=true)
     - C-LSTM-VAE
       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg
       )](https://colab.research.google.com/drive/1pxPQuvcPe5GeFtHdoIUEpSwLEnNkI_WN#offline=true&sandboxMode=true)
       
- Montezuma’s Revenge
     - C-GIA-AE
       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg
    )](https://colab.research.google.com/drive/1jhqtOFqWL-4cbHLr4xnwTsT17C5sbsSg#offline=true&sandboxMode=true)
     - C-LSTM-VAE
       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg
       )](https://colab.research.google.com/drive/1KfmzpITMH8USwjObSMVhIQ-6cfectYi_#offline=true&sandboxMode=true)
       
- Space Invaders
     - C-GIA-AE
       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg
    )](https://colab.research.google.com/drive/1IM4toZY-3UcxanZMsoc8KcGKScE4AbDB#offline=true&sandboxMode=true)
     - C-LSTM-VAE
       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg
       )](https://colab.research.google.com/drive/1fYjIDsAOwo0mlfbtRD26lFcZ9RFK6K3j#offline=true&sandboxMode=true)

### Results on Pacman
[comment]: <> (![Pacman]&#40;./GIFS/original.gif?raw=true "title"&#41;)
#### Best Results 
<table>
  <tr>
     <td>Ground Truth</td>
     <td>Multi Scale GAN [1]</td>
     <td>C-GIA-AE</td>
     <td>C-LSTM-VAE</td>
  </tr>
  <tr>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/original.gif" width=160 height=210 /></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/Multi-Scale GAN.gif" width=160 height=210></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/ConvGIAAE.gif" width=160 height=210></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/ConvLSTMVAE.gif" width=160 height=210></td>
  </tr>
 </table>

#### Worst Results 
<table>
  <tr>
     <td>Ground Truth</td>
     <td>FC-AE</td>
     <td>FC-VAE</td>
  </tr>
  <tr>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/original.gif" width=160 height=210 /></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/DenseAE.gif" width=160 height=210></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/DenseVAE.gif" width=160 height=210></td>
  </tr>
 </table>

#### All models 

<table>
  <tr>
     <td>Ground Truth</td>
     <td>FC-AE</td>
     <td>FC-VAE</td>
     <td>C-AE</td>
     <td>C-VAE</td>
  </tr>
  <tr>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/original.gif" width=80 height=105 /></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/DenseAE.gif" width=80 height=105 ></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/DenseVAE.gif" width=80 height=105 ></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/ConvAE.gif" width=80 height=105 ></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/ConvVAE.gif" width=80 height=105 ></td>
  </tr>

  <tr>
     <td>Ground Truth</td>
     <td>C-LSTM-AE</td>
     <td>C-LSTM-VAE</td>
     <td>C-TD-AE</td>
     <td>C-TD-VAE</td>
  </tr>
  <tr>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/original.gif" width=80 height=105 /></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/ConvLSTMAE.gif" width=80 height=105 ></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/ConvLSTMVAE.gif" width=80 height=105 ></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/TimeDistAE.gif" width=80 height=105 ></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/TimeDistVAE.gif" width=80 height=105 ></td>
  </tr>

  <tr>
     <td>Ground Truth</td>
     <td>C-GIA-AE</td>
     <td>FC-GIA-AE</td>
  </tr>
  <tr>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/original.gif" width=80 height=105 /></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/ConvGIAAE.gif" width=80 height=105 ></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/DenseGIAAE.gif" width=80 height=105 ></td>
  </tr>
 </table>

### Results on Pinball
<table>
  <tr>
     <td>Ground Truth</td>
     <td>C-GIA-AE</td>
     <td>C-LSTM-VAE</td>
  </tr>
  <tr>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/other_atari/original_pinball.gif" width=160 height=210 /></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/other_atari/ConvGIAAE_pinball.gif" width=160 height=210></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/other_atari/ConvLSTMVAE_pinball.gif" width=160 height=210></td>
  </tr>
 </table>

### Results on Q*bert

<table>
  <tr>
     <td>Ground Truth</td>
     <td>C-GIA-AE</td>
     <td>C-LSTM-VAE</td>
  </tr>
  <tr>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/other_atari/original_qbert.gif" width=160 height=210 /></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/other_atari/ConvGIAAE_qbert.gif" width=160 height=210></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/other_atari/ConvLSTMVAE_qbert.gif" width=160 height=210></td>
  </tr>
 </table>

### Results on Montezuma’s Revenge

<table>
  <tr>
     <td>Ground Truth</td>
     <td>C-GIA-AE</td>
     <td>C-LSTM-VAE</td>
  </tr>
  <tr>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/other_atari/original_revenge.gif" width=160 height=210 /></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/other_atari/ConvGIAAE_revenge.gif" width=160 height=210></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/other_atari/ConvLSTMVAE_revenge.gif" width=160 height=210></td>
  </tr>
 </table>

### Results on Space Invaders

<table>
  <tr>
     <td>Ground Truth</td>
     <td>C-GIA-AE</td>
     <td>C-LSTM-VAE</td>
  </tr>
  <tr>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/other_atari/original_spaceinvaders.gif" width=160 height=210 /></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/other_atari/ConvGIAAE_spaceinvaders.gif" width=160 height=210></td>
    <td><img src="https://github.com/azeghost/Generative-Models-for-Video-Prediction/raw/main/GIFS/other_atari/ConvLSTMVAE_spaceinvaders.gif" width=160 height=210></td>
  </tr>
 </table>


### Overall 


### Installation instructions required packages
Requirements: **Tensorflow, Keras, Colorlog, Pillow, Tensorflow-probabilities, Jupyter-notebooks**

Google Colaboratory examples given above.
Downloading datasets : 

    - Linux based systems:  Script_dir = 'data'+sep_local+'download_atari_datasets.sh'
                            Script call to download using dataset_name 
                            !/bin/bash $Script_dir -f $DATA_DOWN_PATH -d $dataset_name

    - Windows based systems:    No script created Manually downloadable from 
                                https://github.com/yobibyte/atarigrandchallenge
## References
[1] M. Mathieu, C. Couprie, and Y. LeCun., “Deep multi-scale video predictionbeyond mean square error.,”ICLR., Feb. 2016.

[2] L. Liyuan, J. Haoming, H. Pengcheng, C. Weizhu, L. Xiaodong, G. Jian-feng, and H. Jiawei, “On the variance of the adaptive learning rate andbeyond,”ICLR 2019, April. 2019.

[3] A. Kar, “Future image prediction using artificial neural networks.,”book,2012.

[4] N. K. Verma, “Future image frame generation using artificial neural networkwith selected features.,”AIPR., Oct. 2012.

[5] N. Srivastava, E. Mansimov, and R. Salakhutdinov., “Unsupervised learningof video representations using lstms.,”ICML., 2015.

[6] X. Shi, Z. Chen, H. Wang, and D.-Y. Yeung., “Convolutional lstm network:A machine learning approach for precipitation nowcasting.,”book, 2015.

[7] A. Dosovitskiy and T. Brox, “Generating images with perceptual similaritymetrics based on deep networks.,”ICLR., Feb. 2016.

[8] Z. Wang, E. P. Simonselli, and A. C. Bovik., “Image quality assessment:From error visibility to structural similarity.,”IEEE Trans. Image Process-ing, no. 13.4, pp. 600 – 612, Apr. 2004.

[9] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Multi-scalestructural similarity for image quality assessment.,”IEEE Conference onSignals, Systems and Computers, Nov. 2003.

[10] H. Zhao, O. Gallo, I. Frosio, and J. Kautz, “Loss functions for neural net-works for image processing.,”book, June 2016.

[11] V. Kurin, S. Nowozin, K. Hofmann, L. Beyer, and B. Leibe, “The atarigrand challenge dataset.,”arXiv:1705.10998., 2017.
