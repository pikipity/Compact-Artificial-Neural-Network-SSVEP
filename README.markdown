# Compact Artificial Neural Network Based on Task Attention for Individual SSVEP Recognition with Less Calibration

This study proposes a **compact artifical neural network** (ANN) architecture to reduce the number of trainable parameters and thus **avoid the over-fitting issue** of the ANNs in the individual SSVEP recognition. 

The related paper has been accepted by [IEEE Transactions on Neural Systems & Rehabilitation Engineering](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=7333).

This repository follows the license [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en).

## Breif introduction to proposed compact neural network

The core part is the proposed task attention architecture shown in the following figure. It contains 3 layers for each stimulus:

1. Temporal filtering layer: This layer measures the similarities of the processed EEG signals and the kernels. In this study, the kernels are pre-defined as the SSVEP reference and/or template signals.
2. Spatial filtering layer: This layer combines features along the spatial direction.
3. Feature combination layer: This layer combines resting features.

![Proposed task attention architecture](./images/1.png)

Following the ensemble technique of the eCCA, this study proposes to use the multi-head task attention layer. The multi-head task attention layer incorporates multiple task attention layers with different kernels in the temporal filtering layer as well as different trainable weights in the spatial filtering layer and the feature combination layer, which is shown in the following figure.

![Multi-head task attention](./images/2.png)

For the SSVEP recognition with multiple stimulus targets, the multi-head task attention architectures correponding to all stimuli are parallel concatenated together as shown in the following figure.

![Task attention based SSVEP recognition architecture](./images/3.png)

The filter-bank approach is also adopted in this study. The sub-band weights present the importance of the spatial information. These sub-band weights are also trainable parameters. The whole architecture is shown below.

![Task attention based SSVEP recognition architecture with filter-bank approach](./images/4.png)

## Run simulations

After downloading this repository, you can follow the follow steps to perform the stimulations. The main codes are based on MATLAB and Python3.9.

1. This repository contains the simulations on the Benchmark Dataset and the BETA Dataset:
   
   + For Benchmark Dataset, please directly download data from the [webpage](http://bci.med.tsinghua.edu.cn/download.html), and put all subjects' data in the folder `BenchmarkData`.
   + For BETA Dataset, please directly download data from the [webpage](http://bci.med.tsinghua.edu.cn/download.html), and put all subjects' data in the folder `BetaData`.

2. Run `rearrange_BenchmarkData.m` and `rearrange_BetaData.m` for Benchmark Dataset and BETA Dataset respectively to generate the filter-bank data. Note: These two codes may gnerate files with the same name. So please do NOT run simulations of these two datasets in the same path. 
3. Run `test_BenchmarkDataset.py` and `test_BetaDataset.py` for Benchmark Dataset and BETA Dataset respectively to do the simulations. Note:

    + The proposed model and related simulations are based on Python3.9 and Tensorflow2.0. Before you run the simulations, you need to install the related packages. You can find the required packages in `environment.yml`.
    + The proposed model is defined in `Compact_model.py`.
    + The results will be stored in `test_BenchmarkDataset.mat` and `test_BetaDataset.mat`. The accuracy is stored in the variable `test_acc_store`, which has 4 dimensions (signal length * block * subject * epoch).

## Simulation results

The proposed ANN architecture is compared with the prominent CA-based and DNN-based SSVEP recognition methods:

+ CA-based methods:
  
  + sCCA: X. Chen et al., “Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain-computer interface,” J. Neural Eng., vol. 12, no. 4, p. 046008, 2015. DOI: [10.1088/1741-2560/12/4/046008](https://doi.org/10.1088/1741-2560/12/4/046008).
  + eCCA: X. Chen, Y. Wang, M. Nakanishi, X. Gao, T.-P. Jung, and S. Gao, “High-speed spelling with a noninvasive brain-computer interface,” Proc. Natl. Acad. Sci., vol. 112, no. 44, pp. E6058-E6067, 2015. DOI: [10.1073/pnas.1508080112](https://doi.org/10.1073/pnas.1508080112).
  + (e)TRCA: M. Nakanishi, Y. Wang, X. Chen, Y.-T. Wang, X. Gao, and T.-P. Jung, “Enhancing detection of SSVEPs for a high-speed brain speller using task-related component Analysis,” IEEE Trans. Biomed. Eng., vol. 65, no. 1, pp. 104-112, 2018. DOI: [10.1109/TBME.2017.2694818](https://doi.org/10.1109/TBME.2017.2694818).

+ DNN-based methods:

    + Guney-DNN: O. B. Guney, M. Oblokulov, and H. Ozkan, “A deep neural network for SSVEP-based brain-computer interfaces,” IEEE Trans. Biomed. Eng., vol. 69, no. 2, pp. 932–944, Feb. 2022, doi: [10.1109/TBME.2021.3110440](https://doi.org/10.1109/TBME.2021.3110440).
    + Conv-CA: Y. Li, J. Xiang, and T. Kesavadas, “Convolutional correlation analysis for enhancing the performance of SSVEP-based brain-computer interface,” IEEE Trans. Neural Syst. Rehabil. Eng., vol. 28, no. 12, pp. 2681–2690, Dec. 2020, doi: [10.1109/TNSRE.2020.3038718](https://doi.org/10.1109/TNSRE.2020.3038718).

Note: Because this study focuses on the individual SSVEP recognition, the calibration data size in this study is much smaller than that in original papers of the Guney-DNN and the Conve-CA. Therefore, the recognition results of the Guney-DNN and the Conve-CA are different from these papers.

![Classification performance](./images/5.png)