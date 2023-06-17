# Vision-Position Multi-Modal Beam Prediction Using Real Millimeter Wave Datasets
This is a python code package related to the following article:
Gouranga Charan, Tawfik Osman, Andrew Hredzak, Ngwe Thawdar, and Ahmed Alkhateeb, "[Vision-Position Multi-Modal Beam Prediction Using Real Millimeter Wave Datasets](https://ieeexplore.ieee.org/document/9771835/),", in 2022 IEEE Wireless Communications and Networking Conference (WCNC), 2022, pp. 2727-2731

# Abstract of the Article
Enabling highly-mobile millimeter wave (mmWave) and terahertz (THz) wireless communication applications requires overcoming the critical challenges associated with the large antenna arrays deployed at these systems. In particular, adjusting the narrow beams of these antenna arrays typically incurs high beam training overhead that scales with the number of antennas. To address these challenges, this paper proposes a multi-modal machine learning based approach that leverages positional and visual (camera) data collected from the wireless communication environment for fast beam prediction. The developed framework has been tested on a real-world vehicular dataset comprising practical GPS, camera, and mmWave beam training data. The results show the proposed approach achieves more than 75% top-1 beam prediction accuracy and close to 100% top-3 beam prediction accuracy in realistic communication scenarios.

# Code Package Content 
The scripts for generating the results of the ML solutions in the paper. This script adopts Scenarios 5 and 6 of DeepSense6G dataset.

**To reproduce the results, please follow these steps:**
1. Download [beam prediction dataset of DeepSense 6G/Scenario 5 and 6]
	a. https://deepsense6g.net/position-aided-beam-prediction/
	b. https://deepsense6g.net/vision-aided-beam-prediction/
2. Download (or clone) the repository into a directory
3. Extract the dataset into the repository directory 
4. Use the downsample_pwr_gen_train_test_data.py code to generate the train, test and validation csv files for each scenario
5. Use the evaluation code provided in each sub-folders to generate the top-3 beam prediction accuracy

If you have any questions regarding the code and used dataset, please contact [Gouranga Charan](mailto:gcharan@asu.edu?subject=[GitHub]%20Beam%20prediction%20implementation).

# License and Referencing
This code package is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). 
If you in any way use this code for research that results in publications, please cite our original article:
> Gouranga Charan, Tawfik Osman, Andrew Hredzak, Ngwe Thawdar, and Ahmed Alkhateeb, "[Vision-Position Multi-Modal Beam Prediction Using Real Millimeter Wave Datasets](https://ieeexplore.ieee.org/document/9771835/),", in 2022 IEEE Wireless Communications and Networking Conference (WCNC), 2022, pp. 2727-2731, doi: 10.1109/WCNC51071.2022.9771835.

If you use the [DeepSense 6G dataset](www.deepsense6g.net), please also cite our dataset article:
> A. Alkhateeb, G. Charan, T. Osman, A. Hredzak, and N. Srinivas, “DeepSense 6G: large-scale real-world multi-modal sensing and communication datasets,” to be available on arXiv, 2022. [Online]. Available: https://www.DeepSense6G.net
