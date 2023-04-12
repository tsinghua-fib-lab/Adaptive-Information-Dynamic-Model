# Adaptive-Information-Dynamic-Model

Thank you for reviewing our manuscript "Human-AI Adaptive Dynamics Drive Emergence of Information Cocoons”, submitted to Nature Machine Intelligence.

The repository contains the source code and the simulated dataset. The source code is implemented in Python.

---

## System requirements
```
python == 3.8.11
numpy == 1.23.5
scipy == 1.10.1
scikit-learn ==  1.2.2
matplotlib == 3.7.1
seaborn == 0.12.2
jupyter notebook == 6.4.8
```
---
## Installation Guide

* Install Anaconda https://docs.anaconda.com/anaconda/install/index.html
* Create an experimental environment: create -n exp python=3.8.11
* Install packages: 
    ```shell
    conda install numpy
    conda install scipy
    conda install scikit-learn
    conda install matplotlib
    conda install seaborn
    conda install tqdm
    conda install setproctitle
    conda install jupyter notebook
    ```

The installation takes less than 10 minutes under good network conditions.

---

## Demo and Reproduction

We take Figure 2 (a,b) and Figure 3 (a,b) in Main Text as the demo. For your convenience, we provide three ways to reproduce the simulation results.

**(1) We provide the code and the dataset for the demo. The usage is listed as follows:**

* Open the demo script: ./figs_demo.ipynb
* Run the script in the created experimental environment.
* Expected output is Figure 2 (a,b) and Figure 3 (a,b) in Main Text.

The running takes less than 5 minutes on a "normal" desktop computer.

**(2) We also provide the code to reproduce the simulated data which have been used in the demo. The usage is listed as follows:**

* Prepare the experimental environment, the result path, and the log file.
    ```shell
    conda activate exp 
    mkdir results
    mkdir pic
    mkdir tmp_re
    touch exp_record.txt
    ```
* Reproduce the data in the demo.
    ```shell
    python run_main_text_demo.py
    ```
* Open the demo script: ./figs_demo.ipynb
* Uncomment Cell 2 in the demo script.
* Run the script.
* Expected output is Figure 2 (a,b) and Figure 3 (a,b) in Main Text.

The running takes about 6-12 hours on a "normal" desktop computer.

**(3) We also provide the code to reproduce all the simulation results in our manuscript. The usage is listed as follows:**

* Prepare the experimental environment, the result path, and the log file.
    ```shell
    conda activate exp 
    mkdir results
    mkdir pic
    mkdir tmp_re
    touch exp_record.txt
    ```
* Reproduce simulation results of the video dataset.
    ```shell
    python run_main_video.py
    ```
* Reproduce simulation results of the news dataset.
    ```shell
    python run_SI_news.py
    ```
 
Note that reproducing all the results requires large computing resources. Therefore, we recommend using CPU workstations rather than desktop computers.

---

## File Description

* Folders
    * Data: initialization data
    * theory_results: data prepared for visualization of theoretical predictions
    * fig_demo: data prepared for demo
* Scripts
    * demo_utils.py: functions for demo
    * draw_pic.py: functions for visualization
    * run_main_text_demo.py: a script for reproducing the simulated data in the demo.
    * run_main_video.py: a script for reproducing simulation results of the video dataset.
    * run_SI_news.py: a script for reproducing simulation results of the news dataset.
    * simulator_optimized.py: base codes of simulation.
    * utils.py: functions for simulation.
















