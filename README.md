# Adaptive-Information-Dynamic-Model

Thank you for reviewing our manuscript "Human-AI Adaptive Dynamics Drive Emergence of Information Cocoons”, submitted to Nature Machine Intelligence.

The repository contains the source code and the simulated dataset. The source code is implemented in Python.

**Update@20230609**: In response to the Reviewer's valuable suggestion, we add a Docker image to the repository. 

**Update@20230726**: After careful anonymization procedures and reviews, we update our empirical codes and datasets.


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

### Option: From Docker Image (Update@20230609)

* Install [Docker](https://docs.docker.com/install/)
* Download the Docker Image: [Image](https://drive.google.com/drive/folders/1Kq3ha3FKKsuqgYHdpr2KY9gYCJ8onHQ0?usp=sharing)
* Import the Docker Image:
    ```shell
    docker import - new_exp<exp_environment.tar
    ```
* Run the Docker Image:
    ```shell
    docker run -t -i -p 8844:8844 new_exp /bin/bash
    ```

Indeed, as the novice of Docker, we follow the base image created by [continuumio](https://hub.docker.com/r/continuumio/anaconda3). We would like to thank the overall community for the support. If there are any issues regarding the enviroment, please feel free to contact us.

---

## Demo and Reproduction

We take Figure 1, Figure 2 (a,b) and Figure 3 (a,b) in Main Text as the demo. For your convenience, we provide three ways to reproduce the simulation results. We also provide the two raw empirical datasets, the preprocessing code, the processed datasets, and the drawing codes.

**(1) We provide the code and the dataset for the demo. The usage is listed as follows:**

* Open the demo script: ./figs_demo_empirical.ipynb or ./figs_demo_model.ipynb
* Run the script in the created experimental environment.
* Expected output is Figure 1, Figure 2 (a,b) and Figure 3 (a,b) in Main Text.

The running takes less than 5 minutes on a "normal" desktop computer.

### Option: Docker (Update@20230609)

* Activate the environment and enter the folder.
    ```shell
    conda activate exp
    cd ./NMI-submissions
    ```

* Start a Jupyter Notebook.
    ```shell
    jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --port 8844
    ```
* Copy the link like http://127.0.0.1:8844/?token=2a22165cd3917ad2f925b76c7d643688b493e48b88a03333 in your browse. 
* Open the demo script: ./figs_demo.ipynb
* Run the script in the created experimental environment.
* Expected output is Figure 1, Figure 2 (a,b) and Figure 3 (a,b) in Main Text.


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
* Rerun the script.
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

**(4)  We also provide the two raw empirical datasets, the preprocessing code, the processed datasets, and the drawing codes:**

* Prepare the experimental environment: 
    ```shell
    conda activate exp 
    ```
* Preprocess the raw datasets:
    ```shell
    python preprocessing.py
    ```

* Draw Figures: Open ./figs_demo_empirical.ipynb and Rerun it.

---

## File Description

* Folders
    * Data: initialization data
    * theory_results: data prepared for visualization of theoretical predictions
    * fig_demo: data prepared for demo
    * raw_data: two raw datasets. Note that following the "less than 30M" instructions, we seperate the video dataset into two partitions. 
    * processed_data: preprocessed empirical data using the script "preprocessing.py".

* Scripts
    * demo_utils.py: functions for demo
    * draw_pic.py: functions for visualization
    * run_main_text_demo.py: a script for reproducing the simulated data in the demo.
    * run_main_video.py: a script for reproducing simulation results of the video dataset.
    * run_SI_news.py: a script for reproducing simulation results of the news dataset.
    * simulator_optimized.py: base codes of simulation.
    * utils.py: functions for simulation.
    * preprocessing.py: a script for preprocessing the raw datasets.
    * figs_demo_model.ipynb: a scirpt for reproducing demo figures of model parts.
    * figs_demo_empirical.ipynb: a scirpt for reproducing demo figures of empircal parts.
* Data Format for Raw Empirical Datasets
    * news_raw_data.csv
        * userid: anonymized user identifiers
        * final_entropy: informantion entropy
        * normalized_entropy: normalized information entropy
        * sim_strength: strength of similarity-based matching
        * pos_ratio: ratio of positive feedback samples
        * neg_ratio: ratio of negative feedback samples
        * pos_num: number of positive feedback samples
        * neg_num: number of negative feedback samples
    * video_raw_data_p1.csv, video_raw_data_p2.csv 
        * userid: anonymized user identifiers
        * pos_num: number of positive feedback samples
        * neg_num: number of negative feedback samples
        * sim_strength: strength of similarity-based matching
        * pos_ratio: ratio of positive feedback samples
        * neg_ratio: ratio of negative feedback samples
        * normalized_entropy: normalized information entropy
        * valid_activated: indicators of valid initial activation
        * delta_entropy: changes in information entropy
        * entropy_squeezed_sequence_0 ... entropy_squeezed_sequence_10: the sequence of information entropy that have been chronologically grouped

* Docker Image: https://drive.google.com/drive/folders/1Kq3ha3FKKsuqgYHdpr2KY9gYCJ8onHQ0?usp=sharing


















