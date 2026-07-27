# ALER: An Active Learning Hybrid System for Efficient Entity Resolution
This repository contains the source code and datasets for the paper: "ALER: An Active Learning Hybrid System for Efficient Entity Resolution" by D. Karapiperis (International Hellenic University), L. Akritidis (International Hellenic University), Panayiotis Bozanis (International Hellenic University), and V.S. Verykios (Hellenic Open University) published in PVLDB 19(8), p. 1782-1790, 2026. It also contains the scripts for the extension paper "ALER: A Scalable Active Learning Architecture for Entity Resolution" authored by the same set of authors.

## 📖 Abstract
Entity Resolution (ER) is a critical task for data integration, yet state-of-the-art supervised deep learning models remain impractical for many real-world applications due to their need for massive, expensive-to-obtain labeled datasets. While Active Learning (AL) offers a potential solution to this "label scarcity" problem, existing  approaches introduce severe scalability bottlenecks. Specifically, they achieve high accuracy but incur prohibitive computational costs by re-training complex models from scratch or solving NP-hard selection problems in every iteration. In this paper, we propose ALER, a novel, semi-supervised pipeline designed to bridge the gap between semantic accuracy and computational scalability. ALER eliminates the training bottleneck by using a frozen bi-encoder architecture to generate static embeddings once, iteratively training a lightweight classifier on top. To address the memory bottleneck associated with large-scale candidate pools, we first generate a representative sample of the data and then use K-Means to partition this sample into semantically coherent chunks, enabling an efficient AL loop. We further propose a hybrid query strategy that combines "confused" and "confident" pairs to efficiently refine the decision boundary while correcting high-confidence errors. Extensive evaluation on benchmark datasets demonstrates that ALER outperforms state-of-the-art baselines while reducing significantly training and resolution times. On the large-scale DBLP dataset, ALER demonstrates superior scalability, completing the training loop 2.3x faster than the most efficient baseline. This advantage is even more pronounced in the resolution task, where ALER outperforms the fastest baseline by a factor of 3.8.


## 📊 Datasets

The experiments were conducted on a diverse suite of nine real-world and semi-synthetic datasets:

* **Product matching:** ABT-BUY, AMAZON-WALMART, AMAZON-GOOGLE
* **Bibliographic matching:** ACM-DBLP, SCHOLAR-DBLP
* **Movies matching:** IMDB-DBPEDIA
* **Resaturants matching:** Restaurants
* **Large-Scale Synthetic:** DBLP, VOTERS

All experiments were run using the `MiniLM-L6-v2` model for embeddings generation.

## ⚙️ Setup and Installation

The implementations rely on several key open-source libraries. You can install them using pip:
 ```bash
    pip install -r requirements.txt
 ```

   
## ▶️ Running the Experiments

The repository is structured to allow for easy replication of the results presented in the paper. Scripts suffixed with "_par" implement the PAR-ALER pipeline, whereas those suffixed with "_wrs" correspond to the WRS-ALER pipeline.

**Running a Single Experiment:** You can run the evaluation for a specific dataset using the corresponding script. For example:
```bash
   python abt_par.py  
```
uses the ABT-BUY paired dataset.
The large scale datasets—DBLP and VOTERS—can be found [here](https://drive.google.com/drive/folders/1IM9Ot8zpx11YcwXe_4ZTVeEx6wFaHiOo?usp=sharing).


