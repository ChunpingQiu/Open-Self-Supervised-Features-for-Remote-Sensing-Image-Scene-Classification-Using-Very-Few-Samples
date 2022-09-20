# Setup
Big models, large datasets, and self-supervised learning (SSL) have recently gained substantial research interest due to their potential to alleviate our reliance on annotations. Considering the current high generalization ability of self-supervised models in literature, we explore in the letter how helpful SSL can be for a crucial task in remote sensing (RS), image scene classification, when forced to rely on only a few labeled samples. We proposed a simple prototype-based classification procedure without training and fine-tuning, which uses open self-supervised features from the Contrastive Language-Image Pre-Training (CLIP). 

Please follow the official setup of CLIP here: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)

# Folder Structure
  ```
  pytorch-template/
  ├── euro_run4clipmodels1.sh - the main file 
  ├── eurosat_fs.py - main file
  │
  │
  │
  ├── util/ - (temporary) results folder
  │   ├── similarity.py - functions to calculate feature distance
  │   ├── dataloder_path.py - dataloader
  │   ├── mis.py - select bands from multi-spectral bands and apply max-min normalization
  │   └── ...
  │   
  ├── ...  
 ```     

# Data

To reproduce the results in Table 1, please download data from: [https://madm.dfki.de/files/sentinel/EuroSATallBands.zip](https://madm.dfki.de/files/sentinel/EuroSATallBands.zip). Then put the folder "EuroSATallBands" into the root folder.

In the folder "experiment_split," there is a random split of query and support for each class. The test results are always reported on the query set, while the few labels used can only be from the support set.
```
/root 
|-- experiment_split
     |-- EuroSATallBands_query_q9_0_AnnualCrop.txt
     |-- EuroSATallBands_support_p1_0_AnnualCrop.txt
     |-- ...         
|-- EuroSATallBands
     |-- AnnualCrop
         |-- AnnualCrop_1.tif
         |-- ...
     |-- ...    
```

After setting up, run: ```bash euro_run4clipmodels1.sh``` and results will be saved in the folder "/root/acc_github.csv".

# Cite:

If you use this code for your research, please cite:

```
```