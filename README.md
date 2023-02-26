# LAF-DBSCAN
This is the codebase for LAF-DBSCAN and LAF-DBSCAN++ in our EDBT paper "[Learned Accelerator Framework for Angular-Distance-Based High-Dimensional DBSCAN
](https://arxiv.org/abs/2302.03136)"

### 1. Requirements
To compile the library, please make sure you have Cython installed. You can install it via this command with `pip`:
```
pip install Cython
```

### 2. Installation
Install the dependencies:
```
pip install -r requirements.txt
```
Then compile the Cython backend:
```
make laf
```
Finally copy the generated library file (the `.so` file located in the directory `build/lib.*/`) to your python package installation directory. You can use this command to check the package directory:
```
python -m site
```
An example of the python package directory is `/usr/local/lib/python3.8/dist-packages/`, and this could be different in your computer. 

### 3. Codebase structure
(1) `ds/`: including the demo data used in `evaluation.ipynb`, which is the original testing set of the `MS-50k` dataset in our paper. Each row in the numpy array is a data point.

(2) `prediction/`: including the predictions made by the cardinality estimator (i.e., RMI in our paper), with shape (n_samples, 2), where the first column includes the estimated cardinality for each data point, while the second column is placeholders that are all zeros.  

(3) `laf.pyx`: the main source code file. 

(4) `*.h`: facility files for `laf.pyx`.

(5) `evaluation.ipynb`: a demo notebook, as well as a mini-version of our evaluation script used by the paper.


### 4. Usage
(1) `DBSCAN.fit_predict`: running the standard/naive DBSCAN

(2) `DBSCAN.fit_predict_with_card_est_with_postproc`: running LAF-DBSCAN

(3) `DBSCANPP.fit_predict`: running DBSCAN++, which is one of the evaluation baselines and the base for DBSCAN/LAF-DBSCAN/LAF-DBSCAN++ implementations.

(4) `DBSCANPP.fit_predict_with_card_est_with_postproc`: running LAF-DBSCAN++


### 5. Demo
Please see `evaluation.ipynb` for a demo about how to evaluate the four methods above. This notebook is also a mini-version of the evaluation script we used in our paper. You can reproduce our experiment results for MS-50k by running the notebook. Note that DBSCAN and LAF-DBSCAN++ both include randomized parts, so the ARI and AMI scores may not be exactly reproducible for them.   

### 6. Developer
Please refer to our paper and the comments in source code if you need to further develop this codebase. The main source file is `laf.pyx`.

### Note:
(1) Function `updateNonCorePointIfNerghbored` in `laf.pyx` corresponds to the `UpdatePartialNeighbors` function in our paper.

(2) There is no individual function for postprocessing, instead, the postprocessing is included in the LAF-DBSCAN and LAF-DBSCAN++ implementations. 

(3) Equivalent Euclidean distance eps (instead of the original cosine distance eps) is passed into the main methods in `evaluation.ipynb` as the DBSCAN++ codebase only supports Euclidean distance metric.

## Citation

If you use this codebase, or otherwise found our work valuable, please cite:

```
@misc{https://doi.org/10.48550/arxiv.2302.03136,
  doi = {10.48550/ARXIV.2302.03136},
  url = {https://arxiv.org/abs/2302.03136},
  author = {Wang, Yifan and Wang, Daisy Zhe},
  keywords = {Information Retrieval (cs.IR), Databases (cs.DB), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Learned Accelerator Framework for Angular-Distance-Based High-Dimensional DBSCAN},
  publisher = {arXiv},
  year = {2023}, 
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
