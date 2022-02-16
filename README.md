# DL-mo
## A benchmark study of deep learning based multi-omics data fusion methods for cancer
***
![Multi-Omics](/img/Multi-Omics.jpg "Multi-Omics")  
We here compare the performances of seven deep learning methods in three contexts: 
1. samples clustering from multi-omics simulated data
2. ability to identify factors associated with survival or clinical annotations and metagenes associated with biological annotations (Reactome, GO, Hallmarks) in bulk multi-omics TCGA data from 10 cancer types.
3. cells clustering based on scRNA-seq and scATAC-seq data from three cell lines.       

We use `python` and `R` to code the programs.   
***
## Seven deep learning methods
* [lfAE](./python-scripts/runCancerAE2.py)
* [Integrative NMF (intNMF)](https://cran.r-project.org/web/packages/IntNMF/index.html) 
* [Joint and individual variation explained (JIVE)](https://cran.r-project.org/web/packages/r.jive/index.html) 
* [Multiple co-inertia analysis (MCIA)](https://bioconductor.org/packages/release/bioc/html/omicade4.html) 
* [Multi-Omics Factor Analysis (MOFA)](https://github.com/bioFAM/MOFA)
* [Multi-Study Factor Analysis (MSFA)](https://github.com/rdevito/MSFA) 
* [Regularized Generalized Canonical Correlation Analysis (RGCCA)](https://cran.r-project.org/web/packages/RGCCA/index.html) 
* [matrix-tri-factorization (scikit-fusion)](https://github.com/marinkaz/scikit-fusion) 
* [tensorial Independent Component Analysis (tICA)](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-018-1455-8)
***
## Install the R software environment
Use conda to create a new environment: `conda create -n momix -c conda-forge -c bioconda -c lcantini momix r-irkernel`
***
## Install the python software environment
You need to build a virtual environment for python.    
You need to install the following main libraries: `Python==3.7.0,Tensorflow==1.15.0, scikit-learn==0.20.0, Jupyter==1.0.0`.

