# nemotronL40

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         nemotronl40 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── nemotronl40   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes nemotronl40 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

# How to Download the Weights
1. Run this command
$ bsub -Is -P acc_EHR_ML -q premium -n 4 -R "rusage[mem=8000]" -W 10:00 bash
2. Run this command
$ tmux new -s nemotron-FP8-download
3. Set the path
$ export HF_HOME=/sc/arion/projects/EHR_ML/lia38/nemotronL40/.cache
4. Set up the environment
$ ml purge && ml proxies && ml load miniforge3/26.1.1-3 cuda/12.8.0 
5. Activate conda environment
$ conda activate /sc/arion/projects/EHR_ML/lia38/nemotronL40/.venvs/nemotron-env
6. Download the weights from huggingface
$ hf download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8
7. Close the LSF job run

# Note that your cache folder might reach disk quota because of vLLM. We can solve this by using Symlinks
1. Rename your current cache directory
$ mv ~/.cache ~/.cache_old
2. Create a new cache in the folder you want it to be in
$ mkdir -p /sc/arion/work/lia38/.cache
3. Create a symbolic link between the new cache and the original cache
$ ln -s /sc/arion/work/lia38/.cache ~/.cache
4. This vertifies if it worked. Should see something like "lrwxrwxrwx"  where l l indicates it's a symbolic link and the -> shows where it points.
$ ls -ld ~/.cache

# How to run the code?
1. Go to the jobs folder
$ cd /sc/arion/projects/EHR_ML/lia38/nemotronL40/jobs
2. Start a Ray cluster. If there is an port error, you have to change PORT number in the file. In your /sc/arion/scratch/<Your User Id>/ folder there will be a file called "ray_address" that consist the address of the Ray cluster. You can view the status of the cluster by doing $ ray status --address=<ray_address>
$ bsub < start_ray_cluster.lsf
3. add_ray_worker.lsf adds a Ray worker to the cluster with the GPU you want. This code only works for 1 B200 GPU and 4 L40s GPUs. Change the bsub options accordingly. Then, add as many workers as you want by resubmitting the bsub. It is advisable to use an even number of workers
$ bsub < add_ray_worker.lsf
4. Lastly, type the command below to run the code in main.py
$ touch /sc/arion/scratch/<Your User Id>/run_code
5. To ensure the code is running, monitor the cluster by doing
$ bpeek <Job ID of start_ray_cluster.lsf>

# How to use Ray Dashboard?
1. Get your Ray address, which is in the format of <IP:PORT>, e.g. 172.28.19.213:25052
2. In a separate terminal,
$ ssh -L 25053:<IP>:25053 <Mount Sinai ID>@minerva.hpc.mssm.edu
3. Open http://localhost:25053/ on your web browser. 

You can also open the Ray Dashboard in VSCode instead
4. Go to the Ports tab at the bottom panel of VS Code (next to the Terminal tab).
5. Click Add Port and enter the port of the dashboard port (in this case, 25053)
6. Press Ctrl + Shift + P (or Cmd + Shift + P on Mac) to open the Command Palette.
7. Type and select: Browser: Open Integrated Browser.
8. Enter the http://localhost:<PORT>
