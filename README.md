# Evidence_Gap_Clinical_Trials

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Project Organization

```
Evidence_Gap_Clinical_Trials/
├── README.md
├── .cache/                     <- Where the weights of the Nemotron lies
├── data/
│   ├── RxClass/
│   ├── clinical_trials/
│   └── responses/
├── jobs/
│   ├── start_ray_cluster.lsf/  <- Submit this job to start a Ray cluster
│   └── add_ray_worker.lsf/     <- Submit this job as many times as you like to add Ray workers to the Ray cluster
├── main.py                     <- Main pipeline script that processes patient notes, fetches trials, and evaluates relevance.
└── tools/
    ├── ClinicalTrialGov.py     <- Fetches, extracts, and downloads clinical trial data from the ClinicalTrials.gov API.
    ├── Ray.py                  <- Handles Ray cluster initialization and actor pool creation for the Nemotron models.
    ├── RxClass.py              <- Interfaces with the RxNav API to retrieve drug classes and their members.
    ├── logger.py               <- Handles logging pipeline stages and dumping raw outputs to disk.
    ├── prompts.py              <- Stores the prompt templates for RxClass selection, query generation, and evaluation.
    └── tasks.py                <- Contains the core logic for batching Nemotron chats and executing pipeline tasks.
```

--------

# How to create the conda environment
1. cd to the project directory
$ cd Evidence_Gap_Clinical_Trials
2. Run the command below
$ conda env create -f environment.yml -y

# How to Download the Weights
1. Run this command
$ bsub -Is -P acc_EHR_ML -q premium -n 4 -R "rusage[mem=8000]" -W 10:00 bash
2. Run this command
$ tmux new -s nemotron-FP8-download
3. Set the path
$ export HF_HOME=/sc/arion/projects/EHR_ML/$USER/Evidence_Gap_Clinical_Trials/.cache
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
2. In a separate terminal or Git Bash,
$ ssh -L 25053:<IP>:25053 <Mount Sinai ID>@minerva.hpc.mssm.edu
3. Open http://localhost:25053/ on your web browser. 

You can also open the Ray Dashboard in VSCode instead
4. Go to the Ports tab at the bottom panel of VS Code (next to the Terminal tab).
5. Click Add Port and enter the port of the dashboard port (in this case, 25053)
6. Press Ctrl + Shift + P (or Cmd + Shift + P on Mac) to open the Command Palette.
7. Type and select: Browser: Open Integrated Browser.
8. Enter the http://localhost:<PORT>
