<div align="center">

# Deep Learning on HPC

</div>

# Objective:
The goal is to run a deep learning task on containers in an HPC environment using Slurm. 

# BackGround:
Slurm is an alternative compute environment for Kubernetes. The advantage of the former is the native job scheduling capability, and is a key player in the HPC industry. 

In this work, we create an HPC environment on AWS using AWS ParallelCluster. We then fine-tune a Llama 3.1 (8B) model using SFT. The task itself will run inside a container.


## Setting up Dev Environment

We install the ParallelCluster CLI tool via

```bash
pip install "aws-parallelcluster"
```

## Provisioning Cluster


To provision the cluster, you can generate the configuration via :
```bash
pcluster configure --config config.yaml
```

The naive configuration may not suit for most purposes - especially deep learning.

A modified version of the configuration can be found [here](./config.yaml)
```bash
pcluster create-cluster --cluster-name dlhpc --cluster-configuration config.yaml
```

This confguration
- specifies the VPC/subnet to fasten provisioning
- installs pyxis and enroot (Nvidia's alternative for Singularity/Apptainer) to run containers

You can see the status of the cluster via:
```bash
pcluster list-clusters
```

Do not SSH before the cluster/cloudformation reads Ready/Created even if HeadNode is ready! This is because the Slurm commands or other tools might not be available.

## Login Into Cluster (Head Node)
You can login to the head node via
```bash
pcluster ssh --cluster-name dlhpc -i ~/.ssh/<your_key>.pem
```

Inside the head-node, you can ssh to other compute nodes via
```bash
ssh <name_of_compute_node>
```

This might be useful for say monitoring GPU usage or other debugging issues.

To get the name of the compute node, you can use the `sinfo` command.

The `sinfo` command will list the status of all of the compute nodes.

## Preparing Container
We need to provision the container that will run the training code. Having a container is not a pre-requisite since we can install our Python dependencies directly on the host itself. While this practice might work for our AWS environment, it might not be recommended in a shared HPC environment.

We will assume that a PyTorch image is already built and hosted on an image repository e.g. DockerHub, ghcr, etc. A sample image can be found in the `docker_image` folder. (Since we are on AWS, it is better to use a PyTorch image built by/for AWS).

```bash
enroot import -o llm.sqsh docker://<username>/<image>:<tag>
```

This command must either be run inside the home directory (which is shared among all nodes) or on a shared file system so that the image is accessible across the entire cluster. For large model training, it is better to create FSx for Lustre as the shared file system due to its high-performance.

Also, since our code will
- push model to hub &
- log metrics to WandB

  we can install the huggingface-cli and wandb on our head node via
```bash
pip install 'huggingface_hub[cli]' wandb
```

and then log in via the CLIs. This will save the credentials somewhere within the home directory, which will be accessible not only to the compute nodes but also to the containers.

## Training Script
Create your code and save it in one of the shared path. For our task, we can use the default SFT script provided by HuggingFace.

```bash
wget https://raw.githubusercontent.com/huggingface/trl/main/examples/scripts/sft.py
```

## Job Submission
 You have to create a batch script - a sample shown [here](./task.sbatch) in order to submit a batch job.
 
Among other things, it
- specifies the training container via the `--container-image` argument
- mounts the home directory inside the container via `--container-mount-home`
- sets up the working directory inside the container to `--container-workdir` inside the container image.

If you need to run tasks before (after) the training job (e.g. to install python packages), you can append the `--task-prolog` (`--task-epilog`) command to the `srun` command e.g. `--task-prolog="setup.sh"`

To submit a job, run
```bash
sbatch task.sbatch
```


## Job Status

You can see the job queue and the status of the jobs via 
```bash
squeue
```

To view the output generate by the runs, you can see the \*.err and \*.out files in the working directory.


## Delete Cluster
Finally, you can delete the cluster and all the resources via

```bash
pcluster delete-cluster --cluster-name dlhpc
```

