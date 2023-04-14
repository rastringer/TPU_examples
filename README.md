# TPU Examples 
Example notebooks for machine learning on TPUs

Why use TPUs?

![TPU Architecture](https://cloud.google.com/static/tpu/docs/images/image4_5pfb45w.gif)

Google designed Cloud TPUs as a matrix processor focused making training and inference of neural networks faster, and more power efficient. The TPU is built for massive matrix processing, and its systolic array architecture assigns thousands of interconnected multiply-accumulators to the task. Cloud TPU v3, contain two systolic arrays of 128 x 128 ALUs, on a single processor. For workloads bound by matmul, TPU can generate significant efficiencies.

### Getting started

(If you're already up and running with a TPU VM, skip ahead to the Notebooks section).

The easiest way to experiment with TPUs is to do so for free on Colab. Simply switch the `Runtime` to `TPU`.
For more demanding workloads or experimentation, the next step is to set up a TPU VM on Google Cloud. The VM has advantages over the TPU Node, namely the ability to SSH into the VM with root access, making available training logs, debugging etc.

Prerequisite: a Google Cloud project.

There are several ways to run the commands below. 

*Vertex AI: If you're running notebooks from within Vertex Workbench, simply open a terminal within the notebook instance and run the commands.  
* Local: [Download](https://cloud.google.com/sdk/docs/install) the gcloud SDK, or open a shell from within Cloud console.
* Compute Engine VM: run the commands to set up a TPU VM.

Setting up the VM
First, run the following command to enable the TPU API, and set your user and project configuration:
```
gcloud services enable tpu.googleapis.com
gcloud config set account your-email-account
gcloud config set project your-project
```

### Create the TPU VM

For more information, an extensive guide can be found [here](https://cloud.google.com/tpu/docs/run-calculation-pytorch#create-vm).

```
gcloud compute tpus tpu-vm create tpu-name \
  --zone=zone \
  --accelerator-type=2-8 \
  --version=tpu-vm-pt-1.11
```

To see a list of versions (such as TensorFlow, other PyTorch versions), replace with zone with the zone of yoru project (eg us-central1-b) and run:

```
gcloud compute tpus tpu-vm versions list --zone <ZONE>
```

For all TPU types, the version is followed by the number of TensorCores (e.g., 8, 32, 128). For example, --accelerator-type=v2-8 specifies a TPU v2 with 8 TensorCores and v3-1024 specifies a v3 TPU with 1024 TensorCores (a slice of a v3 Pod).

For the notebooks in this repository, we used a v2-8 TPU.

### Connecting to a TPU VM

From one of the options above (Workbench, local terminal, Compute Engine VM etc), adjust the VM name and zone placeholders:

```
gcloud compute tpus tpu-vm ssh <your-tpu-vm-name> --zone <your-zone>
```

#### Configure the Torch-XLA environment.

By checking `Compute Engine` -> `TPUs` in the GCP console you'll find the TPU VM ip address. Copy this into the command below:

```
export TPU_IP_ADDRESS=ip-address; \
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
```

Paste and run the above command in the terminal when logged into the VM.

#### Set TPU runtime configuration

GCP recommends PJRT in the absence of reason to use XRT.

```
export PJRT_DEVICE=TPU
```

#### Test 

Perform a simple calculation.
Copy the following code into a file named test.py:

```
import torch
import torch_xla.core.xla_model as xm

dev = xm.xla_device()
t1 = torch.randn(3,3,device=dev)
t2 = torch.randn(3,3,device=dev)
print(t1 + t2)
```

And run:

```
python3 test.py
```

The result should look like this:
```
tensor([[-0.2121,  1.5589, -0.6951],
        [-0.7886, -0.2022,  0.9242],
        [ 0.8555, -1.8698,  1.4333]], device='xla:1')
```

### Notebooks

(With thanks to Meta for the colabs [here](https://github.com/pytorch/xla/tree/master/contrib/colab), a mix of which I refactored for these examples).

Now we're ready to run some notebooks. Learn how the basics of the pytorch_xla library while training a model on a single TPU core in [pytorch_resnet_singlecore.ipynb](https://github.com/rastringer/TPU_examples/blob/main/pytorch_mnist_singlecore.ipynb).
Move on to writing a multi-core training job in [pytorch_resnet_multicore.ipynb](https://github.com/rastringer/TPU_examples/blob/main/pytorch_mnist_multicorecore.ipynb).
