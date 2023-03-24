#!/usr/bin/env python
# coding: utf-8

# ### Multi-core Training FashionMNIST on Cloud TPU
# 
# Now we are familiar with adjusting code to run on a single TPU core, we will walkthrough expanding the data loading, model training and evaluation across all eight cores of the TPU. 
# Multi-core operations will predictibly be faster and able to handle larger loads (datasets, batch sizes).
# 
# Beyond setting `device = xm.xla_device()`, below are some of the torch_xla functions we will use:
# * Data is only being downloaded once by a master worker by checking `xm.is_master_ordinal()`.
# * Subsets of the data are being loaded efficiently across all processes using `DistributedSampler`.
# * `xm.optimizer_step(optimizer)` to consolidates gradients between cores during training.  
# * We run the training using `xmp.spawn()` to enable replication across multiple processes.

# ## TODO: add launch as Colab and pip installs, wheel etc.

# In[ ]:


import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import time
import sys

def trainer():
    
    # Random seed for initialization
    torch.manual_seed(flags['seed'])
    # Sets device to Cloud TPU core
    device = xm.xla_device()

    # Normalization for dataloader 
    # TorchVision models require RGB (3 x H x W) images
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
    to_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
    resize = transforms.Resize((224, 224))
    my_transform = transforms.Compose([resize, to_rgb, transforms.ToTensor(), normalize])

    # Checks if current process is the master ordinal (0)
    # Other workers wait for master to complete download
    if not xm.is_master_ordinal():
        xm.rendezvous('download_only_once')

    train_dataset = datasets.FashionMNIST(
        "/tmp/fashionmnist",
        train=True,
        download=True,
        transform=my_transform)

    test_dataset = datasets.FashionMNIST(
        "/tmp/fashionmnist",
        train=False,
        download=True,
        transform=my_transform)

    if xm.is_master_ordinal():
        xm.rendezvous('download_only_once')

    # DistributedSampler restricts data loading to a subset of the dataset
    # for each process
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False)

    # Dataloaders load data in batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=flags['batch_size'],
        sampler=train_sampler,
        num_workers=flags['num_workers'],
        drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=flags['batch_size'],
        sampler=test_sampler,
        shuffle=False,
        num_workers=flags['num_workers'],
        drop_last=True)

    # Model, optimizer, and loss function 
    # We use a resnet18 model for the 10 classes of the 
    # FashionMNIST dataset
    # Each process has its own copy of the model
    net = torchvision.models.resnet18(num_classes=10).to(device).train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    def train(loader):
        train_start = time.time()
        for batch_num, batch in enumerate(loader):
            data, targets = batch
            # Get prediction
            output = net(data)
            # Loss function
            loss = loss_fn(output, targets)
            # Update model
            optimizer.zero_grad()
            loss.backward()
            # xm.optimizer_step(optimizer) consolidates the gradients between cores
            # and issues the XLA device step computation.
            xm.optimizer_step(optimizer)

        elapsed_train_time = time.time() - train_start
        print(f"Finished training. Train time was: {elapsed_train_time}")

    def test(loader):
        net.eval()
        eval_start = time.time()
        with torch.no_grad():
            num_correct = 0
            total_guesses = 0

        for batch_num, batch in enumerate(loader):
            data, targets = batch
            output = net(data)
            best_guesses = torch.argmax(output, 1)
            # Calculate accuracy
            num_correct += torch.eq(targets, best_guesses).sum().item()
            total_guesses += flags['batch_size']

        accuracy = 100.0 * num_correct / total_guesses
        elapsed_eval_time = time.time() - eval_start
        print(f"Finished evaluation. Evaluation time was: {elapsed_eval_time}")
        print(f"Guessed {num_correct} of {total_guesses} correctly for {accuracy} % accuracy.")

        return accuracy, data, targets
    
    accuracy = 0.0
    data, targets = None, None
    
    # Loop through epochs, calling the train and eval functions above
    for epoch in range(flags['num_epochs']):
        # ParallelLoader wraps a DataLoader with background data upload
        para_train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
        train(para_train_loader)
        xm.master_print("Finished training epoch {}".format(epoch))
        para_test_loader = pl.ParallelLoader(test_loader, [device]).per_device_loader(device)
        accuracy, data, pred, targets  = test(para_test_loader)

    return accuracy, data, targets
    


# In[ ]:


# Training parameters
flags = {}
flags['batch_size'] = 32
flags['num_workers'] = 8
flags['num_epochs'] = 1
flags['seed'] = 1


# The *multiprocess function* runs the trainer, takes the index of each process and the flags defined above 

# In[ ]:


def _mp_fn(index, flags):
    torch.set_default_tensor_type('torch.FloatTensor')
    accuracy, data, targets = trainer()

# Enable replication across multiple processes     
xmp.spawn(_mp_fn, args=(flags,), nprocs=8, start_method='fork')


# In this notebook, we trained and evaluated a ResNet18 model on the FashionMNIST dataset.
# We adjusted the code from the single-core example to include various data and model parallelization features that should have sped up our operations as they were distributed across 8 TPU cores. We can check the time against the single-core example, and predictably should see that the epochs train faster.
# 
# Try experimenting with the `batch_size` - how high can it go before the training starts to slow down?
# 
# We can also experiment with different model architectures, such as EfficientNet or ConvNeXt, just by changing this line:
# 
# ```
# net = torchvision.models.resnet18(num_classes=10).to(device).train()
# 
# ```
# 
# to swap `resnet18` to others listed [here](https://pytorch.org/vision/stable/models.html#classification).

# To save this notebook as a Python file, run the following commmand from a terminal in the same folder:
# ```
# jupyter nbconvert --to script pytorch_resnet_multicore.ipynb
# ```
# 
# We can then load the .py file into a Cloud Storage bucket, then import it via `gsutil` into the TPU VM.
# ```
# gsutil cp pytorch_resnet_multicore.ipynb gs://<your-unique-bucket-name>
# ```

# In[ ]:




