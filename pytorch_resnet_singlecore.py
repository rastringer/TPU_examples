#!/usr/bin/env python
# coding: utf-8

# ### Single-core Training FashionMNIST on Cloud TPU
# 
# In this notebook we explore the simple code changes necessary to train and evaluate a model on a TPU.
# 
# Some of the some of the torch_xla functions we will use include:
# * `device = xm.xla_device()` to set the device to TPU.
# * `xm.is_master_ordinal()` to make sure the current process is the master ordinal (0).
# * `xm.optimizer_step()` for evaluation each time gradients are updated.

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

    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    test_sampler = torch.utils.data.RandomSampler(test_dataset)

    # Dataloaders load data in batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=flags['batch_size'],
        sampler=train_sampler
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=flags['batch_size'],
        sampler=test_sampler
    )

    # Model, optimizer, and loss function 
    # We use a resnet18 model for the 10 classes of the FashionMNIST dataset
    net = torchvision.models.resnet18(num_classes=10).to(device).train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    def train():
        train_start = time.time()
        for data, targets in iter(train_loader):
            # Sends data and targets to device
            data = data.to(device)
            targets = targets.to(device)
            # Get prediction
            output = net(data)
            # Loss function
            loss = loss_fn(output, targets)
            # Update model
            optimizer.zero_grad()
            loss.backward()
            # The 'barrier' here forces evaluation each time gradients are 
            # updates, keeping graphs manageable.
            xm.optimizer_step(optimizer, barrier=True)

        elapsed_train_time = time.time() - train_start
        print(f"Finished training. Train time was: {elapsed_train_time}")

    def test():
        net.eval()
        eval_start = time.time()
        with torch.no_grad():
            num_correct = 0
            total_guesses = 0

        for data, targets in iter(test_loader):
            # Sends data and targets to device
            data = data.to(device)
            targets = targets.to(device)
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
        tic = time.time()
        train()
        xm.master_print("Finished training epoch {}".format(epoch))
        toc = time.time()
        accuracy, data, targets  = test()
        # Calculate training time per epoch
        epoch_time = round(toc-tic, 2)
        print(f"Epoch trained on a single TPU in {epoch_time} seconds")

    return accuracy, data, targets
    


# In[1]:


# Training parameters
flags = {}
flags['batch_size'] = 8
flags['num_epochs'] = 1
flags['seed'] = 1


# In[ ]:


trainer()


# In this notebook, we trained and evaluated a ResNet18 model on the FashionMNIST dataset.
# We used various functions from the pytorch_XLA library to run on a Cloud TPU.
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
#                                            
# The next notebook, `pytorch_resnet_multicore.ipynb`, shows how we can adapt our code to run on multiple TPU cores, which should
# make training more efficient.

# To save this notebook as a Python file, run the following commmand from a terminal in the same folder:
# ```
# jupyter nbconvert --to script pytorch_resnet_singlecore.ipynb
# ```
# 
# We can then load the .py file into a Cloud Storage bucket, then import it via `gsutil` into the TPU VM.
# ```
# gsutil cp pytorch_resnet_singlecore.ipynb gs://<your-unique-bucket-name>
# ```

# In[ ]:




