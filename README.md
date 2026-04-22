# AI-based-masking-of-bones
AI based masking of bones in clinical CT images

# 3D Segmentation Pipeline (VNet)

This repository provides a configurable pipeline for **3D image segmentation** using a VNet-based architecture.  
The workflow is fully controlled via JSON configuration files.

---

## Overview

The pipeline consists of three main stages:

1. **Data Preparation** – preprocess and split input data  
2. **Training** – train the segmentation model  
3. **Testing** – run a test on a trained model  

Each stage is controlled by its own configuration file.

---

## 📁 Project Structure
```
AI-based-masking-of-bones/
├── configs/                # Folder for all config files
│   ├── generator.json      # Config for generating new images for training/testing from the original
│   ├── main_config.json    # Config for training the model
│   └── test_config.json    # Config if extra testing is wanted after training has been finished
├── Callbacks.py            # Python code to define all Callbacks which are used to save data during the training or testing process
├── graphing.Rmd            # R code to draw graphs from the training and testing output
├── Helper.py               # Python code to define helper functions needed by the training and testing process
├── Image8.py               # Python code to define an Image with all its data (including the voxel array)
├── ImageGenerator.py       # Python code to define functions to generate the images for training and testing from the original
├── main.py                 # general starting point of the model
├── Patching.py             # Python code to define the necessary functions which are used by training/testing for the patching of the whole image
├── RandomPatchCallbacks.py # Python code to define all Callbacks which are used to save data during the training or testing process. This python file is used only if random sample extraction is activated
├── README.md               # Readme to illustrate a general documentation
├── requirements.txt        # Requirements needed to run the code, includes all pip installations
├── Testing.py              # Python code to define all functions needed for the testing process
├── Training.py             # Python code to define all functions needed for the training process
├── VNet10.py               # Python code to define the archtecture of the used VNet
```

---

## ⚙️ Configuration Files

### 1. Data Configuration (`generator.json`)

Controls preprocessing and dataset splitting.

```json
{
  "Directories": {
    "BaseDirectory": null,
    "InputDirectory": ..., // Directory name of the original training directory (string)
    "TrainDirectory": ..., // Directory name, where the training images are saved (string)
    "TestDirectory": ..., // Directory name, where the test images are saved (string)
    "PredDirectory": ..., // Directory name, where the prediciton images are saved (string)
  },
  "Settings": {
    "Random": ..., // Defines if random patching should be actived. This means, that random patch samples for training in a new directory are generated from the train dataset. The train dataset must have been created with random = false before to generate with true. Needed parameters are defined in RandomGeneratorSpecifics (boolean)
    "Downsampling": ..., // Definition of a downsampling factor if wanted, should be 1 if not used (int)
    "OrderName": ..., // Definition how voxels should be interpolated when downsampling is active (string)
    "ZeroPaddingTo": ..., // How much the image should be zero-padded, should align with the patch_size used in the training and testing setup e.g.: [1536, 640, 512] (array)
    "PatchSize": ..., // Patch size, defining the size of one patch that traverses the total image e.g: [64, 64, 64] (array)
    "TrainDataSize": ..., // Size of the training set (int 0-1)
    "Seed": ... // Seed that should be used in the project for reproducibility (int)
  },
  "RandomGeneratorSpecifics": {
        "ZeroMaxPercentage": ..., // Defines the max percentage of background voxels in an image to be accepted. This is needed for FgPatchesPercentage amount of the images (float 0-1)
        "RetryAmount": ..., // Defines the number retries that are done if the cut out sample does have more zeros in the image than the percentage defined in ZeroMaxPercentage (int), If retryAmount is reached --> a log message shows up that image"number" was not created
        "FgPatchesPercentage": ..., // Defines the percentage for the amount of sample images where there have to be less zeros in the image than ZeroMaxPercentage defines
        "TrainSamples": ... // Defines the number of train samples that should be created from the original train dataset in the new traing dataset directory
    }
}
```
### 2. Training Configuration (`main_config.json`)

Controls training and inbetween testing if wanted.

```json
{
  "ProjectName": ..., // Project name configuration (string)
  "Settings": {
    "ModelName": ..., // Name of the model, used to generate directories (string)
    "SegmentationClasses": ..., // Segmentation classes of the output e.g.: [0,1,2] (array)
    "InputChannels": ..., // Should be 1 as we are working with black and white images (int)
    "BatchSize": ..., // Batch size, defining how many images are processed at once
    "Testing": ..., // Testing which is done during the training process either full/end/not (string)
    "SummaryLogDir": ..., // Directory where the results is saved like csvs and confusion matrix (string)
    "Network": {
      "Epochs": ..., // Training epochs of the netowrk (int)
      "Dropout": ..., // Should dropout be used (boolean)
      "DropoutRate": ..., // Defines the dropoutrat if "Dropout" is true (float 0-1)
      "Initializer": ..., // Defines the initializer either normal/xavier/he/orthogonal (string)
      "Loss": ..., // Defines the loss either DiceLossCategoricalCrossentropyLoss/CategoricalCrossentropyLoss/DiceLoss (string)
      "ActivationFunction": ..., // Defines the activation function either prelu/lrelu/sigmoid/tanh/softmax/relu (string)
      "NetworkSavingRange": ..., // The epoch range in which the network is saved. If equal to epochs then at the end of every patch (int)
      "NetworkSavingPath": ..., // Directory where the network is saved as keras file (string)
      "ExistingModel": ..., // If training should be resumed from an exisiting model a path needs to be added here (string)
      "Optimizer": {
        "Name": ..., // Defines the optimizer either SGD/Adam/RMSprop/Adagrad (string)
        "LearningRate": ..., // Learning rate used in the model training (int 0-1)
      }
    }
  }
}
```

### 3. Testing Configuration (`test_config.json`)

Controls testing, after training has been finished.

```json
{
  "ProjectName": ..., // Project name configuration (string)
  "Settings": {
    "ModelName": ..., // Name of the model, used to generate directories (string)
    "SegmentationClasses": ..., // segmentation classes of the output e.g.: [0,1,2] (array)
    "PatchSize": ..., // Patch size, defining the size of one patch that traverses the total image e.g: [64, 64, 64] (array)
    "SummaryLogDir": ..., // Directory where the results is saved like csvs and confusion matrix (string)
    "TestFrequency": ..., // Defines after which patch testing should be analyzed
    "Network": {
      "ActivationFunction": ..., // Defines the activation function either prelu/lrelu/sigmoid/tanh/softmax/relu (string)
      "NetworkSavingPath": ..., // The epoch range in which the network is saved. If equal to epochs then at the end of every patch (int)
      "ExistingModel":  ..., // Defines the model the tests should be executed with (string)
      "Optimizer": {
        "Name": ..., // Defines the optimizer either SGD/Adam/RMSprop/Adagrad (string)
      }
    }
  }
}
```

