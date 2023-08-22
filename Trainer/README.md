# Model Training

* [**Introduction**](#introduction)
* [**Trainer**](#trainer)

## <a name="introduction">Introduction

The subject training script is used to process data and subject deep learning model. After training process is completed, resources are saved into some persistent storage.

## <a name="trainer">Trainer

Training on the subject data can be automatically done and handled by training pipeline. This [script][trainer] takes following environment variables as input:

```bash
export saveDir=/output          # Absolute path of output directory
export rootDir=/data            # Absolute path of model input data
export batchSize=10             # Batch size for model training
export epochs=10                # Number of epochs for model training
export workers=8                # Number of workers for model training
```

Also, the training data, directory structure should be as follows:

```text
/data
    |_ images
    |  |_ train
    |  |  |_ 000000.jpg
    |  |  |_ 000001.jpg
    |  |  |_ ...
    |  |
    |  |_ val
    |     |_ 000000.jpg
    |     |_ 000001.jpg
    |     |_ ...
    |
    |_ images
```

[trainer]: ./trainer.py
