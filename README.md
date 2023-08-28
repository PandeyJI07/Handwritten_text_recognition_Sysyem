# Handwritten_text_recognition_Sysyem
Handwritten Text Recognition System with User Interface using CSS
# Handwritten Text Recognition with TensorFlow

- Handwritten Text Recognition (HTR) system implemented with TensorFlow (TF) and trained on the IAM off-line HTR dataset.
- The model takes **images of single words as input** and **outputs the recognized text**.
- 3/4 of the words from the validation-set are correctly recognized, and the character error rate is around 10%.

![htr](./doc/htr.png)

## Run demo:

- Go to the `src` directory
- Run inference code:
  - Execute `python main.py` to run the model on an image of a word

The input images, and the expected outputs are shown below when the text line model is used.

![test](./data/word.png)

```
> python main.py
Init with stored values from ../model/snapshot-33
Recognized: "word"
Accuracy: 0.9513834118843079
```

## Train model on IAM dataset

### Preparing dataset

- Register for free at this [website](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
- Download `words/words.tgz`
- Download `ascii/words.txt`
- Create a directory for the dataset on your disk, and create two subdirectories: `img` and `gt`
- Put `words.txt` into the `gt` directory
- Put the content (directories `a01`, `a02`, ...) of `words.tgz` into the `img` directory

### Run training

- Delete files from `model` directory if you want to train from scratch
- Go to the `src` directory and execute `python main.py --mode train --data_dir path/to/IAM`
- The IAM dataset is split into 95% training data and 5% validation data
- If the option `--line_mode` is specified,
  the model is trained on text line images created by combining multiple word images into one
- Training stops after a fixed number of epochs without improvement

The pretrained word model was trained with this command on a GTX 1050 Ti:

```
python main.py --mode train --fast --data_dir path/to/iam  --batch_size 500 --early_stopping 15
```

And the line model with:

```
python main.py --mode train --fast --data_dir path/to/iam  --batch_size 250 --early_stopping 10
```

### Fast image loading

Loading and decoding the png image files from the disk is the bottleneck even when using only a small GPU.
The database LMDB is used to speed up image loading:

- Go to the `src` directory and run `create_lmdb.py --data_dir path/to/iam` with the IAM data directory specified
- A subfolder `lmdb` is created in the IAM data directory containing the LMDB files
- When training the model, add the command line option `--fast`

The dataset should be located on an SSD drive.
Using the `--fast` option and a GTX 1050 Ti training on single words takes around 3h with a batch size of 500.
Training on text lines takes a bit longer.

## Information about model

- It consists of 5 CNN layers, 2 RNN (LSTM) layers and the CTC loss and decoding layer.

