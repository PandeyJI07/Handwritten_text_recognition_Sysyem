# MODEL TRAINING:

## Modules used:

- argparse - This is a standard Python library used for parsing command-line arguments.
- pickle - This is a standard Python library used for serializing and deserializing Python objects.
- path - This is the Path library, which provides an object-oriented approach to working with file and directory paths. It is not a standard Python library and needs to be installed separately.
- NumPy : NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
- cv2 : cv2 is a Python library that is used to perform computer vision and image processing operations. It is a library of Python bindings designed to solve computer vision problems.
- OpenCV: An open-source computer vision library used to process images and videos in real-time.
- lmdb : The Lightning Memory-Mapped Database (LMDB) is a software library that provides a high-performance transactional database supporting fully concurrent read/write access to millions of keys and values.
- collections : The collections module is a built-in module in Python. It provides alternatives to built-in types that are more efficient than the alternatives in certain situations.
- typing : The typing module defines a standard notation for Python function and variable type annotations.
- random: A module in the Python standard library that implements pseudo-random number generators for various uses.
- dataloader_iam: A custom module that provides functions to load data from the IAM Handwriting Database.
- json: A lightweight data interchange format used to store and exchange data in JSON files, which is used in this code to write the training summary file for the neural network.
- Editdistance: A python library used to calculate the Levenshtein distance between two strings, which is used in this code to calculate character error rates.
- TensorFlow: It is an open-source machine learning library developed by Google. It is designed to support the creation and deployment of machine learning models, including deep neural networks, on a variety of platforms and devices.

# STEPS:

## create_lmdb.py file:
Here's what is happening in the code step by step:

- The required libraries are imported, including argparse, pickle, cv2, lmdb, and Path.
- An argparse parser object is created and used to parse the command-line arguments. The parser expects a required argument --data_dir, which is of type Path.
- The code checks whether an LMDB directory exists inside the data directory provided as an argument. If it does, the code raises an assertion error.
- The LMDB database is created with a size of 2GB using lmdb.open() function.
- The code goes over all the PNG files inside the 'img' directory, which is a subdirectory inside the data directory. It uses the Path object to recursively find all the PNG files inside the 'img' directory.
- For each PNG file, the code reads the image using OpenCV cv2.imread() function and converts it to grayscale using the cv2.IMREAD_GRAYSCALE flag.
- The basename of the PNG file is extracted using the Path.basename() function.
- The grayscale image is pickled using the pickle.dumps() function.
- The pickled image and the basename are stored as a key-value pair in the LMDB database using the txn.put() function.
- After processing all the images, the database is closed using env.close() function.

## dataloader_iam.py:
- The given code is a Python script for loading and processing data in the IAM format (handwriting database). It defines a DataLoaderIAM class that loads image and text data from the IAM dataset and provides functions for iterating over the data in batches.

- The DataLoaderIAM class takes in the following parameters during initialization:
- data_dir: Path object indicating the directory containing the dataset files.
- batch_size: Integer value representing the number of samples to include in a single batch.
- data_split: A float value indicating the proportion of samples to be used for training (default is 0.95).
- fast: A boolean flag to indicate whether to use the fast (LMDB) or slow (OpenCV) image loading method (default is True).
- The DataLoaderIAM class has the following methods:
- train_set(): Switches to the randomly chosen subset of the training set.
- validation_set(): Switches to the validation set.
- get_iterator_info(): Returns the current batch index and the overall number of batches.
- has_next(): Returns True if there is a next element, False otherwise.
- _get_img(i): Private method that loads an image using either the fast or slow method, depending on the value of the fast flag.
- get_next(): Returns a batch of images and corresponding ground truth texts, where a batch is a named tuple with fields imgs, gt_texts, and batch_size.
- The code uses the lmdb library to create a fast key-value store for images, which is used if the fast flag is set to True. It also uses the pickle library to serialize and deserialize the images stored in the LMDB database.
- The namedtuple class is used to define two named tuples: Sample and Batch. Sample represents a single sample from the dataset, and has fields gt_text and file_path representing the ground truth text and the file path of the image. Batch represents a batch of samples, and has fields imgs, gt_texts, and batch_size representing the images, ground truth texts, and the size of the batch.

Overall, the DataLoaderIAM class provides a convenient way to load and process data from the IAM dataset, and can be used for training and evaluating handwriting recognition models.