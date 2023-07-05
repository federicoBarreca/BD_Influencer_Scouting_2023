import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.group import *
from pyspark import SparkContext, SparkConf
from pyspark.sql.window import Window
from pyspark.ml.classification import *
from pyspark.ml.feature import *
from pyspark.ml import *
from pyspark.ml.evaluation import *
from sklearn.metrics import *
from pyspark.ml.tuning import *
from pyspark.ml.linalg import VectorUDT, Vectors
import sparknlp 
from sparknlp.base import *
from sparknlp.annotator import *        
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import os
from PIL import Image
import io
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import *
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import *
from sparktorch import serialize_torch_obj, SparkTorch
from PIL import Image
import math
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
from torch import optim

# Define a custom dataset class to load the input images
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        label_tensor = torch.tensor(self.labels[idx])
        return image_tensor, label_tensor