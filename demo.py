import numpy as np
import matplotlib.pyplot as plt

from test_terrain import test_terrain


if __name__ == '__main__':
  print("Demo to validate the FPFH-ICP algorithm starts...")
  print("Estimated run time: 1 min")
  [_, _] = test_terrain(4, 10, 0)