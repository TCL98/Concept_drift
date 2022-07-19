import math
from Utils import normalize
import numpy as np
import random

def splitBlocksByIndxs(df, indxs):
    """
    Splits the data in blocks distributed by the indexes selected.
    :param df: DataFrame object with all the wind energy data.
    :param indxs: Indexes selected by the concept drift detection tecniques.

    :return:
    :x_trains: List containing all the train sets.
    :y_trains: List containing all the results for the train data sets.
    :x_tests: List containing all the test sets.
    :y_tests: List containing all the results for the test data sets.
    :Anorm_params: Parameters to normalize and denormalize data.
    """
    x_trains, y_trains, x_tests, y_tests, Anorm_params = [], [], [], [], []
    
    for i in range(0, len(indxs)-1):
      df_to_scale = df[indxs[i]:indxs[i+1]].copy()
      limit_train = math.floor(len(df_to_scale)*0.8)
      df_to_scale_train = df_to_scale[:limit_train].copy()
      df_to_scale_test = df_to_scale[limit_train:].copy()
        
      norm_params = {}
      norm_params['mean'] = df_to_scale_train.mean().mean()
      norm_params['std'] = df_to_scale_train.std().std()
      norm_params['max'] = df_to_scale_train.max().max()
      norm_params['min'] = df_to_scale_train.min().min()
                
      scaled_data = normalize(df_to_scale_train, norm_params)
      scaled_data_test = normalize(df_to_scale_test, norm_params)
        
      x_train2, y_train2 = [], []
      x_test2, y_test2 = [], []
          
      forecast_horizon = 1
      past_history = 4
      
      
      for i in range(0, len(scaled_data) - past_history - forecast_horizon):
          a = scaled_data[i:(i+past_history)]
          x_train2.append(a)
          y_train2.append(scaled_data[i + past_history: i + past_history + forecast_horizon])
          
      x_train2, y_train2 = np.asarray(x_train2), np.asarray(y_train2)
      x_test2, y_test2 = [], []
            
      for i in range(0, len(scaled_data_test) - past_history - forecast_horizon):
          a = scaled_data_test[i:(i+past_history)]
          x_test2.append(a)
          y_test2.append(scaled_data_test[i + past_history: i + past_history + forecast_horizon])
    
      x_test2, y_test2 = np.asarray(x_test2), np.asarray(y_test2)
      x_trains.append(x_train2)
      y_trains.append(y_train2)
      x_tests.append(x_test2)
      y_tests.append(y_test2)
      
      Anorm_params.append(norm_params)
  
    return x_trains, y_trains, x_tests, y_tests, Anorm_params

def generateIndexes(lenDf2, indxsLenght):
    """
    Generate Indexes for random blocks.
    :param lenDf2: Number of entries for the DataFrame object that contains all the wind generation data.
    :param indxsLenght: Number of indexes detected by the concept drift detection tecnique.
    :return indexes: New random indexes.
    """
    # Initialize indexes list with 0 which should be the start of the first random block.
    indexes = [0]

    # Define the maximum distance we can find between indexes if we want to achieve the same number of indexes than he concept drift split.
    maxDivision = math.floor(lenDf2 / indxsLenght - 1)

    # Define the minimum distance we can find between indexes if we want to be able of splitting data in train and test data and perform validate tecniques.
    minimumDivision = 30

    # Define a variable to accumulate the maximum index we can select.
    accumulatedDivision = maxDivision

    # Generate every index following the minDivision, maxDivision and the accumulatedDivision restrictions.
    for i in range(0, indxsLenght - 1):
        # Generate a new random index
        newIndex = random.randrange(minimumDivision, accumulatedDivision)
        indexes.append(indexes[i] + newIndex)

        # Update accumulatedDivision
        accumulatedDivision = accumulatedDivision - newIndex + maxDivision

    # Finally, we add the last entry of the data.
    indexes.append(8639)
    return indexes