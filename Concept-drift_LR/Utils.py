import os
import csv
import pandas as pd
import numpy as np

def normalize(ts, norm_params):
  """
  Apply min-max normalization
  :param ts: time series
  :param norm_params: tuple with params mean, std, max, min
  :return: normalized time series
  """
        
  return (ts - norm_params['min']) / (norm_params['max'] - norm_params['min'])

def initializeResults(modelName, type):
    """
    Initilize the results files to save the accuracy metrics for the models.
    :param modelName: String, represents the name of the model used to achieve this results. It can be LR_ADWIN, LR_KSWIN or LR_PH.
    :param type: String, type of the model used to achieve this results. It can be Concept_Drive or Random_Blocks.
    :return: writer object, allows to write the results in the initialized files.
    """
    ruta = os.getcwd()
    archivo = "/resultados_" + modelName + "_" + type + ".csv"
    if(archivo in os.listdir(os.getcwd())):
        os.remove(archivo)

    f = open(ruta + archivo, 'a')
    writer = csv.writer(f)
    writer.writerow(['Carpeta', 'Mae', 'Wape', 'Tiempo_Media'])
    return writer

def escribeDatos(writer, block, mae, wape, timeB):
    """
    Write the accuracy metrics using a writer object that is already pointing at that file.
    :param writer: writer object, it is already pointing the file where we will save the data.
    :param block: String, number of the block that we have validated at this moment.
    :param mae: float, accuracy metric MAE.
    :param wape: float, accuracy metric WAPE.
    :param timeB: float, time metric for this block validation.

    """
    writer.writerow([block, mae, wape, timeB])
    
def loadData():
    """
    Load data from esios dataset called eol.csv.
    :return: Dataframe object with the wind energy generation data from esios.
    """
    return pd.read_csv('eol.csv')

def denormalize(ts, norm_params):
  """
  Apply min-max normalization
  :param data: time series
  :param norm_params: tuple with params mean, std, max, min
  :return: normalized time series
  """
  return ts * (norm_params["max"] - norm_params["min"]) + norm_params["min"]


def evaluate_error(actual, predicted):
    """
  Evaluate the MAE and WAPE error for a folder
  :param actual: List with the real values.
  :param predicted: List with the predicted values.
  :return: List with the MAE and WAPE error for the folder.
  """
    metrics = []
    EPSILON = 1e-10
    mae = np.mean(np.abs(np.asarray(actual) - np.asarray(predicted)))
    metrics.append(mae)
    wape = mae / (np.mean(actual) + EPSILON)
    metrics.append(wape)
    
    return metrics

def getResults(registrosBloque, prediccionBloque, maeWape):
    """
  Calculate MAE and WAPE error between the real values and the one you want to compare.
  :param registrosBloque: List with the real values for the folder.
  :param prediccionBloque: List with the Esios' predictions or our model's predictions
  values for the folder.
  :param maeWape: List with the MAE and WAPE errors for the fodler.
  
  """
    results_metrics = evaluate_error(np.asarray(registrosBloque), np.asarray(prediccionBloque))
    result_mae = results_metrics[0]
    result_wape = results_metrics[1]
    maeWape[0].append(result_mae)
    maeWape[1].append(result_wape)
  
def resetFiles():
  """
  Reset results files to save the new 
  """
  archivos = os.listdir(os.getcwd())
  for archivo in archivos:
    if("resultados" in archivo):
      os.remove(archivo)

def initializeIndxs():
  """
  Initialize the indexes for every Concept Drift tecnique.
  :return indxs_ADWIN, indxs_KSWIN, indxs_PH: Lists containing the indexes for ADWIN, KSWIN and Page Hinkley concept drift detection tecniques.
  """
  indxs_ADWIN = [31, 63, 8639]
  
  indxs_KSWIN = [72, 104, 136, 168, 200, 232, 264, 296, 328, 360, 399, 431, 
                463, 498, 530, 562, 594, 626, 658, 690, 722, 762, 794, 826, 
                858, 890, 922, 954, 986, 1018, 1050, 1082, 1114, 1146, 1178, 
                1210, 1243, 1275, 1307, 1342, 1374, 1406, 1438, 1470, 1502, 
                1542, 1574, 1606, 1638, 1677, 1709, 1741, 1773, 1805, 1837, 
                1869, 1901, 1933, 1965, 1999, 2031, 2063, 2095, 2127, 2161, 
                2193, 2226, 2258, 2290, 2322, 2355, 2387, 2419, 2455, 2488, 
                2521, 2553, 2585, 2617, 2649, 2681, 2741, 2773, 2805, 2837, 
                2870, 2902, 2934, 2966, 2998, 3030, 3062, 3094, 3126, 3158, 
                3190, 3222, 3266, 3298, 3332, 3364, 3420, 3452, 3484, 3516, 
                3555, 3587, 3619, 3651, 3697, 3729, 3761, 3793, 3828, 3869, 
                3901, 3933, 3965, 4008, 4040, 4072, 4104, 4136, 4168, 4202, 
                4239, 4271, 4303, 4335, 4367, 4399, 4431, 4463, 4495, 4527, 
                4559, 4591, 4623, 4655, 4690, 4722, 4762, 4794, 4826, 4867, 
                4899, 4931, 4963, 4995, 5027, 5059, 5091, 5123, 5155, 5187, 
                5219, 5251, 5283, 5315, 5347, 5379, 5418, 5451, 5483, 5515, 
                5571, 5603, 5635, 5667, 5703, 5735, 5767, 5811, 5843, 5881, 
                5913, 5946, 5988, 6020, 6052, 6084, 6121, 6161, 6195, 6227, 
                6269, 6301, 6333, 6365, 6397, 6429, 6461, 6493, 6525, 6557, 
                6612, 6644, 6676, 6708, 6751, 6783, 6834, 6866, 6904, 6936, 
                6983, 7042, 7074, 7106, 7138, 7170, 7202, 7241, 7285, 7357, 
                7389, 7431, 7463, 7500, 7532, 7564, 7596, 7628, 7660, 7692, 
                7724, 7756, 7788, 7820, 7852, 7896, 7928, 7960, 7996, 8028, 
                8060, 8094, 8126, 8158, 8191, 8223, 8255, 8287, 8319, 8365, 
                8397, 8429, 8461, 8493, 8534, 8566, 8605, 8637]

  indxs_PH = [179, 208, 237, 278, 311, 340, 369, 398, 428, 457, 488, 776, 805, 
            834, 863, 892, 921, 950, 979, 1008, 1042, 1071, 1100, 1251, 1283, 
            1312, 1341, 1370, 1399, 1429, 1459, 1488, 1517, 1673, 1702, 1731, 
            1760, 1814, 2038, 2135, 2164, 2193, 2227, 2256, 2285, 2318, 2347, 
            2376, 2405, 2451, 2480, 2509, 2538, 2567, 2646, 2678, 2711, 2740, 
            2769, 2800, 2829, 2865, 2894, 2923, 2952, 2986, 3015, 3158, 3187, 
            3235, 3264, 3307, 3350, 3391, 3420, 3449, 3488, 3521, 3550, 3579, 
            3608, 3644, 3673, 3703, 3732, 3761, 3791, 3838, 3867, 3896, 3925, 
            3968, 3997, 4026, 4055, 4084, 4114, 4143, 4183, 4212, 4241, 4270, 
            4299, 4328, 4357, 4386, 4419, 4448, 4477, 4506, 4537, 4580, 4609, 
            4652, 4681, 4723, 4755, 4799, 4828, 4885, 4914, 4943, 4972, 5026, 
            5126, 5226, 5277, 5306, 5341, 5370, 5399, 5428, 5457, 5486, 5537, 
            5566, 6294, 6329, 6358, 6387, 6462, 6491, 6521, 6553, 6583, 6680, 
            6709, 6743, 6772, 6801, 6830, 6876, 6905, 6934, 6969, 7004, 7033, 
            7062, 7091, 7120, 7149, 7222, 7296, 7328, 7357, 7386, 7431, 7460, 
            7518, 7553, 7582, 7611, 7640, 7669, 7755, 7785, 7814, 7843, 7915, 
            7944, 7978, 8520, 8549, 8584, 8613, 8639]
            
  return indxs_ADWIN, indxs_KSWIN, indxs_PH