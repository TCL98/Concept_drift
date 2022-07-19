import time
from sklearn.linear_model import LinearRegression
from Utils import initializeResults, getResults, escribeDatos, denormalize
from blocksUtils import splitBlocksByIndxs, generateIndexes
import numpy as np

def LR_CD(df2, indxs, modelName):
    """
    Initialize the LR model, divide the data following the condept drift detection tecniques indexes
    and write accuracy metrics and time results in a file.
    :param df2: DataFrame object, contains all the wind energy generation collected from esios.
    :param indxs: List, contains the concept drifts detected by the dectection tecniques.
    :param modelName: String, it should be LR_ADWIN, LR_KSWIN or LR_PH.
    """
    writer = initializeResults(modelName, "Concept_Drive")
    times, maes, wapes = [], [], []
    for i in range(0, len(indxs)-1):
        x_trains, y_trains, x_tests, y_tests, Anorm_params = splitBlocksByIndxs(df2, indxs)
        model = LinearRegression()
        maeWape = [[], []]
        
        new_shape_x_train = (x_trains[i].shape[0], 4)
        new_shape_y_train = (y_trains[i].shape[0], 1)
        new_shape_x_test = (x_tests[i].shape[0], 4)
                
        x_train = np.reshape(x_trains[i], new_shape_x_train)
        y_train = np.reshape(y_trains[i], new_shape_y_train)
                
        start = time.process_time()
        model.fit(x_train, y_train)
        stop = time.process_time()
        times.append(stop-start)
                
        preds = model.predict(np.reshape(x_tests[i], new_shape_x_test))
        preds = denormalize(preds, Anorm_params[i])
        real = denormalize(y_tests[i], Anorm_params[i])
        getResults(real, preds, maeWape)
        escribeDatos(writer, "Carpeta " + str(i), maeWape[0], maeWape[1], times[-1])
        maes.append(maeWape[0])
        wapes.append(maeWape[1])

        mediaMAE = np.mean(maes)
        mediaWAPE = np.mean(wapes)
        mediaTime = np.mean(times)
        escribeDatos(writer, "Total Media", mediaMAE, mediaWAPE, mediaTime)
        
    print("Concept Drift", modelName, "Iteration completed")
    print("Media Total:")
    print("MAE:", mediaMAE)
    print("WAPE", mediaWAPE)
    print("Time:", mediaTime)
    print()
    

def LR_RB(df2, indxsCD, modelName):
    """
    Initialize the LR model, divide the data randomly and write accuracy metrics and time results in a file.
    :param df2: DataFrame object, contains all the wind energy generation collected from esios.
    :param indxsCD: List, contains the concept drifts detected by the dectection tecniques.
    :param modelName: String, it should be LR_ADWIN, LR_KSWIN or LR_PH.
    """
    times, maes, wapes = [], [], []
    for loop in range(0, 50):    
        writer = initializeResults(modelName, "Random_Blocks")
        indxs = generateIndexes(len(df2), len(indxsCD))
        x_trains, y_trains, x_tests, y_tests, Anorm_params = splitBlocksByIndxs(df2, indxs)
        Ltimes, mae, wape = [], [], []
        model = LinearRegression()
            
        for i in range(0, len(indxs)-1):
            maeWape = [[], []]
            new_shape_x_train = (x_trains[i].shape[0], 4)
            new_shape_y_train = (y_trains[i].shape[0], 1)
            new_shape_x_test = (x_tests[i].shape[0], 4)
                
            x_train = np.reshape(x_trains[i], new_shape_x_train)
            y_train = np.reshape(y_trains[i], new_shape_y_train)
                
            start = time.process_time()
            model.fit(x_train, y_train)
            stop = time.process_time()
            times.append(stop-start)
                
            preds = model.predict(np.reshape(x_tests[i], new_shape_x_test))
            preds = denormalize(preds, Anorm_params[i])
            real = denormalize(y_tests[i], Anorm_params[i])
            getResults(real, preds, maeWape)
            escribeDatos(writer, "Carpeta " + str(i), maeWape[0], maeWape[1], times[-1])
            mae.append(maeWape[0])
            wape.append(maeWape[1])
        maes.append(mae)
        wapes.append(wape)
        Ltimes.append(time)
    
    mediaMAE = np.mean(maes)
    mediaWAPE = np.mean(wapes)
    mediaTime = np.mean(times)
    escribeDatos(writer, "Total Media", mediaMAE, mediaWAPE, mediaTime)
    
    print("Random Blocks", modelName, "Iteration completed")
    print("Media Total:")
    print("MAE:", mediaMAE)
    print("WAPE", mediaWAPE)
    print("Time:", mediaTime)
    print()