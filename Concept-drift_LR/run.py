from LR import LR_CD, LR_RB
from Utils import loadData, initializeIndxs, resetFiles
from ConceptDriftDetection import executeDriftDetection

# Load the data
df2 = loadData()

#Load indexes precalculated.
indxs_ADWIN, indxs_KSWIN, indxs_PH = initializeIndxs()
resetFiles()

#Execute Concept Drfit detection methods
executeDriftDetection()

# Execute LR models with the indxs information. 
LR_CD(df2, indxs_ADWIN, "LR_ADWIN")
LR_RB(df2, indxs_ADWIN, "LR_ADWIN")
LR_CD(df2, indxs_KSWIN, "LR_KSWIN")
LR_RB(df2, indxs_KSWIN, "LR_KSWIN")
LR_CD(df2, indxs_PH, "LR_PH")
LR_RB(df2, indxs_PH, "LR_PH")