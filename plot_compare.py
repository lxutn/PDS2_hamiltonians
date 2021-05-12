# -*- coding: utf-8 -*-
"""
Plot and compare various results obtained by run.py !!! results have to be 1D !!!
"""

# ----- Libraries ----- #
#Data
import pickle
import pandas as pd
#Plot
import seaborn as sns
import matplotlib.pyplot as plt
plt.close("all")

# ----- Constants ----- #
COL_NAMES = ["train accuracy","test accuracy","train loss","test loss"]

# ----- Parameters ----- #
LOAD_PATHS = ["TWO_IDS_3it", "TWO_IDS_nobreg_3it", "TWO_IDS_hdiv_3it"]
NAMES = ["Enhanced Two layers","No b in regularisation","Division by h"]
for i in range(len(LOAD_PATHS)):
    LOAD_PATHS[i] = LOAD_PATHS[i]+"/data/"
BOXPLOT_LOG = True                  #plot boxplot in log scale
    
    
    
# ----- Load data & create pandas dataframe  ----- #
#Assume same dataset for all
DATASET = pickle.load(open(LOAD_PATHS[0]+"DATASET.pkl", "rb"))

#Create df for final perfs
columns = ["network","iteration"]+COL_NAMES
df = pd.DataFrame(columns=columns)
#Fill df for final perfs
for i, LOAD_PATH in enumerate(LOAD_PATHS):
    #Load data
    STATS_REPETS = pickle.load(open(LOAD_PATH+"STATS_REPETS.pkl", "rb"))
    train_acc = pickle.load(open( LOAD_PATH+"train_acc.pkl", "rb" ))
    test_acc = pickle.load(open( LOAD_PATH+"test_acc.pkl", "rb" ))
    train_loss = pickle.load(open( LOAD_PATH+"train_loss.pkl", "rb" ))
    test_loss = pickle.load(open( LOAD_PATH+"test_loss.pkl", "rb" ))
    NAMES[i] = NAMES[i]+" with "+str(STATS_REPETS)+" iter." #add the nb of iteration in the name
    #Add data
    for it in range(STATS_REPETS):
        df = df.append({"network":NAMES[i],"iteration":it, 
                   "train accuracy":train_acc[0][0][it],"test accuracy":test_acc[0][0][it],
                   "train loss":train_loss[0][0][it], "test loss":test_loss[0][0][it]},
                  ignore_index=True)
        
#Create df for evolution along epochs
columns = ["network","iteration","epoch"]+COL_NAMES
df_e = pd.DataFrame(columns=columns)
#Fill df for evolution along epochs
for i, LOAD_PATH in enumerate(LOAD_PATHS):
    #Load data
    STATS_REPETS = pickle.load(open(LOAD_PATH+"STATS_REPETS.pkl", "rb"))
    train_acc_e = pickle.load(open( LOAD_PATH+"train_acc_e.pkl", "rb" ))
    test_acc_e = pickle.load(open( LOAD_PATH+"test_acc_e.pkl", "rb" ))
    train_loss_e = pickle.load(open( LOAD_PATH+"train_loss_e.pkl", "rb" ))
    test_loss_e = pickle.load(open( LOAD_PATH+"test_loss_e.pkl", "rb" ))
    #Add data
    for it in range(STATS_REPETS):
        nb_epochs = len(train_acc_e[0][0][it]) #nb of epochs for this specific trial
        for e in range(nb_epochs):
            df_e = df_e.append({"network":NAMES[i],"iteration":it, 
                                "epoch":e,"train accuracy":train_acc_e[0][0][it][e],
                                "test accuracy":test_acc_e[0][0][it][e],
                                "train loss":train_loss_e[0][0][it][e], 
                                "test loss":test_loss_e[0][0][it][e]},
                               ignore_index=True)
        
# ----- Plot ----- #
#Box plots
for col_name in COL_NAMES:
    #Boxplots grid
    plt.figure()
    plt.title(DATASET+": "+col_name)
    if BOXPLOT_LOG:
        plt.yscale("log")
    sns.boxplot(data = df, x="network", y=col_name)
    plt.show()
    
#Line plots
for measure in ["accuracy","loss"]:
    plt.figure()
    plt.title(DATASET+": "+measure+" along epoch")
    df_e_tmp = df_e.rename(columns={"train "+measure:"train","test "+measure:"test"})
    df_e_tmp = df_e_tmp.melt(id_vars=["epoch","network"], value_vars = ["train","test"], 
                             var_name='Dataset', value_name=measure)
    sns.lineplot(data=df_e_tmp, x="epoch", y=measure, hue="network", style = "Dataset")
    plt.show()