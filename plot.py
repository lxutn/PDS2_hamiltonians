# -*- coding: utf-8 -*-
"""
Plot the results obtained by run.py
"""

# ----- Libraries ----- #
#Data
import pickle
import pandas as pd
#Plot
import seaborn as sns
import matplotlib.pyplot as plt
plt.close("all")
#Utils
import numpy as np

# ----- Constants ----- #
COL_NAMES = ["train accuracy","test accuracy","train loss","test loss"]

# ----- Parameters ----- #
LOAD_PATH = "TWO_IDS_H_GRID_3it"+"/data/"
BOXPLOT_LOG = True                  #plot boxplot in log scale

# ----- Load the data ----- #
# Global problem definition
DATASET = pickle.load(open(LOAD_PATH+"DATASET.pkl", "rb"))
STATS_REPETS = pickle.load(open(LOAD_PATH+"STATS_REPETS.pkl", "rb"))
ham_params = pickle.load(open(LOAD_PATH+"ham_params.pkl", "rb"))
net_params = pickle.load(open(LOAD_PATH+"net_params.pkl", "rb"))
learn_params = pickle.load(open(LOAD_PATH+"learn_params.pkl", "rb"))
#Grid search values
params_names = pickle.load(open(LOAD_PATH+"params_names.pkl", "rb"))
param1_values = pickle.load(open(LOAD_PATH+"param1_values.pkl", "rb"))
param2_values = pickle.load(open(LOAD_PATH+"param2_values.pkl", "rb"))
#Performances
train_acc = pickle.load(open( LOAD_PATH+"train_acc.pkl", "rb" ))
test_acc = pickle.load(open( LOAD_PATH+"test_acc.pkl", "rb" ))
train_loss = pickle.load(open( LOAD_PATH+"train_loss.pkl", "rb" ))
test_loss = pickle.load(open( LOAD_PATH+"test_loss.pkl", "rb" ))
train_acc_e = pickle.load(open( LOAD_PATH+"train_acc_e.pkl", "rb" ))
test_acc_e = pickle.load(open( LOAD_PATH+"test_acc_e.pkl", "rb" ))
train_loss_e = pickle.load(open( LOAD_PATH+"train_loss_e.pkl", "rb" ))
test_loss_e = pickle.load(open( LOAD_PATH+"test_loss_e.pkl", "rb" ))

print(test_acc)

# ----- Print the problem ----- #
print("Testing on {} dataset with {} repetitions\n".format(DATASET, STATS_REPETS))
print("Learning parameters:\n"+str(learn_params)+"\n")
print("Network parameters:\n"+str(net_params)+"\n")
print("Hamiltonian parameters:\n"+str(ham_params)+"\n")
print("Grid search over:")
print(params_names[0]+" with values "+str(param1_values))
print(params_names[1]+" with values "+str(param2_values)+"\n")

# ----- Convert to pandas dataframe ----- #
#Create df for final perfs
columns = [params_names[0],params_names[1],"iteration"]+COL_NAMES
df = pd.DataFrame(columns=columns)
#Fill df for final perfs
for i in range(len(param1_values)):
    for j in range(len(param2_values)):
        for it in range(STATS_REPETS):
            df = df.append({params_names[0]:param1_values[i],params_names[1]:param2_values[j],"iteration":it, 
                       "train accuracy":train_acc[i][j][it],"test accuracy":test_acc[i][j][it],
                       "train loss":train_loss[i][j][it], "test loss":test_loss[i][j][it]},
                      ignore_index=True)
#Create df for evolution along epochs
columns = [params_names[0],params_names[1],"iteration","epoch"]+COL_NAMES
df_e = pd.DataFrame(columns=columns)
#Fill df for evolution along epochs
for i in range(len(param1_values)):
    for j in range(len(param2_values)):
        for it in range(STATS_REPETS):
            nb_epochs = len(train_acc_e[i][j][it]) #nb of epochs for this specific trial
            for e in range(nb_epochs):
                df_e = df_e.append({params_names[0]:param1_values[i],params_names[1]:param2_values[j],
                                    "iteration":it,"epoch":e,"train accuracy":train_acc_e[i][j][it][e],
                                    "test accuracy":test_acc_e[i][j][it][e],
                                    "train loss":train_loss_e[i][j][it][e], 
                                    "test loss":test_loss_e[i][j][it][e]},
                                   ignore_index=True)

# ----- Plot ----- #
if len(param1_values) == 1 and len(param2_values) == 1:
    #Just plot the performance

    #Print perf
    for col_name in COL_NAMES:
        print(col_name+": "+str(df[col_name].mean()*100)+"% with std "+str(df[col_name].std()*100)+"%")
    #Line plot for accuracy and loss
    for measure in ["accuracy","loss"]:
        plt.figure()
        plt.title(DATASET+": "+measure+" along epoch"+" with "+str(STATS_REPETS)+" iter.")
        df_e_tmp = df_e.rename(columns={"train "+measure:"train","test "+measure:"test"})
        df_e_tmp = df_e_tmp.melt(id_vars=['epoch'], value_vars = ["train","test"], var_name='Dataset', 
                                 value_name=measure)
        sns.lineplot(data=df_e_tmp, x="epoch", y=measure,hue="Dataset")
        plt.show()
        
elif len(param1_values) > 1 and len(param2_values) > 1:
    #Print the result of the grid search
    for col_name in COL_NAMES:
        #Perf grid
        plt.figure()
        plt.title(DATASET+": "+col_name+" w.r.t. "+str(params_names[0])+" and "+str(params_names[1])
                  +" with "+str(STATS_REPETS)+" iter.")
        if BOXPLOT_LOG:
            plt.yscale("log")
        x = df.pivot_table(index=params_names[0], columns=params_names[1], values=col_name, aggfunc=np.mean)
        sns.heatmap(x, annot = True, cbar_kws={'label': col_name})
        plt.show()
        #Std grid
        plt.figure()
        plt.title(DATASET+": "+col_name+" std w.r.t. "+str(params_names[0])+" and "+str(params_names[1])
                  +" with "+str(STATS_REPETS)+" iter.")
        x = df.pivot_table(index=params_names[0], columns=params_names[1], values=col_name, aggfunc=np.std)
        sns.heatmap(x, annot = True, cbar_kws={'label': col_name+" std"})
        plt.show()
        
    #Line plot for accuracy and loss
    for measure in ["accuracy","loss"]:
        plt.figure()
        plt.title(DATASET+": "+measure+" along epoch w.r.t. "+str(params_names[0])+" and "
                  +str(params_names[1])+" with "+str(STATS_REPETS)+" iter.")
        df_e_tmp = df_e.rename(columns={"train "+measure:"train","test "+measure:"test"})
        df_e_tmp = df_e_tmp.melt(id_vars=["epoch",params_names[0], params_names[1]], 
                                 value_vars = ["train","test"], var_name='Dataset', value_name=measure)
        sns.lineplot(data=df_e_tmp, x="epoch", y=measure, hue=params_names[0], style = "Dataset", 
                     size=params_names[1])
        plt.show()
        
else:
    #Only one direction search
    #choose the good parameter to plot along
    if len(param1_values)>1: 
        param_name = params_names[0]
        param = param1_values
    else:
        param_name = params_names[1]
        param = param2_values
    
    #Plot all the values
    for col_name in COL_NAMES:
        #Boxplots grid
        plt.figure()
        plt.title(DATASET+": "+col_name+" w.r.t. "+str(param_name)+" with "+str(STATS_REPETS)+" iter.")
        if BOXPLOT_LOG:
            plt.yscale("log")
        sns.boxplot(data = df, x=param_name, y=col_name)
        plt.show()
        
    #Line plot for accuracy and loss
    for measure in ["accuracy","loss"]:
        plt.figure()
        plt.title(DATASET+": "+measure+" along epoch w.r.t. "+str(param_name)+" with "
                  +str(STATS_REPETS)+" iter.")
        df_e_tmp = df_e.rename(columns={"train "+measure:"train","test "+measure:"test"})
        df_e_tmp = df_e_tmp.melt(id_vars=["epoch",param_name], value_vars = ["train","test"], 
                                 var_name='Dataset', value_name=measure)
        sns.lineplot(data=df_e_tmp, x="epoch", y=measure, hue=param_name, style = "Dataset")
        plt.show()
    
