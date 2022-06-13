# -*- coding: utf-8 -*-
"""
Plot and compare various results obtained by run.py
Note that the results compared can not be grid searches
Line 24 one can select the various to compare
"""

# ----- Libraries ----- #
# Data
import pickle
import pandas as pd
# Plot
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.close("all")

# Change the font for the plots
font = {'size': 22}
matplotlib.rc('font', **font)

# ----- Parameters ----- #
# Where load the data (/data will be added in an automated way)
LOAD_PATHS = ["TwoLayersPaper/vanilla_5it", "TwoLayersIDS/vanilla_5it",
              "J1/ReLU/vanilla_1it", "J1/Tanh/vanilla_1it", "J2/vanilla_1it"]
# Name to attribute to each file
NAMES = ["TwoLayers Paper", "TwoLayers IDS", "J1 ReLU", "J1 Tanh", "J2"]

# ----- Constants ----- #
COL_NAMES = ["train accuracy", "test accuracy", "train loss", "test loss"]

# ----- Load data & create pandas dataframe  ----- #
for i in range(len(LOAD_PATHS)):
    LOAD_PATHS[i] = LOAD_PATHS[i]+"/data/"

# Assume same dataset for all
DATASET = pickle.load(open(LOAD_PATHS[0]+"DATASET.pkl", "rb"))

# Create df for final perfs
columns = ["network", "iteration"]+COL_NAMES
df = pd.DataFrame(columns=columns)
# Fill df for final perfs
for i, LOAD_PATH in enumerate(LOAD_PATHS):
    # Load data
    STATS_REPETS = pickle.load(open(LOAD_PATH+"STATS_REPETS.pkl", "rb"))
    train_acc = pickle.load(open(LOAD_PATH+"train_acc.pkl", "rb"))
    test_acc = pickle.load(open(LOAD_PATH+"test_acc.pkl", "rb"))
    train_loss = pickle.load(open(LOAD_PATH+"train_loss.pkl", "rb"))
    test_loss = pickle.load(open(LOAD_PATH+"test_loss.pkl", "rb"))
    # Add data
    for it in range(STATS_REPETS):
        df = df.append({"network": NAMES[i], "iteration": it,
                        "train accuracy": train_acc[0][0][it], "test accuracy": test_acc[0][0][it],
                        "train loss": train_loss[0][0][it], "test loss": test_loss[0][0][it]},
                       ignore_index=True)

# Create df for evolution along epochs
columns = ["network", "iteration", "epoch"]+COL_NAMES
df_e = pd.DataFrame(columns=columns)
# Fill df for evolution along epochs
for i, LOAD_PATH in enumerate(LOAD_PATHS):
    # Load data
    STATS_REPETS = pickle.load(open(LOAD_PATH+"STATS_REPETS.pkl", "rb"))
    train_acc_e = pickle.load(open(LOAD_PATH+"train_acc_e.pkl", "rb"))
    test_acc_e = pickle.load(open(LOAD_PATH+"test_acc_e.pkl", "rb"))
    train_loss_e = pickle.load(open(LOAD_PATH+"train_loss_e.pkl", "rb"))
    test_loss_e = pickle.load(open(LOAD_PATH+"test_loss_e.pkl", "rb"))
    # Add data
    for it in range(STATS_REPETS):
        # nb of epochs for this specific trial
        nb_epochs = len(train_acc_e[0][0][it])
        for e in range(nb_epochs):
            df_e = df_e.append({"Network": NAMES[i], "iteration": it,
                                "epoch": e, "train accuracy": train_acc_e[0][0][it][e],
                                "test accuracy": test_acc_e[0][0][it][e],
                                "train loss": train_loss_e[0][0][it][e],
                                "test loss": test_loss_e[0][0][it][e]},
                               ignore_index=True)

# ----- Plot ----- #
# Plot the grid (test and train on the same image)
for i, col_name in enumerate(COL_NAMES):
    if i == len(COL_NAMES)-1:
        continue
    plt.figure()
    title = ""
    for name in NAMES:
        title = title+name+" "
    plt.title(DATASET+" "+title)
    sns.lineplot(data=df, x="network", y=col_name)
    sns.lineplot(data=df, x="network",
                 y=COL_NAMES[i], color=u'#1f77b4', label="Test")
    sns.lineplot(data=df, x="network",
                 y=COL_NAMES[i+1], color=u'#1f77b4', linestyle="--", label="Train")
    plt.xticks(NAMES)
    plt.legend()
    plt.show()


# Line plots
for measure in ["accuracy", "loss"]:
    plt.figure()
    plt.title(DATASET+": "+measure+" along epochs")
    df_e_tmp = df_e.rename(
        columns={"train "+measure: "train", "test "+measure: "test"})
    df_e_tmp = df_e_tmp.melt(id_vars=["epoch", "Network"], value_vars=["train", "test"],
                             var_name='Dataset', value_name=measure)
    sns.lineplot(data=df_e_tmp, x="epoch", y=measure,
                 hue="Network", style="Dataset")
    if measure == "accuracy":
        plt.legend(loc='lower right')
    else:
        plt.legend(loc='upper right')
    plt.show()
