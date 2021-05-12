# -*- coding: utf-8 -*-
"""
Fusion some dataset
"""

import pickle

LOAD_PATH = "TWO_IDS_H_GRID_3it"+"/data/"
LOAD_PATH_2 = "h_025"+"/data/"
SAVE_ = "aaaa/"

param1_values = pickle.load(open(LOAD_PATH+"param1_values.pkl", "rb"))

print(param1_values)

param1_values.append(0.25)

print(param1_values)

train_acc = pickle.load(open( LOAD_PATH+"train_acc.pkl", "rb" ))
test_acc = pickle.load(open( LOAD_PATH+"test_acc.pkl", "rb" ))
train_loss = pickle.load(open( LOAD_PATH+"train_loss.pkl", "rb" ))
test_loss = pickle.load(open( LOAD_PATH+"test_loss.pkl", "rb" ))
train_acc_e = pickle.load(open( LOAD_PATH+"train_acc_e.pkl", "rb" ))
test_acc_e = pickle.load(open( LOAD_PATH+"test_acc_e.pkl", "rb" ))
train_loss_e = pickle.load(open( LOAD_PATH+"train_loss_e.pkl", "rb" ))
test_loss_e = pickle.load(open( LOAD_PATH+"test_loss_e.pkl", "rb" ))


train_acc_2 = pickle.load(open( LOAD_PATH_2+"train_acc.pkl", "rb" ))
test_acc_2 = pickle.load(open( LOAD_PATH_2+"test_acc.pkl", "rb" ))
train_loss_2 = pickle.load(open( LOAD_PATH_2+"train_loss.pkl", "rb" ))
test_loss_2 = pickle.load(open( LOAD_PATH_2+"test_loss.pkl", "rb" ))
train_acc_e_2 = pickle.load(open( LOAD_PATH_2+"train_acc_e.pkl", "rb" ))
test_acc_e_2 = pickle.load(open( LOAD_PATH_2+"test_acc_e.pkl", "rb" ))
train_loss_e_2 = pickle.load(open( LOAD_PATH_2+"train_loss_e.pkl", "rb" ))
test_loss_e_2 = pickle.load(open( LOAD_PATH_2+"test_loss_e.pkl", "rb" ))


train_acc.append(train_acc_2[0])
test_acc.append(test_acc_2[0])
train_loss.append(train_loss_2[0])
test_loss.append(test_loss_2[0])
train_acc_e.append(train_acc_e_2[0])
test_acc_e.append(test_acc_e_2[0])
train_loss_e.append(train_loss_e_2[0])
test_loss_e.append(test_loss_e_2[0])

pickle.dump(param1_values,open( SAVE_+"param1_values.pkl", "wb" ))
pickle.dump(train_acc,open( SAVE_+"train_acc.pkl", "wb" ))
pickle.dump(test_acc,open( SAVE_+"test_acc.pkl", "wb" ))
pickle.dump(train_loss,open( SAVE_+"train_loss.pkl", "wb" ))
pickle.dump(test_loss,open( SAVE_+"test_loss.pkl", "wb" ))
pickle.dump(train_acc_e,open( SAVE_+"train_acc_e.pkl", "wb" ))
pickle.dump(test_acc_e,open( SAVE_+"test_acc_e.pkl", "wb" ))
pickle.dump(train_loss_e,open( SAVE_+"train_loss_e.pkl", "wb" ))
pickle.dump(test_loss_e,open( SAVE_+"test_loss_e.pkl", "wb" ))