#all required imports

import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score,classification_report,cohen_kappa_score
from sklearn.metrics import roc_curve,auc,precision_score,recall_score
from statsmodels.stats.weightstats import _zconfint_generic, _tconfint_generic


def metrics(l):
    mean = np.mean(l)
    std = np.std(l,ddof=1)
    mean_std = np.std(l,ddof=1)/np.sqrt(len(l))
    tconf = _tconfint_generic(mean, mean_std, len(l) - 1, 0.05, 'two-sided')
    return mean, std, mean_std, tconf

def print_f(l, metric):
    mean, std, mean_std, tconf = metrics(l)
    print("Model " + metric + ": mean %.3f, std %.3f" % (mean, std))
    print("Model " + metric + " 95%% confidence interval:", np.round(tconf, 3))

def results_full_test(optimizer, threshold, X_test , y_test):
    model = optimizer.best_estimator_

    (X0_50, X50_100, y0_50, y50_100) = train_test_split(X_test , y_test , test_size=0.5, stratify=y_test )
    
    (X0_25, X25_50, y0_25, y25_50) = train_test_split(X0_50, y0_50, test_size=0.5, stratify=y0_50)
    (X50_75, X75_100, y50_75, y75_100) = train_test_split(X50_100, y50_100, test_size=0.5, stratify=y50_100)
    
    (X0_12, X12_25, y0_12, y12_25) = train_test_split(X0_25, y0_25, test_size=0.5, stratify=y0_25)
    (X25_37, X37_50, y25_37, y37_50) = train_test_split(X25_50, y25_50, test_size=0.5, stratify=y25_50)
    (X50_62, X62_75, y50_62, y62_75) = train_test_split(X50_75, y50_75, test_size=0.5, stratify=y50_75)
    (X75_87, X87_100, y75_87, y87_100) = train_test_split(X75_100, y75_100, test_size=0.5, stratify=y75_100)

    f1_list=[]
    acc_list=[]
    rec_list=[]
    prec_list =[]
    X = [X0_12,X12_25,X25_37, X37_50, X50_62,X62_75,X75_87, X87_100]
    y = [y0_12,y12_25,y25_37, y37_50, y50_62,y62_75,y75_87, y87_100]
    for X_test,y_test in zip(X,y):
        y_proba = model.predict_proba(X_test)
        y_pred = (y_proba [:,1] >= threshold).astype('int')
        acc= (accuracy_score(y_test, y_pred))
        prec=precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1=f1_score(y_test, y_pred)
        f1_list.append(f1)
        acc_list.append(acc)
        rec_list.append(rec)
        prec_list.append(prec)
    
    print_f(acc_list, 'accuracy')    
    print_f(prec_list, 'precision')    
    print_f(rec_list, 'recall')    
    print_f(f1_list, 'f1-score')    