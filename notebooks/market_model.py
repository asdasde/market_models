#!/usr/bin/env python
# coding: utf-8

# In[14]:


import time
import dataframe_image as dfi
import numpy as np
import pandas as pd
import xgboost
import os
import shap
import csv

from fpdf import FPDF
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.inspection import PartialDependenceDisplay as PDP
from sklearn.inspection import partial_dependence as pdep
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

xgboost.set_config(verbosity = 0)


# In[15]:


def prepareDir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)
    for file in os.listdir(dir):
        os.remove(dir + file)
        
def detect_csv_delimiter(file_path):
    """
    Detects the delimiter used in a CSV file.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - str: The detected delimiter.
    """
    with open(file_path, 'r', newline='') as file:
        dialect = csv.Sniffer().sniff(file.read(1024))
        return dialect.delimiter

def read_file(file_path):
    """
    Reads a file using the appropriate pandas read function based on the file extension.

    Parameters:
    - file_path (str): The path to the file.

    Returns:
    - pd.DataFrame: The DataFrame containing the data from the file.
    """
    file_extension = file_path.split('.')[-1].lower()

    if file_extension == 'csv':
        return pd.read_csv(file_path, sep = detect_csv_delimiter(file_path))
    elif file_extension in ['xls', 'xlsx']:
        return pd.read_excel(file_path)
    elif file_extension == 'json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


# In[16]:


####### PATHS

INPUT = '../data/raw/'
INTERMEDIATE = '../data/interim/'
OUTPUT = '../processed/'

DATA_PATH = INPUT + 'newprices.csv'
FEATURES_PATH = INPUT + 'netrisk_casco15369_features.txt'
KPI_DATA_PATH = INTERMEDIATE + 'pdf_data/'
PDF_PATH = OUTPUT + DATA_PATH.split('/')[2].replace('.xlsx', '') + '_'


####### CONSTANTS

TEST_SIZE = 0.1
RANDOM_STATE = 42
KPI_TABLE_COLUMNS = ['MAE', 'sd MAE', 'RMSE', 'sd RMSE', 'MAPE', 'sd MAPE', 'Avg target']

TARGET_VARIABLE = 'ALLIANZ_price'
PRED_TARGET_VARIABLE = 'predicted' + TARGET_VARIABLE[0].capitalize() +TARGET_VARIABLE[1 : ]


PARAMS_GRID = {
    'objective': ['reg:squarederror'],
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [5, 6, 7, 8, 9, 10],
    'enable_categorical': [False, True],
    'eval_metric': ['mae'],
    'n_estimators': [200, 400, 600, 800, 1000, 1200],
    'eta': [0.05, 0.1, 0.15]
}


# In[17]:


data = read_file(DATA_PATH)

#with open(FEATURES_PATH) as file:
#    features = file.readlines()
#    features = [feature.replace('\n', '') for feature in features]
#    feature_dtypes = {feature.split(',')[0] : feature.split(',')[1] for feature in features}
#    features = [feature.split(',')[0] for feature in features]

#for feature in features:
    #print(feature, feature_dtypes[feature].replace('object', 'category'))
#    data[feature] = data[feature].astype(feature_dtypes[feature].replace('object', 'category'))


# In[18]:


data


# In[69]:


# Make last column name a variable to generelize

data = data[features + [TARGET_VARIABLE]]
data = data.dropna()


# In[28]:


# Make data overview

describe = pd.concat([data.describe(), pd.DataFrame(np.array([len(data[col].unique()) for col in data.columns]).reshape(1, -1), index = ['unique'], columns = data.columns)])
describeStyle = describe.T.style.format(precision = 2)
dfi.export(describeStyle, KPI_DATA_PATH + 'dataOverview.png', dpi = 200)


# In[41]:


# Various model related methods

def modelFit(trData, n_estimators = 1200, max_depth = 8):
    model = xgboost.XGBRegressor(n_estimators = n_estimators, max_depth = max_depth, eta=0.1, gamma = 5, subsample=1, colsample_bytree=0.8, min_child_weight = 1 , eval_metric = 'mae', enable_categorical = True)
    model.fit(transposed_train[features], transposed_train[ TARGET_VARIABLE])
    return model

def makeDMatrix(features, target):
    return xgboost.DMatrix(features, target, enable_categorical = True)

def modelTrain(trData, teData):

    dtrain = xgboost.DMatrix(trData[features], trData[TARGET_VARIABLE], enable_categorical = True)
    dtest = xgboost.DMatrix(teData[features], teData[TARGET_VARIABLE], enable_categorical = True)
    param = {'max_depth' : 10, 'eta' : 0.2, 'eval_metric' : 'mae' }
    evallist  = [(dtest,'eval'), (dtrain,'train')]
    num_round = 400
    return xgboost.train( param, dtrain, num_round, evals = evallist, verbose_eval = False)

def mergePredictions(model):
    output = transposed_test.copy()
    try:
        output[ PRED_TARGET_VARIABLE] = model.predict(transposed_test[features])
    except Exception as e:
        dtest = xgboost.DMatrix(transposed_test[features], transposed_test[TARGET_VARIABLE], enable_categorical = True)
        output[ PRED_TARGET_VARIABLE] = model.predict(dtest)
    output['error'] = output[TARGET_VARIABLE] - output[ PRED_TARGET_VARIABLE]
    output['percentageError'] = output['error'] / output[TARGET_VARIABLE] * 100
    return output

def plotHistErrorPercenage(preds):
    plt.hist(preds.percentageError, range = [-100, 100], bins = 40, weights = np.ones(len(preds.percentageError)) / len(preds.percentageError))
    plt.xlabel('Error percentage')
    plt.ylabel('Percent of errors')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.savefig(KPI_DATA_PATH + 'ErrorDistribution.jpg')
    plt.show()

def plotFeatureDistribution(data, feature):
    plt.figure(figsize = (10, 10))
    if feature in feature_dtypes.keys() and feature_dtypes[feature] == 'bool':
        data[feature].value_counts().plot(kind = 'bar', title = feature)
        plt.savefig(KPI_DATA_PATH + feature + 'Distribution.jpg', bbox_inches = 'tight')
    else:
        plt.hist(data[feature], bins = 40, weights = np.ones(len(data[feature])) / len(data[feature]), alpha=0.5)
        plt.xlabel(feature)
        plt.ylabel('Percent of values')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.savefig(KPI_DATA_PATH + feature + 'Distribution.jpg', bbox_inches = 'tight')
    plt.close()



# In[30]:


def kFoldCrossValidation(k = 5):
    kf = KFold(n_splits=k)
    maes = []
    mses = []
    mapes = []

    for transposed_train, transposed_test in kf.split(data):
        transposed_train, transposed_test = data.iloc[transposed_train], data.iloc[transposed_test]
        trainModel = modelTrain(transposed_train, transposed_test)
        dmat = makeDMatrix(transposed_test[features], transposed_test[TARGET_VARIABLE])
        mae = mean_absolute_error(transposed_test[TARGET_VARIABLE].values, trainModel.predict(dmat))
        mse = mean_squared_error(transposed_test[TARGET_VARIABLE].values, trainModel.predict(dmat))
        mape = mean_absolute_percentage_error(transposed_test[TARGET_VARIABLE].values, trainModel.predict(dmat))
        print("Mean absolute error is {}, which is {}% of mean {}.".format(mae, round(mae / data[TARGET_VARIABLE].mean() * 100, 3), TARGET_VARIABLE))
        print("Mean square error is {}.".format(mse))
        print("Mean absolute percentage error is {}%.".format(round(mape * 100, 3)))
        maes.append(mae)
        mses.append(mse)
        mapes.append(mape)
    return maes, mses, mapes


# In[31]:


maes, mses, mapes = kFoldCrossValidation(k = 10)
mMae, sMae = np.mean(maes), np.std(maes)
mRMse, sRMse = np.mean(np.sqrt(mses)), np.std(np.sqrt(mses))
mMape, sMape = np.mean(mapes), np.std(mapes)
meanPrice = data[TARGET_VARIABLE].mean()

rmMae, rsMae = round(mMae, 2), round(sMae / mMae * 100, 3)
rmRMse, rsRMse = round(mRMse, 2), round(sRMse / mRMse * 100, 2)
rmMape, rsMape = round(mMape * 100, 2), round(sMape / mMape, 3)

print("Mean MAE over 5 fold Cross-validation is {} ± {}%, which is {} ± {}% percent of mean {}.".format(rmMae, rsMae, round(mMae / meanPrice * 100, 3), round(sMae / meanPrice * 100, 3), TARGET_VARIABLE))
print("Mean RMSE over 5 fold Cross-validation is {} ± {}%.".format(rmRMse, rsRMse))
print("Mean MAPE over 5 fold Cross-validation is {} ± {}%.".format(rmMape, rsMape))


# In[32]:


kpi_data = [[rmMae, rsMae, rmRMse, rsRMse, rmMape, rsMape, round(data[TARGET_VARIABLE].mean(), 1)]]
kpi = pd.DataFrame(data = kpi_data, columns = KPI_TABLE_COLUMNS).astype('str')


# In[42]:


for col in data.columns:
    plotFeatureDistribution(data, col)


# In[43]:


transposed_train, transposed_test = train_test_split(data, test_size = 0.2, random_state = 42)
model = modelTrain(transposed_train, transposed_test)


# In[45]:


scModel = xgboost.XGBRegressor(n_estimators = 800)
scModel.fit(transposed_train[features].select_dtypes(include='number'), transposed_train[TARGET_VARIABLE])


# In[46]:


def pdpSingle(feature):
    return PDP.from_estimator(scModel, transposed_train[features].select_dtypes(include = 'number'), [feature], kind='both', ice_lines_kw = {"color" : "black"}, centered = True, pd_line_kw = {"color" : "red", "lw" : 3, "linestyle" : "--"})

for col in transposed_train[features].select_dtypes(include = 'number').columns:
    pdpSingle(col).figure_.savefig(KPI_DATA_PATH + 'PDPandICE' + col + 'Plot.jpg')
    plt.close()


# In[47]:


partialDepenadanceFeatureImportance = []
for col in transposed_train[features].select_dtypes(include = 'number').columns:
    partialDepenadanceFeatureImportance.append(np.std(pdep(scModel, transposed_train[features].select_dtypes(include = 'number'), [col])['average']))

cols = transposed_train[features].select_dtypes(include = 'number').columns
ind = [x for _, x in sorted(zip(partialDepenadanceFeatureImportance, cols))]
plt.figure(figsize = (10, 20))
plt.barh(y = ind, width = sorted(partialDepenadanceFeatureImportance))
plt.ylabel('Feature')
plt.xlabel('Feature importance')
plt.savefig(KPI_DATA_PATH + 'PDPimportancePlot.jpg', bbox_inches = 'tight')


# In[50]:


def getQauntSplit(step = 50):
    return [i / 1000 for i in range(0, 1001, step)]

quant = getQauntSplit(int(np.ceil(100000 / len(transposed_train))))
qr = transposed_test[TARGET_VARIABLE].quantile(quant)
qp = pd.Series(model.predict(makeDMatrix(transposed_test[features], transposed_test[TARGET_VARIABLE]))).quantile(quant)

plt.scatter(qr, qp, alpha = 0.8, s = 40)
x = np.linspace(qr.min(),  qr.max())
plt.plot(x, x, c = 'r')
plt.xlabel('Real quantiles for training data')
plt.ylabel('Predicted quantiles for training data')
plt.savefig(KPI_DATA_PATH + 'QQplot.jpg', bbox_inches = 'tight')
plt.close()


# In[52]:


def quantPlot(out, feature, numQuant = 20):
    type = True # Does the feature have more than 20 unique values in train data, 20 because it looks bad with more
    fQuant = None
    cats = None
    if len(out[feature].unique()) > 30 and feature in feature_dtypes.keys() and feature_dtypes[feature] == 'object':
        return
    if len(out[feature].unique()) < numQuant or (feature in feature_dtypes.keys() and feature_dtypes[feature] == 'object'):
        out['QuantCat'] = out[feature]
        type = False
    else:
        fQuant = out[feature].quantile(getQauntSplit(1000 // numQuant)).values
        cats = list(pd.Series(fQuant).unique())
        def getQId(x):
            for i in range(0, len(cats) - 1):
                if x >= cats[i] and x <= cats[i + 1]:
                    return i
            return len(cats) - 1
        out['QuantCat'] = out[feature].apply(lambda x: getQId(x))
        cats = [str(round(cats[i], 2)) + "-" + str(round(cats[i + 1], 2)) for i in range(0, len(cats) - 1)]


    agg_age = out.groupby('QuantCat')

    fig, ax = plt.subplots(figsize = (14, 14))
    if type:
        ax.plot(sorted(list(out['QuantCat'].unique())), agg_age[TARGET_VARIABLE].mean().dropna(), marker = 'o', color = 'b', alpha=0.6)
        ax.plot(sorted(list(out['QuantCat'].unique())), agg_age[ PRED_TARGET_VARIABLE].mean().dropna(), marker = 'o', color = 'r', alpha=0.6)
        ax.set_xticks(sorted(list(out['QuantCat'].unique())))
    else:
        ax.plot(sorted(list(out['QuantCat'].unique())), agg_age[TARGET_VARIABLE].mean().dropna(), marker = 'o', color = 'b', alpha=0.6)
        ax.plot(sorted(list(out['QuantCat'].unique())), agg_age[ PRED_TARGET_VARIABLE].mean().dropna(), marker = 'o', color = 'r', alpha=0.6)
        if feature == 'CarMaker':
            ax.set_xticklabels(sorted([str(x)[:4] for x in list(out['QuantCat'].unique())]))
        else:
            ax.set_xticklabels(sorted(list(out['QuantCat'].unique())))




    plt.title('Mean Real vs Predicted price aggregated by {}'.format(feature))
    ax.legend(['Real', 'Predicted'])
    plt.savefig(KPI_DATA_PATH + 'quantPlot' + col + '.jpg', bbox_inches = 'tight')
    plt.close()
    ret =  pd.DataFrame(agg_age[TARGET_VARIABLE].mean() - agg_age[ PRED_TARGET_VARIABLE].mean())
    #print(ret)
    if type:
        #print(cats, out['QuantCat'].unique())
        ret.index = [cats[i] for i in range(len(cats)) if i in out['QuantCat'].unique()]

    return ret

out = mergePredictions(model)
agg_errors = {}
for col in transposed_test.columns:
    agg_errors[col] = quantPlot(out, col)


# In[53]:


# Function to compare two models based on quantile mean aggregated by a feature

def quantPlots(outs, feature, numQuant = 20):
    type = True
    fQuant = None
    featureCol = outs[0][feature]
    if len(featureCol.unique()) < numQuant:
        for i in range(0, len(outs)):
            outs[i]['QuantCat'] = featureCol
        type = False
    else :
        fQuant = featureCol.quantile(getQauntSplit(1000 // numQuant)).values
        print(fQuant)
        def getQId(x):
            id = 0
            while id < len(fQuant) - 1 and x > fQuant[id]:
                id += 1
                if x <= fQuant[id]:
                    break
            return max(0, id - 1)
        quantCat = featureCol.apply(lambda x : getQId(x))
        for i in range(0, len(outs)):
            outs[i]['QuantCat'] = quantCat

    agg_ages = [outs[i].groupby('QuantCat') for i in range(0, len(outs))]
    fig, ax = plt.subplots(figsize = (14, 14))
    ax.plot(sorted(list(outs[0]['QuantCat'].unique())), agg_ages[0][TARGET_VARIABLE].mean(), marker = 'o', color = 'b', alpha=0.6)
    for agg_age in agg_ages:
        ax.plot(sorted(list(outs[0]['QuantCat'].unique())), agg_age[ PRED_TARGET_VARIABLE].mean(), marker = 'o', color = 'r', alpha=0.6)
    if type:
        ax.set_xticks(range(0, numQuant + 1))
    else:
        ax.set_xticks(sorted(list(outs[0]['QuantCat'].unique())))

    plt.title('Mean Real vs Predicted price aggregated by {}'.format(feature))
    ax.legend(['Real'] + ['Predicted' + str(i) for i in range(1, len(outs) + 1)])
    plt.savefig(KPI_DATA_PATH + 'quantPlots' + col + '.jpg', bbox_inches = 'tight')
    plt.show()
    ret =  pd.DataFrame(np.array([agg_age[TARGET_VARIABLE].mean() - agg_age[PRED_TARGET_VARIABLE].mean() for agg_age in agg_ages])).T
    ret.columns = ['Model' + str(i) + ' error' for i in range(1, len(outs) + 1)]
    print(ret.index)
    if type:
        cats = list(pd.Series(fQuant).unique())
        cats = [str(round(cats[i], 2)) + "-" + str(round(cats[i + 1], 2)) for i in range(0, len(cats) - 1)]
        for i in range(len(cats)):
            print(feature, cats[i], agg_age[TARGET_VARIABLE].mean().iloc[i])
        ret.index = cats

    return ret


# In[54]:


# Inherits FPDF class

class PDF(FPDF):
    def __init__(self):
        super().__init__()
    def header(self):
        self.set_font('Arial', '', 12)
        self.cell(0, 8,'Model', 0, 1, 'C')
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', '', 12)
        self.cell(0, 8, f'Page {self.page_no()}', 0, 0, 'C')
# Create the KPI report PDF

def makePDF():
    ch = 8
    QUANT_W = 120
    QUANT_H = 120


    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)

    for j in range(len(kpi.columns)):
            pdf.cell(w = 25, h = ch, txt = kpi.columns[j], border = 1, ln = j == (len(kpi.columns) - 1), align = 'C')
    for j in range(len(kpi.columns)):
            pdf.cell(w = 25, h = ch, txt = kpi[kpi.columns[j]].iloc[0] + ('%' if j % 2 else ''), border = 1, ln = 0, align = 'C')

    pdf.set_font('Arial', '', 12)
    pdf.cell(w = 40, h = 3 * ch, txt = ' ', border = 0, ln = 1, align = 'C')
    pdf.cell(w = 40, h = ch, txt = 'Parameter', border = 1, ln = 0, align = 'C')
    pdf.cell(w = 40, h = ch, txt = 'Value', border = 1, ln = 0, align = 'C')
    pdf.cell(w = 75, h = ch, txt = 'GridSpace', border = 1, ln = 1, align = 'C')
#    for col, val in params.items():
#        pdf.cell(w = 40, h = ch, txt = col, border = 1, ln = 0, align = 'C')
#        pdf.cell(w = 40, h = ch, txt = str(val), border = 1, ln = 0, align = 'C')
#        pdf.cell(w = 75, h = ch, txt = str(paramsGrid[col]), border = 1,  ln = 1, align = 'C')


    pdf.add_page()

    pdf.image(KPI_DATA_PATH + 'dataOverview.png', w = 200, h = 120, type = 'PNG')

    pdf.add_page()

    idx = 1
    for col in features:
        pdf.image(KPI_DATA_PATH + col + 'Distribution.jpg', x = 10, y = 5 + (idx - 1) * 90, w = 150, h = 90, type = 'JPG')
        idx += 1
        if idx == 4:
            idx = 1
            pdf.add_page()

    if idx > 1:
        pdf.add_page()

    pdf.add_page()

    pdf.image(KPI_DATA_PATH + 'PDPimportancePlot.jpg', w = 160, h = 200, type = 'JPG')
    pdf.image(KPI_DATA_PATH + 'QQplot.jpg', w = 150, h = 150, x = 100, type = 'JPG')
    #pdf.image('quantPlotminPrice.jpg', w = 150, h = 150, x = 100, type = 'JPG')


    pdf.add_page()
    idx = 1
    for col in transposed_train[features].select_dtypes(include = 'number').columns:
        pdf.image(KPI_DATA_PATH + 'PDPandICE' + col + 'Plot.jpg', x = 10, y = 5 + (idx - 1) * 90, w = 150, h = 90, type = 'JPG')
        idx += 1
        if idx == 4:
            idx = 1
            pdf.add_page()

    if idx > 1:
        pdf.add_page()

    idx = 1
    for col in [transposed_test.columns[-1]] + list(transposed_test.columns[ : -1]):
        try :
             pdf.image(KPI_DATA_PATH + 'quantPlot' + col + '.jpg', x = 10, y = 5 + (idx - 1) * (QUANT_H + 20), w = QUANT_W, h = QUANT_H, type = 'JPG')
        except Exception as e:
            continue
        # pdf.image(col + 'Distribution.jpg', x = 10, y = 5 + (idx - 1) * (QUANT_H + 120), w = QUANT_W  / 2, h = QUANT_H / 2, type = 'JPG')
        pdf.set_xy(x = QUANT_W + 20, y = 10 + (idx - 1) * (QUANT_H + 20))
        pdf.set_font('Arial', '', 8)
        pdf.cell( w = 29, h = ch, txt = col + ' agg', border = 1, ln = 0, align = 'C')
        pdf.cell(w = 29, h = ch, txt = 'Mean error', border = 1, ln = 1, align = 'C')
        #print(agg_errors[col].iloc[0])
        mx = agg_errors[col].iloc[  :, 0].max()
        mn = agg_errors[col].iloc[ : , 0].min()

        for i in range(0, len(agg_errors[col])):
            pdf.set_x(QUANT_W + 20)
            cat = agg_errors[col].iloc[i].name
            val = agg_errors[col].iloc[i].values[0]
            print(cat, val)
            if val == mn:
                pdf.set_fill_color(r = 255, g = 0, b = 0)
            elif val == mx:
                pdf.set_fill_color(r = 0, g = 0, b = 255)
            else:
                pdf.set_fill_color(r = 255, g = 255, b = 255)

            pdf.cell(w = 29, h = ch / 2, txt = str(cat), border = 1, fill = 1, ln = 0, align = 'C')
            pdf.cell(w = 29, h = ch / 2, txt = str(round(val, 0)), fill = 1, border = 1, ln = 1, align = 'C')

        idx += 1
        if idx == 3:
            idx = 1
            pdf.add_page()

    return pdf


# In[64]:


makePDF().output(PDF_PATH + TARGET_VARIABLE + '_summary.pdf')


# In[70]:


dmat = xgboost.DMatrix(data[features], data[TARGET_VARIABLE], enable_categorical = True)
param = {'max_depth' : 10, 'eta' : 0.2, 'eval_metric' : 'mae' }
finalModel = xgboost.train(param, dmat, 400)
finalModel.save_model(PDF_PATH + TARGET_VARIABLE +  "_model.json")


# In[ ]:




