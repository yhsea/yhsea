{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "f5ef6386",
   "metadata": {},
   "source": [
    "# Predict cases_recovered"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bc2d90a4",
   "metadata": {},
   "source": [
    "Import libraries and file path"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e9ebaf29",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.neighbors import KNeighborsRegressor\n",
    "\n",
    "from sklearn.preprocessing import PolynomialFeatures\n",
    "from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler\n",
    "\n",
    "from sklearn.metrics import mean_squared_error\n",
    "\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.linear_model import SGDRegressor\n",
    "from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV\n",
    "\n",
    "#ignore warnings\n",
    "from warnings import simplefilter\n",
    "from sklearn.exceptions import ConvergenceWarning\n",
    "simplefilter(\"ignore\", category=ConvergenceWarning)\n",
    "simplefilter(\"ignore\", category=FutureWarning)\n",
    "\n",
    "%matplotlib inline\n",
    "filepath = 'cases_malaysia.csv'"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "07556f67",
   "metadata": {},
   "source": [
    "# Classes"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cc78f6eb",
   "metadata": {},
   "source": [
    "Plot and print error metric"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a720e1cf",
   "metadata": {},
   "outputs": [],
   "source": [
    "def rmse(actual, pred):\n",
    "    return np.sqrt(mean_squared_error(actual, pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6874c2d5",
   "metadata": {},
   "outputs": [],
   "source": [
    "#----------Plot Prediction against Actual---------------\n",
    "def plotPredAct_plot(title, train, train_pred, test, test_pred, row_num, col_num, counterTitle=None, counterList=[], ):\n",
    "    fig, axs = plt.subplots(row_num * 2, col_num, sharex='col', sharey='row')\n",
    "    clr = ['blue', 'red']\n",
    "    lbl = ['train', 'test']\n",
    "    \n",
    "    for i in range(col_num):\n",
    "        for j in range(row_num):\n",
    "            for k in range(2):\n",
    "                data = [train[i * 2 + j], test[i * 2 + j]]\n",
    "                pred = [train_pred[i * 2 + j], test_pred[i * 2 + j]]\n",
    "                if col_num == 1:\n",
    "                    ax = axs[k + j * 2]\n",
    "                else:\n",
    "                    ax = axs[k + j * 2, i]\n",
    "                #show only one legends\n",
    "                if i == 0 and j == 0:\n",
    "                    label = lbl[k]\n",
    "                else:\n",
    "                    label = '_nolegend_'\n",
    "                if k == 0 and counterTitle != None: \n",
    "                    subTitle = counterTitle + str(counterList[i * 2 + j])\n",
    "                    ax.set_title(subTitle, fontsize='small')\n",
    "                ax.scatter(data[k], pred[k], s=5, c=clr[k], label=label)\n",
    "                ax.label_outer()\n",
    "            \n",
    "    #set title, axis title\n",
    "    fig.legend(loc='lower right')\n",
    "    fig.suptitle(str(title))\n",
    "    fig.supylabel('predicted cases')\n",
    "    fig.supxlabel('actual cases')\n",
    "    fig.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d846ba14",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plotPredAct(X_train, X_test, y_train, y_test, fit_model, title, subtitle, loop_range, row_num, col_num, isDegree=False):\n",
    "    train_plt, train_pred_plt, test_plt, test_plt = [], [], [], []\n",
    "    counter = []\n",
    "    for i in loop_range:\n",
    "        degree = i\n",
    "        counter.append(degree)\n",
    "        if(isDegree==False):\n",
    "            y_train_pred, y_test_pred =  fit_model(X_train, X_test, y_train, y_test)\n",
    "        else:\n",
    "            y_train_pred, y_test_pred =  fit_model(degree, X_train, X_test, y_train, y_test)\n",
    "        train_plt.append(train[label_cols])\n",
    "        train_pred_plt.append(y_train_pred)\n",
    "        test_plt.append(test[label_cols])\n",
    "        test_plt.append(y_test_pred)\n",
    "\n",
    "    plotPredAct_plot(title, train_plt, train_pred_plt, test_plt, test_plt, row_num, col_num, subtitle, counter)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "86d986e8",
   "metadata": {},
   "source": [
    "Skew Logarithm Transform"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "27e20c59",
   "metadata": {},
   "outputs": [],
   "source": [
    "def skewTrans(skew_limit=0.75, train=None, target=None):\n",
    "    skew_vals = train.skew() #save skew value into a list\n",
    "    mask = (skew_vals >= skew_limit) & (skew_vals.index != str(target)) #create skew mask >= 0.75\n",
    "    skew_cols = skew_vals[mask] #filter column index with the mask\n",
    "    \n",
    "    log_train = train.copy()  #make a copy\n",
    "    #apply transform to one that exceed skew_limit\n",
    "    for item in train[skew_cols.index]:\n",
    "        log_train[item] = np.log1p(log_train[item])\n",
    "    \n",
    "    return pd.DataFrame(log_train) #return log transformed list"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "82d5020c",
   "metadata": {},
   "source": [
    "Split test, train and feature, label\n",
    "Before spliting the testing and training features are scalled by given scaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c3d304b9",
   "metadata": {},
   "outputs": [],
   "source": [
    "def split_sc(train, test, feature_cols, label_cols, scaler=None):\n",
    "    if(scaler==None): #if no scaler, just split it\n",
    "        return train[feature_cols], test[feature_cols], train[label_cols], test[label_cols]\n",
    "    \n",
    "    #if scaler is available, scaled it before spliting the test train data\n",
    "    X_train_scale = scaler.fit_transform(train[feature_cols])\n",
    "    X_test_scale = scaler.transform(test[feature_cols])\n",
    "    return X_train_scale, X_test_scale, train[label_cols], test[label_cols]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "44f0f846",
   "metadata": {},
   "source": [
    "Error metric"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "956dd919",
   "metadata": {},
   "outputs": [],
   "source": [
    "#print error metric where:\n",
    "#raw: data without preprocessing, log: data being logarithm transformed, mm_sc: MinMaxScaler(), \n",
    "#ss_sc: StandardScaler(), ma_sc: MaxAbsScaler()\n",
    "def errorMetrics(fit_model, title, coefficient=None):\n",
    "    cols = ['raw', 'log', 'mm_sc','mm_sc+log', 'ss_sc', 'ss_sc+log', 'ma_sc', 'ma_sc+log']\n",
    "    models = [None, MinMaxScaler(), StandardScaler(), MaxAbsScaler()]\n",
    "    trains = [train, log_train]\n",
    "    tests = [test, log_test]\n",
    "    \n",
    "    label_cols = 'cases_recovered' #target column\n",
    "    feature_cols = [x for x in train.columns if x != label_cols] #feature column\n",
    "\n",
    "    rmses = pd.DataFrame() #create rmse data frame\n",
    "    \n",
    "    for i in range(0, 4):\n",
    "        for j in range(0, 2):\n",
    "            X_train, X_test, y_train, y_test = split_sc(trains[j], tests[j], feature_cols, label_cols, models[i])\n",
    "            #fit trainning and testing data into model\n",
    "            if coefficient == None:\n",
    "                y_train_pred, y_test_pred =  fit_model(X_train, X_test, y_train, y_test)\n",
    "            else:\n",
    "                y_train_pred, y_test_pred =  fit_model(coefficient, X_train, X_test, y_train, y_test)\n",
    "\n",
    "            #append error metric\n",
    "            new_rmses = pd.Series()\n",
    "            new_rmses['train'] = rmse(y_train_pred, y_train)\n",
    "            new_rmses['test'] = rmse(y_test_pred, y_test)\n",
    "            rmses[cols[i* 2 + j]] = pd.Series(new_rmses)\n",
    "    \n",
    "    rmses_out = rmses.style.set_caption(str(title)).format('{:.2e}')\n",
    "                                               \n",
    "    #return error metric with style\n",
    "    return rmses_out"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "09d67b57",
   "metadata": {},
   "source": [
    "### used to test errorMetrics()\n",
    "\n",
    "display(errorMetrics(knn_reg,'RMSE | KNN (K=' + str(K) + ')' , K))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b867e832",
   "metadata": {},
   "source": [
    "# Initialization"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8063f0b3",
   "metadata": {},
   "source": [
    "Check shape of raw data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "18cfde84",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(filepath)\n",
    "print(\"Row & Column of \" + str(filepath) + \": \" + str(df.shape))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d22d8515",
   "metadata": {},
   "source": [
    "Get data from March 2021 to Auguest 2021"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "60a6e206",
   "metadata": {},
   "outputs": [],
   "source": [
    "#convert date string to date object in dataframe\n",
    "df['date'] = pd.to_datetime(df['date'])\n",
    "\n",
    "#get data from March 2021 to Auguest 2021\n",
    "date_mask = (df['date'] >= \"2021-03-01\") & (df['date'] <= \"2021-08-31\")\n",
    "df_masked = df.loc[date_mask]\n",
    "\n",
    "print(\"Row & Column of data from March 2021 to Auguest 2021: \" + str(df_masked.shape))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b2ac6200",
   "metadata": {},
   "source": [
    "Explore each column, check if any data is null"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2f5c6b10",
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "df_sum = pd.isnull(df_masked) \n",
    "df_sum.sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d5214298",
   "metadata": {},
   "source": [
    "Check column type of the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a4ab4466",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_masked.columns"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "396d460b",
   "metadata": {},
   "source": [
    "Seperate data into features and labels."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8150d0ee",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = df_masked.copy() #copy the whole data range \n",
    "del data['date'] #drop date column\n",
    "\n",
    "train, test = train_test_split(data, test_size=0.3, random_state=42) #split train test data\n",
    "label_cols = 'cases_recovered' #target column\n",
    "feature_cols = [x for x in train.columns if x != label_cols] #feature columns"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c57bfb5e",
   "metadata": {},
   "source": [
    "Check the skewness before and after the logarithm transform on the features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9022136f",
   "metadata": {},
   "outputs": [],
   "source": [
    "skew_limit = 0.75\n",
    "#log transform the train and test date excluding 'cases_recovered'\n",
    "log_train = skewTrans(skew_limit, train, 'cases_recovered')\n",
    "log_test = skewTrans(skew_limit, test, 'cases_recovered')\n",
    "#display transformed and original skew list of train data\n",
    "skew_list = pd.DataFrame()\n",
    "skew_list['transformed'] = log_train.skew()\n",
    "skew_list['original'] = train.skew()\n",
    "skew_list"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "0616f573",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: title={'center': 'log'}>"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#display histogram of 'cases_new' before and after log transform\n",
    "fig, axs = plt.subplots(1, 2)\n",
    "fig.suptitle('cases_new')\n",
    "axs[0].set_title('original', size='small')\n",
    "axs[1].set_title('log', size='small')\n",
    "train['cases_new'].hist(ax=axs[0])\n",
    "log_train['cases_new'].hist(ax=axs[1])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b2ee075f",
   "metadata": {},
   "source": [
    "## K-Nearest Neighbor Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "305221f6",
   "metadata": {},
   "outputs": [],
   "source": [
    "def knn_reg(K, X_train, X_test, y_train, y_test):\n",
    "    #Create a N-Nearest Neighbors classifier of 5 neighbors\n",
    "    knn = KNeighborsRegressor(n_neighbors=K)\n",
    "\n",
    "    #Fill the model to the training set\n",
    "    knn.fit(X_train, y_train)\n",
    "\n",
    "    #prediction on the test set\n",
    "    y_train_pred = knn.predict(X_train)\n",
    "    y_test_pred = knn.predict(X_test)\n",
    "\n",
    "    return (y_train_pred, y_test_pred)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a8018577",
   "metadata": {},
   "source": [
    "Generate table of RMSE with KNN model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "42ea7af3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style type=\"text/css\">\n",
       "</style>\n",
       "<table id=\"T_db68e\">\n",
       "  <caption>RMSE | KNN (K=1)</caption>\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th class=\"blank level0\" >&nbsp;</th>\n",
       "      <th id=\"T_db68e_level0_col0\" class=\"col_heading level0 col0\" >raw</th>\n",
       "      <th id=\"T_db68e_level0_col1\" class=\"col_heading level0 col1\" >log</th>\n",
       "      <th id=\"T_db68e_level0_col2\" class=\"col_heading level0 col2\" >mm_sc</th>\n",
       "      <th id=\"T_db68e_level0_col3\" class=\"col_heading level0 col3\" >mm_sc+log</th>\n",
       "      <th id=\"T_db68e_level0_col4\" class=\"col_heading level0 col4\" >ss_sc</th>\n",
       "      <th id=\"T_db68e_level0_col5\" class=\"col_heading level0 col5\" >ss_sc+log</th>\n",
       "      <th id=\"T_db68e_level0_col6\" class=\"col_heading level0 col6\" >ma_sc</th>\n",
       "      <th id=\"T_db68e_level0_col7\" class=\"col_heading level0 col7\" >ma_sc+log</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th id=\"T_db68e_level0_row0\" class=\"row_heading level0 row0\" >train</th>\n",
       "      <td id=\"T_db68e_row0_col0\" class=\"data row0 col0\" >0.00e+00</td>\n",
       "      <td id=\"T_db68e_row0_col1\" class=\"data row0 col1\" >0.00e+00</td>\n",
       "      <td id=\"T_db68e_row0_col2\" class=\"data row0 col2\" >0.00e+00</td>\n",
       "      <td id=\"T_db68e_row0_col3\" class=\"data row0 col3\" >0.00e+00</td>\n",
       "      <td id=\"T_db68e_row0_col4\" class=\"data row0 col4\" >0.00e+00</td>\n",
       "      <td id=\"T_db68e_row0_col5\" class=\"data row0 col5\" >0.00e+00</td>\n",
       "      <td id=\"T_db68e_row0_col6\" class=\"data row0 col6\" >0.00e+00</td>\n",
       "      <td id=\"T_db68e_row0_col7\" class=\"data row0 col7\" >0.00e+00</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th id=\"T_db68e_level0_row1\" class=\"row_heading level0 row1\" >test</th>\n",
       "      <td id=\"T_db68e_row1_col0\" class=\"data row1 col0\" >8.16e+02</td>\n",
       "      <td id=\"T_db68e_row1_col1\" class=\"data row1 col1\" >2.87e+03</td>\n",
       "      <td id=\"T_db68e_row1_col2\" class=\"data row1 col2\" >1.40e+03</td>\n",
       "      <td id=\"T_db68e_row1_col3\" class=\"data row1 col3\" >2.33e+03</td>\n",
       "      <td id=\"T_db68e_row1_col4\" class=\"data row1 col4\" >1.42e+03</td>\n",
       "      <td id=\"T_db68e_row1_col5\" class=\"data row1 col5\" >2.54e+03</td>\n",
       "      <td id=\"T_db68e_row1_col6\" class=\"data row1 col6\" >1.39e+03</td>\n",
       "      <td id=\"T_db68e_row1_col7\" class=\"data row1 col7\" >4.25e+03</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n"
      ],
      "text/plain": [
       "<pandas.io.formats.style.Styler at 0x258034f2e90>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style type=\"text/css\">\n",
       "</style>\n",
       "<table id=\"T_55dba\">\n",
       "  <caption>RMSE | KNN (K=4)</caption>\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th class=\"blank level0\" >&nbsp;</th>\n",
       "      <th id=\"T_55dba_level0_col0\" class=\"col_heading level0 col0\" >raw</th>\n",
       "      <th id=\"T_55dba_level0_col1\" class=\"col_heading level0 col1\" >log</th>\n",
       "      <th id=\"T_55dba_level0_col2\" class=\"col_heading level0 col2\" >mm_sc</th>\n",
       "      <th id=\"T_55dba_level0_col3\" class=\"col_heading level0 col3\" >mm_sc+log</th>\n",
       "      <th id=\"T_55dba_level0_col4\" class=\"col_heading level0 col4\" >ss_sc</th>\n",
       "      <th id=\"T_55dba_level0_col5\" class=\"col_heading level0 col5\" >ss_sc+log</th>\n",
       "      <th id=\"T_55dba_level0_col6\" class=\"col_heading level0 col6\" >ma_sc</th>\n",
       "      <th id=\"T_55dba_level0_col7\" class=\"col_heading level0 col7\" >ma_sc+log</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th id=\"T_55dba_level0_row0\" class=\"row_heading level0 row0\" >train</th>\n",
       "      <td id=\"T_55dba_row0_col0\" class=\"data row0 col0\" >6.88e+02</td>\n",
       "      <td id=\"T_55dba_row0_col1\" class=\"data row0 col1\" >1.90e+03</td>\n",
       "      <td id=\"T_55dba_row0_col2\" class=\"data row0 col2\" >7.21e+02</td>\n",
       "      <td id=\"T_55dba_row0_col3\" class=\"data row0 col3\" >9.08e+02</td>\n",
       "      <td id=\"T_55dba_row0_col4\" class=\"data row0 col4\" >6.74e+02</td>\n",
       "      <td id=\"T_55dba_row0_col5\" class=\"data row0 col5\" >9.78e+02</td>\n",
       "      <td id=\"T_55dba_row0_col6\" class=\"data row0 col6\" >7.21e+02</td>\n",
       "      <td id=\"T_55dba_row0_col7\" class=\"data row0 col7\" >1.07e+03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th id=\"T_55dba_level0_row1\" class=\"row_heading level0 row1\" >test</th>\n",
       "      <td id=\"T_55dba_row1_col0\" class=\"data row1 col0\" >8.35e+02</td>\n",
       "      <td id=\"T_55dba_row1_col1\" class=\"data row1 col1\" >2.72e+03</td>\n",
       "      <td id=\"T_55dba_row1_col2\" class=\"data row1 col2\" >8.38e+02</td>\n",
       "      <td id=\"T_55dba_row1_col3\" class=\"data row1 col3\" >1.95e+03</td>\n",
       "      <td id=\"T_55dba_row1_col4\" class=\"data row1 col4\" >1.05e+03</td>\n",
       "      <td id=\"T_55dba_row1_col5\" class=\"data row1 col5\" >1.60e+03</td>\n",
       "      <td id=\"T_55dba_row1_col6\" class=\"data row1 col6\" >8.43e+02</td>\n",
       "      <td id=\"T_55dba_row1_col7\" class=\"data row1 col7\" >3.89e+03</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n"
      ],
      "text/plain": [
       "<pandas.io.formats.style.Styler at 0x258077a1350>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style type=\"text/css\">\n",
       "</style>\n",
       "<table id=\"T_302f4\">\n",
       "  <caption>RMSE | KNN (K=7)</caption>\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th class=\"blank level0\" >&nbsp;</th>\n",
       "      <th id=\"T_302f4_level0_col0\" class=\"col_heading level0 col0\" >raw</th>\n",
       "      <th id=\"T_302f4_level0_col1\" class=\"col_heading level0 col1\" >log</th>\n",
       "      <th id=\"T_302f4_level0_col2\" class=\"col_heading level0 col2\" >mm_sc</th>\n",
       "      <th id=\"T_302f4_level0_col3\" class=\"col_heading level0 col3\" >mm_sc+log</th>\n",
       "      <th id=\"T_302f4_level0_col4\" class=\"col_heading level0 col4\" >ss_sc</th>\n",
       "      <th id=\"T_302f4_level0_col5\" class=\"col_heading level0 col5\" >ss_sc+log</th>\n",
       "      <th id=\"T_302f4_level0_col6\" class=\"col_heading level0 col6\" >ma_sc</th>\n",
       "      <th id=\"T_302f4_level0_col7\" class=\"col_heading level0 col7\" >ma_sc+log</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th id=\"T_302f4_level0_row0\" class=\"row_heading level0 row0\" >train</th>\n",
       "      <td id=\"T_302f4_row0_col0\" class=\"data row0 col0\" >8.03e+02</td>\n",
       "      <td id=\"T_302f4_row0_col1\" class=\"data row0 col1\" >2.23e+03</td>\n",
       "      <td id=\"T_302f4_row0_col2\" class=\"data row0 col2\" >8.03e+02</td>\n",
       "      <td id=\"T_302f4_row0_col3\" class=\"data row0 col3\" >1.05e+03</td>\n",
       "      <td id=\"T_302f4_row0_col4\" class=\"data row0 col4\" >8.04e+02</td>\n",
       "      <td id=\"T_302f4_row0_col5\" class=\"data row0 col5\" >1.14e+03</td>\n",
       "      <td id=\"T_302f4_row0_col6\" class=\"data row0 col6\" >8.10e+02</td>\n",
       "      <td id=\"T_302f4_row0_col7\" class=\"data row0 col7\" >1.27e+03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th id=\"T_302f4_level0_row1\" class=\"row_heading level0 row1\" >test</th>\n",
       "      <td id=\"T_302f4_row1_col0\" class=\"data row1 col0\" >7.75e+02</td>\n",
       "      <td id=\"T_302f4_row1_col1\" class=\"data row1 col1\" >2.60e+03</td>\n",
       "      <td id=\"T_302f4_row1_col2\" class=\"data row1 col2\" >9.51e+02</td>\n",
       "      <td id=\"T_302f4_row1_col3\" class=\"data row1 col3\" >1.53e+03</td>\n",
       "      <td id=\"T_302f4_row1_col4\" class=\"data row1 col4\" >9.35e+02</td>\n",
       "      <td id=\"T_302f4_row1_col5\" class=\"data row1 col5\" >1.53e+03</td>\n",
       "      <td id=\"T_302f4_row1_col6\" class=\"data row1 col6\" >9.60e+02</td>\n",
       "      <td id=\"T_302f4_row1_col7\" class=\"data row1 col7\" >3.65e+03</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n"
      ],
      "text/plain": [
       "<pandas.io.formats.style.Styler at 0x25806bc0450>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style type=\"text/css\">\n",
       "</style>\n",
       "<table id=\"T_72dff\">\n",
       "  <caption>RMSE | KNN (K=10)</caption>\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th class=\"blank level0\" >&nbsp;</th>\n",
       "      <th id=\"T_72dff_level0_col0\" class=\"col_heading level0 col0\" >raw</th>\n",
       "      <th id=\"T_72dff_level0_col1\" class=\"col_heading level0 col1\" >log</th>\n",
       "      <th id=\"T_72dff_level0_col2\" class=\"col_heading level0 col2\" >mm_sc</th>\n",
       "      <th id=\"T_72dff_level0_col3\" class=\"col_heading level0 col3\" >mm_sc+log</th>\n",
       "      <th id=\"T_72dff_level0_col4\" class=\"col_heading level0 col4\" >ss_sc</th>\n",
       "      <th id=\"T_72dff_level0_col5\" class=\"col_heading level0 col5\" >ss_sc+log</th>\n",
       "      <th id=\"T_72dff_level0_col6\" class=\"col_heading level0 col6\" >ma_sc</th>\n",
       "      <th id=\"T_72dff_level0_col7\" class=\"col_heading level0 col7\" >ma_sc+log</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th id=\"T_72dff_level0_row0\" class=\"row_heading level0 row0\" >train</th>\n",
       "      <td id=\"T_72dff_row0_col0\" class=\"data row0 col0\" >8.74e+02</td>\n",
       "      <td id=\"T_72dff_row0_col1\" class=\"data row0 col1\" >2.37e+03</td>\n",
       "      <td id=\"T_72dff_row0_col2\" class=\"data row0 col2\" >8.21e+02</td>\n",
       "      <td id=\"T_72dff_row0_col3\" class=\"data row0 col3\" >1.11e+03</td>\n",
       "      <td id=\"T_72dff_row0_col4\" class=\"data row0 col4\" >8.70e+02</td>\n",
       "      <td id=\"T_72dff_row0_col5\" class=\"data row0 col5\" >1.22e+03</td>\n",
       "      <td id=\"T_72dff_row0_col6\" class=\"data row0 col6\" >8.17e+02</td>\n",
       "      <td id=\"T_72dff_row0_col7\" class=\"data row0 col7\" >1.30e+03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th id=\"T_72dff_level0_row1\" class=\"row_heading level0 row1\" >test</th>\n",
       "      <td id=\"T_72dff_row1_col0\" class=\"data row1 col0\" >7.98e+02</td>\n",
       "      <td id=\"T_72dff_row1_col1\" class=\"data row1 col1\" >2.51e+03</td>\n",
       "      <td id=\"T_72dff_row1_col2\" class=\"data row1 col2\" >9.12e+02</td>\n",
       "      <td id=\"T_72dff_row1_col3\" class=\"data row1 col3\" >1.40e+03</td>\n",
       "      <td id=\"T_72dff_row1_col4\" class=\"data row1 col4\" >8.89e+02</td>\n",
       "      <td id=\"T_72dff_row1_col5\" class=\"data row1 col5\" >1.49e+03</td>\n",
       "      <td id=\"T_72dff_row1_col6\" class=\"data row1 col6\" >9.31e+02</td>\n",
       "      <td id=\"T_72dff_row1_col7\" class=\"data row1 col7\" >3.61e+03</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n"
      ],
      "text/plain": [
       "<pandas.io.formats.style.Styler at 0x258077afe90>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style type=\"text/css\">\n",
       "</style>\n",
       "<table id=\"T_273e1\">\n",
       "  <caption>RMSE | KNN (K=13)</caption>\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th class=\"blank level0\" >&nbsp;</th>\n",
       "      <th id=\"T_273e1_level0_col0\" class=\"col_heading level0 col0\" >raw</th>\n",
       "      <th id=\"T_273e1_level0_col1\" class=\"col_heading level0 col1\" >log</th>\n",
       "      <th id=\"T_273e1_level0_col2\" class=\"col_heading level0 col2\" >mm_sc</th>\n",
       "      <th id=\"T_273e1_level0_col3\" class=\"col_heading level0 col3\" >mm_sc+log</th>\n",
       "      <th id=\"T_273e1_level0_col4\" class=\"col_heading level0 col4\" >ss_sc</th>\n",
       "      <th id=\"T_273e1_level0_col5\" class=\"col_heading level0 col5\" >ss_sc+log</th>\n",
       "      <th id=\"T_273e1_level0_col6\" class=\"col_heading level0 col6\" >ma_sc</th>\n",
       "      <th id=\"T_273e1_level0_col7\" class=\"col_heading level0 col7\" >ma_sc+log</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th id=\"T_273e1_level0_row0\" class=\"row_heading level0 row0\" >train</th>\n",
       "      <td id=\"T_273e1_row0_col0\" class=\"data row0 col0\" >9.43e+02</td>\n",
       "      <td id=\"T_273e1_row0_col1\" class=\"data row0 col1\" >2.39e+03</td>\n",
       "      <td id=\"T_273e1_row0_col2\" class=\"data row0 col2\" >8.93e+02</td>\n",
       "      <td id=\"T_273e1_row0_col3\" class=\"data row0 col3\" >1.24e+03</td>\n",
       "      <td id=\"T_273e1_row0_col4\" class=\"data row0 col4\" >9.08e+02</td>\n",
       "      <td id=\"T_273e1_row0_col5\" class=\"data row0 col5\" >1.32e+03</td>\n",
       "      <td id=\"T_273e1_row0_col6\" class=\"data row0 col6\" >8.79e+02</td>\n",
       "      <td id=\"T_273e1_row0_col7\" class=\"data row0 col7\" >1.40e+03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th id=\"T_273e1_level0_row1\" class=\"row_heading level0 row1\" >test</th>\n",
       "      <td id=\"T_273e1_row1_col0\" class=\"data row1 col0\" >8.44e+02</td>\n",
       "      <td id=\"T_273e1_row1_col1\" class=\"data row1 col1\" >2.43e+03</td>\n",
       "      <td id=\"T_273e1_row1_col2\" class=\"data row1 col2\" >9.77e+02</td>\n",
       "      <td id=\"T_273e1_row1_col3\" class=\"data row1 col3\" >1.42e+03</td>\n",
       "      <td id=\"T_273e1_row1_col4\" class=\"data row1 col4\" >1.02e+03</td>\n",
       "      <td id=\"T_273e1_row1_col5\" class=\"data row1 col5\" >1.44e+03</td>\n",
       "      <td id=\"T_273e1_row1_col6\" class=\"data row1 col6\" >1.00e+03</td>\n",
       "      <td id=\"T_273e1_row1_col7\" class=\"data row1 col7\" >3.59e+03</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n"
      ],
      "text/plain": [
       "<pandas.io.formats.style.Styler at 0x2587f76ced0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style type=\"text/css\">\n",
       "</style>\n",
       "<table id=\"T_987cb\">\n",
       "  <caption>RMSE | KNN (K=16)</caption>\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th class=\"blank level0\" >&nbsp;</th>\n",
       "      <th id=\"T_987cb_level0_col0\" class=\"col_heading level0 col0\" >raw</th>\n",
       "      <th id=\"T_987cb_level0_col1\" class=\"col_heading level0 col1\" >log</th>\n",
       "      <th id=\"T_987cb_level0_col2\" class=\"col_heading level0 col2\" >mm_sc</th>\n",
       "      <th id=\"T_987cb_level0_col3\" class=\"col_heading level0 col3\" >mm_sc+log</th>\n",
       "      <th id=\"T_987cb_level0_col4\" class=\"col_heading level0 col4\" >ss_sc</th>\n",
       "      <th id=\"T_987cb_level0_col5\" class=\"col_heading level0 col5\" >ss_sc+log</th>\n",
       "      <th id=\"T_987cb_level0_col6\" class=\"col_heading level0 col6\" >ma_sc</th>\n",
       "      <th id=\"T_987cb_level0_col7\" class=\"col_heading level0 col7\" >ma_sc+log</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th id=\"T_987cb_level0_row0\" class=\"row_heading level0 row0\" >train</th>\n",
       "      <td id=\"T_987cb_row0_col0\" class=\"data row0 col0\" >9.77e+02</td>\n",
       "      <td id=\"T_987cb_row0_col1\" class=\"data row0 col1\" >2.48e+03</td>\n",
       "      <td id=\"T_987cb_row0_col2\" class=\"data row0 col2\" >9.61e+02</td>\n",
       "      <td id=\"T_987cb_row0_col3\" class=\"data row0 col3\" >1.39e+03</td>\n",
       "      <td id=\"T_987cb_row0_col4\" class=\"data row0 col4\" >9.65e+02</td>\n",
       "      <td id=\"T_987cb_row0_col5\" class=\"data row0 col5\" >1.45e+03</td>\n",
       "      <td id=\"T_987cb_row0_col6\" class=\"data row0 col6\" >9.60e+02</td>\n",
       "      <td id=\"T_987cb_row0_col7\" class=\"data row0 col7\" >1.48e+03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th id=\"T_987cb_level0_row1\" class=\"row_heading level0 row1\" >test</th>\n",
       "      <td id=\"T_987cb_row1_col0\" class=\"data row1 col0\" >8.43e+02</td>\n",
       "      <td id=\"T_987cb_row1_col1\" class=\"data row1 col1\" >2.42e+03</td>\n",
       "      <td id=\"T_987cb_row1_col2\" class=\"data row1 col2\" >9.97e+02</td>\n",
       "      <td id=\"T_987cb_row1_col3\" class=\"data row1 col3\" >1.37e+03</td>\n",
       "      <td id=\"T_987cb_row1_col4\" class=\"data row1 col4\" >1.04e+03</td>\n",
       "      <td id=\"T_987cb_row1_col5\" class=\"data row1 col5\" >1.49e+03</td>\n",
       "      <td id=\"T_987cb_row1_col6\" class=\"data row1 col6\" >9.81e+02</td>\n",
       "      <td id=\"T_987cb_row1_col7\" class=\"data row1 col7\" >3.50e+03</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n"
      ],
      "text/plain": [
       "<pandas.io.formats.style.Styler at 0x258077a1290>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style type=\"text/css\">\n",
       "</style>\n",
       "<table id=\"T_6aa00\">\n",
       "  <caption>RMSE | KNN (K=19)</caption>\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th class=\"blank level0\" >&nbsp;</th>\n",
       "      <th id=\"T_6aa00_level0_col0\" class=\"col_heading level0 col0\" >raw</th>\n",
       "      <th id=\"T_6aa00_level0_col1\" class=\"col_heading level0 col1\" >log</th>\n",
       "      <th id=\"T_6aa00_level0_col2\" class=\"col_heading level0 col2\" >mm_sc</th>\n",
       "      <th id=\"T_6aa00_level0_col3\" class=\"col_heading level0 col3\" >mm_sc+log</th>\n",
       "      <th id=\"T_6aa00_level0_col4\" class=\"col_heading level0 col4\" >ss_sc</th>\n",
       "      <th id=\"T_6aa00_level0_col5\" class=\"col_heading level0 col5\" >ss_sc+log</th>\n",
       "      <th id=\"T_6aa00_level0_col6\" class=\"col_heading level0 col6\" >ma_sc</th>\n",
       "      <th id=\"T_6aa00_level0_col7\" class=\"col_heading level0 col7\" >ma_sc+log</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th id=\"T_6aa00_level0_row0\" class=\"row_heading level0 row0\" >train</th>\n",
       "      <td id=\"T_6aa00_row0_col0\" class=\"data row0 col0\" >1.05e+03</td>\n",
       "      <td id=\"T_6aa00_row0_col1\" class=\"data row0 col1\" >2.61e+03</td>\n",
       "      <td id=\"T_6aa00_row0_col2\" class=\"data row0 col2\" >1.05e+03</td>\n",
       "      <td id=\"T_6aa00_row0_col3\" class=\"data row0 col3\" >1.43e+03</td>\n",
       "      <td id=\"T_6aa00_row0_col4\" class=\"data row0 col4\" >1.05e+03</td>\n",
       "      <td id=\"T_6aa00_row0_col5\" class=\"data row0 col5\" >1.61e+03</td>\n",
       "      <td id=\"T_6aa00_row0_col6\" class=\"data row0 col6\" >1.05e+03</td>\n",
       "      <td id=\"T_6aa00_row0_col7\" class=\"data row0 col7\" >1.58e+03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th id=\"T_6aa00_level0_row1\" class=\"row_heading level0 row1\" >test</th>\n",
       "      <td id=\"T_6aa00_row1_col0\" class=\"data row1 col0\" >8.99e+02</td>\n",
       "      <td id=\"T_6aa00_row1_col1\" class=\"data row1 col1\" >2.52e+03</td>\n",
       "      <td id=\"T_6aa00_row1_col2\" class=\"data row1 col2\" >1.05e+03</td>\n",
       "      <td id=\"T_6aa00_row1_col3\" class=\"data row1 col3\" >1.36e+03</td>\n",
       "      <td id=\"T_6aa00_row1_col4\" class=\"data row1 col4\" >1.02e+03</td>\n",
       "      <td id=\"T_6aa00_row1_col5\" class=\"data row1 col5\" >1.46e+03</td>\n",
       "      <td id=\"T_6aa00_row1_col6\" class=\"data row1 col6\" >1.06e+03</td>\n",
       "      <td id=\"T_6aa00_row1_col7\" class=\"data row1 col7\" >3.42e+03</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n"
      ],
      "text/plain": [
       "<pandas.io.formats.style.Styler at 0x258077acc90>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style type=\"text/css\">\n",
       "</style>\n",
       "<table id=\"T_454c9\">\n",
       "  <caption>RMSE | KNN (K=22)</caption>\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th class=\"blank level0\" >&nbsp;</th>\n",
       "      <th id=\"T_454c9_level0_col0\" class=\"col_heading level0 col0\" >raw</th>\n",
       "      <th id=\"T_454c9_level0_col1\" class=\"col_heading level0 col1\" >log</th>\n",
       "      <th id=\"T_454c9_level0_col2\" class=\"col_heading level0 col2\" >mm_sc</th>\n",
       "      <th id=\"T_454c9_level0_col3\" class=\"col_heading level0 col3\" >mm_sc+log</th>\n",
       "      <th id=\"T_454c9_level0_col4\" class=\"col_heading level0 col4\" >ss_sc</th>\n",
       "      <th id=\"T_454c9_level0_col5\" class=\"col_heading level0 col5\" >ss_sc+log</th>\n",
       "      <th id=\"T_454c9_level0_col6\" class=\"col_heading level0 col6\" >ma_sc</th>\n",
       "      <th id=\"T_454c9_level0_col7\" class=\"col_heading level0 col7\" >ma_sc+log</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th id=\"T_454c9_level0_row0\" class=\"row_heading level0 row0\" >train</th>\n",
       "      <td id=\"T_454c9_row0_col0\" class=\"data row0 col0\" >1.17e+03</td>\n",
       "      <td id=\"T_454c9_row0_col1\" class=\"data row0 col1\" >2.60e+03</td>\n",
       "      <td id=\"T_454c9_row0_col2\" class=\"data row0 col2\" >1.19e+03</td>\n",
       "      <td id=\"T_454c9_row0_col3\" class=\"data row0 col3\" >1.57e+03</td>\n",
       "      <td id=\"T_454c9_row0_col4\" class=\"data row0 col4\" >1.19e+03</td>\n",
       "      <td id=\"T_454c9_row0_col5\" class=\"data row0 col5\" >1.67e+03</td>\n",
       "      <td id=\"T_454c9_row0_col6\" class=\"data row0 col6\" >1.18e+03</td>\n",
       "      <td id=\"T_454c9_row0_col7\" class=\"data row0 col7\" >1.69e+03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th id=\"T_454c9_level0_row1\" class=\"row_heading level0 row1\" >test</th>\n",
       "      <td id=\"T_454c9_row1_col0\" class=\"data row1 col0\" >9.52e+02</td>\n",
       "      <td id=\"T_454c9_row1_col1\" class=\"data row1 col1\" >2.57e+03</td>\n",
       "      <td id=\"T_454c9_row1_col2\" class=\"data row1 col2\" >1.09e+03</td>\n",
       "      <td id=\"T_454c9_row1_col3\" class=\"data row1 col3\" >1.41e+03</td>\n",
       "      <td id=\"T_454c9_row1_col4\" class=\"data row1 col4\" >1.11e+03</td>\n",
       "      <td id=\"T_454c9_row1_col5\" class=\"data row1 col5\" >1.54e+03</td>\n",
       "      <td id=\"T_454c9_row1_col6\" class=\"data row1 col6\" >1.09e+03</td>\n",
       "      <td id=\"T_454c9_row1_col7\" class=\"data row1 col7\" >3.29e+03</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n"
      ],
      "text/plain": [
       "<pandas.io.formats.style.Styler at 0x258077a4290>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "for i in range(1, 23, 3):\n",
    "    K = i\n",
    "    display(errorMetrics(knn_reg,'RMSE | KNN (K=' + str(K) + ')' , K))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c3a88880",
   "metadata": {},
   "source": [
    "Generate plots of predicted vs actual cases with KNN models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "60cb6d9f",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAoAAAAHlCAYAAABlHE1XAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAACUjUlEQVR4nO3deVxU5f4H8M+wDYswgopILkCaS5iVlUsqkgIuaN5btxIj7ZbVTTRTszQVsMKtW1lWdu+tbJW6ZdYtIzEVJdDM0jTTFhdcQFzYZRt4fn+c3wwzw6wwwzBzPu/Xa14wZ54588zMd57znGc7CiGEABERERHJhoezM0BEREREbYsVQCIiIiKZYQWQiIiISGZYASQiIiKSGVYAiYiIiGSGFUAiIiIimWEFkIiIiEhmWAEkIiIikhlWAImIiIhkhhVAonZkw4YNUCgU+OGHH/S2X7x4ETfddBM6dOiA7OxsAEBaWhoUCgVCQ0NRUVHRbF8RERFITEzU26ZQKKBQKLBy5UqrX9vQzp07tftRKBTw9PREly5dMGnSJIvPdSeaz+vkyZNOef36+nr069fP6HfpDMnJyZgyZYqzs0FEVmIFkKidO3PmDEaOHInjx49j27ZtiIuL03v8woULWL16tU37XLlyJS5fvtyqfGVkZCA/Px87d+7E0qVLkZeXh5iYGPz++++t2q+rmDhxIvLz89GtWzenvP5rr72GkpISzJ492ymvbygtLQ1fffUVtm/f7uysEJEVWAEkasd+//133HrrrSgrK0NOTg6GDh3aLM24cePw4osvoqioyKp9jh07FlVVVXjuuedalbc+ffpg6NChGDlyJObMmYMXX3wRV65cwfvvv9+q/bbElStX2vw1u3TpgqFDh0KpVLb5a6vVaqxZswZ///vfERAQYDZtW302V199NcaNG9duWiSJyDxWAInaqQMHDmDEiBHw8vJCbm4uBg4caDTds88+C7VajbS0NKv227dvXzzwwAN49dVXcerUKbvl96abbgIAnD9/Xm/777//jqSkJISGhkKpVKJ///549dVXmz3/l19+QXx8PPz9/dGlSxfMmjULX331FRQKBXbu3KlNN3r0aERHR2PXrl0YPnw4/P398fe//x0AUF5ejgULFiAyMhI+Pj646qqrMHfuXFRVVem91n//+18MGTIEKpUK/v7+iIqK0u4DABobG/Hss8+ib9++8PPzQ8eOHXHddddh7dq12jSmuoDfeustDBo0CL6+vggJCcFf/vIX/Prrr3ppZsyYgQ4dOuCPP/7AhAkT0KFDB/To0QPz589HbW2txc/6iy++wNmzZ5GcnKy3XTMs4Mcff8Sdd96J4OBgXH311QCAH374Affccw8iIiLg5+eHiIgITJ06VS8GysvL4eXlhTVr1mi3Xbx4ER4eHlCpVFCr1drtc+bMQZcuXSCE0G5LTk7Gtm3b8Oeff1p8D0TkXKwAErVDubm5GD16NEJDQ5Gbm4uoqCiTaXv16oVHH30Ub775Jn777Ter9p+WlgZPT08sXbrUXlnGiRMnAADXXHONdtuRI0dw88034/Dhw/jnP/+JL7/8EhMnTsScOXOQnp6uTVdYWIiYmBgcO3YMr7/+Ot59911UVFQgJSXF6GsVFhbi3nvvRVJSErZs2YJHH30UV65cQUxMDN555x3MmTMHX3/9NZ588kls2LABkydP1lZU8vPzcffddyMqKgqZmZn46quvsGzZMr3KzerVq5GWloapU6fiq6++wkcffYQHHngApaWlZj+DFStW4IEHHsC1116LTZs2Ye3atfj5558xbNiwZl3j9fX1mDx5MsaMGYPPP/8cf//73/Hiiy9i1apVFj/rr776CqGhoRgwYIDRx//617+id+/e+O9//4v169cDAE6ePIm+ffvipZdewjfffINVq1ahsLAQN998My5evAgACAoKws0334xt27Zp9/Xtt99CqVSioqIC33//vXb7tm3bcNttt0GhUGi3jR49GkIIbNmyxeJ7ICInE0TUbrz99tsCgAAgVCqVKC4uNpk2NTVVABAXLlwQFy9eFCqVStxxxx3ax3v16iUmTpyo9xwAYtasWUIIIZ5++mnh4eEhDh48qPfa+/btM5vHHTt2CADio48+EvX19eLKlSviu+++E3379hUDBgwQJSUl2rQJCQmie/fuoqysTG8fKSkpwtfXV1y+fFkIIcQTTzwhFAqF+OWXX/TSJSQkCABix44d2m0xMTECgPj222/10q5YsUJ4eHg0y/8nn3wiAIgtW7YIIYR4/vnnBQBRWlpq8j0mJiaK66+/3uznoPm8Tpw4IYQQoqSkRPj5+YkJEybopSsoKBBKpVIkJSVpt02fPl0AEB9//LFe2gkTJoi+ffuafV0hhOjfv78YN25cs+2amFi2bJnFfajValFZWSkCAgLE2rVrtduXLFki/Pz8RE1NjRBCiAcffFCMGzdOXHfddSI9PV0IIcTZs2cFAPGvf/2r2X6vuuoqcffdd1t8fSJyLrYAErVDkydPRllZGebOnYuGhgaL6Tt16oQnn3wSn376Kfbu3WvVayxcuBAhISF48sknW5THu+++G97e3vD398ett96K8vJyfPXVV+jYsSMAoKamBt9++y3+8pe/wN/fH2q1WnubMGECampqsGfPHgBATk4OoqOjm7VoTZ061ehrBwcH47bbbtPb9uWXXyI6OhrXX3+93mslJCTodSPffPPNAIC77roLH3/8Mc6ePdts/7fccgsOHjyIRx99FN988w3Ky8stfh75+fmorq7GjBkz9Lb36NEDt912G7799lu97QqFApMmTdLbdt1111nVLX/u3DmEhoaafPyOO+5otq2yshJPPvkkevfuDS8vL3h5eaFDhw6oqqrS66IeM2YMqqurkZeXBwDaiUdjx47VzkDXtBCOHTu22euEhoYa/UyJqH1hBZCoHVq6dCmWLVuGDz/8EPfee69VlcC5c+ciPDwcCxcutOo1goKCsGTJEmRlZWHHjh0253HVqlXYt28fcnJy8PTTT+P8+fOYMmWKdgzbpUuXoFar8corr8Db21vvNmHCBADQdj1eunQJXbt2bfYaxrYBMDrz9vz58/j555+bvVZgYCCEENrXGjVqFDZv3gy1Wo377rsP3bt3R3R0NDZu3Kjd16JFi/D8889jz549GD9+PDp16oQxY8aYXebm0qVLJvMWHh6ufVzD398fvr6+etuUSiVqampMvoZGdXV1s+fqMpaHpKQkrFu3Dg8++CC++eYbfP/999i3bx+6dOmC6upqbTrNuMpt27bhjz/+wMmTJ7UVwL1796KyshLbtm1DVFQUIiMjm72Or6+v3v6IqH3ycnYGiMi49PR0KBQKpKeno7GxER988AG8vEz/ZP38/JCWloaHHnoIX331lVWv8Y9//ANr167Fk08+iX/84x825S8qKko78WPUqFHw8/PDkiVL8Morr2DBggUIDg6Gp6cnkpOTMWvWLKP70FQgOnXq1GzyCACTM5t1x51pdO7cGX5+fnjrrbeMPqdz587a/2+//XbcfvvtqK2txZ49e7BixQokJSUhIiICw4YNg5eXF+bNm4d58+ahtLQU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TU012lxPjuNKn7sr5dWduMrn7ir5dDeu9Lm7Ul7dhat85pwEYgevvfYaVq9ejcLCQkRHR+PFF1/EKC6rT0RERO0UK4BEREREMsMxgEREREQywwogERERkcywAkhEREQkM6wAEhEREckMK4BEREREMsMKIBEREZHMsAJIREREJDOsABIRERHJDCuARERERDLDCiARERGRzLACSERERCQzrAASERERyQwrgEREREQywwogERERkcywAkhEREQkM6wAEhEREckMK4BEREREMsMKIBEREZHMsAJIREREJDOsABIRERHJDCuARERERDLDCiARERGRzLACSERERCQzrAASERERyQwrgEREREQywwogERERkcywAkhEREQkM6wAEhEREckMK4BEREREMsMKIBEREZHMeDk7A3LW2NiIc+fOITAwEAqFwtnZIRsJIVBRUYHw8HB4eDjmXIox4roYH2QJY4TMcXR8sALoROfOnUOPHj2cnQ1qpdOnT6N79+4O2TdjxPUxPsgSxgiZ46j4YAXQiQIDAwFIX25QUJCTc0NYuRJYsaLp/qJFwFNPmUxeXl6OHj16aL9HR2CMtCOMD7KEMULmtLP4cKkK4IoVK7Bp0yYcPXoUfn5+GD58OFatWoW+fftq0wghkJ6ejn/9618oKSnBkCFD8Oqrr+Laa6/VpqmtrcWCBQuwceNGVFdXY8yYMXjttdf0atglJSWYM2cOvvjiCwDA5MmT8corr6Bjx47aNAUFBZg1axa2b98OPz8/JCUl4fnnn4ePj49V70fTHB8UFMQfZnvwww/N71vxvTiyW4Ux0o4wPsgSxgiZ087iw6UmgeTk5GDWrFnYs2cPsrOzoVarER8fj6qqKm2a1atX44UXXsC6deuwb98+hIWFIS4uDhUVFdo0c+fOxWeffYbMzEzk5uaisrISiYmJaGho0KZJSkrCgQMHkJWVhaysLBw4cADJycnaxxsaGjBx4kRUVVUhNzcXmZmZ+PTTTzF//vy2+TDI/kaMADQ/NIVCuk+kwfggSxgjZE57iw/hwoqLiwUAkZOTI4QQorGxUYSFhYmVK1dq09TU1AiVSiXWr18vhBCitLRUeHt7i8zMTG2as2fPCg8PD5GVlSWEEOLIkSMCgNizZ482TX5+vgAgjh49KoQQYsuWLcLDw0OcPXtWm2bjxo1CqVSKsrIyq/JfVlYmAFidnhysvl6I9HQh4uKkv/X1ZpO3xffHGGlHGB9kCWOEzGln8eFSXcCGysrKAAAhISEAgBMnTqCoqAjx8fHaNEqlEjExMcjLy8PDDz+M/fv3o76+Xi9NeHg4oqOjkZeXh4SEBOTn50OlUmHIkCHaNEOHDoVKpUJeXh769u2L/Px8REdHIzw8XJsmISEBtbW12L9/P2JjY5vlt7a2FrW1tdr75eXl9vswqPW8vIBly5yaBcZIO8b4IEsYI2ROO4gPXS7VBaxLCIF58+ZhxIgRiI6OBgAUFRUBALp27aqXtmvXrtrHioqK4OPjg+DgYLNpQkNDm71maGioXhrD1wkODoaPj482jaEVK1ZApVJpb5yZRYYYI2QO44MsYYyQtVy2ApiSkoKff/4ZGzdubPaY4YBJIYTFQZSGaYylb0kaXYsWLUJZWZn2dvr0abN5IjtQq4Hly4H4eOmvWu3sHJnFGGljjA+yhDFC5rhYfOhyyS7g2bNn44svvsCuXbv0Zu6GhYUBkFrnunXrpt1eXFysba0LCwtDXV0dSkpK9FoBi4uLMXz4cG2a8+fPN3vdCxcu6O1n7969eo+XlJSgvr6+WcughlKphFKpbMlbppbKyADS0gAhgG3bpG3tqAneEGOkjTE+yBLGCJnjYvGhy6VaAIUQSElJwaZNm7B9+3ZERkbqPR4ZGYmwsDBkZ2drt9XV1SEnJ0dbuRs8eDC8vb310hQWFuLw4cPaNMOGDUNZWRm+//57bZq9e/eirKxML83hw4dRWFioTbN161YolUoMHjzY/m+erGN4NrZ7t/TDBKS/ubnOzR85F+ODLGGMkDluFB8u1QI4a9YsfPjhh/j8888RGBioHWunUqng5+cHhUKBuXPnIiMjA3369EGfPn2QkZEBf39/JCUladM+8MADmD9/Pjp16oSQkBAsWLAAAwcOxNixYwEA/fv3x7hx4zBz5ky88cYbAICHHnoIiYmJ2jUH4+PjMWDAACQnJ2PNmjW4fPkyFixYgJkzZ3KtpbamVktnYbm50v87dkjbt20DRo+WptsL0T6m3ZNzaGLknXeA48elbYwP0mAZQpa4YxnikLnFDgLA6O3tt9/WpmlsbBSpqakiLCxMKJVKMWrUKHHo0CG9/VRXV4uUlBQREhIi/Pz8RGJioigoKNBLc+nSJTFt2jQRGBgoAgMDxbRp00RJSYlemlOnTomJEycKPz8/ERISIlJSUkRNTY3V74fT8+0kPV0IhUII6SeofxszxqZp97bgEg4uxFSMMD5ICJYhZJkbliEu1QIoNM2sZigUCqSlpSEtLc1kGl9fX7zyyit45ZVXTKYJCQnB+++/b/a1evbsiS+//NJinsjBcnObmuB1KRTAqFEuMx6DHMhYjDA+SINlCFnihmWIS40BJDI648pwdfXYWCAuThqYu3ixU7NLbczUjDzdGAGAqCjGhxxZEx8sQ+RNRmWIS7UAkszV1AD9+wMnT0r3NRN5ND/A3FzpR7p4sbTgJsmLqfhYtowxQtKBfOxYICdHup+dDTQ0AOnpjA+SyKwMce3ck7xMmND0w9TIzW13q6uTk5iKD4AxQtIAfk3lT+P996UKIOODANmVITZ3AZ87dw7Hjh3T3m9oaMDq1atxzz334K233rJr5oj0HDzYfJsrzbgix2J8kDkutDwHOYnMyhCbK4APP/wwXn75Ze39Z555Bk899RS2bt2KmTNnWpw4QdRigwbp34+IcOnxF2RnjA8yx9iBPDm57fNB7ZfMyhCbK4A//vgjYmNjtff//e9/4/HHH8fly5fx0EMP4dVXX7VrBom0tmyRBmeHhEh/f/3V5cdgkB0xPsicxYulLryoKOmWmgosWeLsXFF7IrMyxOZ3dunSJe0l13799VcUFhZixowZAIA77rgDH330kV0zSKTl6wts3+7sXFB7xfggc7y8pPF+6enOzgm1VzIrQ2xuAVSpVCguLgYA7Nq1CyEhIRg4cCAAaQ2+uro6++aQiIiIiOzK5hbAW265BatWrYK3tzfWrl2L+Ph47WPHjx9HeHi4XTNIRERERPZlcwvgM888g+PHj+P222/H+fPn8fTTT2sf27x5M2655Ra7ZpCIiIiI7MvmFsDrr78ep06dwtGjR9G7d28EBQVpH3v00UfRp08fu2aQ3ITuxdbdZBFNsiPGB1nCGCFzGB82a9Gn4+/vjxtvvLHZ9okTJ7Y6Q+SmMjKky+YIAWzbJm1zs0U1qRUYH2QJY4TMYXzYrEXXAr5w4QIWLVqEYcOG4ZprrsEvv/wCAHjjjTfw008/2TWD5CZ0L6QtBBdlJX2MD7KEMULmMD5sZnMF8MSJE7juuuvw8ssvQ6FQ4M8//0RtbS0A4Oeff9ZbJJpkyNqLrbvx6upkgbEYYXyQBssQsoRliF3Y3AW8cOFCBAcHY//+/QgNDYWPj4/2sREjRiA1NdWuGSQXY6oZ3tiFtEmejMUI44M0WIaQJSxD7MLmFsBvv/0WqampCA8Ph0JT2/5/3bp1w7lz5+yWOWN27dqFSZMmaV9/8+bNeo8LIZCWlobw8HD4+flh9OjR2i5qjdraWsyePRudO3dGQEAAJk+ejDNnzuilKSkpQXJyMlQqFVQqFZKTk1FaWqqXpqCgAJMmTUJAQAA6d+6MOXPmcB1EU83wmgtpb90q/eXgXPkyFiOMD9JgGUKWsAyxC5srgDU1NQgJCTH6WFVVFTw8WjSs0GpVVVUYNGgQ1q1bZ/Tx1atX44UXXsC6deuwb98+hIWFIS4uDhUVFdo0c+fOxWeffYbMzEzk5uaisrISiYmJaGho0KZJSkrCgQMHkJWVhaysLBw4cADJOteNbGhowMSJE1FVVYXc3FxkZmbi008/xfz58x335l0Bm+HJEsYImcP4IEsYI/YhbHT99deLhQsXCiGEUKvVQqFQiP379wshhFi4cKEYNmyYrbtsMQDis88+095vbGwUYWFhYuXKldptNTU1QqVSifXr1wshhCgtLRXe3t4iMzNTm+bs2bPCw8NDZGVlCSGEOHLkiAAg9uzZo02Tn58vAIijR48KIYTYsmWL8PDwEGfPntWm2bhxo1AqlaKsrMyq/JeVlQkAVqd3CfX1QqSnCxEXJ/2tr3d2jhymLb4/xojrYny0kEziQwjGSIvJJEYc/d3Z3EY6c+ZMzJs3D+Hh4Zg2bRoAoK6uDp988glee+01ky1zbeHEiRMoKirSuzqJUqlETEwM8vLy8PDDD2P//v2or6/XSxMeHo7o6Gjk5eUhISEB+fn5UKlUGDJkiDbN0KFDoVKpkJeXh759+yI/Px/R0dF6Vz5JSEhAbW0t9u/fj9jY2Gb5q62t1U6YAYDy8nJ7fwSOZ2mtJU0zPLWIy8eINWtxMUZazOXjA2AZ4mAuHyMsQ9qMzRXARx99FAcOHMDjjz+u7e4cMWIEhBCYOXMmpk+fbvdMWquoqAgA0LVrV73tXbt2xalTp7RpfHx8EBwc3CyN5vlFRUUIDQ1ttv/Q0FC9NIavExwcDB8fH20aQytWrEC6q16IXPOjfOcd4PhxaRvXWrI7l4+R+Hhgxw7pfna29JfxYTcuHx8sQxzO5WOEZUibadGAvX/961/Iy8vDokWL8OCDD2LhwoXYvXs31q9fb+/8tYjh5BQhRLNthgzTGEvfkjS6Fi1ahLKyMu3t9OnTZvPUrmhmXWkKboBrLTmAy8eIpuDWYHzYlcvHB8sQh3P5GGEZ0mZaPE1m6NChGDp0qD3z0mphYWEApNa5bt26abcXFxdrW+vCwsJQV1eHkpISvVbA4uJiDB8+XJvm/PnzzfZ/4cIFvf3s3btX7/GSkhLU19c3axnUUCqVUCqVrXiHTqQ760qDg2/tzuVjxBDjw65cPj5Yhjicy8eIIcaHw9jcAnju3DkcO3ZMe7+hoQGrV6/GPffcg7feesuumbNVZGQkwsLCkK1pNoY0PjEnJ0dbuRs8eDC8vb310hQWFuLw4cPaNMOGDUNZWRm+//57bZq9e/eirKxML83hw4dRWFioTbN161YolUoMHjzYoe/T4SwtsgkAUVHS2TzXWpIfaxbqBYDYWMaHXLEMIXNYhrQLNrcAPvzww+jZsydeffVVAMAzzzyD5cuXo2PHjvjvf/8LHx8f3HvvvXbPqEZlZSX++OMP7f0TJ07gwIEDCAkJQc+ePTF37lxkZGSgT58+6NOnDzIyMuDv74+kpCQAgEqlwgMPPID58+ejU6dOCAkJwYIFCzBw4ECMHTsWANC/f3+MGzcOM2fOxBtvvAEAeOihh5CYmIi+ffsCAOLj4zFgwAAkJydjzZo1uHz5MhYsWICZM2ciKCjIYe/fYXQH3qrVwM6dlhfZ5DpL8mEpPkwt1MsYkQ+WIWQOy5D2x9Zpw+Hh4eK///2v3v158+YJIYR45JFHxNChQ+0yPdmUHTt2CADNbtOnTxdCSEvBpKamirCwMKFUKsWoUaPEoUOH9PZRXV0tUlJSREhIiPDz8xOJiYmioKBAL82lS5fEtGnTRGBgoAgMDBTTpk0TJSUlemlOnTolJk6cKPz8/ERISIhISUkRNTU1Vr+XdjU9Pz1dCIVCCOknqX+Li3N27tolWS3hwPiwmaziQwjGSAvIKkYYHzZrd8vAXLp0STvW7tdff0VhYSFmzJgBALjjjjvw0UcftbZOatbo0aMhDMeR6FAoFEhLS0NaWprJNL6+vnjllVfwyiuvmEwTEhKC999/32xeevbsiS+//NJintst3TOyP/9sPj4H4BgdOWN8kDmGy3Xs2sUYIX0sQ9o1myuAKpUKxcXFAKTLsoWEhGDgwIEApMqX7C+F5kp0r6doKDZWanrnNRXli/FB5hhej3X0aOlgrokXxgixDGnXbK4A3nLLLVi1ahW8vb2xdu1avQWVjx8/rrcwMrUzNTXA+PHA3r2Ary/QsaP+DzMqCrj6ao69kKuaGmDCBODgQSAoCCgtZXyQPt0ypL5e/3qsnp7SwZ7jt+SLZYhLsfnTf+aZZxAXF4fbb78dwcHBePrpp7WPbd68GbfccotdM0h2oFYDzz4LrFol/UABoLoaKClpSqNQANOnc8FNOdJ007z0UlNMXL6sn4bxIW/GyhBdCgUwciTjQ65YhrgkmyuA119/PU6dOoWjR4+id+/eejNeH330UfTp08euGaRW0BTaL7+sX9nTZXhGRvKiVgNjxwI5OabThIQAjz3G+JAja8oQxoe8sQxxWS1qf/X398eNN97YbPvEiRNbnSGyk8pKoEcPqQneHJ6RyZNaLXXXrVrVtAaXMQqFVHAzRuTH2jKE8SFPLENcXos74MvKyvDbb7+hurq62WOjRo1qVaaoFWpqgHHjTM/I0/D1BRYs4BmZ3GgWYF2zxnhXnq6ICOD++xkjcsMyhMxhGeI2bK4AqtVqPPLII3j33XfR0NBgNI2p7dQGJkww3xQfHAzMng0sXcoBuHKUkQE884z5NJGRwIwZHKQtVyxDyByWIW7D5m/mxRdfxP/+9z+89dZbuO+++/Dqq6/C29sb//73v1FWVoaXX37ZEfkkc3TXWtqzx3S6mBhpuQb+IOVHEyNr15pO4+UlFdg8sMsPyxAyx3A9P1NYhrgUm7+h9957D08//TSmTp2K++67D0OGDMGNN96IBx98EAkJCdixY4fe0jDkQJop93v2SLN6TfHyAhYuBNLT+aOUE8MlGU6dMt2l16sXcPgw0KFD2+aRnMeW+GAZIk/WHmMAliEuyMPWJxw/fhyDBg2Ch4f01BqdMQCPPPIIPvjgA/vljozTjMEIDwd27Gj+w+zYEfDzk7pqli6VHn/uORbccqJWA/37S/Fx+TJw8qT+wd3LS4qRmBgpPk6eZMEtJ5big2UI6caI4TEmMlKKDZYhLs3mCmBAQADq6uqgUCgQEhKCU6dOaR/z8/PDpUuX7JpB0lFZKf3wlEogNdX4sgwKBfD448CVK1LBvnw5C225UKulmXYhIVLBfPKk8XQKhXRQv3JFuiC7r29b5pKcSbcMMRcfLEPkSRMf3t5Aly7GY0ShkMb3Xb7MMsTF2fyr7tevH06cOAEAGD58OF544QWMHDkSPj4+WL16Nfr27Wv3TMqephl+507zs/KCg4G5cznjSo5qaqSzdVMHdUCakdenD9d8lCNryhDGh3xp1nt87rmmJV2MLf/DY4xbsbkCePfdd+O3334DAKSnp2PUqFHo1asXAMDb2xubNm2ybw7lrLRUKpTLysyn8/MDhg4FtmzhmZic1NQACQnA7t3mTwwAKY5+/ZXxITfWliGMD3mqrASio6Xxn5YwRtyOzRXARx99VPv/DTfcgCNHjmDz5s1QKBSIi4tjC6A9aMb4Pfus5XW4Fi3iVHs5qqyUumgsrcOle8bOGJEPzTV7d+40n45liHxVVkrlg7lFnDt2BG6+mdfudVOt/jZ79OiB2bNn2yMvVFMjXVLnu+/Mp1MopBlXhw5x0K3clJZK3315ufl0Hh7StVmzsnjGLicsQ8iSixeB7t2B2lrz6Tp2BE6fZny4MZsngezZswcff/yx0cc+/vhj7N27t9WZkq0JEywX3B07SgNvT5zgD1OOIiMtV/4iIoCqKg7OliOWIWRJjx6WK38xMcCFC4wPN2dzBXDx4sU4dOiQ0ceOHDmCJUuWtDpTsnXwoPnHR44ECgt5UJczc9dlVSqBp58Gfv+dMSJXLEPIEnPDRlQqaYUALvYtCzZ/wz///DPmz59v9LEhQ4bgtddea3WmZGvQIGnNJUNKpXTgZ6FNHTsarwRWVPBsnUyXIUFBwKVLPKiTdBwxrAQqFFLPAssQWbG5NKiqqoKXiULEw8MDFRUVrc6UXIj/n+BRrunSy8wEJk8GdLvRw8OBffuAujrpRu2G5nsTlmbgtkKzGDl4UJq1p/mdKRTAsWNAY6PlrmFqU06JD3NlyJUrDssHtYxTYuTIEaBfv6bjiZcX8NtvLEPaIUfHh80VwMjISOzYsQMJCQnNHtuxY4d2SRiyTLNodo8ePUwnOncOuOqqNsoRtURFRQVUKpVD9m0xRoQArrnGIa9N9uHU+ABYhrgAp8aIWg1ERTnktck+HBUfNlcA77nnHjz33HPo27cv7r//fu32DRs24KWXXsKiRYvsmkF3FhISAgAoKChw2I+fmisvL0ePHj1w+vRpBAUFtXg/QghUVFQgPDzcjrnTxxhxDnvECOPDfbEMIXNcJT4Uwsa2xbq6OowbNw47d+6En58fwsPDce7cOdTU1GD06NH4+uuv4ePj45DMupvy8nKoVCqUlZW1KkjINq70ubtSXt2Jq3zurpJPd+NKn7sr5dVduMpnbnMLoI+PD7Kzs/Hhhx8iKysLFy5cwC233ILx48dj6tSp8PT0dEQ+iYiIiMhOWjQlzNPTE8nJyUhOTrZ3foiIiIjIwWxeB5DsR6lUIjU1FUql0tlZkRVX+txdKa/uxFU+d1fJp7txpc/dlfLqLlzlM7d5DCARERERuTa2ABIRERHJDCuARERERDLDCiARERGRzLACSERERCQzVi0D8/e//93qHSoUCrz55pstzhAREREROZZVFcDt27dDoVBo75eWlqKsrAxeXl7o1KkTLl26BLVaDZVKheDgYIdlloiIiIhaz6ou4JMnT+LEiRM4ceIEPv74Y3To0AEffPABqqurUVhYiOrqarz//vsICAhAZmamo/NMRERERK1g8zqAo0aNwh133IHHHnus2WMvvvgiPvnkE3z33Xd2yyARERER2ZfNk0D279+P6Ohoo48NHDgQBw4caG2eiIiIiMiBbK4ABgUFYdu2bUYf27ZtG4KCglqdKSIiIiJyHKsmgehKTk7GmjVroFarkZSUhLCwMBQVFeGDDz7ASy+9hHnz5jkin0RERERkJzaPAVSr1XjwwQfx7rvv6s0MFkLg3nvvxVtvvQUvL5vrlURERETURmyuAGocO3YM27dvx+XLl9GpUyeMHj0a/fr1s3f+iIiIiMjOWlwBJCIiIiLX1KJLwdXW1uKNN97A1KlTER8fj99//x0A8Pnnn+P48eN2zSC1PxEREdizZ4/2fnp6Onr37o0zZ87YtJ+ysjKEhYVh3Lhx9s4itbHWxsQnn3yCIUOGQKlU4pFHHmn2+Ndff43evXsjICAAt99+O0pKSuyWd3I8R8dHYWEh7rzzTqhUKnTq1AkLFy60W97J8VobH6tWrUK/fv0QGBiIAQMGYNOmTdrH8vPzERsbi+DgYHTr1g0pKSmoq6uz+3twRTZXAC9evIibbroJ//jHP5CTk4Nvv/0WFRUVAIDNmzfj+eeft3smqf1avnw53n33XezYsQPdu3e36bmpqano3bu3g3JGztKSmAgJCcHChQvx4IMPNnusuLgYSUlJeOWVV1BcXIzAwECj65CSa7B3fAghMHnyZAwZMgTnzp3D2bNnce+999o729RGWhIfnp6e+O9//4uysjKsX78e999/P/78808AUkPDY489hjNnzuDw4cP4+eefsWrVKke+BZdhcwVw4cKFKC0txQ8//ICCggLo9iDHxsYiJyfHrhmk9uuZZ57Bhg0bsGPHDvTo0cOm5x4+fBh5eXk2XWea2r+WxsRtt92GO+64A126dGn22GeffYahQ4di/PjxCAgIQHp6Ov773/+itrbWnlmnNuCI+Pj666/h4+ODJ554AgEBAfD19cV1111nz2xTG2lpfCxYsAADBw6Eh4cHRo0ahejoaO2axOPGjcOUKVMQEBCATp06ITk5Gd9//72D3oFrsXm67pdffolVq1bhxhtvRENDg95j3bt3t7kbkFzT6tWr8dNPP2Hnzp3o2bMnACA3NxeJiYkmn1NaWqr9f86cOfjnP/+pPUsj19famDDlyJEjGDhwoPb+1VdfDS8vLxw/fhz9+/dvdb6pbTgqPvbt24fIyEgkJCRg//79GDhwINatW4drr73WXlmnNmCv+KioqMAvv/yCAQMGGH1OXl4eY+P/2VwBLC8vR69evYw+Vl9fD7Va3epMUfuXnZ2Nu+++Wy8WRowYYVWBnZmZidDQUIwcOZIVQDfSmpgwp7KyslnLT1BQECorK1u1X2pbjoqPs2fP4qOPPsIXX3yBMWPGYP369bj99tvx66+/wtvbu5W5prZir/h4+OGHMXnyZKMnh19//TW+/vprHDx4sLXZdQs2dwFHRkYiPz/f6GPff/89+vbt2+pMUfv35ptv4ptvvsGyZctsel5VVRVSU1OxZs0aB+WMnKWlMWFJhw4dUF5erretvLwcHTp0sOvrkGM5Kj78/PwwcuRIjB8/Hj4+PpgzZw7Ky8vx22+/2fV1yLHsER9PPfUUCgoK8MYbbzR7bN++fbj//vuxefNmdO3atTVZdRs2twBOmzYNq1atQnR0NCZOnAgAUCgU2LdvH9auXYunn37a7pmk9qdnz57Izs7GqFGjEBwcjMcffxy7d+/G+PHjTT6nsrISv//+O06cOIGbb74ZAFBdXY2amhpce+21+OWXX9oq++QALY0JSwYMGIDNmzdr7x8/fhxqtRpRUVH2yDa1EUfFR3R0NA4dOmTPrJITtDY+1qxZg//973/Izc2Fn5+fXrpff/0VkydPxltvvYWhQ4c67D24HGGjuro6MW7cOKFQKERISIhQKBSiS5cuwsPDQ0yYMEE0NDTYuktyMb169RL5+flCCCH2798vVCqVeOutt6x6bn19vSgsLNTeXnrpJREbGyvOnz/vyCyTg7UmJoQQQq1Wi+rqavH000+LBx98UFRXV4v6+nohhBDnz58XHTt2FFlZWaKqqkrce++9Ijk52SHvgxzDkfFx8eJFERISIrZu3SrUarVYt26duPrqq0VdXZ1D3gvZX2vj48033xQ9e/YUp0+fbvZYQUGB6NGjh9iwYYPd8usubK4ACiFEY2Oj2Lhxo7j33ntFXFycmDp1qvjggw9Y+ZMJ3R+rEELs2rVLdOjQQXzyySc27+vtt98WCQkJ9sweOUFrY+Ltt98WAPRuqamp2se/+uorERUVJfz8/MSkSZPE5cuX7f0WyIEcHR/bt28X/fr1Ex06dBAjRowQhw4dsvdbIAdqbXxEREQIb29vERAQoL0999xzQggh0tLShEKh0HtswIABDnkfroZXAiEiIiKSGZsngXh6eppcQ2f//v3w9PRsdaaIiIiIyHFsrgCaazBsbGyEQqFoVYaIiIiIyLFadC1gU5W8/fv3Q6VStSpDRERERORYVi0Ds3btWqxduxaAVPmbMmUKlEqlXprq6moUFxfjzjvvtH8uiYiIiMhurKoAhoaGai+dcvLkSURFRaFjx456aZRKJQYOHMiLtBMRERG1czbPAo6NjcXrr7+Ofv36OSpPRERERORAXAbGiRobG3Hu3DkEBgZy8owLEkKgoqIC4eHh8PBo0XBaixgjrovxQZYwRsgcR8eHzZeCe/vtt3Hq1CmkpaU1eywtLQ1RUVG477777JE3t3fu3Dn06NHD2dmgVjp9+jS6d+/ukH0zRlwf44MsYYyQOY6KD5srgC+//DJmzJhh9LHOnTvj5ZdfZgXQSoGBgQCkLzcoKMjJuZEntRp4/nlgzx5g6FBgwQLAy8u6NOXl5ejRo4f2e3QExojrYnyQJYwRMsfh8WHrpUM6dOggtm3bZvSxb7/9VgQFBbXsmiRWyMjIEDfddJPo0KGD6NKli7j99tvF0aNH9dI0NjaK1NRU0a1bN+Hr6ytiYmLE4cOH9dLU1NSIlJQU0alTJ+Hv7y8mTZrU7BqCly9fFvfee68ICgoSQUFB4t577xUlJSV6aU6dOiUSExOFv7+/6NSpk5g9e7aora21+v2UlZUJAKKsrMy2D4LsJj1dCIVCCED6m55ufZq2+P4YI66L8UGWMEbIHEd/dy3qVC4rKzO5Xa1Wt6giao2cnBzMmjULe/bsQXZ2NtRqNeLj41FVVaVNs3r1arzwwgtYt24d9u3bh7CwMMTFxaGiokKbZu7cufjss8+QmZmJ3NxcVFZWIjExEQ0NDdo0SUlJOHDgALKyspCVlYUDBw4gOTlZ+3hDQwMmTpyIqqoq5ObmIjMzE59++inmz5/vsPdP9pebC2hGwQoh3W9JGnIdajWwbBkQEgL4+wOjRwOVlcDy5UB8vPRXU4yp1UBaGnD11VL6qChg6VJpmyZtTY3x5xIRtWu21hiHDRsm/va3vxl97G9/+5sYMmRIq2ul1iouLhYARE5OjhBCav0LCwsTK1eu1KapqakRKpVKrF+/XgghRGlpqfD29haZmZnaNGfPnhUeHh4iKytLCCHEkSNHBACxZ88ebZr8/HwBQNviuGXLFuHh4SHOnj2rTbNx40ahVCqtrq3zzMwxqquFGD1aCD8/IYKDhViyRNqWni5EXJz0t75eSpuaKrXsaW6pqdJjummXLWMLoDtJT9f/zgEhIiKMf8fG0ureFAohYmMZH9QyjBEyx9Hfnc1jAFNSUnDvvfdi+vTpePTRR9G9e3ecOXMGr7/+Oj799FO8++67dqucWqJpiQwJCQEAnDhxAkVFRYiPj9emUSqViImJQV5eHh5++GHs378f9fX1emnCw8MRHR2NvLw8JCQkID8/HyqVCkOGDNGmGTp0KFQqFfLy8tC3b1/k5+cjOjoa4eHh2jQJCQmora3F/v37ERsb2yy/tbW1qK2t1d4vLy+334dBWhMmADt3Sv9XVwPPPgt89520TQhg2zbpsWXLmlr2NN59F9i1Sz/tsmVSi09uLjBiBLB4sePyzhhxPGMtuGfO6LfyvvOO9D1bau0VAjhwoO1aiBkfZAljhKxlcwUwKSkJR48exYoVK/D+++9rt3t4eGDJkiWYNm2aXTNoihAC8+bNw4gRIxAdHQ0AKCoqAgB07dpVL23Xrl1x6tQpbRofHx8EBwc3S6N5flFREUJDQ5u9ZmhoqF4aw9cJDg6Gj4+PNo2hFStWID093da3SjY6eND4NmMH+Px8/XQnTkg3DSGAvDxg61bH5VcXY8TxRowAsrP1t3XvDpw82XT/+HEgI8N4WkMqFVBaKsWKQiE9x1EYH/ajVkvf8a5dQGMj4OkJjBwplQtqNTB+PLB3L+DrC8yaBTz9NLByJfDee9Lz771XGg6gVksnnQcPAkFBQK9e0uOnT0t/p02T4iI/v+kE0nCimT0xRshaLQrD5cuX4+9//zuys7Nx4cIFdOnSBfHx8eilifw2kJKSgp9//hm5Rk63Ddc6EkJYXP/IMI2x9C1Jo2vRokWYN2+e9r5mhg/Z16BBwI4d5rfpHuC3bWveEqjLkQd0Q4wRx9Mc4Netk8bv3XIL8OWXUowcP96ULjcX2LJFqhy89x5QUgJ07CjFim5lMSoKuP/+tmkhZnzYT0aG1LKv+9vftq2p9d9YL4JuGbJ8uVRp3Lmzafvly/qxAQDPPKO/f0DqVXAUxghZq8XnIREREZg5c6Y982K12bNn44svvsCuXbv01sYJCwsDILXOdevWTbu9uLhY21oXFhaGuro6lJSU6LUCFhcXY/jw4do058+fb/a6Fy5c0NvP3r179R4vKSlBfX19s5ZBDaVS2ewaymR/W7YYP3u/9lrjB3hAahHUfUwjKsqxB3RDjBH70LTu5OYCw4ZJLTB5eU0VtOXLpZuu6dObKgSaljwvL2mb7rKny5frp4uJcewBXRfjw350J3fp2rED8PNrvt1Yz0JurvHtprTFJDLGCFnNISMLHaSxsVHMmjVLhIeHi99++83o42FhYWLVqlXabbW1tUYngXz00UfaNOfOnTM6CWTv3r3aNHv27DE6CeTcuXPaNJmZmZwE0o6ZW/JFM/EjKkp/gL+xZWE0OIC7/dL9rg0nbZj6Tg0n/2gmCrU0HeOjfVu2zPTknuDg5ttiY5tvS083vt3cpCHd+GOMtF/19VKMREVJk8QiIqS4iIiQJhnGxAjRsaM02TAmRoiKCmkSYVSUdFu2TNpHdbUUIyEh0nNjY6WJiaNHS9tiY6XnGitTHP3dWVUB9PDw0FaGFAqF8PDwMHnz9PR0SEaFEOIf//iHUKlUYufOnaKwsFB7u3LlijbNypUrhUqlEps2bRKHDh0SU6dOFd26dRPl5eXaNI888ojo3r272LZtm/jxxx/FbbfdJgYNGiTUarU2zbhx48R1110n8vPzRX5+vhg4cKBITEzUPq5Wq0V0dLQYM2aM+PHHH8W2bdtE9+7dRUpKitXvhz/M1rP2YGxtWlv2x8K7/YqLM30Qjooy/73aC+OjfTNcAcBwVnhERPOVBKw5wMfESDdNuqVLpbTGyhTGSPtlaQUAYzHT0hMEU6sQtItZwMuWLdN2tS5btsxp1xN8/fXXAQCjR4/W2/72229rr06ycOFCVFdX49FHH0VJSQmGDBmCrVu36q2k/eKLL8LLywt33XUXqqurMWbMGGzYsAGenp7aNB988AHmzJmjnS08efJkrFu3Tvu4p6cnvvrqKzz66KO49dZb4efnh6SkJDz//PMOevekq6ZGGni9Z480RgeQBuvv3ClN2DA2yNrLy3JXnTVpqP0zN3lDM/6T37O85eXp34+Kkv4ePy6N41MopK5+3TgxHA4ASGXG9u2Oyyc5hu4wEWOTc2ztqj9zpvk2a4cIGK5C0GZrzTqkWklW4ZlZy5k7q4qKstx6Zw88e2+/6uv1u/Od0QrI+GjfjA0JMWw5jotzbB4YI85j6SpQcmgBbNGVQIgcTa02f3WFAwdMP/f4ceksPSPDkTmk9szLS5rUYaqzQtMKSO7JUvkBSC0+aWlAXJz0d/FiqSVIEzOOXtKHnMvUFZ40sbNrV9OSPhq+vkBEhHT1oJgYaVUAPz/p/0OHgNRUqSU5KkpqOV68WJpoGBsrXUkoIkL6f8kSaR8hIdL9Q4eax2JbsKoL2NbFne+7774WZYZIQ3eJBmNLJ6hU0rIcprRpMzq1OUvdNwCwcKE0JODgQWDgQKCgQH+NR8aH+7JUfgDGh3toDrxtsaQPtQ1TZYXuEmC6lX1jywNpjBxpfk3Y1gwRcMaQFKsqgJrxdRqaMYBC5xPSHRfICiC1lqXr70ZF6a+3FRkp/Yg1S7nw7N29WXOAX726aU23XbukM+6TJ5sX+OR+rLl+t6mKAceGuhdTZYWpyr6p5YHcscywqgJ4Que0uaioCHfffTcSEhKQlJSEsLAwFBUV4YMPPsDWrVvx0UcfOSyz5H5sOTvTTavZrvk7Y4b0XMN9kXsy132jiYE//9RP4+nZdpf0I+cy1bqjGx9qtfHLQ5J7MVVWmKrsG14cIDa26ZjkbmWGVRVA3St8PPXUU/jLX/6CF198Ubutb9++iImJweOPP44XXniBlUCymi1nZ4ZN84Y/TJ69ux97dd8oFFL3DePDvZiKD1OtO6big0NGXJel4SCmygpTjMWOIy/d50w2v62vv/4an3zyidHHJkyYgL/97W+tzhTJhy1nZ4ZN815ebXeNXnKO1nbfREUBV1/tnmfvcmHuAG8qPkydDMqpe08uLA0HsXVcp5waEmyuADY2NuL333/H2LFjmz32+++/640LJLLElrMzW8/kyPW1pvtGoZBmAsulMHdX5g7w1oz10yWn7j25sBQDcqrQ2crmCuC4cePw9NNPo2fPnpg4caJ2+5dffoklS5YgISHBrhkk92bL2Rln6MmPPbpvyPWYG8upe4Bn9578aGJj926goUF/MiAbBmxjc+ivXbsWY8aMweTJkxEYGIiuXbvi/PnzqKioQJ8+fbB27VpH5JPclC1nZzyTcz+Wxu+w+0ZeNPHwzjtNM/p1GR7gGR/yY2ocZ1SU1OLPkz7r2VwB7NatG3788Uds2LABO3fuxKVLl3DDDTcgNjYW9913H/z8/ByRTyJyA5oDfE6OdIAvKpIu6wcYH7/DA7a8mDu4GxvLyfiQH1PjOK++mrFgqxY1fvv6+uKRRx7BI488Yu/8EJEbM7fIKmdikqmDe69enPBFEsNxnBrDhzsnP66sxZeCO3r0KN544w0899xzKCoqAgCcO3cO1dXVdsscuTa1WjrYX31109lZaqr5yzORezN1gAc4fof0L8Wm68ABlhkk0VzCLypKf/t77zFGbGVzC2BDQwMeeughbNiwAUIIKBQKjB8/HmFhYXj44Ydxww03YPny5Y7IK7mYjAwgPb3p/jPPNP3PhVflydTZe3AwMHcux+/Ineb7NxwDWFIiHfQbGwEPD07ikDNNt39urn6MaK4Bv3Nn8/VhyTibP5rnnnsOH374IdasWYNx48YhOjpa+9j48eOxYcMGVgBlSnd2Vn09sHev6bTs7pMnzQFeMwawtFS6oLrh2TzJk+bgrln4fe1a4PJl6TEhpFaeEyd49Q4yfjIpBLBjh/Q/48MymyuAGzZswNKlSzFv3jw0NDToPRYZGal32TiSF3Pjuwyxu0+eDAftL18uxczJk9IF099+WxouMGoUz97lTDdOdMuUkhLb1v0j92WqtViD8WGZzWMAz549i2HDhhl9zNfXFxUVFa3OFLkmc+O7NCIjgbg4qVBndx8ZxszJk8C330rxkZHhrFxRe7F4MTB6dNP9kpKm/3kSKW+ak4Rjx6QFvQ0xPiyzuQIYGhqK48aq2wCOHTuG7t27tzpT5uzatQuTJk1CeHg4FAoFNm/erPe4EAJpaWkIDw+Hn58fRo8ejV9++UUvTW1tLWbPno3OnTsjICAAkydPxpkzZ/TSlJSUIDk5GSqVCiqVCsnJySgtLdVLU1BQgEmTJiEgIACdO3fGnDlzUFdX54i37RJMDeDWUCiAGTOk2XyayzWRvJkqoHn2ToBURhiWE1FRPImkJsZiJCSE8WENmyuAEyZMwHPPPYezZ89qtykUCpSVleHll1/GpEmT7JpBQ1VVVRg0aBDWrVtn9PHVq1fjhRdewLp167Bv3z6EhYUhLi5Or2Vy7ty5+Oyzz5CZmYnc3FxUVlYiMTFRr0s7KSkJBw4cQFZWFrKysnDgwAEkJydrH29oaMDEiRNRVVWF3NxcZGZm4tNPP8X8+fMd9+bbOc3srLFjgZgYqbUvKkr6f8wY/iCpucWLefZO5umeWGou78eTSNJlGCOPPcb4sIqwUVFRkejRo4cICgoSt99+u/Dw8BDjxo0T3bt3F5GRkeLSpUu27rLFAIjPPvtMe7+xsVGEhYWJlStXarfV1NQIlUol1q9fL4QQorS0VHh7e4vMzExtmrNnzwoPDw+RlZUlhBDiyJEjAoDYs2ePNk1+fr4AII4ePSqEEGLLli3Cw8NDnD17Vptm48aNQqlUirKyMqvyX1ZWJgBYnZ7al7b4/uQQI/X1QqSnCzF2rBCxsUKMGSPdr693ds5ah/FhH5r4iItzj7jQxRixD3eNEUd/dzbXj7t27Yp9+/YhNTUVX331FTw9PXHw4EEkJiZi+fLlCAkJsVvl1FYnTpxAUVER4uPjtduUSiViYmKQl5eHhx9+GPv370d9fb1emvDwcERHRyMvLw8JCQnIz8+HSqXCkCFDtGmGDh0KlUqFvLw89O3bF/n5+YiOjkZ4eLg2TUJCAmpra7F//37EGmnWqK2tRW1trfZ+eXm5vT8CcnGuHCOWLutmCq/mYD1Xjg+gZTHC+LCNK8cIy5C2ZVMFsKamBsuXL8cdd9yB9evXOypPLaZZkLpr165627t27YpTp05p0/j4+CA4OLhZGs3zi4qKEBoa2mz/oaGhemkMXyc4OBg+Pj7aNIZWrFiBdN2F8VyM7o9z2DCpqT0vj+st2ZMrx4juLHAuweAYrhwfgH6MZGdzzTZHcOUYYRnStmwaA+jr64sXX3wRVVVVjsqPXSgMZiKI/1+w2hzDNMbStySNrkWLFqGsrEx7O336tNk8OZNaLS3RoXvVDs2PMztb2paeLv3PGZv240oxYkh3Rq8Q0vIMvOqLfbX3+NAtN9LSml/5x3DW944dLEPsrb3HiDksQ9qWzedb/fv3x4kTJzBq1ChH5KdVwsLCAEitc926ddNuLy4u1rbWhYWFoa6uDiUlJXqtgMXFxRj+/xcTDAsLw/nz55vt/8KFC3r72Wuw0nFJSQnq6+ubtQxqKJVKKJXKVrzDtmPsTMzUMi+csWk/7T1GdBf7bmiQrsqgWbPPcGHW48elG8/k7ac9x4daLR2sNQvxZmc3PaaJAVNXgmEZYj/tOUZ0GevuZRnStmyuAC5duhQLFy7EiBEjcPXVVzsiTy0WGRmJsLAwZGdn44YbbgAA1NXVIScnB6tWrQIADB48GN7e3sjOzsZdd90FACgsLMThw4exevVqAMCwYcNQVlaG77//HrfccgsAYO/evSgrK9NWEocNG4bnnnsOhYWF2srm1q1boVQqMXjw4DZ9345geCam+ZEaK7w5Y1M+jC32vX279Fczwzs3F/jzz6bFWXlwl4eMjKbKnyFNDGzZIt3PzZUqADt3So+xDJEPTcVPdwFnTQWPZUjbsrkC+Pbbb+PKlSvo378/rrvuOnTr1q1Zt+jnn39u10zqqqysxB9//KG9f+LECRw4cAAhISHo2bMn5s6di4yMDPTp0wd9+vRBRkYG/P39kZSUBABQqVR44IEHMH/+fHTq1AkhISFYsGABBg4ciLFjxwKQWjnHjRuHmTNn4o033gAAPPTQQ0hMTETfvn0BAPHx8RgwYACSk5OxZs0aXL58GQsWLMDMmTMRFBTksPfvKGq1dK3e99+X7vfo0fSYpnDW/XEaGwNIrsvS+E6g6dJcplpvdAdia67wwYO7+zE1UN/cAVoTA7oxYmw/5L6MVfw0WIY4ia3Thnv16iUiIiJM3iIjI+09U1nPjh07BIBmt+nTpwshpKVgUlNTRVhYmFAqlWLUqFHi0KFDevuorq4WKSkpIiQkRPj5+YnExERRUFCgl+bSpUti2rRpIjAwUAQGBopp06aJkpISvTSnTp0SEydOFH5+fiIkJESkpKSImpoaq99Le5qen54uhPRTa7oFBwsRFSXEsmXuM63entxpCYf0dCEUiuYxAEhLsyxbZvpxhUJ6vi53XZbBFu4UH7rfZ2xs8/jQPK67ffRoKW7kHAOWuFOM6DL2+zdXxrAMMc7R351CCGuu3EqOUF5eDpVKhbKyMoe3Ghpr4dF0wxQUAKdPGx9kq1BIZ2Ece9FcW3x/jnwN3ZjQ7W4xpmNHQPdCOH5+wC23SGfsvG6vca4eH7p0W2OMiY2VundXr7Z9CQ85c6cY0S1P1Gr94QAREdKFAYwNEYiKkhb3Zrw05+jvjh+3m9LMxlu3DqipAby9Ac1yULqDsy3h2Av39eyz0kxuaxhcBRHV1dIEEA+bryVErqSmBpgwAdi1y/x1vnfsAIKDAaVS+qsZ36c7SYgHd/eWkSHN+jbm5Enpr0LRFEes+Dlfiz72hoYGfPzxx9ixYwcuXbqETp06ITY2Fn/729/gxW+yXcjIkMb0aVRXt2w/HHvhntRq4OWXW7cPzdk8Z+i5n8pKYODApgO3NWpqpFtZmf7zNJOEGB/uo6YGGD8e2LNHKks8PMyfIABSTHh4SCcJ3boB06ZJKwlMmMAWY2ex+eO+ePEixo0bhx9//BFeXl7o1KkTLl26hP/85z94/vnn8c0336Bz586OyCsZMDUYu6YGeOml1u8/JES6piIHZ7sPtVrqylu5Uip87YGtxK7NsByZNw/o1Amoq7PP/hkfrs9wCNE77wD/f20FmzQ2SsenEyf0Gyh4EukcNlcAH3/8cRw7dgwffPAB7rrrLnh6eqKhoQEfffQRHnnkETz++ON47733HJFXMmBq1fQJE4CSktbvX3NBbXJNxtbsa2w0vVSHLSIi9Lt12ErsejQz/195pam8yM4G3n7bfpU/gPHhykzFiL3xJME5bK4A/u9//8Ozzz6LqVOnard5enoiKSkJxcXFSEtLs2f+yAzdcTlCABs2SGfvu3bZvi+lEggLk8YJBgcDycls+XN15sbktNbVVwP3388lPFxZRoY0TtjQmTP22X9wMHDjjU1jAMn1mIoRe+NJgnPYXAEUQuDaa681+lh0dDQ4qbjtNDbq3z9xQhq3Y2vXnkIhFdBs7XMvjjyjHjWK8eLqTMVHhw7NJ/3o8vS0XMYoFMDcuYwRV+fIMiQmRrrl5/Mk0llsrgCOHTsW27Zt0y6arCs7OxujR4+2R77ICp6ezbdZOnt/+mnAx0dqJWxslPYxciR/fO5oxAjbu2tGjZK6infubP5YTIwUOyys3YOx+PD11a/8KRTSovDnzwP19dL/P/wA3HUXcOAAEBQkxUtpqbRUUM+e+ksDkWsbPtz2MqRXL2nJF8MypGNHqVVYoQDuvRdYupSTPpytRZeC++tf/4qGhgYkJSUhLCwMRUVF+OCDD7Bp0yZs2rQJly9f1qYPCQmxa4apyciRTWP/NAICpFl4uiIigD59ONNKbhYvlgph3TF/vr7SIGyNqCipMDY8C8/IMH6SwNhxH4sXSzN0c3KatunGBgCMHQts3dr8uZqZveR+dCd8WDMWdNQoqVJ36BAwaJC0HqSXl/6VpZKTgSVLWH60NzZ/HTfeeCMA4J///CdeeOEF7XZN16/hdXAb7DXVkLQDct97r2lAru66SkDzyl/PnsCvv0oHfnJ/uoX3LbdICzzrxojhAX76dOPddOy6c39eXtL6oKZwXJY8Gbvety6FAhg9Woofc40K6enWrzNKzmFzBXDZsmV61/6lttOSAbl9+7LyJyfPPNMUI5a6bmJj2U0nN4YnkZWVxtMFBwOzZzM+5Mjcot9eXlKjwqhRbNFzBzZ/fZzl2/Y0V/VYscK25/EMXl7Uamm5Bmt5ebEAlwO1Wrrqy3vvAZcvm5/goVFaKnX9Mz7kQ9N78OOP5tMcPy4djzw82FPg6vjzbudKS6XlWWprbXue7mV2yL2o1VLXyrp1QEWFVNFvaLC8Er8unhy4t5oaYNw4YO9eqeywdXEGrsvm3nRbgi9fBqqqjF8L3hTGh3tgBbAdqqwEoqOBggLrC24fH+CJJ4Dvv+dkD3eXkSG16Nhq0SJpOADX7nN/EyboT+6whpdX04kETxDcm+5QkZZgfLgHVhHamdJS6RJstrbmPP00m+PloKYGePFF258XGSkV+DwpcG81NUB8vHT1F1t1787FveXg4sWWVf4iIqRepbw8xoe74OGgHVGrgdBQ6yt/ERHSFRm45pZ8xMdbN4bL0IwZrPzJwfjxLav8AcB99/EkUg569LD9OQqFdHLA+HAvPCS0IxkZ0mKr1hg1Cvj2Wx7U5ea776xLp+mi8fHhCYKc7N1rfdqRI6Xyw8ODMSInhktBmeLhIS0ErVQyPtwVqw/tiDWDaiMjpdYcjvGTJ8PL/xkKDgbmzOESDXLl6wtUV5tP06sX8McfjA+5MlwM3pjYWGkBcMaIe/NwdgaoialBtUqlNKs3NRX47TepGZ4/THnq1cv0Y6mpQHGxtIgr40OeZs0y/Zifn3Q5v6NHGR9ydvq08QXAPT2ly7UtXcrKn1zwK3YizdVTysvLAQApKdIAXd213IYMAb74omkx5ytX2jqXZIrmexO2rrFhA8MYycuTrvBx9mxTGm9v4NgxoFMnxkd74oz4ePxxaRWBl15qSuPlJZ04duok3a+rs+4SX+R4zogRHx/pOGMOy5H2wdHxwQqgE126dAkA0MPMqNy9e4GuXdsqR9QSFRUVUKlUDtm3NTFSXy+1EFP75Oz4UKsZH+2ds2OE2jdHxQcrgE4UEhICACgoKHDYj5+aKy8vR48ePXD69GkEBQW1eD9CCFRUVCA8PNyOudPHGHEOe8QI48N9sQwhc1wlPlgBdCIPD2kIpkqlalWQUMsEBQW1+nN3dIHKGHGu1sYI48O9sQwhc9p7fHASCBEREZHMsAJIREREJDOsADqRUqlEamoqlEqls7MiK670ubtSXt2Jq3zurpJPd+NKn7sr5dVduMpnrhCOnH9ORERERO0OWwCJiIiIZIYVQCIiIiKZYQWQiIiISGZYASQiIiKSGVYAiYiIiGSGFUAiIiIimWEFkIiIiEhmWAEkIiIikhlWAImIiIhkhhVAIiIiIplhBZCIiIhIZlgBJCIiIpIZVgCJiIiIZIYVQCIiIiKZYQWQiIiISGZYASQiIiKSGVYAiYiIiGSGFUAiIiIimWEFkIiIiEhmWAEkIiIikhlWAImIiIhkhhVAIiIiIplhBZCIiIhIZlgBJCIiIpIZVgCJiIiIZIYVQCIiIiKZYQWQiIiISGZYASQiIiKSGVYAiYiIiGSGFUAiIiIimfFydgbkrLGxEefOnUNgYCAUCoWzs0M2EkKgoqIC4eHh8PBwzLkUY8R1MT7IEsYImePo+GAF0InOnTuHHj16ODsb1EqnT59G9+7dHbJvxojrY3yQJYwRMsdR8cEKoBMFBgYCkL7coKAgJ+eGAAArVwIrVjTdX7QIeOopo0nLy8vRo0cP7ffoCIwR18X4IEsYI2SOo+ODFUAn0jTHBwUF8YfZXvzwQ/P7Fr4bR3arMEZcH+ODLGGMkDmOig+XmgSyYsUK3HzzzQgMDERoaCimTJmCY8eO6aURQiAtLQ3h4eHw8/PD6NGj8csvv+ilqa2txezZs9G5c2cEBARg8uTJOHPmjF6akpISJCcnQ6VSQaVSITk5GaWlpXppCgoKMGnSJAQEBKBz586YM2cO6urqHPLeqY2MGAFofmwKhXSfiMgYtRpYvhyIj5f+qtXOzhGR1VyqApiTk4NZs2Zhz549yM7OhlqtRnx8PKqqqrRpVq9ejRdeeAHr1q3Dvn37EBYWhri4OFRUVGjTzJ07F5999hkyMzORm5uLyspKJCYmoqGhQZsmKSkJBw4cQFZWFrKysnDgwAEkJydrH29oaMDEiRNRVVWF3NxcZGZm4tNPP8X8+fPb5sMgx1i8GEhLA+LipL+LFzs7R0TUXmVkSOVEdrb0NyPD2Tmi9kCtluLh6qulW2pq+zw5EC6suLhYABA5OTlCCCEaGxtFWFiYWLlypTZNTU2NUKlUYv369UIIIUpLS4W3t7fIzMzUpjl79qzw8PAQWVlZQgghjhw5IgCIPXv2aNPk5+cLAOLo0aNCCCG2bNkiPDw8xNmzZ7VpNm7cKJRKpSgrK7Mq/2VlZQKA1empfWmL748x0saqq4WIjRUiJET6W13d4l0xPtyUbowEBwsBNN3i4mzaFWPEDdXXS/GhGxeAEOnpNu/K0d+dS7UAGiorKwMAhISEAABOnDiBoqIixMfHa9MolUrExMQgLy8PALB//37U19frpQkPD0d0dLQ2TX5+PlQqFYYMGaJNM3ToUKhUKr000dHRCA8P16ZJSEhAbW0t9u/fbzS/tbW1KC8v17sR6WKMOJFaDfTvD+zYAVy+LP2dMMHZudLD+GgHJkxoipGSkqbt7WTICGPEyTIypPgwlJvb9nmxwGUrgEIIzJs3DyNGjEB0dDQAoKioCADQtWtXvbRdu3bVPlZUVAQfHx8EBwebTRMaGtrsNUNDQ/XSGL5OcHAwfHx8tGkMrVixQjumUKVScWo+NcMYcaKMDODkSf1tBw86JSumMD7aAcOY8PNrV0NGGCNtzHAc6O7dxtO1g5MDQy5bAUxJScHPP/+MjRs3NnvMcMaMEMLiLBrDNMbStySNrkWLFqGsrEx7O336tNk8USvU1AC33QZ06iT9ralxdo6swhhpY7qF9zvvNH980KC2z5MZjI82ZGqCh2FMDB0KbN0KLFsGeDl/YQ3GSBszHAfa0NA0kRAAgoOl2GgHJweGnB+tLTB79mx88cUX2LVrl97iiGFhYQCk1rlu3bpptxcXF2tb68LCwlBXV4eSkhK9VsDi4mIMHz5cm+b8+fPNXvfChQt6+9m7d6/e4yUlJaivr2/WMqihVCqhVCpb8pbJVppuGqCpK2/7dufmyQqMkTamKbyFaP5YRASwZUtb58gsxkcb0o2NbdukbcuWSTExYYLUEjhoEGNE7nJzm8oPIQAPDylucnOlVr/Fi9vFiYExLtUCKIRASkoKNm3ahO3btyMyMlLv8cjISISFhSE7O1u7ra6uDjk5OdrK3eDBg+Ht7a2XprCwEIcPH9amGTZsGMrKyvD9999r0+zduxdlZWV6aQ4fPozCwkJtmq1bt0KpVGLw4MH2f/NkmW6rn2EzfDvryqN2QrfwBoCoKKk7Lz0d+P13wNfXeXkj5zI8sGvGcPn6SieTly5Jfxkj8ma4dNioUdKJQjtqFTal/ebMiFmzZuHDDz/E559/jsDAQO1YO5VKBT8/PygUCsydOxcZGRno06cP+vTpg4yMDPj7+yMpKUmb9oEHHsD8+fPRqVMnhISEYMGCBRg4cCDGjh0LAOjfvz/GjRuHmTNn4o033gAAPPTQQ0hMTETfvn0BAPHx8RgwYACSk5OxZs0aXL58GQsWLMDMmTO52GZbUqulM/XcXOmAbTiGS6OddeVRG9GND2Nn4yNGSK07QkiF9/TpUqFNZBgb7XAMF7UhU2WJpmtXd7urcMjcYgcBYPT29ttva9M0NjaK1NRUERYWJpRKpRg1apQ4dOiQ3n6qq6tFSkqKCAkJEX5+fiIxMVEUFBTopbl06ZKYNm2aCAwMFIGBgWLatGmipKREL82pU6fExIkThZ+fnwgJCREpKSmipqbG6vfD6fmtUF8vTauPimo+3V5z8/Kyy3IepnAJBxeQni6EQiHFg0LRfCkGTRzFxUl/6+vt9tKMj3bO0nfvwNjQYIy4EEtliQM4+rtTCGFs8Au1hfLycqhUKpSVlbHV0FbLl5seu6URG+vQcX9t8f0xRlpA90z9zz+B48ebHouLk7pm2gDjo53TLUMUCun/Nm79ZYy0Y4Ytfrt3N40FBdqkLHH0d+dSXcBEWoZjtwCpEO/VCygvb5eDs6mNmJrYwW48qqlpmsCh6SsA9Mf4EQHNJwGNHi2VIW40JIAVQHJNuuNzAGnw/vTp7XrGFbURYxM7rr7a9cbnkP3prg6gy00O6NRKhr0Hlmb3ujgeKck1GRt4y4qf/BgbmM2JHWSKsUWcR4xwmwM6tZK53gPN7F43wiMmuSYvL7f7MVILGFurzZVn5ZFjDRqk3wKoWcSZCJBd7wErgETkuoyt1caTAzKlnS/iTE4ms94DVgCJyHVxrTayhWYRZyJjZNZ7wAogOZelhXqJzJFZgU1GsAwhc2yJD5n1HvBXQs5l6nqbRNYU3DIrsMkIliFkDuPDJFYAyblMXW+TiAU3WYNlCJnD+DDJw9kZIJkzvJA2x3CRBgtusgbLEDKH8WESWwDJsXRX3tfMuvP1bXqcY7jIFE7wIGuwDCFzGB8msQJIjqW78v6OHdJ93Vl4HMNFprDgJmuwDCFzGB8msQJIjmW48r7hfZInSy3DAAtuIiIH4hhAcqxBg8zfJ/lQq4Hly4H4eKB/f6lF+PLlppZhIkA/TpYvl+4Tkd2xBZAciyvvk4ap62wCbBmmpmV/3nkHOH5c2sbZ3wRwrUcHcbkWwF27dmHSpEkIDw+HQqHA5s2b9R4XQiAtLQ3h4eHw8/PD6NGj8csvv+ilqa2txezZs9G5c2cEBARg8uTJOHPmjF6akpISJCcnQ6VSQaVSITk5GaWlpXppCgoKMGnSJAQEBKBz586YM2cO6urqHPG2XZdm5f1Ll6S/ht18JB+G19nUxZZhedJt7YuPl04QNJU/gLO/5U4TH337AqmpQHa2FCMZGc7OmVtwuQpgVVUVBg0ahHXr1hl9fPXq1XjhhRewbt067Nu3D2FhYYiLi0NFRYU2zdy5c/HZZ58hMzMTubm5qKysRGJiIhoaGrRpkpKScODAAWRlZSErKwsHDhxAcnKy9vGGhgZMnDgRVVVVyM3NRWZmJj799FPMnz/fcW+eyJXpLscAABERQEgIEBvLlmG50rQKZ2dLQwEMTxA4+1veNPHBkwLHEC4MgPjss8+09xsbG0VYWJhYuXKldltNTY1QqVRi/fr1QgghSktLhbe3t8jMzNSmOXv2rPDw8BBZWVlCCCGOHDkiAIg9e/Zo0+Tn5wsA4ujRo0IIIbZs2SI8PDzE2bNntWk2btwolEqlKCsrM5rfmpoaUVZWpr2dPn1aADCZntq3srIyu39/bh0j9fVCpKcLERcn/a2vd3aOHIrxYYW4OCGkQ3rzW1SU28cJY8QETVkREtI8LhQK6TEZcER86HK5FkBzTpw4gaKiIsTHx2u3KZVKxMTEIC8vDwCwf/9+1NfX66UJDw9HdHS0Nk1+fj5UKhWGDBmiTTN06FCoVCq9NNHR0QgPD9emSUhIQG1tLfbv3280fytWrNB2KatUKvTo0cN+b74tcHC2w7l8jJijmdW7dav0l2N4bOZ28WHYKhwbC8TFAenpwLFjjJMWcIsY0bT8Xb6svz0qStrOJaHswq1+WUVFRQCArl276m3v2rUrTp06pU3j4+OD4ODgZmk0zy8qKkJoaGiz/YeGhuqlMXyd4OBg+Pj4aNMYWrRoEebNm6e9X15e7lo/Tl6ay+FcPkbIodwuPoyt9cgKX6u4RYwYjhcOCQEee4zxYWdu+UkqdM8oIU0MMdxmyDCNsfQtSaNLqVRCqVSazUe7xktzOZzLxwg5lNvFB9d6tDu3iBHDqwA99hjjxAHcqgs4LCwMAJq1wBUXF2tb68LCwlBXV4eSkhKzac6fP99s/xcuXNBLY/g6JSUlqK+vb9Yy6DZ4TUUiInK0xYul3qa4OHb5OpBbVQAjIyMRFhaG7Oxs7ba6ujrk5ORg+PDhAIDBgwfD29tbL01hYSEOHz6sTTNs2DCUlZXh+++/16bZu3cvysrK9NIcPnwYhYWF2jRbt26FUqnE4MGDHfo+nYY/SiIicjSOF24TLvepVlZW4o8//tDeP3HiBA4cOICQkBD07NkTc+fORUZGBvr06YM+ffogIyMD/v7+SEpKAgCoVCo88MADmD9/Pjp16oSQkBAsWLAAAwcOxNixYwEA/fv3x7hx4zBz5ky88cYbAICHHnoIiYmJ6Nu3LwAgPj4eAwYMQHJyMtasWYPLly9jwYIFmDlzJoKCgtr4U2kj7K4hIiJyCy5XAfzhhx8QGxurva8Z7Dp9+nRs2LABCxcuRHV1NR599FGUlJRgyJAh2Lp1KwIDA7XPefHFF+Hl5YW77roL1dXVGDNmDDZs2ABPT09tmg8++ABz5szRzhaePHmy3tqDnp6e+Oqrr/Doo4/i1ltvhZ+fH5KSkvD88887+iOwH2uux0pERERuRyGEqaX5ydHKy8uhUqlQVlbWtq2GlZVAdDTw/zOjtWJjpat1kFXa4vtzeoycPg34+ADz50vd/uyKsZpbx4daDTz7LPDee9L9e+8Fli5lfNjI7WOEl29rFUd/d/w25GjgwOaVP4DXYyWgtFS6QkdZWdO2mhrgueekiiCHAMjbxYtA9+5Aba3+9uXLAU9PxofcqdXSieI//ymVGxqaMfeMj3bFrSaBkBm6izgXFBhPw+uxypdaDTz9NBAcrF/508Vlf+StshIIDW1e+dNgfMibWg2MHSudLOpW/jQYH+0OWwDdnVoNPPMM8MorgMHSN3p8fXk9VrmqrAR69JBa/8zhsj/ypOnKW7Gi+bV6dTE+5EnT6rdqlfmrQzE+2h1WAN2VpR+lr2/TWVqPHsCRI5wAIlcDB5qv/Hl5AU8+yWV/5KimBujfHzh50nQaLy8pNhgf8qNWS71KO3aYT9exI+OjHWIF0B1Z86NctIjjMUhy5ozpxxYtkoYOcPC2fOgO3v/9d/OVP83JY4cObZY9cjLd+FCrLVf+NDHCMqTd4TfiLnSXdGlsNN+iExvLszFq0r1784N8RARw6BAP7HKke81vYzw8gJ49GR9yZSk+AOlKUaNGAVlZ7FlqxzgJxF1MmCCdiV2+bLry5+UFpKZKq6vzbEw+dCcALV/efEjAoUNShc/LS/pbUQGcOMGDu1zpXvPbUESENAmE8SFf5uJDoQB69QLKy4GdO1n5a+dYC3BllZXS+K0zZ4CGBvNpe/UCDh9moS0nmlbhPXuA6mpp27Zt0l/d7v8OHaQDOhEgDdbftq3pIB8RIR3QNYvF8+RRPoyt5acbHwoFMHq0FBNc68/l8JtyZQMHmh+fA0jdvWzxk6eEBGDXLv1tQnA5BpJODhISgLw86f6ttzZ112mGh3ABX3kzHEuuWcuP8eE2+K25EsOzMcPB+wqFtI5bUBAQGSmdmfHHKR+aJX/ef1+6b+zkQKHgcgxypilDXnpJf1monByptXj7dl7zW850F3KurW3e1Zuby/hwI6wZuALNZbkKCpp+kNnZ0tR63fF+vXqxK0+uLl4EwsIsDwVIS+MEIDmqqQHi4sy3/vJKQPJlTXwAPHl0M5wE0p6p1cDChUBgoHTpNsOzseuv1x+8f+iQEzJJTqWZ4NG1q+XKX0yMdObOFmH5qKyUegP8/Cwf3HklIPmpqZF6ivz9LccHV49wOzwStEemrrdpKDaWTfFyVVraNNvOHMMB/CQPNTXAuHFS1645Hh7S7dZbGR9yolYDzz4rXSjA2GXbdPn6SuuBcjiR2+G32Z5oxl8895zltDwbkye1GnjqKWmMjiWjR0tDBVhoy0tpKRASYn6dNkA6sFdUMD7k6PHHgXXrLKfj6hFujV3A7UlGhnWVv6ef5sxeucrIsFz5UyikS7ex8idPkZGWK39KJVBYyPiQK0uVP5VKmiR08iQrf26Mv/72xNIYDE9PaeZvWFjb5IfaH3Mx0rEjcOECD+pyZ2oheF9fYOjQpt4DxgkZGjUK+PZbxoZMsAWwPTE1w2rECKC+Xur+Y+VP3kzFSM+ewOnTLLhJOhEwpFBI3b07dnAiEBmXmsrKn8zwm3Yi8f/dNOWagfwpKVIh/fLLQF2d1E0ze7Y0APfKFSfmlIzRfG/CUndbKxiNkQsX9LtwjhwBrrpKuga0pUkh1GacEh+AtJzLgAFAVZV038MDOHqUZUg75LQY2bcPuPlm/fvXXMMYaWccHR8K4cjII7OOHz+Oq6++2tnZoFY6ffo0unfv7pB9M0ZcH+ODLGGMkDmOig+2ADpRSEgIAKCgoAAqlcrJuZGP8vJy9OjRA6dPn0ZQUFCL9yOEQEVFBcLDw+2YO32MEeewR4wwPtwXyxAyx1XigxVAJ/LwkIZgqlSqVgUJtUxQUFCrP3dHF6iMEedqbYwwPtwbyxAyp73HByeBEBEREckMK4BEREREMsMKoBMplUqkpqZCqVQ6Oyuy4kqfuyvl1Z24yufuKvl0N670ubtSXt2Fq3zmnAVMREREJDNsASQiIiKSGVYAiYiIiGSGFUAiIiIimWEFkIiIiEhmWAF0otdeew2RkZHw9fXF4MGDsXv3bmdnqV1asWIFbr75ZgQGBiI0NBRTpkzBsWPH9NIIIZCWlobw8HD4+flh9OjR+OWXX/TS1NbWYvbs2ejcuTMCAgIwefJknDlzRi9NSUkJkpOToVKpoFKpkJycjNLSUr00BQUFmDRpEgICAtC5c2fMmTMHdXV1dn/fjA/ryDU+AMaIteQaI4wP68g1PiDIKTIzM4W3t7f497//LY4cOSIee+wxERAQIE6dOuXsrLU7CQkJ4u233xaHDx8WBw4cEBMnThQ9e/YUlZWV2jQrV64UgYGB4tNPPxWHDh0Sd999t+jWrZsoLy/XpnnkkUfEVVddJbKzs8WPP/4oYmNjxaBBg4RardamGTdunIiOjhZ5eXkiLy9PREdHi8TERO3jarVaREdHi9jYWPHjjz+K7OxsER4eLlJSUuz6nhkf1pNjfAjBGLGFHGOE8WE9OcaHEEKwAugkt9xyi3jkkUf0tvXr10889dRTTsqR6yguLhYARE5OjhBCiMbGRhEWFiZWrlypTVNTUyNUKpVYv369EEKI0tJS4e3tLTIzM7Vpzp49Kzw8PERWVpYQQogjR44IAGLPnj3aNPn5+QKAOHr0qBBCiC1btggPDw9x9uxZbZqNGzcKpVIpysrK7PYeGR8tJ4f4EIIx0hpyiBHGR8vJIT6EEIJdwE5QV1eH/fv3Iz4+Xm97fHw88vLynJQr11FWVgag6SLnJ06cQFFRkd7nqVQqERMTo/089+/fj/r6er004eHhiI6O1qbJz8+HSqXCkCFDtGmGDh0KlUqllyY6Olrv4twJCQmora3F/v377fL+GB+t4+7xATBGWsvdY4Tx0TruHh8arAA6wcWLF9HQ0ICuXbvqbe/atSuKioqclCvXIITAvHnzMGLECERHRwOA9jMz93kWFRXBx8cHwcHBZtOEhoY2e83Q0FC9NIavExwcDB8fH7t9d4yPlpNDfACMkdaQQ4wwPlpODvGh4WXXvZFNFAqF3n0hRLNtpC8lJQU///wzcnNzmz3Wks/TMI2x9C1JYw+MD9vJKT6MvRZjxDI5xQjjw3Zyig+2ADpB586d4enp2aw2X1xc3KzmT01mz56NL774Ajt27ED37t2128PCwgDA7OcZFhaGuro6lJSUmE1z/vz5Zq974cIFvTSGr1NSUoL6+nq7fXeMj5aRS3wAjJGWkkuMMD5aRi7xocEKoBP4+Phg8ODByM7O1tuenZ2N4cOHOylX7ZcQAikpKdi0aRO2b9+OyMhIvccjIyMRFham93nW1dUhJydH+3kOHjwY3t7eemkKCwtx+PBhbZphw4ahrKwM33//vTbN3r17UVZWppfm8OHDKCws1KbZunUrlEolBg8ebJf3y/iwjdziA2CM2EpuMcL4sI3c4kP3jZMTaKbov/nmm+LIkSNi7ty5IiAgQJw8edLZWWt3/vGPfwiVSiV27twpCgsLtbcrV65o06xcuVKoVCqxadMmcejQITF16lSjU/S7d+8utm3bJn788Udx2223GZ2if91114n8/HyRn58vBg4caHSK/pgxY8SPP/4otm3bJrp37+6wJRwYH5bJMT6EYIzYQo4xwviwnhzjQwguA+NUr776qujVq5fw8fERN954o3bKOekDYPT29ttva9M0NjaK1NRUERYWJpRKpRg1apQ4dOiQ3n6qq6tFSkqKCAkJEX5+fiIxMVEUFBTopbl06ZKYNm2aCAwMFIGBgWLatGmipKREL82pU6fExIkThZ+fnwgJCREpKSmipqbG7u+b8WEducaHEIwRa8k1RtwhPtRqtaiurnborVevXkZvH374oTbNlStXxJo1a8RNN90k+vTpI+68805x8OBBvf2UlpaKp59+WgwaNEj07dtXzJgxQxw/flwvTWFhoUhJSREDBgwQAwYMECkpKeL8+fN6af78808xffp00bdvXzFo0CDx9NNPi7KyMpvfV0NDg9nPViGEEPZtUyQiIiJqOSEEioqKml0lg6zn4eGByMhI+Pj4GH2cFUAiIiJqVwoLC1FaWorQ0FD4+/tz9rKNGhsbce7cOXh7e6Nnz55GPz8uA0NERETtRkNDg7by16lTJ2dnx2V16dIF586dg1qthre3d7PHOQuYiIiI2o36+noAgL+/v5Nz4to0Xb8NDQ1GH2cFkIiIiNoddvu2jqXPjxVAIiIiIplhBZCIiIionYmIiMBLL73ksP1zEggRERGRHYwePRrXX3+9XSpu+/btQ0BAQOszZQIrgERERERtQAiBhoYGeHlZrn516dLFoXlhFzARERFRK82YMQM5OTlYu3YtFAoFFAoFNmzYAIVCgW+++QY33XQTlEoldu/ejT///BO33347unbtig4dOuDmm2/Gtm3b9PZn2AWsUCjwn//8B3/5y1/g7++PPn364IsvvmhxflkBJCIiImqltWvXYtiwYZg5cyYKCwtRWFiIHj16AAAWLlyIFStW4Ndff8V1112HyspKTJgwAdu2bcNPP/2EhIQETJo0CQUFBWZfIz09HXfddRd+/vlnTJgwAdOmTcPly5dblF9WAImIiMjtqNXA8uVAfLz0V6127OupVCr4+PjA398fYWFhCAsLg6enJwBg+fLliIuLw9VXX41OnTph0KBBePjhhzFw4ED06dMHzz77LKKioiy26M2YMQNTp05F7969kZGRgaqqKnz//fctyi/HABIREZHbycgA0tIAIQBN7+qyZc7Jy0033aR3v6qqCunp6fjyyy+1V+uorq622AJ43XXXaf8PCAhAYGAgiouLW5QnVgCJiIjI7eTmSpU/QPqbm+u8vBjO5n3iiSfwzTff4Pnnn0fv3r3h5+eHO++8E3V1dWb3Y3hJN4VCgcbGxhbliRVAIiIicjsjRkgtf0IACoV039F8fHxMXnpN1+7duzFjxgz85S9/AQBUVlbi5MmTDs6dPlYAiYiIyO0sXiz9zc2VKn+a+44UERGBvXv34uTJk+jQoYPJ1rnevXtj06ZNmDRpEhQKBZYuXdrilryW4iQQIiIicjteXtKYv61bpb9WLL3XagsWLICnpycGDBiALl26mBzT9+KLLyI4OBjDhw/HpEmTkJCQgBtvvNHxGdShEELTQ05ERETkXDU1NThx4gQiIyPh6+vr7Oy4LEufI1sAiYiIiGSGFUAiIiIimWEFkIiIiEhmWAEkIiIikhlWAImIiIhkhhVAIiIiIplhBZCIiIhIZlgBJCIiIpIZVgCJiIiIZIYVQCIiIiKZYQWQiIiIyA5Gjx6NuXPn2m1/M2bMwJQpU+y2P12sABJRm7py5QrS0tKwc+fONnm9GTNmICIiok1ei4jIVbACSERt6sqVK0hPT2+zCiARUVuYMWMGcnJysHbtWigUCigUCpw8eRJHjhzBhAkT0KFDB3Tt2hXJycm4ePGi9nmffPIJBg4cCD8/P3Tq1Aljx45FVVUV0tLS8M477+Dzzz/X7s+e5SYrgERERESttHbtWgwbNgwzZ85EYWEhCgsL4e3tjZiYGFx//fX44YcfkJWVhfPnz+Ouu+4CABQWFmLq1Kn4+9//jl9//RU7d+7EX//6VwghsGDBAtx1110YN26cdn/Dhw+3W35ZASQii/744w/cf//96NOnD/z9/XHVVVdh0qRJOHToULO0paWlmD9/PqKioqBUKhEaGooJEybg6NGjOHnyJLp06QIASE9P157VzpgxA4Dp7tq0tDQoFAq9ba+++ipGjRqF0NBQBAQEYODAgVi9ejXq6+tb/D6zsrIwZswYqFQq+Pv7o3///lixYoX28R9++AH33HMPIiIi4Ofnh4iICEydOhWnTp3S28+VK1ewYMECREZGwtfXFyEhIbjpppuwceNGvXQ//PADJk+ejJCQEPj6+uKGG27Axx9/3KJ9EZEBtRpYvhyIj5f+qtUOfTmVSgUfHx/4+/sjLCwMYWFheOONN3DjjTciIyMD/fr1ww033IC33noLO3bswG+//YbCwkKo1Wr89a9/RUREBAYOHIhHH30UHTp0QIcOHeDn5welUqndn4+Pj93y62W3PRGR2zp37hw6deqElStXokuXLrh8+TLeeecdDBkyBD/99BP69u0LAKioqMCIESNw8uRJPPnkkxgyZAgqKyuxa9cu7dlrVlYWxo0bhwceeAAPPvggAGgrhbb4888/kZSUhMjISPj4+ODgwYN47rnncPToUbz11ls27+/NN9/EzJkzERMTg/Xr1yM0NBS//fYbDh8+rE1z8uRJ9O3bF/fccw9CQkJQWFiI119/HTfffDOOHDmCzp07AwDmzZuH9957D88++yxuuOEGVFVV4fDhw7h06ZJ2Xzt27MC4ceMwZMgQrF+/HiqVCpmZmbj77rtx5coVbaXYmn0RkREZGUBaGiAEsG2btG3ZsjbNwv79+7Fjxw506NCh2WN//vkn4uPjMWbMGAwcOBAJCQmIj4/HnXfeieDgYMdnThAR2UitVou6ujrRp08f8fjjj2u3L1++XAAQ2dnZJp974cIFAUCkpqY2e2z69OmiV69ezbanpqYKc8VVQ0ODqK+vF++++67w9PQUly9ftrhPXRUVFSIoKEiMGDFCNDY2mk2rS61Wi8rKShEQECDWrl2r3R4dHS2mTJli9rn9+vUTN9xwg6ivr9fbnpiYKLp16yYaGhqs3heRO6murhZHjhwR1dXVrdtRXJwQUvVPusXF2SeDZsTExIjHHntMe3/cuHHir3/9q/j999+b3SorK4UQQjQ2Norc3FyxbNkyMXDgQNGlSxdx/PhxIYRUft1+++0tyoulz5FdwERkkVqtRkZGBgYMGAAfHx94eXnBx8cHv//+O3799Vdtuq+//hrXXHMNxo4d6/A8/fTTT5g8eTI6deoET09PeHt747777kNDQwN+++03m/aVl5eH8vJyPProo826mnVVVlbiySefRO/eveHl5QUvLy906NABVVVVep/DLbfcgq+//hpPPfUUdu7cierqar39/PHHHzh69CimTZsGQPp8NbcJEyagsLAQx44ds2pfRGTCiBGA5vesUEj3HczHxwcNDQ3a+zfeeCN++eUXREREoHfv3nq3gICA/8+aArfeeivS09Px008/wcfHB5999pnR/dkTK4BEZNG8efOwdOlSTJkyBf/73/+wd+9e7Nu3D4MGDdKrkFy4cAHdu3d3eH4KCgowcuRInD17FmvXrsXu3buxb98+vPrqqwBgcyXpwoULAGAx70lJSVi3bh0efPBBfPPNN/j++++xb98+dOnSRe81X375ZTz55JPYvHkzYmNjERISgilTpuD3338HAJw/fx4AsGDBAnh7e+vdHn30UQDQzhK0tC8iMmHxYqkLOC5O+rt4scNfMiIiAnv37sXJkydx8eJFzJo1C5cvX8bUqVPx/fff4/jx49i6dSv+/ve/o6GhAXv37kVGRgZ++OEHFBQUYNOmTbhw4QL69++v3d/PP/+MY8eO4eLFi60a42yIYwCJyKL3338f9913HzIyMvS2X7x4ER07dtTe79KlC86cOdPi1/H19UVtbW2z7bpLJgDA5s2bUVVVhU2bNqFXr17a7QcOHGjR62rGIJrLe1lZGb788kukpqbiqaee0m6vra3F5cuX9dIGBAQgPT0d6enpOH/+vLYFb9KkSTh69Kh2rOCiRYvw17/+1ejracZVWtoXEZng5dXmY/4WLFiA6dOnY8CAAaiursaJEyfw3Xff4cknn0RCQgJqa2vRq1cvjBs3Dh4eHggKCsKuXbvw0ksvoby8HL169cI///lPjB8/HgAwc+ZM7Ny5EzfddBMqKyuxY8cOjB492i55ZQWQiCxSKBRQKpV627766iucPXsWvXv31m4bP348li1bhu3bt+O2224zui/Nfoy10kVERKC4uBjnz59H165dAQB1dXX45ptvmuVHd18AIITAv//97xa8O2D48OFQqVRYv3497rnnHqPdwAqFAkKIZp/Df/7zH7NdNF27dsWMGTNw8OBBvPTSS7hy5Qr69u2LPn364ODBg80q1eYY25e/v7/1b5SIHOqaa65Bfn5+s+2bNm0ymr5///7Iysoyub8uXbpg69atdsufLlYAiciixMREbNiwAf369cN1112H/fv3Y82aNc26TOfOnYuPPvoIt99+O5566inccsstqK6uRk5ODhITExEbG4vAwED06tULn3/+OcaMGYOQkBB07twZERERuPvuu7Fs2TLcc889eOKJJ1BTU4OXX365WQUrLi4OPj4+mDp1KhYuXIiamhq8/vrrKCkpadH769ChA/75z3/iwQcfxNixYzFz5kx07doVf/zxBw4ePIh169YhKCgIo0aNwpo1a7T5zcnJwZtvvqnXCgoAQ4YMQWJiIq677joEBwfj119/xXvvvYdhw4ZpK2xvvPEGxo8fj4SEBMyYMQNXXXUVLl++jF9//RU//vgj/vvf/1q9LyIim7VoagkRyUpJSYl44IEHRGhoqPD39xcjRowQu3fvFjExMSImJqZZ2scee0z07NlTeHt7i9DQUDFx4kRx9OhRbZpt27aJG264QSiVSgFATJ8+XfvYli1bxPXXXy/8/PxEVFSUWLdundFZwP/73//EoEGDhK+vr7jqqqvEE088Ib7++msBQOzYsUObzppZwLqvHRMTIwICAoS/v78YMGCAWLVqlfbxM2fOiDvuuEMEBweLwMBAMW7cOHH48GHRq1cvvffw1FNPiZtuukkEBwcLpVIpoqKixOOPPy4uXryo93oHDx4Ud911lwgNDRXe3t4iLCxM3HbbbWL9+vU274vIXdhtFrDMWfocFUII4eQ6KBEREREAoKamBidOnNAufk4tY+lz5CxgIiIiIplhBZCIiIjaHXZQto6lz48VQCIiImo3vL29AUjXwaaWq6urAwB4enoafZyzgImIiKjd8PT0RMeOHVFcXAwA8Pf3N3uFHmqusbERFy5cgL+/P7y8jFf1WAEkIiKidiUsLAwAtJVAsp2Hhwd69uxpsvLMWcBERETULjU0NNj18mdy4uPjAw8P0yP9WAEkIiIikhlOAiEiIiKSGVYAiYiIiGSGFUAiIiIimWEFkIiIiEhm/g8k0wqHs7XFDgAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 16 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "X_train, X_test, y_train, y_test = split_sc(train, test, feature_cols, label_cols, None)\n",
    "plotPredAct(X_train, X_test, y_train, y_test, knn_reg, 'KNN Regression (raw)', 'K=', range(1, 23, 3), 2, 4, True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "777458eb",
   "metadata": {},
   "source": [
    "## Polynomial Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "ab0b0fcc",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "def poly_reg(degree, X_train, X_test, y_train, y_test):\n",
    "    pf = PolynomialFeatures(degree=degree, include_bias=False) #creating an instance of PolynomialFeature\n",
    "    poly_model = LinearRegression() #create an instance of LinearRegression\n",
    "    \n",
    "    # transform the train and test data into polynomial form\n",
    "    poly_X_train = pf.fit_transform(X_train)\n",
    "    poly_X_test = pf.fit_transform(X_test)\n",
    "    \n",
    "    # train the model\n",
    "    poly_model.fit(poly_X_train, y_train)\n",
    "    \n",
    "    # prediction\n",
    "    y_train_pred = poly_model.predict(poly_X_train)\n",
    "    y_test_pred = poly_model.predict(poly_X_test)\n",
    "    \n",
    "    return (y_train_pred, y_test_pred)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b7f67358",
   "metadata": {},
   "source": [
    "Get RMSE with linear model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "0778eafb",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = split_sc(train, test, feature_cols, label_cols, None)\n",
    "y_train_pred, y_test_pred = poly_reg(1, X_train, X_test, y_train, y_test)\n",
    "Linear_rmse = rmse(y_test_pred, y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c9917d35",
   "metadata": {},
   "source": [
    "Generate table of RMSE with Polynomial model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "36f21d98",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style type=\"text/css\">\n",
       "</style>\n",
       "<table id=\"T_286b4\">\n",
       "  <caption>RMSE | Polynomial Regression(degree=1)</caption>\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th class=\"blank level0\" >&nbsp;</th>\n",
       "      <th id=\"T_286b4_level0_col0\" class=\"col_heading level0 col0\" >raw</th>\n",
       "      <th id=\"T_286b4_level0_col1\" class=\"col_heading level0 col1\" >log</th>\n",
       "      <th id=\"T_286b4_level0_col2\" class=\"col_heading level0 col2\" >mm_sc</th>\n",
       "      <th id=\"T_286b4_level0_col3\" class=\"col_heading level0 col3\" >mm_sc+log</th>\n",
       "      <th id=\"T_286b4_level0_col4\" class=\"col_heading level0 col4\" >ss_sc</th>\n",
       "      <th id=\"T_286b4_level0_col5\" class=\"col_heading level0 col5\" >ss_sc+log</th>\n",
       "      <th id=\"T_286b4_level0_col6\" class=\"col_heading level0 col6\" >ma_sc</th>\n",
       "      <th id=\"T_286b4_level0_col7\" class=\"col_heading level0 col7\" >ma_sc+log</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th id=\"T_286b4_level0_row0\" class=\"row_heading level0 row0\" >train</th>\n",
       "      <td id=\"T_286b4_row0_col0\" class=\"data row0 col0\" >8.84e+02</td>\n",
       "      <td id=\"T_286b4_row0_col1\" class=\"data row0 col1\" >1.09e+03</td>\n",
       "      <td id=\"T_286b4_row0_col2\" class=\"data row0 col2\" >8.84e+02</td>\n",
       "      <td id=\"T_286b4_row0_col3\" class=\"data row0 col3\" >1.09e+03</td>\n",
       "      <td id=\"T_286b4_row0_col4\" class=\"data row0 col4\" >8.84e+02</td>\n",
       "      <td id=\"T_286b4_row0_col5\" class=\"data row0 col5\" >1.09e+03</td>\n",
       "      <td id=\"T_286b4_row0_col6\" class=\"data row0 col6\" >8.84e+02</td>\n",
       "      <td id=\"T_286b4_row0_col7\" class=\"data row0 col7\" >1.09e+03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th id=\"T_286b4_level0_row1\" class=\"row_heading level0 row1\" >test</th>\n",
       "      <td id=\"T_286b4_row1_col0\" class=\"data row1 col0\" >9.38e+02</td>\n",
       "      <td id=\"T_286b4_row1_col1\" class=\"data row1 col1\" >9.54e+03</td>\n",
       "      <td id=\"T_286b4_row1_col2\" class=\"data row1 col2\" >9.38e+02</td>\n",
       "      <td id=\"T_286b4_row1_col3\" class=\"data row1 col3\" >9.54e+03</td>\n",
       "      <td id=\"T_286b4_row1_col4\" class=\"data row1 col4\" >9.38e+02</td>\n",
       "      <td id=\"T_286b4_row1_col5\" class=\"data row1 col5\" >9.54e+03</td>\n",
       "      <td id=\"T_286b4_row1_col6\" class=\"data row1 col6\" >9.38e+02</td>\n",
       "      <td id=\"T_286b4_row1_col7\" class=\"data row1 col7\" >9.54e+03</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n"
      ],
      "text/plain": [
       "<pandas.io.formats.style.Styler at 0x25806bb2590>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style type=\"text/css\">\n",
       "</style>\n",
       "<table id=\"T_87517\">\n",
       "  <caption>RMSE | Polynomial Regression(degree=2)</caption>\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th class=\"blank level0\" >&nbsp;</th>\n",
       "      <th id=\"T_87517_level0_col0\" class=\"col_heading level0 col0\" >raw</th>\n",
       "      <th id=\"T_87517_level0_col1\" class=\"col_heading level0 col1\" >log</th>\n",
       "      <th id=\"T_87517_level0_col2\" class=\"col_heading level0 col2\" >mm_sc</th>\n",
       "      <th id=\"T_87517_level0_col3\" class=\"col_heading level0 col3\" >mm_sc+log</th>\n",
       "      <th id=\"T_87517_level0_col4\" class=\"col_heading level0 col4\" >ss_sc</th>\n",
       "      <th id=\"T_87517_level0_col5\" class=\"col_heading level0 col5\" >ss_sc+log</th>\n",
       "      <th id=\"T_87517_level0_col6\" class=\"col_heading level0 col6\" >ma_sc</th>\n",
       "      <th id=\"T_87517_level0_col7\" class=\"col_heading level0 col7\" >ma_sc+log</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th id=\"T_87517_level0_row0\" class=\"row_heading level0 row0\" >train</th>\n",
       "      <td id=\"T_87517_row0_col0\" class=\"data row0 col0\" >1.66e-07</td>\n",
       "      <td id=\"T_87517_row0_col1\" class=\"data row0 col1\" >7.66e-08</td>\n",
       "      <td id=\"T_87517_row0_col2\" class=\"data row0 col2\" >3.60e-11</td>\n",
       "      <td id=\"T_87517_row0_col3\" class=\"data row0 col3\" >2.56e-11</td>\n",
       "      <td id=\"T_87517_row0_col4\" class=\"data row0 col4\" >3.07e-11</td>\n",
       "      <td id=\"T_87517_row0_col5\" class=\"data row0 col5\" >4.86e-11</td>\n",
       "      <td id=\"T_87517_row0_col6\" class=\"data row0 col6\" >5.42e-11</td>\n",
       "      <td id=\"T_87517_row0_col7\" class=\"data row0 col7\" >8.51e-11</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th id=\"T_87517_level0_row1\" class=\"row_heading level0 row1\" >test</th>\n",
       "      <td id=\"T_87517_row1_col0\" class=\"data row1 col0\" >5.15e+03</td>\n",
       "      <td id=\"T_87517_row1_col1\" class=\"data row1 col1\" >3.02e+04</td>\n",
       "      <td id=\"T_87517_row1_col2\" class=\"data row1 col2\" >2.62e+03</td>\n",
       "      <td id=\"T_87517_row1_col3\" class=\"data row1 col3\" >5.04e+03</td>\n",
       "      <td id=\"T_87517_row1_col4\" class=\"data row1 col4\" >2.70e+03</td>\n",
       "      <td id=\"T_87517_row1_col5\" class=\"data row1 col5\" >7.33e+03</td>\n",
       "      <td id=\"T_87517_row1_col6\" class=\"data row1 col6\" >2.63e+03</td>\n",
       "      <td id=\"T_87517_row1_col7\" class=\"data row1 col7\" >1.03e+04</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n"
      ],
      "text/plain": [
       "<pandas.io.formats.style.Styler at 0x25807c22e90>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style type=\"text/css\">\n",
       "</style>\n",
       "<table id=\"T_fb303\">\n",
       "  <caption>RMSE | Polynomial Regression(degree=3)</caption>\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th class=\"blank level0\" >&nbsp;</th>\n",
       "      <th id=\"T_fb303_level0_col0\" class=\"col_heading level0 col0\" >raw</th>\n",
       "      <th id=\"T_fb303_level0_col1\" class=\"col_heading level0 col1\" >log</th>\n",
       "      <th id=\"T_fb303_level0_col2\" class=\"col_heading level0 col2\" >mm_sc</th>\n",
       "      <th id=\"T_fb303_level0_col3\" class=\"col_heading level0 col3\" >mm_sc+log</th>\n",
       "      <th id=\"T_fb303_level0_col4\" class=\"col_heading level0 col4\" >ss_sc</th>\n",
       "      <th id=\"T_fb303_level0_col5\" class=\"col_heading level0 col5\" >ss_sc+log</th>\n",
       "      <th id=\"T_fb303_level0_col6\" class=\"col_heading level0 col6\" >ma_sc</th>\n",
       "      <th id=\"T_fb303_level0_col7\" class=\"col_heading level0 col7\" >ma_sc+log</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th id=\"T_fb303_level0_row0\" class=\"row_heading level0 row0\" >train</th>\n",
       "      <td id=\"T_fb303_row0_col0\" class=\"data row0 col0\" >1.01e-06</td>\n",
       "      <td id=\"T_fb303_row0_col1\" class=\"data row0 col1\" >1.08e-08</td>\n",
       "      <td id=\"T_fb303_row0_col2\" class=\"data row0 col2\" >1.33e-10</td>\n",
       "      <td id=\"T_fb303_row0_col3\" class=\"data row0 col3\" >5.50e-11</td>\n",
       "      <td id=\"T_fb303_row0_col4\" class=\"data row0 col4\" >7.55e-11</td>\n",
       "      <td id=\"T_fb303_row0_col5\" class=\"data row0 col5\" >3.84e-11</td>\n",
       "      <td id=\"T_fb303_row0_col6\" class=\"data row0 col6\" >8.30e-11</td>\n",
       "      <td id=\"T_fb303_row0_col7\" class=\"data row0 col7\" >6.41e-11</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th id=\"T_fb303_level0_row1\" class=\"row_heading level0 row1\" >test</th>\n",
       "      <td id=\"T_fb303_row1_col0\" class=\"data row1 col0\" >5.48e+04</td>\n",
       "      <td id=\"T_fb303_row1_col1\" class=\"data row1 col1\" >3.27e+05</td>\n",
       "      <td id=\"T_fb303_row1_col2\" class=\"data row1 col2\" >2.29e+03</td>\n",
       "      <td id=\"T_fb303_row1_col3\" class=\"data row1 col3\" >4.64e+03</td>\n",
       "      <td id=\"T_fb303_row1_col4\" class=\"data row1 col4\" >2.27e+03</td>\n",
       "      <td id=\"T_fb303_row1_col5\" class=\"data row1 col5\" >3.41e+03</td>\n",
       "      <td id=\"T_fb303_row1_col6\" class=\"data row1 col6\" >2.33e+03</td>\n",
       "      <td id=\"T_fb303_row1_col7\" class=\"data row1 col7\" >4.50e+03</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n"
      ],
      "text/plain": [
       "<pandas.io.formats.style.Styler at 0x25807c16350>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style type=\"text/css\">\n",
       "</style>\n",
       "<table id=\"T_45c79\">\n",
       "  <caption>RMSE | Polynomial Regression(degree=4)</caption>\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th class=\"blank level0\" >&nbsp;</th>\n",
       "      <th id=\"T_45c79_level0_col0\" class=\"col_heading level0 col0\" >raw</th>\n",
       "      <th id=\"T_45c79_level0_col1\" class=\"col_heading level0 col1\" >log</th>\n",
       "      <th id=\"T_45c79_level0_col2\" class=\"col_heading level0 col2\" >mm_sc</th>\n",
       "      <th id=\"T_45c79_level0_col3\" class=\"col_heading level0 col3\" >mm_sc+log</th>\n",
       "      <th id=\"T_45c79_level0_col4\" class=\"col_heading level0 col4\" >ss_sc</th>\n",
       "      <th id=\"T_45c79_level0_col5\" class=\"col_heading level0 col5\" >ss_sc+log</th>\n",
       "      <th id=\"T_45c79_level0_col6\" class=\"col_heading level0 col6\" >ma_sc</th>\n",
       "      <th id=\"T_45c79_level0_col7\" class=\"col_heading level0 col7\" >ma_sc+log</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th id=\"T_45c79_level0_row0\" class=\"row_heading level0 row0\" >train</th>\n",
       "      <td id=\"T_45c79_row0_col0\" class=\"data row0 col0\" >3.62e-06</td>\n",
       "      <td id=\"T_45c79_row0_col1\" class=\"data row0 col1\" >1.18e-06</td>\n",
       "      <td id=\"T_45c79_row0_col2\" class=\"data row0 col2\" >3.86e-10</td>\n",
       "      <td id=\"T_45c79_row0_col3\" class=\"data row0 col3\" >9.54e-11</td>\n",
       "      <td id=\"T_45c79_row0_col4\" class=\"data row0 col4\" >1.77e-10</td>\n",
       "      <td id=\"T_45c79_row0_col5\" class=\"data row0 col5\" >1.77e-10</td>\n",
       "      <td id=\"T_45c79_row0_col6\" class=\"data row0 col6\" >1.01e-09</td>\n",
       "      <td id=\"T_45c79_row0_col7\" class=\"data row0 col7\" >1.07e-10</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th id=\"T_45c79_level0_row1\" class=\"row_heading level0 row1\" >test</th>\n",
       "      <td id=\"T_45c79_row1_col0\" class=\"data row1 col0\" >7.34e+05</td>\n",
       "      <td id=\"T_45c79_row1_col1\" class=\"data row1 col1\" >9.98e+05</td>\n",
       "      <td id=\"T_45c79_row1_col2\" class=\"data row1 col2\" >2.85e+03</td>\n",
       "      <td id=\"T_45c79_row1_col3\" class=\"data row1 col3\" >3.54e+03</td>\n",
       "      <td id=\"T_45c79_row1_col4\" class=\"data row1 col4\" >4.66e+03</td>\n",
       "      <td id=\"T_45c79_row1_col5\" class=\"data row1 col5\" >3.06e+03</td>\n",
       "      <td id=\"T_45c79_row1_col6\" class=\"data row1 col6\" >2.54e+03</td>\n",
       "      <td id=\"T_45c79_row1_col7\" class=\"data row1 col7\" >2.51e+03</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n"
      ],
      "text/plain": [
       "<pandas.io.formats.style.Styler at 0x25807dabdd0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "for i in range(1, 5):\n",
    "    degree = i\n",
    "    display(errorMetrics(poly_reg, 'RMSE | Polynomial Regression(degree=' + str(degree)+')',degree))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "17ad55ed",
   "metadata": {},
   "source": [
    "Generate plots of predicted vs actual cases with Polynomial models with raw data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "fe86eaf8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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xYsvvohevXqhoqICaWlp2L17N+bNm4eBAwdiyZIluHbtGkaMGAEvLy/k5uZi1apVsLa2RlBQEABg7Nix6nUmXV1dceXKFXzwwQfo3LlzjYtcN+f/ediwYVi2bFm91/jT93MwYsQIlJeXY+/evVpLDY0cORLR0dGQyWQ61/o7deoU7t69i+Dg4EadH5G5YgsgUSPJZDJs3boVc+fOxbp16zB69Gj1kjD79u2r1hJSkyeffBIAalyOpL4WLFiAL774AidPnsSjjz6KqKgo9OzZE0lJSU1+1Yxvv/0WwcHBePnll/H444/jt99+Q3x8PBQKhcFf65///CcGDhyI9957D3l5eejRoweOHTsGf39/LFq0CKGhoZg6dSp++OEHrRYxT09PJCYm4t5778X06dMxceJE2NjYYNmyZQBQr2C1TZs2OHjwIKZMmYLPP/8cY8aMwZNPPokPP/wQXl5e6hbAgQMHIjMzE6+88gpCQ0Px73//G3Z2dti3bx969uwJQFpY/MCBA5g+fTpCQkKwaNEijBgxAomJibC2tq4xD831f54wYQLKy8uxY8eOej9Hn89Bnz591F3Wmi19qr/79OlTbewrAGzduhXt27dHaGiovqdERABkQmisrElERtOvXz/IZDIcPXrU2FkxS//+97+xceNGZGdnw8bGxtjZaVHGjh0LpVKJXbt2GTsrAIDy8nJ06dIFEyZMwBtvvGHs7BC1SuwCJjKi/Px8nDlzBtu3b0dqaip+/PFHY2fJLCxbtgyenp7w8/NDYWEhtm/fji+++AKLFi1i8KfDihUr0KdPHxw9erTOSU3NYcOGDSgsLMRLL71k7KwQtVoMAImM6NixYwgODoaLiwuio6Px6KOPGjtLZsHa2hpvv/02rl27BqVSia5du+K9997DCy+8YOystUj+/v5Yt25djTOlm1tFRQW++eYbg48tJTIn7AImIiIiMjOcBEJERERkZhgAEhEREZkZBoBEREREZoYBIBEREZGZYQBIREREZGYYABIRERGZGQaARERERGaGASARERGRmWEASERERGRmGAASERERmRkGgERERERmhgEgERERkZlhAEhERERkZhgAEhEREZkZBoBEREREZoYBIBEREZGZYQBIREREZGYYABIRERGZGQaARERERGaGASARERGRmWEASATAx8cH9vb2aNu2LVxcXDBy5Ehs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r+plGmvTuTNw5gy7eYlaI7YANtJ7772HqVOn4tlnn0X37t3xwQcfwNvbG59++mmz58XKCujTp/Y0CgXg7CxdSi04WBqcPWyY1NKn6ZVX2JVDZI7q+gHZubPU2ieENKGMwR9R68Sv90YoLS1FamoqXn31Va3toaGhSEpKqpa+pKQEJRqrpObl5QGQpnUbyqBBNV8xw9sb+O236ivtv/SSNNj7nXeAlBTpGLNnAwbMFhGhsqw3dPWt5qhDFAppXb+qHByk1r527aRVBlg/EDW/xtYhWgQ12PXr1wUA8euvv2ptf+ONN8S9995bLX10dLQAwBtvvJn57erVqw2qc1iH8MYbb0DD6xBNXAi6EdLT09GxY0ckJSVh8ODB6u1vvPEGvv76a5w7d04rfdVf7xUVFbh9+zZcXFwgk8lqfJ38/Hx4e3vj6tWrzbbYa1MypfPhubRMLfVchBAoKCiAp6cnLCz0H4HT0DqkJi31fWoqPF/TZg7n29g6RBO7gBuhffv2sLS0RGZmptb2rKwsuLu7V0svl8shl8u1trVr167er+fo6GhSH2pTOh+eS8vUEs9FoVA0+LmNrUNq0hLfp6bE8zVtpn6+jalDNHESSCPY2Nigb9++iI+P19oeHx+PIUOGGClXRERERLVjC2AjzZ07F5GRkejXrx8GDx6Mzz//HGlpaZg+fbqxs0ZERESkEwPARho3bhyys7OxbNkyZGRkwN/fHzt37kTnzp0N9hpyuRzR0dHVun5aK1M6H55Ly2RK59KUzO194vmaNnM738biJBAiIiIiM8MxgERERERmhgEgERERkZlhAEhERERkZhgAEhEREZkZBoBEREREZoYBIBEREZGZYQBIREREZGYYABIRERGZGQaARERERGaGASARERGRmWEASERERGRmGAASERERmRkGgERERERmhgEgERERkZlhAEhERERkZhgAEhEREZkZBoBEREREZoYBIBEREZGZYQBIREREZGYYABIRERGZGQaARERERGaGASARERGRmWEASERERGRmGAASERERmRkGgERERERmhgEgERERkZlhAEhERERkZhgAEhEREZkZBoBEREREZoYBIBEREZGZYQBIREREZGasjJ0Bc1ZRUYH09HQ4ODhAJpMZOztE1MSEECgoKICnpycsLBr/+5t1CJF5MWQdwgDQiNLT0+Ht7W3sbBBRM7t69Sq8vLwafRzWIUTmyRB1CANAI3JwcAAg/SMdHR2NnBsiM7dyJbBiReXjBQuAV1816Evk5+fD29tbXfYbi3UINVgzfN7J8AxZh7SqAHDFihXYsmULzp07Bzs7OwwZMgRvvvkmunXrpk4jhEBMTAw+//xz5OTkYODAgfj444/Rs2dPdZqSkhLMnz8fGzduRFFREUaMGIFPPvlEK5rOycnB7NmzsW3bNgDAww8/jI8++gjt2rVTp0lLS8PMmTOxb98+2NnZYcKECXjnnXdgY2NTr/NRddk4Ojqy8iYytt9+q/64icqlobprWYdQgzXj550MzxB1SKuaBJKYmIiZM2ciJSUF8fHxUCqVCA0NxZ07d9Rp3nrrLbz33ntYvXo1jh49Cg8PD4SEhKCgoECdZs6cOfjxxx+xadMmHDp0CIWFhYiIiEB5ebk6zYQJE3DixAnExcUhLi4OJ06cQGRkpHp/eXk5xowZgzt37uDQoUPYtGkTNm/ejHnz5jXPm0FEhhUYCKgqVZlMekxkqvh5J9GKZWVlCQAiMTFRCCFERUWF8PDwECtXrlSnKS4uFgqFQqxZs0YIIURubq6wtrYWmzZtUqe5fv26sLCwEHFxcUIIIc6ePSsAiJSUFHWa5ORkAUCcO3dOCCHEzp07hYWFhbh+/bo6zcaNG4VcLhd5eXn1yn9eXp4AUO/0RNSEysqEiIkRIiREui8rM/hLGLrMsw6hBmuGzzsZniHLfKvqAq4qLy8PAODs7AwAuHTpEjIzMxEaGqpOI5fLERQUhKSkJDz33HNITU1FWVmZVhpPT0/4+/sjKSkJYWFhSE5OhkKhwMCBA9VpBg0aBIVCgaSkJHTr1g3Jycnw9/eHp6enOk1YWBhKSkqQmpqK4ODgavktKSlBSUmJ+nF+fr7h3gwiahwrK2DJEmPnolasQ8hgWsHnnZpWq+oC1iSEwNy5cxEYGAh/f38AQGZmJgDA3d1dK627u7t6X2ZmJmxsbODk5FRrGjc3t2qv6ebmppWm6us4OTnBxsZGnaaqFStWQKFQqG+cvUdE+mAdQkSG0moDwKioKJw6dQobN26stq/q4EghRJ0DJqum0ZW+IWk0LViwAHl5eerb1atXa80TEWlQKoFly4DQUOleqTR2jpod6xCqhuWCGqhVdgHPmjUL27Ztw4EDB7Rm7np4eACQWuc6dOig3p6VlaVurfPw8EBpaSlycnK0WgGzsrIwZMgQdZobN25Ue92bN29qHefw4cNa+3NyclBWVlatZVBFLpdDLpc35JSJKDYWWLoUEALYs0faZmZdWKxDqBqWC2qgVtUCKIRAVFQUtmzZgn379sHX11drv6+vLzw8PBAfH6/eVlpaisTERHVw17dvX1hbW2ulycjIwJkzZ9RpBg8ejLy8PBw5ckSd5vDhw8jLy9NKc+bMGWRkZKjT7N69G3K5HH379jX8yROZsvq0Yhw6JH3JAdL9oUPNm0ciY9NVTlguqIFaVQvgzJkz8e233+Knn36Cg4ODeqydQqGAnZ0dZDIZ5syZg9jYWHTt2hVdu3ZFbGws7O3tMWHCBHXaqVOnYt68eXBxcYGzszPmz5+PgIAAjBw5EgDQvXt3hIeHY9q0afjss88AAP/+978RERGhXnMwNDQUPXr0QGRkJN5++23cvn0b8+fPx7Rp07geF1FtlEqp1eLQIWnpiYUL69eKERgo7ROCy1aQ+dAsL0olkJAgbVeVE5YLaqhGzyNuRgB03tatW6dOU1FRIaKjo4WHh4eQy+Vi6NCh4vTp01rHKSoqElFRUcLZ2VnY2dmJiIgIkZaWppUmOztbTJw4UTg4OAgHBwcxceJEkZOTo5XmypUrYsyYMcLOzk44OzuLqKgoUVxcXO/z4RIOZJZiYoSQyYQApHvVUhTSV5h0Cwmp/jwTWLaCy8CQ3jTLS9VbSIhJlAuqP0OWeZkQqrZjam75+flQKBTIy8tjqyGZj9BQQGMIBkJCpFYLVQugTCb9bYLjmAxd5lmHmIGq5UXFhMsJ1cyQZb5VdQETUQukq0vXqpaqRVeX1cKF0j7NYxCZg7rKT9XyMmyYtJ/lhBqJASAR6U+pBJYvBzZsAHJypBtQv1mIuoI9LkpL5kipBEaOBBITpcfx8UB5ORATU5mmpvJC1Ej8FBGR/mJjpVmIVdVnFiKDPSJJbGxl8KeyYYN2AMjyQk1E72Vg0tPTcf78efXj8vJyvPXWW3jqqaewdu1ag2aOiFqomoI8zkIkqj8u2UJGpHcA+Nxzz+HDDz9UP16+fDleffVV7N69G9OmTcOGDRsMmkEiaoF0BXl+ftKgdI5LIqofXeUoMrL580FmSe8A8NixYwgODlY//s9//oMXX3wRt2/fxr///W98/PHHBs0gEbVACxdK3VJ+ftItOho4f17axvFJRPWjqxwtWmTsXJGZ0HsZGFtbW+zZsweBgYH4448/0LNnT5w8eRIBAQHYs2cPnnzySdy+fbup8mtSuIQDkXnhMjBE1BiGLPN6twAqFApkZWUBAA4cOABnZ2cEBAQAAGQyGUpLSxuVISIiIiJqWnr31QwYMABvvvkmrK2tsWrVKoSGhqr3Xbx4EZ6engbNIBEREREZlt4tgMuXL8fFixfxyCOP4MaNG3jttdfU+7Zu3YoBAwYYNINEREREZFh6twDef//9uHLlCs6dO4cuXbpo9UHPmDEDXbt2NWgGiQj6X22DiAyLZZBMTIM+vfb29njggQeqbR8zZkyjM0REOsTGVl4rtz5X2yAiw2IZJBOjdxcwANy8eRMLFizA4MGDce+99+L3338HAHz22Wc4fvy4QTNIRJBaHVQT9utztQ0iMiyWQTIxegeAly5dQq9evfDhhx9CJpPh77//RklJCQDg1KlTWotEE1ENlErpUmqhodK9Ull7+sBA6SobAK+2QdRUaiuXLINkYvTuAn755Zfh5OSE1NRUuLm5wcbGRr0vMDAQ0dHRBs0gkUnStztJ1wXhiciwaiuXLINkYvRuAdy7dy+io6Ph6ekJmerX0P/r0KED0tPTDZY5XQ4cOICxY8eqX3/r1q1a+4UQWLp0KTw9PWFnZ4dhw4apu6hVSkpKMGvWLLRv3x5t2rTBww8/jGvXrmmlycnJQWRkJBQKBRQKBSIjI5Gbm6uVJi0tDWPHjkWbNm3Qvn17zJ49m+sgUv3o252kuiD87t282gZRU6mtXLIMkonROwAsLi6Gs7Ozzn137tyBhUWDhhXW2507d9C7d2+sXr1a5/633noL7733HlavXo2jR4/Cw8MDISEhKCgoUKeZM2cOfvzxR2zatAmHDh1CYWEhIiIiUF5erk4zYcIEnDhxAnFxcYiLi8OJEycQqXGNxvLycowZMwZ37tzBoUOHsGnTJmzevBnz5s1rupMn08HuJKKWh+WSzInQ0/333y9efvllIYQQSqVSyGQykZqaKoQQ4uWXXxaDBw/W95ANBkD8+OOP6scVFRXCw8NDrFy5Ur2tuLhYKBQKsWbNGiGEELm5ucLa2lps2rRJneb69evCwsJCxMXFCSGEOHv2rAAgUlJS1GmSk5MFAHHu3DkhhBA7d+4UFhYW4vr16+o0GzduFHK5XOTl5dUr/3l5eQJAvdOTCSkrEyImRoiQEOm+rMzYOaJmYOgyzzrEwFguqYUzZJnXuw172rRpmDt3Ljw9PTFx4kQAQGlpKX744Qd88sknNbbMNYdLly4hMzNT6+okcrkcQUFBSEpKwnPPPYfU1FSUlZVppfH09IS/vz+SkpIQFhaG5ORkKBQKDBw4UJ1m0KBBUCgUSEpKQrdu3ZCcnAx/f3+tK5+EhYWhpKQEqampCA4Orpa/kpIS9YQZQLqmH7UihlwHTNWdRKQH1iENoE+5ZbkkM6L3t9eMGTNw4sQJvPjii+ruzsDAQAghMG3aNEyePNngmayvzMxMAIC7u7vWdnd3d1y5ckWdxsbGBk5OTtXSqJ6fmZkJNze3asd3c3PTSlP1dZycnGBjY6NOU9WKFSsQExPTgDMjo9H88lAqgYQEaTvXASMjYB2iB1XZXb8euHhR2sZyS6TWoOaLzz//HM888wx27NiBGzduoH379oiIiMCQIUMMnb8GqTo5RQhRbVtVVdPoSt+QNJoWLFiAuXPnqh/n5+fD29u71nyRkWnOCtTEdcDICFiH6EFX2WW5JVJr8DSmQYMGYdCgQYbMS6N5eHgAkFrnOnTooN6elZWlbq3z8PBAaWkpcnJytFoBs7Ky1AGsh4cHbty4Ue34N2/e1DrO4cOHtfbn5OSgrKysWsugilwuh1wub8QZUrPTnBWoiQPEyQhYh+hBV9lluSVS03vKbnp6Os6fP69+XF5ejrfeegtPPfUU1q5da9DM6cvX1xceHh6Ij49XbystLUViYqI6uOvbty+sra210mRkZODMmTPqNIMHD0ZeXh6OHDmiTnP48GHk5eVppTlz5gwyMjLUaXbv3g25XI6+ffs26XlSIzR2AebgYCAkRGpZ4DpgRMZV34WbAcDPj+WWSJO+s0YiIiLEjBkz1I+jo6OFTCYTTk5OwsLCQnz99deNnplSm4KCAnH8+HFx/PhxAUC899574vjx4+LKlStCCCFWrlwpFAqF2LJlizh9+rQYP3686NChg8jPz1cfY/r06cLLy0vs2bNHHDt2TAwfPlz07t1bKJVKdZrw8HDRq1cvkZycLJKTk0VAQICIiIhQ71cqlcLf31+MGDFCHDt2TOzZs0d4eXmJqKioep8LZ/A1oZpm88XECCGTCQFI9zExDTsOUQNwFnAjVS2P0dE1l2eWXTJBhizzegeAnp6e4n//+5/W47lz5wohpMBq0KBBjc5UbRISEgSAarfJkycLIaSlYKKjo4WHh4eQy+Vi6NCh4vTp01rHKCoqElFRUcLZ2VnY2dmJiIgIkZaWppUmOztbTJw4UTg4OAgHBwcxceJEkZOTo5XmypUrYsyYMcLOzk44OzuLqKgoUVxcXO9zMbvKuznVFOiFhEjbVLeQEOPmk8wKA8BGqlqu/fxYnsmsGHUZmOzsbPVYuz/++AMZGRmYMmUKAOCJJ57Ad99918g2ydoNGzYMQteYrP8nk8mwdOlSLF26tMY0tra2+Oijj/DRRx/VmMbZ2RkbNmyoNS+dOnXC9u3b68wzNZHalneoaUX/wEBpJqAQHA9E1NJVLeMHDmiXa0AqxyzPRHrTOwBUKBTIysoCIF2WzdnZGQEBAQCk4IuXQqNmU9t1O2sK9Hg9T6LWo2oZHzZMO+CbNAmwtGR5JmoAvQPAAQMG4M0334S1tTVWrVqltaDyxYsXtRZGJmqU4mJg9Gjg5EmgVy+pgj98uLKir+26nTUFelzolahl0izvjo6Ary9w5Yp2Gbe0lAJCQyzGTmTm9C45y5cvR0hICB555BE4OTnhtddeU+/bunUrBgwYYNAMkpnQ7OoZMkSq7D/6CMjJkfbv3y/dgMrWvtq6cxnoEbUOSiXw+uvAm29KQSAA3L4NXL6snU4mAx56iOWayED0DgDvv/9+XLlyBefOnUOXLl3g6Oio3jdjxgx07drVoBkkE6YK+g4eBP76q7LC11iiRydVa9/OndJjdv8QtS41lX1d/PyAe+5hGScysAa1ndvb2+OBBx6otn3MmDGNzhCZONW6XatXA3l5QEWF/sdQtfaxlY+o9SksBLy9gdzcutPKZMDkySznRE2gwYMn8vLy8Oeff6KoqKjavqFDhzYqU2QiVF07X30lVfZOTkCnTpVdufVlZwcMHFh9DCARtQ6aP/xyc3VfXUdTu3ZAnz7SpA+WdaImoXcAqFQqMX36dHz11VcoLy/Xmaam7WRmYmMBzQvX5+RUXpS9NsOGAUOHAsnJHOhNZApiY4Hly+tO5+QEzJoFLF7MMk/UxPQuYe+//z5+/vlnrF27Fk8//TQ+/vhjWFtb4z//+Q/y8vLw4YcfNkU+qSWraT0+fS667uMjjfMZOpQBH5Ep0KwX/v679rS+vsCUKSz7RM1I72sBf/3113jttdcwfvx4AMDAgQPx7LPP4vDhw+jcuTMSEhIMnklqQYqLgeHDARcX6b64uHKtrvh46T42Vkpb06KsQUHSL30rK6mrZ9Ei4MIFaUbvkiX8AiBqjarWDcuWVdYLNbX829pKQd+ff7LsEzUzvUvbxYsX0bt3b1hYSLFjsWraPoDp06fjhRdewIoVKwyXQzI+zV/yFy5UzthLSJDW7bKy0r0e38KF0iQPzTGAkZFSwMeKnsh0KJVA9+7adcOJE9pj/Xx9pXqguBgYMACIi5MCQCIyCr1bANu0aYPS0lLIZDI4OzvjypUr6n12dnbIzs42aAapGSmV0q9wZ2fA3l4ai1e1ha/qcg0nT0otfTKZ9FhzPT4rK+l5Fy9K63r9/bf0mMEfUetWta7o0qV63VBcrF0vTJki1QN370oTwRj8ERmV3t/E9913Hy5dugQAGDJkCN577z089NBDsLGxwVtvvYVu3boZPJPUhKquvq9ZiScmVm/hq6p3b15ejchcqOqLlBRAcwUIjYYAtQEDpK5g1gtELZLeAeC4cePw559/AgBiYmIwdOhQdO7cGQBgbW2NLVu2GDaHZDiFhUBAgFRZW1pKv9xLS7VX36/q5EnghRcqr7gBSBM28vOl4G/nTq7HR2TKlEqp5f7ddyvrirr4+LCLl6iF0zsAnDFjhvrvPn364OzZs9i6dStkMhlCQkLYAtjSFBcDo0ZJ6+eVlFQuvKxUSkFcXWpq4WM3LpHp02fRZh8foGtX1hFErUSjS6i3tzdmzZpliLxQYxUXA2FhwK+/Ag1di7FzZykwVA3UZgsfkXkpLgZGjpTqkbpYWQHW1pzUQdQK6R0ApqSkIC0tDU8++WS1fd9//z06d+6MgQMHGiRzpKfRo4EDBxr2XDs7YNAgKeBjJU5kvkaPrl/w164dkJHB+oKoldJ7FvDChQtx+vRpnfvOnj2LRYsWNTpT1EAnT9Y/rZWVNOnDz09q3cvPB/btY2VOZO7qqkcsLICHHmLwR9TK6R0Anjp1CoMGDdK5b+DAgTipTxBChtW7d+37bW2lNfjKyqRbXp60NEtMDMfrEJGkpnrEygooKJCGlxw4wOCPqJXT+1v/zp07sKohWLCwsEBBQUGjM2UuxP/Pqs2vz2SM+ti0CXjsMWmJBtVkDwCwsQFmzwYWLJAq8bt3DfN6RKQXVVkXNS2rpCeD1yGAVI88/LA0cUzF0xM4elSqVwz5WkSkF0PWIXoHgL6+vkhISEBYWFi1fQkJCeolYahuqmDZ29u7aV+otBR45x3pRkRGV1BQAIVCYZDjAM1Qh6SnAx07Nu1rEFG9GaIO0TsAfOqpp/DGG2+gW7du+Ne//qXe/uWXX+KDDz7AggULGpUhc+Lp6YmrV6/CwcEBMtWK+Trk5+fD29sbV69ehaOjYzPmsGmY0vnwXFqmlnouQggUFBTA09PTIMerbx1Sk5b6PjUVnq9pM4fzNWQdIhN6tiOWlpYiPDwc+/fvh52dHTw9PZGeno7i4mIMGzYMu3btgo2NTaMzRpXy8/OhUCiQl5dnEh9qUzofnkvLZErn0pTM7X3i+Zo2czvfxtK7BdDGxgbx8fH49ttvERcXh5s3b2LAgAEYNWoUxo8fD0tLy6bIJxEREREZSIOmflpaWiIyMhKRkZGGzg8RERERNTG9l4Gh5ieXyxEdHQ25XG7srBiEKZ0Pz6VlMqVzaUrm9j7xfE2buZ1vY+k9BpCIiIiIWje2ABIRERGZGQaARERERGaGASARERGRmWEASERERGRm6rUMzDPPPFPvA8pkMvz3v/9tcIaIiIiIqGnVKwDct2+f1mWGcnNzkZeXBysrK7i4uCA7OxtKpRIKhQJOTk5NllkiIiIiarx6dQFfvnwZly5dwqVLl/D999+jbdu2+Oabb1BUVISMjAwUFRVhw4YNaNOmDTZt2tTUeSYiIiKiRtB7HcChQ4fiiSeewAsvvFBt3/vvv48ffvgBv/76q8EySERERESGpfckkNTUVPj7++vcFxAQgBMnTjQ2T0RERETUhPQOAB0dHbFnzx6d+/bs2QNHR8dGZ4qIiIiImk69JoFoioyMxNtvvw2lUokJEybAw8MDmZmZ+Oabb/DBBx9g7ty5TZFPIiIiIjIQvccAKpVKPPvss/jqq6+0ZgYLITBp0iSsXbsWVlZ6x5VERERE1Ez0DgBVzp8/j3379uH27dtwcXHBsGHDcN999xk6f0RERERkYA0OAImIiIiodWrQpeBKSkrw2WefYfz48QgNDcWFCxcAAD/99BMuXrxo0AwSNQcfHx/Y29ujbdu2cHFxwciRI7Ft2zZjZ0tvWVlZePLJJ+Hu7g5nZ2eMHTsWaWlpxs4WkckzlTpE06ZNmyCTybi+r4nSOwC8desW+vXrh+effx6JiYnYu3cvCgoKAABbt27FO++8Y/BMEjWHffv2obCwEOfOncO4ceMQGRmJNWvWGOz4FRUVqKioMNjxdLlz5w4CAwPx+++/IzMzE126dMG//vWvJn1NIpKYQh2icufOHbz++uvo2bNns7weNT+9A8CXX34Zubm5+O2335CWlgbNHuTg4GAkJiYaNINEzc3V1RXTpk3D8uXLsWjRIpSXl+P06dMYOnQonJyc0LdvX/z222/q9MnJyfD394ejoyOmT5+OoKAg9S/mKVOmYPbs2Rg2bBjatm2LtLS0Wo+VlpaGMWPGwMXFBd27d0dcXJxeeff19cXs2bPRvn172NjYYMaMGThy5Ihh3hgiqpfWXIeoLF++HFOnTkX79u0b92ZQyyX05OrqKtauXSuEEEKpVAqZTCZSU1OFEELs3btXODo66ntIIqPr3LmzSE5O1tp28eJFAUCcOHFCdOzYUWzZskUolUrx448/Ci8vL1FUVCSKi4tFhw4dxBdffCFKS0vFxx9/LKysrMTGjRuFEEJMnjxZuLq6itTUVKFUKkV+fn6NxyovLxe9evUSq1evFmVlZSIpKUm4uLiIjIwMIYQQzz//vFAoFDpvzz//vM7z+vLLL8XAgQOb9s0jIpOqQ86fPy969eolSktLRVBQkDovZFr0DgDlcrnYu3evEKJ6ABgXFyfs7e0Nm0OiZqCr8i4qKhIAxOuvvy5GjRqlta9v375i3759IiEhQXTp0kVrn7e3t1blPWPGDPW+jRs31nis5ORk0b17d619TzzxhPoHl77S0tKEh4eHiI+Pb9Dziaj+TKkOCQ8PF7t27RJCCAaAJkzvBft8fX2RnJyM4cOHV9t35MgRdOvWrdGtkkQtQUZGBgCgvLwce/fuRbt27dT7ysrKkJGRAQsLC3h7e2s9r2PHjlqPvby81H+npaXVeCwrKytcuHBBa59SqUTfvn31zvvt27cRHh6OhQsXYuTIkXo/n4garzXWIT/99BOsrKwQHh5e7+dQ66R3ADhx4kS8+eab8Pf3x5gxYwAAMpkMR48exapVq/Daa68ZPJNExrB9+3a0b98e99xzD8aMGYMtW7ZUS7N//35cu3ZNa9v169e1HmsumN6xY8caj/Xrr78iICAAx44d05mf6dOnY8OGDTr3TZo0ST3YvLCwEKNHj8YjjzyCWbNm1X6SRNRkWmMdkpCQgAMHDsDDwwOA9GPyxIkT+PPPP7FkyZLaT5haF32bDEtLS0V4eLiQyWTC2dlZyGQy4erqKiwsLMTo0aNFeXl5U7RUEjUpze6bW7duiS+++EIoFArx6aefitzcXOHt7S1++uknoVQqxd27d8WuXbtEbm6uevzO2rVrRVlZmfj000+rjd9ZsWKF+nVqO1ZZWZnw9/cXn332mSgpKRElJSXiwIED4sqVK/U+j5KSEhESEiKeeeYZw75BRFQrU6lD8vPzRUZGhvo2ePBgsWbNGpGfn2/YN4yMTu9ZwNbW1ti5cye+/fZbjB49GiNHjsTIkSPx9ddf4+eff4aFRYOWFiQyuuHDh6Nt27bo2rUrNm7ciPXr12P69OlQKBTYvn07Vq1aBVdXV/j4+ODzzz8HAMjlcmzevBnvvvsunJ2dceLECfTv3x9yuVzna9R2LCsrK2zfvh07d+5Ex44d4enpiTfeeEOvZR+Sk5MRHx+PTZs2oW3btuob1wIkanqmUIc4ODjAw8NDfbOxsYFCoYCDg0Pj3yBqUXglECIDEkLAy8sLW7duRf/+/Y2dHSJqZViHUHPRu7nO0tKyxnXFUlNTYWlp2ehMEbUmCQkJuHHjBkpLS/Hmm2/C2toaffr0MXa2iKiVYB1CxqD3JJDaGgwrKiq0BqsSmYPff/8d48ePx507d9CzZ09s2bIFVlZ6Fy0iMlOsQ8gY9O4CtrCwwOHDh3U2Ta9ZswavvfYasrOzDZZBIiIiIjKsev3EWLVqFVatWgVAmo7+6KOPVhugWlRUhKysLPzjH/8wfC6JiIiIyGDqFQC6ubmpLwh9+fJl+Pn5aS00CUgzmQICAvDCCy8YPJNEREREZDh6dwEHBwfj008/xX333ddUeSIiIiKiJsRlYIyooqIC6enpcHBw4OQZIjMghEBBQQE8PT0NsmYq6xAi82LIOkTvaUbr1q3DlStXsHTp0mr7li5dCj8/Pzz99NONypS5SE9Pr3YNSCIyfVevXtW6vmtDsQ4hMk+GqEP0DgA//PBDTJkyRee+9u3b48MPP2QAWE+qldWvXr0KR0dHI+eGiPSlVALvvAOkpACDBgHz5wO1rd6Rn58Pb29vg11VgXUINRV9P9vUPAxZh+j97/zrr7/g7++vc1+PHj1w4cKFRmfKXKi6bBwdHVl5E7VCy5YBK1cCQgD79wO2tsCSJXU/z1DdtaxDqKk09LNNzcMQdUiDOpDz8vJq3K5UKhuVISIiY1MqpS/AkSOB4cOBkBDpcdXq7dAh6QsSkO4PHWr+vBI1hlIJREcD99wj3ZYulbbxs2369G4BDAgIwKZNm/D4449X27dx40YEBAQYJGNERM1FqQRiY6UvucBAoKJCCvg0p8jt3Svda7aCBAYCe/ZI6WQy6TFRS6f5eVcqgYSEyn0xMYCFBT/bZkHo6ZtvvhEymUw8/fTTIiUlRVy7dk2kpKSIyZMnCwsLC7FhwwZ9D1lvsbGxol+/fqJt27bC1dVVPPLII+LcuXNaaSoqKkR0dLTo0KGDsLW1FUFBQeLMmTNaaYqLi0VUVJRwcXER9vb2YuzYseLq1ataaW7fvi0mTZokHB0dhaOjo5g0aZLIycnRSnPlyhUREREh7O3thYuLi5g1a5YoKSmp9/nk5eUJACIvL0+/N4KIGqyoSIjgYCGcnaX7oiIhYmKEkMmEAKR7Pz/p76q3kBDtY5WVSc8NCZHuy8pqf21Dl3nWIVRfZWVCREdLn20nJ92fb83Pub6fbWoehizzegeAQgixePFiYWVlJSwsLNQ3KysrsWTJkkZnqDZhYWFi3bp14syZM+LEiRNizJgxolOnTqKwsFCdZuXKlcLBwUFs3rxZnD59WowbN0506NBB5Ofnq9NMnz5ddOzYUcTHx4tjx46J4OBg0bt3b6FUKtVpwsPDhb+/v0hKShJJSUnC399fREREqPcrlUrh7+8vgoODxbFjx0R8fLzw9PQUUVFR9T4fVt5EzS84WPvLLjhY+pLT3ObnVxkQqm4ymfRF2BgMAMlYYmJqD/o0b439nFPTMXoAKIQQly5dEp9//rl44403xOeffy4uX77c6MzoKysrSwAQiYmJQgip9c/Dw0OsXLlSnaa4uFgoFAqxZs0aIYQQubm5wtraWmzatEmd5vr168LCwkLExcUJIYQ4e/asACBSUlLUaZKTkwUAdYvjzp07hYWFhbh+/bo6zcaNG4VcLq/3P4aVN1Hzc3bW/rJzdq7eArhkibRtxAgpQBw50jCtIAwAyViq/sjRvAUFST96/PykVkK29rVchizzDZ7U7ePjg2nTpjWq+7mxVJNRnJ2dAQCXLl1CZmYmQkND1WnkcjmCgoKQlJSE5557DqmpqSgrK9NK4+npCX9/fyQlJSEsLAzJyclQKBQYOHCgOs2gQYOgUCiQlJSEbt26ITk5Gf7+/vD09FSnCQsLQ0lJCVJTUxEcHFwtvyUlJSgpKVE/zs/PN9ybQWSmqo7fW7iw9uUqevfWHvPUu7f0HKD+xzAW1iFUl5rKQ2AgEB+vndbPD5g8ueV+3qlptdp/uRACc+fORWBgoHpZmszMTACAu7u7Vlp3d3dcuXJFncbGxgZOTk7V0qien5mZCTc3t2qv6ebmppWm6us4OTnBxsZGnaaqFStWICYmRt9TJaIaKJVAaGhlQKf6gqttuYqdO4HRo4GTJ6Xgb+dO6cuvNSxxwTqEalNbeVi4UJrc9PXX0rZJk4DFixn4mbN6LQNjaWmJI0eOSE+wsIClpWWNN6tm+jRFRUXh1KlT2LhxY7V9VdfHEULUuWZO1TS60jckjaYFCxYgLy9Pfbt69WqteSKi2sXGarfmAXUvV2FrC+zbB2RnS/e2tk2XP0NjHUK1qa08WFlJS7z8/bd0i4lh8Gfu6vXvX7JkifqSI0uWLDH6NSdnzZqFbdu24cCBA1qXQvHw8AAgtc516NBBvT0rK0vdWufh4YHS0lLk5ORotQJmZWVhyJAh6jQ3btyo9ro3b97UOs7hw4e19ufk5KCsrKxay6CKXC6HXC5vyCkTmQ2lEnj9damlIicHaNcOiIzU3VqhK9gz5eUqWIeQao3K1auB4mJgwAAgLk76IWNu5YEaqdGjCJtRRUWFmDlzpvD09BR//vmnzv0eHh7izTffVG8rKSnROQnku+++U6dJT0/XOQnk8OHD6jQpKSk6J4Gkp6er02zatImTQIj0oGupiZpmK+qamag5eUM1o7clD2DnJBBqCM1yUnUWu+pzL0TrKw+kvxYxCcQYZs6ciW+//RY//fQTHBwc1GPtFAoF7OzsIJPJMGfOHMTGxqJr167o2rUrYmNjYW9vjwkTJqjTTp06FfPmzYOLiwucnZ0xf/58BAQEYOTIkQCA7t27Izw8HNOmTcNnn30GAPj3v/+NiIgIdOvWDQAQGhqKHj16IDIyEm+//TZu376N+fPnY9q0abwkE1Etqi5Cu3+/9HW1Z4+0v6YuXF3bW8vkDaKGqjquT5eTJ6V7lgfSR70+Gl999ZVeB3366acblJm6fPrppwCAYcOGaW1ft24dpkyZAgB4+eWXUVRUhBkzZiAnJwcDBw7E7t27tS6c/P7778PKygpPPvkkioqKMGLECHz55ZewtLRUp/nmm28we/Zs9Wzhhx9+GKtXr1bvt7S0xI4dOzBjxgw8+OCDsLOzw4QJE/DOO+80ybkTtXaqwG/9euDixer7VZeb0jVbEdDdldVaJm8Q6auu8qKpd2/pnuWB9CETQvNiR7pZWGjPFVGNAdR8qua4wPLyckPlz6Tl5+dDoVAgLy+PrYZkMpRKYPlyaQxfbi7g5CSN4QOqX15Nk0wmDVJfuLD+YwBbG0OXedYhpkWplCZnfPwxUFBQ/drTKkFBwKlT1ccAkukzZJmvV3V66dIl9d+ZmZkYN24cwsLCMGHCBHh4eCAzMxPffPMNdu/eje+++65RGSKi1kPXmmOxsVKgp5KTI32p+fnpDv6CgyvXKVN1WS1dKt2ITF1t1+XVJTgY2L279f8YIuOr10eoc+fO6r9fffVVPPbYY3j//ffV27p164agoCC8+OKLeO+99xgEEpmJ2FgpUKvPGD5AauVTBYFchJZIuwzVhuWFDE3vj9GuXbvwww8/6Nw3evRo/POf/2x0poiodTh0qPKLq64xfJMmAZaWHKBOpEmzDOni5ATMmcPyQoan98epoqICFy5cUM+Y1XThwgXUY0ghEbVQ+l5WLTBQavkTQmrdUz2nvLz6GMBFi/gFRuaptnKlWYYAYOhQ4PRpaXzfwIHArl0c30dNQ+/qODw8HK+99ho6deqEMWPGqLdv374dixYtQlhYmEEzSETNR1eXbm2zCmtadiImRroRUe3liku3kLHo/TFbtWoVRowYgYcffhgODg5wd3fHjRs3UFBQgK5du2LVqlVNkU8iaga6unRrw2UniOpWW7liGSJjqde1gDV16NABx44dw8cff4zw8HB06tQJ4eHh+OSTT3D8+HGtS7ARkXGoLhcVGird17ScRFWBgVJXLlDZpUtEtaurvLFcUUvUoIZmW1tbTJ8+HdOnTzd0fojIAPTtylXR1R1FRLWrq7yxXFFL1OCRBufOnUNiYiJu3bqFqVOnwsPDA+np6XBycoKdnZ0h80hEetK3K1eF3VFE+qurvLFcUUukdxdweXk5pk6dip49e+L555/HkiVLkJ6eDgB47rnnsGLFCoNnkogk9e3aZZcTkeHVVP5Y3qg10rsF8I033sC3336Lt99+G+Hh4fD391fvGzVqFL788kss07wMABHpTXU5tQ0bpMeqZVTq27XLLieixqtaDjt1Avbvl/7WLH8sb9Qa6R0Afvnll1i8eDHmzp1b7Zq/vr6+WpeNIyL9KZVSC4PmJaFiYgALi/p37bLLiajxli/XvqzhxYuVf2uWP5Y3ao30DgCvX7+OwYMH69xna2uLgoKCRmeKyFzoaukDdF8PVNW6UHXhZSIynOJiYPRo4ORJ4O7dmtOx/FFrp3cA6ObmhosXLyI4OLjavvPnz8PLy8sgGSMyRVWvCFBert3CEBMjXfNTF82uJXY1ERmWqmx+8AGQk1NzuuBgqcWP5Y9aO70DwNGjR+ONN95AeHg4PDw8AAAymQx5eXn48MMPMXbsWINnkshUVB3D5+urO51MVtnV6+QEzJ5deYUAdjURGZ5m2azJsGHA7t28UgeZBr0/xsuWLcOuXbvQo0cPBAcHQyaTYeHChThz5gysra2xePHipsgnkUmoOoZPl0mTAEtLXhqKqDlplk1NPj5A164si2R69P4ou7u74+jRo4iOjsaOHTtgaWmJkydPIiIiAsuWLYOzs3NT5JOoRartIu+6VB3DFxkp/V11ti+/ZIgMq66yqlk2AcDODhg0CNi5E7C1NU6eiZqSXl8zxcXFWLZsGZ544gmsWbOmqfJE1KLU9sWh7xU3arrwe0xMk54CkdmpWm4rKqTxtvpcrYM/xMiU6fXxtrW1xfvvv4/w8PCmyg9Ri1NbkKfvFTc4ho+oeegab8urdRBV0vtKIN27d+daf2RWagvyeAUAopZJ13hbllWiSnoHgIsXL8brr7+Ov//+uynyQ9QslEqpdeCee6RbdHTDLqu2cKF0nJAQ6Z7LQhA1r+JiYPhwwMVFui8ulrZXLbeTJrGsEmnSe4TDunXrcPfuXXTv3h29evVChw4dIFOVMkhLwvz0008GzSRRQymVwOuvA19/LT2eNAlYvFjqHtIcd7dsmTTzVt/LqrHbiKh5VS3T5eXAlSvS3wkJ0iLO+/ZxTB9RXfQuDqdOnYKNjQ06duyI7OxsZGdna+3XDAaJmpvqy+Grr4DcXGmb5qKuqkBP11g9XlaNqOVSKqXy+/bbla18upw8Kd2z3BLVTu8A8PLly02QDaKGUV226cQJQKGQgr28vNqfo2oRiI/X3s4xQUQti2ZrX3p67YGfSu/eTZ8vIlPABnFqVQoLAX9/qctH82oZQO2Xb9Kk6g6qqNDuGuaYIKKWobAQ6NkTSEurX3ofHyA/Xwr+du5s0qwRmYwGBYDl5eX4/vvvkZCQgOzsbLi4uCA4OBj//Oc/YcVBFmRgqq6f1aulbt26rqShi5OTdFMFelZW0kDwpUubIMNE1CCqsv7GG9IPtLrY2gIvvyyN6+VXD5F+9C4yt27dQnh4OI4dOwYrKyu4uLggOzsbX3zxBd555x388ssvaN++fVPklUycqnUvLU0ap2dvD1j8/zx11Xg+fdjaAp6evLoGUUuWmyu14NU1dKOqzp2Bc+d4lQ6ihtJ7GZgXX3wR58+fxzfffIOioiJkZGSgqKgIGzZswIULF/Diiy82RT7JRBUXA0FBgLU14OAgde0KIbUE5OdLXw71Df7atZMWe/XzkwZ/FxQAf/8ttfIx+CNqWVRl38mp/sGfpaVUxpcsAf76i8EfUWPo/bX4888/4/XXX8f48ePV2ywtLTFhwgRkZWVhKfvUqAaal2YaMEC6/q1q+YaGkMmkLwQHB2DmTGktPwZ6RC2Xqg7Yvx84eLDmtTersrICHnwQiItj0EdkKHp/XQoh0LNnT537/P39IfQZmEUmrbgYCA8HDh8Gysqk9bpUqs7ArQ9bWynoGzCAXwRErUlxMRAWBhw4UP/nWFgADz3Esk7UVPQOAEeOHIk9e/Zg5MiR1fbFx8dj2LBhhsgXtUJVL76+bx+QmNiwY1lZVY4BdHbmOD6i1khVJ3zwQf1n6dvaAoMGAbt2MfAjakp6f50uXrwYjz/+OMrLyzFhwgR4eHggMzMT33zzDbZs2YItW7bg9u3b6vTOzs4GzTC1HEolsHy51JULAJ06SQGf6uLr+lbe7OYhav0064WcnPoHfkFBLPtEzUkm9OyztbConDeiedUP1WGqXgmkXLPfj7Tk5+dDoVAgLy8Pjo6Oxs5OrVSV+vr1leP2qq7DV5WTU82Vf6dO0vOvXwe8vIDTp4G2bQ2fb6KWxNBlvqXUIUqlNAb33XeBkhL9nuvjw/JPVF+GLPN6twAuWbKEl3szA1WvsCGTAZcuaaepLfiTyYCoKGnMj2oMYEWFNGljyBDgl1/4S5+otVON8/311/pP6FB56CFg927WA0TGoncAyFm+pikzE/D2lipxmQywsan8JV/fLhwACA6WunJ58XUi06RUSuNx33qr/oux+/kBkyezTiBqSVgUzZCqu+a996Rf8BYW2qvuC6FfN47qKhucqEFkuhqyYLOVFfDaa6wXiFoiFkkzonkdXU31ueSSikJR+QVgaSldhmnZMlbuRKZKqZQWXl6xQr/ndeoE/P47x/YRtVT82jZBt25JEyv0HYxdk3btgAcekGbpsQuHyPRlZkp1iD5z+KysgAULpGCRdQRRy8diakJUXbuxsY07jkwmjQe0sgImTeKF1onMSW4u0KGDfs8JDJQWd+eEDqLWQ+9rAVN1n3zyCXx9fWFra4u+ffvi4MGDRslHbGzjgr927aQJHxUVUjfx338DMTEM/ojMia9v/dN27ixdc/vgQQZ/RK0NA8BG+u677zBnzhy89tprOH78OB566CGMGjUKaWlpzZ6XQ4fqn9bCQqqwhw0DioqkiR85OVIQSETmKze39v2qrt6yMuDyZY7xI2qtGAA20nvvvYepU6fi2WefRffu3fHBBx/A29sbn376abW0JSUlyM/P17oZUmBg7fstLSsr7vJyKfBLSOAvd6LWoqnrEKDmH4FyOXDzplR/xMayZ4CotWMRboTS0lKkpqbi1Vdf1doeGhqKpKSkaulXrFiBmJiYatsNVYlHRQH5+cBHHwGlpVKFPWuWFPRpVtZ37xrk5YhIT6qyrucFmNSaug4BgJMngYAAqS6xtAReeEFaykVVhzRBzElE9dTYOkST3peCo0rp6eno2LEjfv31VwwZMkS9PTY2FuvXr8f58+e10peUlKBEY2ru9evX0aNHj2bLLxG1DFevXoWXl5fez2MdQkRAw+sQTWwBNICql8YTQui8XJ5cLodcLlc/btu2La5evQoHB4daL6+Xn58Pb29vXL16tcVfM7g+TOl8eC4tU0s9FyEECgoK4Onp2aDnN7QOqUlLfZ+aCs/XtJnD+Ta2DtHEALAR2rdvD0tLS2RmZmptz8rKgru7e53Pt7Cw0CuCd3R0NKkPtSmdD8+lZWqJ56JQKAx2LH3rkJq0xPepKfF8TZupn6+h6hBOAmkEGxsb9O3bF/Hx8Vrb4+PjtbqEiYiIiFoStgA20ty5cxEZGYl+/fph8ODB+Pzzz5GWlobp06cbO2tEREREOjEAbKRx48YhOzsby5YtQ0ZGBvz9/bFz50507tzZYK8hl8sRHR2tNfanNTOl8+G5tEymdC5NydzeJ56vaTO3820szgImIiIiMjMcA0hERERkZhgAEhEREZkZBoBEREREZoYBIBEREZGZYQBIREREZGYYABIRERGZGQaARERERGaGASARERGRmWEASERERGRmGAASERERmRkGgERERERmhgEgERERkZlhAEhERERkZhgAEhEREZkZBoBEREREZoYBIBEREZGZYQBIREREZGYYABIRERGZGQaARERERGaGASARERGRmWEASERERGRmGAASERERmRkGgERERERmhgEgERERkZlhAEhERERkZhgAEhEREZkZBoBEREREZoYBIBEREZGZYQBIREREZGYYABIRERGZGStjZ8CcVVRUID09HQ4ODpDJZMbODhE1MSEECgoK4OnpCQuLxv/+Zh1CZF4MWYcwADSi9PR0eHt7GzsbRNTMrl69Ci8vr0Yfh3UIkXkyRB3CANCIHBwcAEj/SEdHRyPnhoiaWn5+Pry9vdVlv7FYhxC1citXAitWVD5esAB49dUakxuyDmEAaESqLhtHR0dW3kRmxFDdtaxDiFq5336r/rgeZdkQdQgngRAREREZQ2AgoArmZDLpcTNhCyARERGRMSxcKN0fOiQFf6rHzYABIBEREZExWFkBS5YY5aXZBUxERERkZhgAEhEREZkZBoBEREREZoYBIBEREZGZYQBIREREZGYYABIRERGZGQaARERE1PoolcCyZUBoqHSvVBo7R60K1wEkIiKi1ic2Fli6FBAC2LNH2makNfVaI7YAEhERUetz6JAU/AHS/aFDxs1PK8MAkIiIiFofI15H1xS0qgBwxYoV6N+/PxwcHODm5oZHH30U58+f10ozZcoUyGQyrdugQYO00pSUlGDWrFlo37492rRpg4cffhjXrl3TSpOTk4PIyEgoFAooFApERkYiNzdXK01aWhrGjh2LNm3aoH379pg9ezZKS0ub5NyJiIhIw8KFUhdwSIh034zX0TUFrSoATExMxMyZM5GSkoL4+HgolUqEhobizp07WunCw8ORkZGhvu3cuVNr/5w5c/Djjz9i06ZNOHToEAoLCxEREYHy8nJ1mgkTJuDEiROIi4tDXFwcTpw4gcjISPX+8vJyjBkzBnfu3MGhQ4ewadMmbN68GfPmzWvaN4GIiIgqr6O7e7d0b8VpDXoRrVhWVpYAIBITE9XbJk+eLB555JEan5Obmyusra3Fpk2b1NuuX78uLCwsRFxcnBBCiLNnzwoAIiUlRZ0mOTlZABDnzp0TQgixc+dOYWFhIa5fv65O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",
      "text/plain": [
       "<Figure size 640x480 with 8 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "X_train, X_test, y_train, y_test = split_sc(train, test, feature_cols, label_cols, None)\n",
    "plotPredAct(X_train, X_test, y_train, y_test, poly_reg, 'Polynomial Regression (raw)', 'Degree=', range(1, 5), 2, 2, True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d56402e9",
   "metadata": {},
   "source": [
    "## Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "3fde99da",
   "metadata": {},
   "outputs": [],
   "source": [
    "def log_reg(X_train, X_test, y_train, y_test):\n",
    "\n",
    "    log = LogisticRegression()\n",
    "    log_model = log.fit(X_train, y_train)\n",
    "\n",
    "    y_train_pred = log_model.predict(X_train)\n",
    "    y_test_pred = log_model.predict(X_test)\n",
    "\n",
    "    return (y_train_pred, y_test_pred)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "73fbcdf8",
   "metadata": {},
   "source": [
    "Generate table of RMSE with Polynomial model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "8e8c5ef7",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style type=\"text/css\">\n",
       "</style>\n",
       "<table id=\"T_49d55\">\n",
       "  <caption>RMSE | Logistic Regression</caption>\n",
       "  <thead>\n",
       "    <tr>\n",
       "      <th class=\"blank level0\" >&nbsp;</th>\n",
       "      <th id=\"T_49d55_level0_col0\" class=\"col_heading level0 col0\" >raw</th>\n",
       "      <th id=\"T_49d55_level0_col1\" class=\"col_heading level0 col1\" >log</th>\n",
       "      <th id=\"T_49d55_level0_col2\" class=\"col_heading level0 col2\" >mm_sc</th>\n",
       "      <th id=\"T_49d55_level0_col3\" class=\"col_heading level0 col3\" >mm_sc+log</th>\n",
       "      <th id=\"T_49d55_level0_col4\" class=\"col_heading level0 col4\" >ss_sc</th>\n",
       "      <th id=\"T_49d55_level0_col5\" class=\"col_heading level0 col5\" >ss_sc+log</th>\n",
       "      <th id=\"T_49d55_level0_col6\" class=\"col_heading level0 col6\" >ma_sc</th>\n",
       "      <th id=\"T_49d55_level0_col7\" class=\"col_heading level0 col7\" >ma_sc+log</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th id=\"T_49d55_level0_row0\" class=\"row_heading level0 row0\" >train</th>\n",
       "      <td id=\"T_49d55_row0_col0\" class=\"data row0 col0\" >3.31e+03</td>\n",
       "      <td id=\"T_49d55_row0_col1\" class=\"data row0 col1\" >1.91e+03</td>\n",
       "      <td id=\"T_49d55_row0_col2\" class=\"data row0 col2\" >2.45e+03</td>\n",
       "      <td id=\"T_49d55_row0_col3\" class=\"data row0 col3\" >2.20e+03</td>\n",
       "      <td id=\"T_49d55_row0_col4\" class=\"data row0 col4\" >2.00e+02</td>\n",
       "      <td id=\"T_49d55_row0_col5\" class=\"data row0 col5\" >9.99e+00</td>\n",
       "      <td id=\"T_49d55_row0_col6\" class=\"data row0 col6\" >2.47e+03</td>\n",
       "      <td id=\"T_49d55_row0_col7\" class=\"data row0 col7\" >2.91e+03</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th id=\"T_49d55_level0_row1\" class=\"row_heading level0 row1\" >test</th>\n",
       "      <td id=\"T_49d55_row1_col0\" class=\"data row1 col0\" >3.71e+03</td>\n",
       "      <td id=\"T_49d55_row1_col1\" class=\"data row1 col1\" >5.56e+03</td>\n",
       "      <td id=\"T_49d55_row1_col2\" class=\"data row1 col2\" >2.64e+03</td>\n",
       "      <td id=\"T_49d55_row1_col3\" class=\"data row1 col3\" >1.83e+03</td>\n",
       "      <td id=\"T_49d55_row1_col4\" class=\"data row1 col4\" >1.39e+03</td>\n",
       "      <td id=\"T_49d55_row1_col5\" class=\"data row1 col5\" >2.11e+03</td>\n",
       "      <td id=\"T_49d55_row1_col6\" class=\"data row1 col6\" >2.64e+03</td>\n",
       "      <td id=\"T_49d55_row1_col7\" class=\"data row1 col7\" >3.30e+03</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n"
      ],
      "text/plain": [
       "<pandas.io.formats.style.Styler at 0x2580671ca10>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "display(errorMetrics(log_reg, 'RMSE | Logistic Regression'))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4d9c5d92",
   "metadata": {},
   "source": [
    "Generate plots of predicted vs actual cases with Logistic models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "c67b1905",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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NnkkPJG5Yx9AolZUUVGBxYsXIyMjAzKZDL1798aXX36JJ5980thFM7rCwkLMnz8f2dnZsLa2xv33349du3Yx+avHjBkzkJ+fjwsXLrSJt2BUVFQ0+Eo2mUwGS0tLg5xLo9Gge/fuWLBggUGOR2SOWANIRETN1q1btwanJBo6dCgOHDjQegUiogaxBpCIiJptx44dKC0trXe/vb19K5aGiG6HNYBEREREZoYTQRMRERGZGSaARERERGaGCSARERGRmWECSERERGRmmAASERERmRkmgERERERmhgkgERERkZlhAkhERERkZpgAEhEREZkZJoBEREREZoYJIBEREZGZYQJIREREZGaYABIRERGZGSaARERERGaGCSARERGRmWECSERERGRmmAASERERmRkrYxfAnFVWVuL69euwt7eHTCYzdnGIiIioDRNCoLCwEO7u7rCwaF4dHhNAI7p+/To8PDyMXQwiIiIyIVeuXEGXLl2adQwmgEZkb28PQPsgO3XqZOTSEBERUVtWUFAADw8PKX9oDiaARlTV7NupUycmgERERNQohug2xkEgRERERGaGCSARERGRmWECSERERGRmmAASEREZiUYDLFsGBAdrPzWaurc1FN/U8xiqzCUlhju2KTPkPW4NHARCRERkJDExQHQ0IASwb9/f22tuW7y4/viqffqepzHfa8yxDhzQLoY4tikz5D1uDUwAiYiIjCQxUZswANrPxMS/f665raH4pp7HEGU+ccJwxzZlhrzHrYFNwEREREYSEABUzeghk2nX69rWUHxTz2OoMvfta7hjmzJD3uPWwBpAIiIiI1m4UPuZmKhNGKrW69vWUHxTz9PcMr/0ErBqlWGObcoMeY9bg0yIqgpLam0FBQVQKpVQq9WcCJqIiIgaZMi8gU3AREREZNZMbQSvIbAJmIiIqAEajXaEZ/WmPSv+9TR51Z+rRmN+I5n5nzAREVEDTG16D2qc6s+1OlMYwWsIbAImIiJqgKlN70GNU/25VmcKI3gNgQkgERFRA0xteg9qnOrPFQACA4GgIG2tYFsfwWsIbAImIiJqgKlN70GNU9dzNae+nZwGxog4DQwRERE1FqeBISIiIrOh0WibZrt31y5LlpjHVC0tyYwqO4mIiMgUxcQAS5f+vb5sGWBpydHYzcEaQCIiImrT6hp5zdHYzcMEkIiIiNq0ukZe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9khzAALAu+++CysrK0ycOBHFxcUYMWIENm7cqDOf4JYtWzB79mxptPDYsWOxbt261r1YIiIioiaQCdGYt5L+zdbWFj/88AOCgoJQUVEBa2tr/Prrr7j//vuxc+dOTJo0idObNFJBQQGUSiXUajX7A+pBowFefx3YvFm7HhEBvPqq9ueqaVr8/bX99A4davyULdWnean+/aYci4iIyNAMmTfo/WesZ8+e2LdvH4KCgmrtO3jwIHx8fJpVIKLbiYkBli37e33pUu3r1YC/p2mp1oWz0VO2VJ/mpfr3m3IsIiKitkzvBHD69OmYO3cu3N3dMWXKFADagRTfffcdPvzwQzaDUoura/qVqm111Wc3dsqW6tO81IfTvxARUXug9yjgmTNn4qmnnsKLL74oTasSEBCAxx9/HFOmTMHUqVMNXkii6uqafiUgQHealuoaO2VLfd9vyrGaS6PR1nIGB2s/NZqWPycREZmPJvVk+ve//41nnnkGO3fuRGZmJpydnREWFsYpUKhVLFwIVFTo9gGsPiVLfX0AG3Pcur7flGM1F986QkRELUnvQSBkOBwEQvUJDtbtexgUBOzZY7zyEBGR8Rkyb9C7Cfj69es4d+6ctF5RUYFVq1bhiSeewGeffdaswhCRFt86QkRELUnvJuDnnnsOXbt2xQcffAAAeP3117Fs2TLccccd+PbbbyGXy/Hkk08avKBE5qR6c3RrNTsTEZH50LsG8NixYwgMDJTWP/74Y7z44ovIzc3FjBkzpMSQiJqu6q0je/ZoPznvIBERGZLeCWBOTo40+vfMmTNIT09HZGQkAGDChAk6zcNERERE1PbonQAqlUpkZWUB0E787OjoCF9fXwCATCZDWVmZYUtIJotTmRAREbVNejcsDRw4EG+++Sasra2xdu1a6V24AHDhwgW4u7sbtIBkujiVCRERUdukdw3g66+/jgsXLuDhhx9GZmYmFi1aJO3bvn07Bg4caNACkumq/mYNvkGDiIio7dC7BrBfv364dOkSzp49i7vvvltnHpqZM2fC29vboAUk0xUQoK35E4JTmRAREbUlnAjaiNr7RNAajbYZuPpUJhzNSkRE1DSGzBua/OdYrVbjjz/+QHFxca19Q4YMaVahqH2omsqEiIiI2ha9E0CNRoPnn38eX3zxBSoqKuqMqW87ERERERmf3oNA3n33XezYsQOfffYZhBBYt24dNmzYgAEDBsDb2xs///xzS5STiIiIiAxE7wTwyy+/xKJFizBp0iQAgJ+fH5599lkcPnwYnp6eiI+PN3ghiYiIiMhw9E4AL1y4gL59+8LCQvvVkpISad/zzz+PLVu2GK50RERERGRwevcBtLOzQ1lZGWQyGRwdHXHp0iUMHjwYAGBjY4OcnByDF5L0p9EAS5cC69YBhYWAtTXQuTNw9ap2WhZLS+CBB4Dk5L+/o1AAgwYBw4cDL72kHcG7bh1QUqKNfegh4KuvgPR07TYA8PQEUlOBDh20kz6vXg2UlQEdOwKVlcCtW4CFBfDgg8BPPwHvvAP88gtQXg5cvqydHqZrV215hgzRnnfVKu3I4cGDtWX93/+0x5LJtJ+XLml/fvJJ4LXX9BtZ3JSRydW/4++vPfehQ7W/r9EAy5cDX36pXZ80SfudkyeBvn2BXbu096k56io/oHveyZO1Zaz6f7EpU7TrSUm6Za461i+/aH+ufl8XLPj7OTR1BDdHgRMRtWFCT0OGDBEbNmwQQggRHh4uBgwYIK5cuSIyMzNFYGCguO+++/Q9pNlSq9UCgFCr1QY/9tKlQmjTJ/0XmUyIwMDGx3fr1rjzdeumPfbtzttQTM1l6VL970vV8WWyxn2/+ndqlrf69293DwID9StrY8uvz7OuXub6rquqrPrep8aUlYiIms6QeYPe/z/++OOP448//gAALF26FEOGDIGnpycAwNraGlu3bjVkfkpN1Jy3bggBnDjR+PirVxt3vqrax9udV5+ZKfW9zqa8naT6d6qr+f3bHUufe9qYsjTl7SrVv1PfdQG6z6Gpb3Hhm2CIiNouvfsAzpw5E2+//TYA4L777sPp06exZs0arF27FidOnMDo0aMNXkjSX3PeuiGTaZssG6tLl8adr0sX7bFvd96GYmrS9zoDAv4+fmPfTlL9O9XV/P7tjqXPPW1MWarOr889qF7m+q4L0H0OTX2LS1PuNRERtY5m98jx8PDArFmzDFEWMqCFC7X97FqjD+DJk9q+bWVlrd8HsKoPnD73Bajdh66x36mrD2D1uMrKhvsANld95a9+3sb0Aax+rMb2ATRUWYmIyPj0fhVccnIyLl++jIkTJ9ba95///Aeenp7w8/MzWAHbs/b+KjgiIiIyHEPmDXo3AS9cuBAnT56sc9/p06fx6quvNqtARERERNSy9E4Af//9dwwaNKjOfX5+fjhhiJ7uRERERNRi9E4Ai4qKYFXPZF4WFhYoLCxsdqGIiIiIqOXoPQjEy8sL8fHxCAkJqbUvPj5emhKGbq+q+2VBQYGRS0JERERtXVW+oOfwjTrpnQA+8cQTeOONN9CzZ088/fTT0vaNGzdizZo1WLBgQbMLZS6qaks9PDyMXBIiIiIyFYWFhVAqlc06ht6jgMvKyhAaGooDBw7AxsYG7u7uuH79OkpKSjBs2DD8/PPPkMvlzSqUuaisrMS5c+fQu3dvXLlyhSOBTUxBQQE8PDz47EwMn5vp4rMzTXxuhiOEQGFhIdzd3WFhoXcvPh161wDK5XLs3bsXX331FeLi4pCdnY2BAwdi1KhRmDRpEiwtLZtVIHNiYWGBO++8EwDQqVMn/mKYKD4708TnZrr47EwTn5thNLfmr0qTJoK2tLREREQEIiIiDFIIIiIiImo9zas/JCIiIiKTwwTQyBQKBZYsWQKFQmHsopCe+OxME5+b6eKzM018bm2T3oNAiIiIiMi0sQaQiIiIyMwwASQiIiIyM0wAiYiIiMxMo6aBeeaZZxp9QJlMhk8//bTJBSIiIiKiltWoBHD//v2QyWTSen5+PtRqNaysrODk5IScnBxoNBoolUo4ODi0WGGJiIiIqPka1QR88eJFpKWlIS0tDf/5z3/QsWNHbNmyBcXFxUhPT0dxcTE2b94MOzs7xMbGtnSZiYiIiKgZ9J4GZsiQIZgwYQL++c9/1tr37rvv4rvvvsP//vc/gxWQiIiIiAxL70EgKSkp8PHxqXOfr68vjh8/3twyEREREVEL0jsB7NSpE/bt21fnvn379vFFz0RERERtXKMGgVQXERGBt956CxqNBpMnT4ZKpUJGRga2bNmCNWvWYO7cuS1RTiIiIiIyEL37AGo0Gjz77LP44osvdEYGCyHw5JNP4rPPPoOVld55JRERERG1kia/C/jcuXPYv38/cnNz4eTkhGHDhuGee+4xdPnatcrKSly/fh329vY6yTQRERFRTUIIFBYWwt3dHRYWzXuXR5MTQGq+q1evwsPDw9jFICIiIhNy5coVdOnSpVnHaFJbbWlpKTZu3IgDBw4gJycHH3zwAby9vfHDDz/A19cXd911V7MKZS7s7e0BaB8kB88QERFRQwoKCuDh4SHlD82hdwJ448YNBAYG4tSpU1CpVMjMzERhYSEAYPv27di9ezc+/PDDZhfMHFQ1+3bq1IkJIBERETWKIbqN6d2A/NJLLyE/Px+//vorLl++jOotyIGBgUhISGh2oYiIiIio5ehdA/jTTz/hzTffxP3334+KigqdfV26dMHVq1cNVjgiIiIiMjy9awALCgrg6elZ577y8nJoNJpmF4qIiIiIWo7eCaCXlxeSkpLq3HfkyBH07Nmz2YUiIiIiMgiNBli2DAgO1n6yogpAE5qAp0yZgjfffBM+Pj4YM2YMAG1nxKNHj2Lt2rVYtGiRwQtJRERE1CQxMUB0NCAEUPUq28WLjVqktkDvBPDll1/G//73PzzyyCNwcHAAAISEhCAnJwehoaH45z//afBCEhERETVJYqI2+QO0n4mJxi1PG6F3AmhtbY1du3bhm2++wc6dO5GZmQlnZ2eEhYXhiSeeaPbM1EREREQGExCgrfkTApDJtOvEN4EYU0FBAZRKJdRqNecBJCIiagkajbYZODFRm/wtXAhYNek9GEZnyLxB7ztgaWmJpKQkDBw4sNa+lJQUDBw4sNb0MERERERGYWXFPn910Lu9tqEKw8rKSoPMTk1ERERELadJHfbqS/JSUlKgVCqbVSAiIiIialmNagJeu3Yt1q5dC0Cb/I0bNw4KhUInpri4GFlZWXj00UcNX0oiIiIiMphGJYAuLi649957AQAXL17EXXfdhTvuuEMnRqFQwNfXl9PAEBEREbVxjWoCnjRpEnbs2IEdO3Zg6NCh+Pjjj6X1quW7777DkiVLaiWGhrRixQo88MADsLe3h4uLC8aNG4dz587pxAghEB0dDXd3d9jY2GDYsGE4deqUTkxpaSlmzZoFZ2dn2NnZYezYsbXeYZyXl4eIiAgolUoolUpEREQgPz9fJ+by5csIDw+HnZ0dnJ2dMXv2bJSVlbXItRMREREZit59AOPj43HPPfe0RFluKyEhAS+88AKSk5Oxd+9eaDQaBAcHo6ioSIpZtWoV3nnnHaxbtw5Hjx6FSqVCUFAQCgsLpZg5c+Zg27ZtiI2NRWJiIm7evImwsDCd0cuTJ0/G8ePHERcXh7i4OBw/fhwRERHS/oqKCowZMwZFRUVITExEbGwsvv/+e8ybN691bgYRERFRUwk9ffbZZ2LJkiV17luyZInYtGmTvodssqysLAFAJCQkCCGEqKysFCqVSqxcuVKKKSkpEUqlUqxfv14IIUR+fr6wtrYWsbGxUsy1a9eEhYWFiIuLE0IIcfr0aQFAJCcnSzFJSUkCgDh79qwQQohdu3YJCwsLce3aNSnm66+/FgqFQqjV6jrLW1JSItRqtbRcuXJFAKg3noiIiKiKWq02WN6gdw3ge++9J70CriZnZ2e89957zclH9aJWqwEAjo6OAIC0tDRkZGQgODhYilEoFBg6dCgOHToEQDtSuby8XCfG3d0dPj4+UkxSUhKUSiX8/PykmEGDBkGpVOrE+Pj4wN3dXYoJCQlBaWkpUlJS6izvihUrpCZlpVIJDw8PQ9wGIiIiIr3onQD++eef8PHxqXNf7969cf78+WYXqjGEEJg7dy4CAgKk8mRkZAAAXF1ddWJdXV2lfRkZGZDL5bWS2JoxLi4utc7p4uKiE1PzPA4ODpDL5VJMTQsWLIBarZaWK1eu6HvZRERERM3WpHehVNW81bVdo9E0q0CNFRUVhd9//x2JdbzUueY8hUKI205QXTOmrvimxFSnUChqTZ9DRERE1Nr0rgH09fVFbGxsnfu+/vpr+Pr6NrtQtzNr1iz8+OOPiI+PR5cuXaTtKpUKAGrVwGVlZUm1dSqVCmVlZcjLy2swJjMzs9Z5s7OzdWJqnicvLw/l5eW1agaJiIiI2hK9E8CoqCh89913mDp1Kg4fPoxr167h8OHDiIyMxPfff49Zs2a1RDkBaGvXoqKisHXrVuzfvx9eXl46+728vKBSqbB3715pW1lZGRISEjB48GAAQP/+/WFtba0Tk56ejtTUVCnG398farUaR44ckWIOHz4MtVqtE5Oamor09HQpZs+ePVAoFOjfv7/hL56IiIjIUJoycuS1114TVlZWwsLCQlqsrKzE4sWLmz0qpSH/+Mc/hFKpFAcOHBDp6enScuvWLSlm5cqVQqlUiq1bt4qTJ0+KSZMmCTc3N1FQUCDFPP/886JLly5i37594tixY2L48OGib9++QqPRSDGhoaGiT58+IikpSSQlJQlfX18RFhYm7ddoNMLHx0eMGDFCHDt2TOzbt0906dJFREVFNfp6DDmah4iIiNo3Q+YNMiGEaEriePHiRezduxfZ2dno3LkzgoOD4enpadjstIb6+tZ9/vnniIyMBKCtJVy6dCk2bNiAvLw8+Pn54YMPPtAZuFJSUoJ//etf+Oqrr1BcXIwRI0bgww8/1BmVm5ubi9mzZ+PHH38EAIwdOxbr1q3Tmej68uXLmDlzJvbv3w8bGxtMnjwZb7/9dqP7+RUUFECpVEKtVqNTp0563g0iIiID02iAmBggMREICAAWLgSsmjRcgFqAIfOGJieA1HxMAImIqE1ZtgyIjgaEAGQy7c+LFxu7VPT/DJk36N0HkIiIiNqpxERt8gdoP+uYaYPah0YlgJaWltKACAsLC1haWta7WLGqmIiIyDQFBGhr/gDtZ0CAcctDLaZR2drixYul6VYWL1582zn1iIiIyAQtXKj9rN4HkNol9gE0IvYBJCIiosZiH0AiIiLS0mi0gzeCg7WfrfRGLjJtjWoC/uKLL/Q66FNPPdWkwhAREZGeYmL+Hrm7b592G0fu0m00KgGsmmOvSlUfwOqtx9X7BTIBJCIiaiUcuUtN0Kgm4LS0NGlJSkqCh4cHnn32WcTHx+PMmTOIj4/HtGnT4OHhgUOHDrV0mYmIiKgKR+5SE+g9CGTSpElQqVR49913a+178cUXcf36dXzzzTcGK2B7xkEgRERUi75v4+DbO8yGIfMGvf8L+fnnn/Hdd9/VuW/06NF47LHHmlUgIiIis6Zvnz4rK/b5I73pPQq4srIS58+fr3Pf+fPnwVlliIiImoF9+qgV6J0AhoaGYtGiRdi5c6fO9p9++gmvvvoqQkJCDFY4IiIis8M+fdQK9O4DmJ6ejhEjRuDcuXOwt7eHq6srMjMzUVhYCG9vb8THx8PNza2lytuusA8gERHVwj59VA9D5g1NehNISUkJNm7ciAMHDiAnJwdOTk4IDAzEU089BRsbm2YVyJwwASQiIqLGMnoCSIbBBJCIiIgay6ijgKucPXsWCQkJuHHjBqZNmwaVSoXr16/DwcGBtYBEREREbZjeCWBFRQVmzJiBjRs3QggBmUyGUaNGQaVS4bnnnsN9992HZcuWtURZiYiIiMgA9B4F/MYbb+Crr77CW2+9hdTUVJ1pX0aNGoW4uDiDFpCIiIiIDEvvGsCNGzfitddew9y5c1FRUaGzz8vLC2lpaQYrHBERUZvAkbnUzuj9X++1a9fg7+9f574OHTqgsLCw2YUiIiIyKo0GeP11YPNm7XrXrsCBA9qfG/N2DqI2Tu8E0MXFBRcuXEBgYGCtfefOnUOXLl0MUjAiIiKjiYkBqvdnv3Dh75/5dg5qB/TuAzh69Gi88cYbuHbtmrRNJpNBrVbjvffeQ3h4uEELSERE1OoaSvD4dg5qB/ROAJctWwaNRoPevXtjwoQJkMlkWLhwIXx8fFBSUoLXXnutJcpJRETUeupK8AIDgaAgIDpa2weQyITp3QTs6uqKo0ePYsmSJdi5cycsLS1x4sQJhIWFYdmyZXB0dGyJchIREbWehQuBioq/+wBGRACvvsqBH9Ru6PUmkJKSEixbtgwTJkxA//79W7JcZoFvAiEiIqLGMmTeoFcTcIcOHfDuu++iqKioWSclIiIiIuPRuw9gr169ONcfERERkQnTOwF87bXXsHz5cvz1118tUZ7bOnjwIMLDw+Hu7g6ZTIbt27fr7I+MjIRMJtNZBg0apBNTWlqKWbNmwdnZGXZ2dhg7diyuXr2qE5OXl4eIiAgolUoolUpEREQgPz9fJ+by5csIDw+HnZ0dnJ2dMXv2bJSVlbXEZRMREREZjN69WT///HPcunULvXr1Qp8+feDm5gaZTCbtl8lk+OGHHwxayOqKiorQt29fPP3005gwYUKdMaGhofj888+ldblcrrN/zpw52LFjB2JjY+Hk5IR58+YhLCwMKSkpsLS0BABMnjwZV69elV5tN2PGDERERGDHjh0AtO9EHjNmDDp37ozExETk5ORg6tSpEELg/fffb4lLJyIiIjIIvQaBAEC3bt10Er5aB5TJcKH6hJktSCaTYdu2bRg3bpy0LTIyEvn5+bVqBquo1Wp07twZX375JR5//HEAwPXr1+Hh4YFdu3YhJCQEZ86cQe/evZGcnAw/Pz8AQHJyMvz9/XH27Fn07NkTP//8M8LCwnDlyhW4u7sDAGJjYxEZGYmsrKxGdc7kIBAiIiJqLEPmDXrXAF68eLFZJ2wNBw4cgIuLC+644w4MHToUb7zxBlxcXAAAKSkpKC8vR3BwsBTv7u4OHx8fHDp0CCEhIUhKSoJSqZSSPwAYNGgQlEolDh06hJ49eyIpKQk+Pj5S8gcAISEhKC0tRUpKSp1vSiktLUVpaam0XlBQ0BKXT0RERNQgvfsAtnWjRo3Cli1bsH//fqxevRpHjx7F8OHDpcQrIyMDcrkcDg4OOt9zdXVFRkaGFFOVMFbn4uKiE+Pq6qqz38HBAXK5XIqpacWKFVKfQqVSCQ8Pj2ZfLxEREZG+mjSjZUVFBf7zn/8gPj4eOTk5cHJyQmBgIB577DFYGXmSzKpmXQDw8fHBgAED4OnpiZ07d2L8+PH1fk8IUasvoyFiqluwYAHmzp0rrRcUFDAJJCIioland7Z248YNhIaG4tixY7CysoKTkxNycnLwySef4O2338bu3bvh7OzcEmVtEjc3N3h6euL8+fMAAJVKhbKyMuTl5enUAmZlZWHw4MFSTGZmZq1jZWdnS7V+KpUKhw8f1tmfl5eH8vLyWjWDVRQKBRQKhUGui4iIiKip9G4CfvHFF3Hu3Dls2bIFxcXFSE9PR3FxMTZv3ozz58/jxRdfbIlyNllOTg6uXLkCNzc3AED//v1hbW2NvXv3SjHp6elITU2VEkB/f3+o1WocOXJEijl8+DDUarVOTGpqKtLT06WYPXv2QKFQ8C0pRERE1KbpXQO4Y8cOLF++HJMmTZK2WVpaYvLkycjKykJ0dLQhy1fLzZs38eeff0rraWlpOH78OBwdHeHo6Ijo6GhMmDABbm5uuHjxIhYuXAhnZ2c88sgjAAClUolp06Zh3rx5cHJygqOjI+bPnw9fX1+MHDkSgHay69DQUEyfPh0bNmwAoJ0GJiwsDD179gQABAcHo3fv3oiIiMBbb72F3NxczJ8/H9OnT+eIXiIiImrT9E4AhRC4995769zn4+MDPWeV0duvv/6qM8K2qk/d1KlT8dFHH+HkyZP44osvkJ+fDzc3NwQGBuKbb76Bvb299J13330XVlZWmDhxIoqLizFixAhs3LhRmgMQALZs2YLZs2dLo4XHjh2LdevWSfstLS2xc+dOzJw5Ew8++CBsbGwwefJkvP322y16/URERETNpfc8gBMmTIC3tzdWrlxZa9/LL7+Ms2fPtuhE0O0J5wEkIiKixjLqPICvvfYaxo8fj4qKCkyePBkqlQoZGRnYsmULtm7diq1btyI3N1eKd3R0bFYBiYiIiMiw9K4BtLD4e9xI9elOqg5TcwqUioqK5pSvXWMNIBERETWWUWsAFy9e3OCr4IiIiIiobdO7BpAMhzWARERE1FiGzBva3avgiIiIiKhhTACJiIiIzAwTQCIiIiIzwwSQiIiIyMwwASQiIiIyM0wAiYjaK40GWLYMCA7Wfmo0xi4REbURes8DSEREJiImBoiOBoQA9u3Tblu82KhFIqK2gTWARETtVWKiNvkDtJ+JicYtDxG1GUwAiYjaq4AAoOrNTTKZdp2ICGwCJiJqvxYu1H4mJmqTv6p1IjJ7TACJiNorKyv2+SOiOrEJmIiIiMjMMAEkIiIiMjNMAImIiIjMDBNAIqLWxgmaicjIOAiEiKi1cYJmIjIy1gASEbU2TtBMREbGBJCIqLVxgmYiMjI2ARMRtTZO0ExERsYEkIiotXGCZiIyMjYBExEREZkZJoBEREREZoYJIBEREZGZMbkE8ODBgwgPD4e7uztkMhm2b9+us18IgejoaLi7u8PGxgbDhg3DqVOndGJKS0sxa9YsODs7w87ODmPHjsXVq1d1YvLy8hAREQGlUgmlUomIiAjk5+frxFy+fBnh4eGws7ODs7MzZs+ejbKyspa4bCIiIiKDMbkEsKioCH379sW6devq3L9q1Sq88847WLduHY4ePQqVSoWgoCAUFhZKMXPmzMG2bdsQGxuLxMRE3Lx5E2FhYaioqJBiJk+ejOPHjyMuLg5xcXE4fvw4IiIipP0VFRUYM2YMioqKkJiYiNjYWHz//feYN29ey108ERERkSEIEwZAbNu2TVqvrKwUKpVKrFy5UtpWUlIilEqlWL9+vRBCiPz8fGFtbS1iY2OlmGvXrgkLCwsRFxcnhBDi9OnTAoBITk6WYpKSkgQAcfbsWSGEELt27RIWFhbi2rVrUszXX38tFAqFUKvVjSq/Wq0WABodT0RERObLkHmDydUANiQtLQ0ZGRkIDg6WtikUCgwdOhSHDh0CAKSkpKC8vFwnxt3dHT4+PlJMUlISlEol/Pz8pJhBgwZBqVTqxPj4+MDd3V2KCQkJQWlpKVJSUuosX2lpKQoKCnQWIiIiotbWrhLAjIwMAICrq6vOdldXV2lfRkYG5HI5HBwcGoxxcXGpdXwXFxedmJrncXBwgFwul2JqWrFihdSnUKlUwsPDowlXSURERNQ87SoBrCKresXS/xNC1NpWU82YuuKbElPdggULoFarpeXKlSsNlomImkCjAZYtA4KDtZ8ajbFLRETU5rSrN4GoVCoA2to5Nzc3aXtWVpZUW6dSqVBWVoa8vDydWsCsrCwMHjxYisnMzKx1/OzsbJ3jHD58WGd/Xl4eysvLa9UMVlEoFFAoFM24QiK6rZgYIDoaEALYt0+7jW/dICLS0a5qAL28vKBSqbB3715pW1lZGRISEqTkrn///rC2ttaJSU9PR2pqqhTj7+8PtVqNI0eOSDGHDx+GWq3WiUlNTUV6eroUs2fPHigUCvTv379Fr5PI7DVUy5eYqE3+AO1nYqJxykhE1IaZXA3gzZs38eeff0rraWlpOH78OBwdHdG1a1fMmTMHMTEx8Pb2hre3N2JiYmBra4vJkycDAJRKJaZNm4Z58+bByckJjo6OmD9/Pnx9fTFy5EgAQK9evRAaGorp06djw4YNAIAZM2YgLCwMPXv2BAAEBwejd+/eiIiIwFtvvYXc3FzMnz8f06dPR6dOnVr5rhCZmYZq+QICtNuEAGQy7ToREelq9jjiVhYfHy8A1FqmTp0qhNBOBbNkyRKhUqmEQqEQQ4YMESdPntQ5RnFxsYiKihKOjo7CxsZGhIWFicuXL+vE5OTkiClTpgh7e3thb28vpkyZIvLy8nRiLl26JMaMGSNsbGyEo6OjiIqKEiUlJY2+Fk4DQ9REQUFCaFM87RIU9Pe+8nIhli7Vblu6VLtORNQOGDJvkAlR1VZCra2goABKpRJqtZq1hkT6WLbs7xpAmUz7M/v5EVE7Z8i8weSagImIsHCh9jMxUdvEW7VORESNwgSQiEyPlRVr/IiImqFdjQImIiIiottjAkhERERkZpgAEhEREZkZJoBEREREZoYJIBEZDt/DS0RkEjgKmIgMh+/hJSIyCawBJCLD4Xt4iYhMAhNAIjKcgADtmzkAvoeXiKgNYxMwEdWm0Wibc6u/acOqEf9c8A0dREQmgQkgEdXW1L58fEMHEZFJYBMwEdXGvnxERO0aE0Aiqo19+YiI2jU2ARNRbezLR0TUrjEBJKLa2JePiKhdYxMwUXvFt3IQEVE9WANI1F7xrRxERFQP1gAStVccyUtERPVgAkhkSvRp1uVIXiIiqgebgIlMiT7NuhzJS0RE9WACSNQWlZQAo0cDJ04AffsCu3YBHTro16zLkbxERFQPNgETtUWjRwPx8UBurvZz9GjtdjbrEhGRAbAGkKgt0Gi0zbtVzbUnTujur1pnsy4RERkAE0CitqBm3z5PT23tX5W+fbWfbNYlIiIDYBMwUWurayRvzb59Xl5AYCDg6Kj93LXLuGUmIqJ2hTWARK2hehOvRqPt1wf8PZI3IED7sxDavn3DhrGmj4iIWgwTQKLWUL2Jt7qqkbxVNXzs20dERK2g3TUBR0dHQyaT6SwqlUraL4RAdHQ03N3dYWNjg2HDhuHUqVM6xygtLcWsWbPg7OwMOzs7jB07FlevXtWJycvLQ0REBJRKJZRKJSIiIpCfn98al0htUVWz7siRwPDhQFCQ7kTN1Zt4q6sayVvVt2/PHu2nFf/fjIiIWk67/Ctz7733Yl9V0xoAS0tL6edVq1bhnXfewcaNG9GjRw8sX74cQUFBOHfuHOzt7QEAc+bMwY4dOxAbGwsnJyfMmzcPYWFhSElJkY41efJkXL16FXFxcQCAGTNmICIiAjt27GjFKyWj02iA5cuB994D8vJ09/33v9rPxYvrbuK1smJtHxERGYdoZ5YsWSL69u1b577KykqhUqnEypUrpW0lJSVCqVSK9evXCyGEyM/PF9bW1iI2NlaKuXbtmrCwsBBxcXFCCCFOnz4tAIjk5GQpJikpSQAQZ8+ebXRZ1Wq1ACDUarU+l0htQXm5EEuXCnHXXUJo07q6l6Ag3figIO1neblxy09ERCbHkHlDu2sCBoDz58/D3d0dXl5eeOKJJ3DhwgUAQFpaGjIyMhAcHCzFKhQKDB06FIcOHQIApKSkoLy8XCfG3d0dPj4+UkxSUhKUSiX8/PykmEGDBkGpVEoxdSktLUVBQYHOQiaqqk/f//+3VafqEzWziZeIiNqQdpcA+vn54YsvvsDu3bvx8ccfIyMjA4MHD0ZOTg4yMjIAAK6urjrfcXV1lfZlZGRALpfDwcGhwRgXF5da53ZxcZFi6rJixQqpz6BSqYSHh0ezrpVamEYDLFkCdO+uXaKjb9+nDwC6ddP2BYyOZvMuERG1Se2uGmLUqFHSz76+vvD390f37t2xadMmDBo0CAAgq3qV1v8TQtTaVlPNmLrib3ecBQsWYO7cudJ6QUEBk8C2pObbOCoqtAM5qixdClhY1O7TBwAODtrlySeB115jDR8REbVp7f6vlJ2dHXx9fXH+/HmMGzcOgLYGz83NTYrJysqSagVVKhXKysqQl5enUwuYlZWFwYMHSzGZmZm1zpWdnV2rdrE6hUIBhUJhiMuillDzbRxeXrVjEhO1n3W9ko1JHxERmYh21wRcU2lpKc6cOQM3Nzd4eXlBpVJh79690v6ysjIkJCRIyV3//v1hbW2tE5Oeno7U1FQpxt/fH2q1GkeOHJFiDh8+DLVaLcWQCar5No66sE8fERG1A+3ur9b8+fMRHh6Orl27IisrC8uXL0dBQQGmTp0KmUyGOXPmICYmBt7e3vD29kZMTAxsbW0xefJkAIBSqcS0adMwb948ODk5wdHREfPnz4evry9GjhwJAOjVqxdCQ0Mxffp0bNiwAYB2GpiwsDD07NnTaNdOzVRzqpaICO3Pmzdr90dEsE8fERG1C+0uAbx69SomTZqEGzduoHPnzhg0aBCSk5Ph6ekJAHjppZdQXFyMmTNnIi8vD35+ftizZ480ByAAvPvuu7CyssLEiRNRXFyMESNGYOPGjTrzCW7ZsgWzZ8+WRguPHTsW69ata92LJcOqr1l36VLjlouIiMjAZELU19ZFLa2goABKpRJqtRqdOnUydnHan5qDOthPj4iITJgh8wb+NaT2q+agDkDbX4+IiMjMtftBIGTGag7qqBrBS0REZOaYAFL7FRCgHcwB6L6Vg4iIyMyxCZhMR0kJMHo0cOIE0LcvsGsX0KFD/fF1DeogIiIiDgIxJg4C0dPw4UB8/N/rgYHA/v3GKw8REVErMmTewCZgaps0Gu1r2IKDtZ8ajbbmr7qa60RERNQobAKmtqmuEbx9++rWAPbta5SiERERmTrWAFLbVNcI3l27tM2+jo7az127jFtGIiIiE8UEkFpPSYm2H5+Tk/azpKT+2LpG8HbooO3zl5Oj/WxoAAgRERHVi03A1PKq3sixZg2Ql6fdFh+vHdFb3yAOjuAlIiJqMUwAqeVV789XXUODOKys+NYOIiKiFsImYGp51fvzVcdBHEREREbBBJCaT6PR1vB1765dlizRbqtSvT8fANjYcBAHERGREbEJmJqmql9fYqL25+rTsyxbBlha/t2EW1d/Piv+p0dERGQs/CtMjddQ0ldTYuLfP7M/HxERUZvCBJDqV1IChIYCR45o+/ApFIBa3bjvBgS0bNmIiIioyZgAUv1GjwYSEv5eb2jevqFDgStXtD8/+SSnbSEiImrDmACSVvXm3ap+eg1N0yKTAcOGaZt32a+PiIjIpPAvtrnTaIDXXwfef//vSZr37tV+1nz3bpW77gKmTmXSR0REZKL419vcVE3Zsno1UFYG2NvX3a+v6t271fsAqlTAU08Br73GxI+IiMiE8a94e1dVw/fll0B+PlBcrNuXr75BHVXv3j1woDVKSURERK2ICWB7VVfTbmMFBnIQBxERUTvGBLC9ionRTsjcGEol4OSk/TkiAnj1VTbxEhERtWP8K99eVZ+IuSGenkBqKtCxY8uWh4iIiNoMvgu4vaprIuYOHQAHB8DLS/tmjvJy4OJFJn9ERERmhjWA7dXChUBFBbB5s3adTbtERET0/2RCCGHsQpirgoICKJVKqNVqdOrUydjFISIiojbMkHkDm4CJiIiIzAwTQCIiIiIzwwSQiIiIyMxwRIARVXW/LCgoMHJJiIiIqK2ryhcMMXyDCaARFRYWAgA8PDyMXBIiIiIyFYWFhVAqlc06BkcBG1FlZSXOnTuH3r1748qVKxwJbGIKCgrg4eHBZ2di+NxMF5+daeJzMxwhBAoLC+Hu7g4Li+b14mMNoBFZWFjgzjvvBAB06tSJvxgmis/ONPG5mS4+O9PE52YYza35q8JBIERERERmhgkgERERkZlhAmhkCoUCS5YsgUKhMHZRSE98dqaJz8108dmZJj63tomDQIiIiIjMDGsAiYiIiMwME0AiIiIiM8MEkIiIiMjMMAEkIiIiMjNMAI3oww8/hJeXFzp06ID+/fvjl19+MXaRzEp0dDRkMpnOolKppP1CCERHR8Pd3R02NjYYNmwYTp06pXOM0tJSzJo1C87OzrCzs8PYsWNx9epVnZi8vDxERERAqVRCqVQiIiIC+fn5rXGJ7cbBgwcRHh4Od3d3yGQybN++XWd/az6ry5cvIzw8HHZ2dnB2dsbs2bNRVlbWEpdt8m733CIjI2v9Dg4aNEgnhs+t9a1YsQIPPPAA7O3t4eLignHjxuHcuXM6MfydM31MAI3km2++wZw5c7Bo0SL89ttveOihhzBq1ChcvnzZ2EUzK/feey/S09Ol5eTJk9K+VatW4Z133sG6detw9OhRqFQqBAUFSe9wBoA5c+Zg27ZtiI2NRWJiIm7evImwsDBUVFRIMZMnT8bx48cRFxeHuLg4HD9+HBEREa16naauqKgIffv2xbp16+rc31rPqqKiAmPGjEFRURESExMRGxuL77//HvPmzWu5izdht3tuABAaGqrzO7hr1y6d/XxurS8hIQEvvPACkpOTsXfvXmg0GgQHB6OoqEiK4e9cOyDIKAYOHCief/55nW333HOPeOWVV4xUIvOzZMkS0bdv3zr3VVZWCpVKJVauXCltKykpEUqlUqxfv14IIUR+fr6wtrYWsbGxUsy1a9eEhYWFiIuLE0IIcfr0aQFAJCcnSzFJSUkCgDh79mwLXFX7B0Bs27ZNWm/NZ7Vr1y5hYWEhrl27JsV8/fXXQqFQCLVa3SLX217UfG5CCDF16lTx8MMP1/sdPre2ISsrSwAQCQkJQgj+zrUXrAE0grKyMqSkpCA4OFhne3BwMA4dOmSkUpmn8+fPw93dHV5eXnjiiSdw4cIFAEBaWhoyMjJ0npFCocDQoUOlZ5SSkoLy8nKdGHd3d/j4+EgxSUlJUCqV8PPzk2IGDRoEpVLJZ20grfmskpKS4OPjA3d3dykmJCQEpaWlSElJadHrbK8OHDgAFxcX9OjRA9OnT0dWVpa0j8+tbVCr1QAAR0dHAPyday+YABrBjRs3UFFRAVdXV53trq6uyMjIMFKpzI+fnx+++OIL7N69Gx9//DEyMjIwePBg5OTkSM+hoWeUkZEBuVwOBweHBmNcXFxqndvFxYXP2kBa81llZGTUOo+DgwPkcjmfZxOMGjUKW7Zswf79+7F69WocPXoUw4cPR2lpKQA+t7ZACIG5c+ciICAAPj4+APg7115YGbsA5kwmk+msCyFqbaOWM2rUKOlnX19f+Pv7o3v37ti0aZPUEb0pz6hmTF3xfNaG11rPis/TcB5//HHpZx8fHwwYMACenp7YuXMnxo8fX+/3+NxaT1RUFH7//XckJibW2sffOdPGGkAjcHZ2hqWlZa3/e8nKyqr1fzrUeuzs7ODr64vz589Lo4EbekYqlQplZWXIy8trMCYzM7PWubKzs/msDaQ1n5VKpap1nry8PJSXl/N5GoCbmxs8PT1x/vx5AHxuxjZr1iz8+OOPiI+PR5cuXaTt/J1rH5gAGoFcLkf//v2xd+9ene179+7F4MGDjVQqKi0txZkzZ+Dm5gYvLy+oVCqdZ1RWVoaEhATpGfXv3x/W1tY6Menp6UhNTZVi/P39oVarceTIESnm8OHDUKvVfNYG0prPyt/fH6mpqUhPT5di9uzZA4VCgf79+7fodZqDnJwcXLlyBW5ubgD43IxFCIGoqChs3boV+/fvh5eXl85+/s61E60+7ISEEELExsYKa2tr8emnn4rTp0+LOXPmCDs7O3Hx4kVjF81szJs3Txw4cEBcuHBBJCcni7CwMGFvby89g5UrVwqlUim2bt0qTp48KSZNmiTc3NxEQUGBdIznn39edOnSRezbt08cO3ZMDB8+XPTt21doNBopJjQ0VPTp00ckJSWJpKQk4evrK8LCwlr9ek1ZYWGh+O2338Rvv/0mAIh33nlH/Pbbb+LSpUtCiNZ7VhqNRvj4+IgRI0aIY8eOiX379okuXbqIqKio1rsZJqSh51ZYWCjmzZsnDh06JNLS0kR8fLzw9/cXd955J5+bkf3jH/8QSqVSHDhwQKSnp0vLrVu3pBj+zpk+JoBG9MEHHwhPT08hl8vF/fffLw2xp9bx+OOPCzc3N2FtbS3c3d3F+PHjxalTp6T9lZWVYsmSJUKlUgmFQiGGDBkiTp48qXOM4uJiERUVJRwdHYWNjY0ICwsTly9f1onJyckRU6ZMEfb29sLe3l5MmTJF5OXltcYlthvx8fECQK1l6tSpQojWfVaXLl0SY8aMETY2NsLR0VFERUWJkpKSlrx8k9XQc7t165YIDg4WnTt3FtbW1qJr165i6tSptZ4Jn1vrq+uZARCff/65FNNav3MajUb89ddfYurUqaJnz56ib9++YtGiRUKtVovi4mIuDSwVFRUNPmfZ/z9sIiIiojZBCIGMjAy+NakZLCws4OXlBblcXud+JoBERETUpqSnpyM/Px8uLi6wtbXliF89VVZW4vr167C2tkbXrl3rvH+cBoaIiIjajIqKCin5c3JyMnZxTFbnzp1x/fp1aDQaWFtb19rPUcBERETUZpSXlwMAbG1tjVwS01bV9Fv93cvVMQEkIiKiNofNvs1zu/vHBJCIiIjIzDABJCIiImpjunXrhjVr1rTY8TkIhIiIiMgAhg0bhn79+hkkcTt69Cjs7OyaX6h6MAEkIiIiagVCCFRUVMDK6vbpV+fOnVu0LGwCJiIiImqmyMhIJCQkYO3atZDJZJDJZNi4cSNkMhl2796NAQMGQKFQ4JdffsFff/2Fhx9+GK6urujYsSMeeOAB7Nu3T+d4NZuAZTIZPvnkEzzyyCOwtbWFt7c3fvzxxyaXlwkgERERUTOtXbsW/v7+mD59OtLT05Geng4PDw8AwEsvvYQVK1bgzJkz6NOnD27evInRo0dj3759+O233xASEoLw8HBcvny5wXMsXboUEydOxO+//47Ro0djypQpyM3NbVJ5mQASERFRu6PRAMuWAcHB2k+NpmXPp1QqIZfLYWtrC5VKBZVKBUtLSwDAsmXLEBQUhO7du8PJyQl9+/bFc889B19fX3h7e2P58uW46667blujFxkZiUmTJuHuu+9GTEwMioqKcOTIkSaVl30AiYiIqN2JiQGiowEhgKrW1cWLjVOWAQMG6KwXFRVh6dKl+Omnn6S3dRQXF9+2BrBPnz7Sz3Z2drC3t0dWVlaTysQEkIiIiNqdxERt8gdoPxMTjVeWmqN5//Wvf2H37t14++23cffdd8PGxgaPPvooysrKGjxOzVe6yWQyVFZWNqlMTACJiIio3QkI0Nb8CQHIZNr1liaXy+t99Vp1v/zyCyIjI/HII48AAG7evImLFy+2cOl0MQEkIiKidmfhQu1nYqI2+atab0ndunXD4cOHcfHiRXTs2LHe2rm7774bW7duRXh4OGQyGV577bUm1+Q1FQeBEBERUbtjZaXt87dnj/azEVPvNdv8+fNhaWmJ3r17o3PnzvX26Xv33Xfh4OCAwYMHIzw8HCEhIbj//vtbvoDVyISoaiEnIiIiMq6SkhKkpaXBy8sLHTp0MHZxTNbt7iNrAImIiIjMDBNAIiIiIjPDBJCIiIjIzDABJCIiIjIzTACJiIiIzAwTQCIiIiIzwwSQiIiIyMwwASQiIiIyM0wAiYiIiMwME0AiIiIiM8MEkIiIiMgAhg0bhjlz5hjseJGRkRg3bpzBjlcdE0AialW3bt1CdHQ0Dhw40Crni4yMRLdu3VrlXEREpoIJIBG1qlu3bmHp0qWtlgASEbWGyMhIJCQkYO3atZDJZJDJZLh48SJOnz6N0aNHo2PHjnB1dUVERARu3Lghfe+7776Dr68vbGxs4OTkhJEjR6KoqAjR0dHYtGkTfvjhB+l4hvx3kwkgERERUTOtXbsW/v7+mD59OtLT05Geng5ra2sMHToU/fr1w6+//oq4uDhkZmZi4sSJAID09HRMmjQJzzzzDM6cOYMDBw5g/PjxEEJg/vz5mDhxIkJDQ6XjDR482GDlZQJIRLf1559/4umnn4a3tzdsbW1x5513Ijw8HCdPnqwVm5+fj3nz5uGuu+6CQqGAi4sLRo8ejbNnz+LixYvo3LkzAGDp0qXS/9VGRkYCqL+5Njo6GjKZTGfbBx98gCFDhsDFxQV2dnbw9fXFqlWrUF5e3uTrjIuLw4gRI6BUKmFra4tevXphxYoV0v5ff/0VTzzxBLp16wYbGxt069YNkyZNwqVLl3SOc+vWLcyfPx9eXl7o0KEDHB0dMWDAAHz99dc6cb/++ivGjh0LR0dHdOjQAffddx/+85//NOlYRFSDRgMsWwYEB2s/NZoWPZ1SqYRcLoetrS1UKhVUKhU2bNiA+++/HzExMbjnnntw33334bPPPkN8fDz++OMPpKenQ6PRYPz48ejWrRt8fX0xc+ZMdOzYER07doSNjQ0UCoV0PLlcbrDyWhnsSETUbl2/fh1OTk5YuXIlOnfujNzcXGzatAl+fn747bff0LNnTwBAYWEhAgICcPHiRbz88svw8/PDzZs3cfDgQen/XuPi4hAaGopp06bh2WefBQApKdTHX3/9hcmTJ8PLywtyuRwnTpzAG2+8gbNnz+Kzzz7T+3iffvoppk+fjqFDh2L9+vVwcXHBH3/8gdTUVCnm4sWL6NmzJ5544gk4OjoiPT0dH330ER544AGcPn0azs7OAIC5c+fiyy+/xPLly3HfffehqKgIqampyMnJkY4VHx+P0NBQ+Pn5Yf369VAqlYiNjcXjjz+OW7duSUlxY45FRHWIiQGiowEhgH37tNsWL27VIqSkpCA+Ph4dO3aste+vv/5CcHAwRowYAV9fX4SEhCA4OBiPPvooHBwcWr5wgohITxqNRpSVlQlvb2/x4osvStuXLVsmAIi9e/fW+93s7GwBQCxZsqTWvqlTpwpPT89a25csWSIa+ueqoqJClJeXiy+++EJYWlqK3Nzc2x6zusLCQtGpUycREBAgKisrG4ytTqPRiJs3bwo7Ozuxdu1aabuPj48YN25cg9+95557xH333SfKy8t1toeFhQk3NzdRUVHR6GMRtSfFxcXi9OnTori4uHkHCgoSQpv+aZegIMMUsAFDhw4V//znP6X10NBQMX78eHH+/Play82bN4UQQlRWVorExESxePFi4evrKzp37iwuXLgghND++/Xwww83qSy3u49sAiai29JoNIiJiUHv3r0hl8thZWUFuVyO8+fP48yZM1Lczz//jB49emDkyJEtXqbffvsNY8eOhZOTEywtLWFtbY2nnnoKFRUV+OOPP/Q61qFDh1BQUICZM2fWamqu7ubNm3j55Zdx9913w8rKClZWVujYsSOKiop07sPAgQPx888/45VXXsGBAwdQXFysc5w///wTZ8+exZQpUwBo72/VMnr0aKSnp+PcuXONOhYR1SMgAKj6fZbJtOstTC6Xo6KiQlq///77cerUKXTr1g133323zmJnZ/f/RZPhwQcfxNKlS/Hbb79BLpdj27ZtdR7PkJgAEtFtzZ07F6+99hrGjRuHHTt24PDhwzh69Cj69u2rk5BkZ2ejS5cuLV6ey5cv46GHHsK1a9ewdu1a/PLLLzh69Cg++OADANA7ScrOzgaA25Z98uTJWLduHZ599lns3r0bR44cwdGjR9G5c2edc7733nt4+eWXsX37dgQGBsLR0RHjxo3D+fPnAQCZmZkAgPnz58Pa2lpnmTlzJgBIowRvdywiqsfChdom4KAg7efChS1+ym7duuHw4cO4ePEibty4gRdeeAG5ubmYNGkSjhw5ggsXLmDPnj145plnUFFRgcOHDyMmJga//vorLl++jK1btyI7Oxu9evWSjvf777/j3LlzuHHjRrP6ONfEPoBEdFubN2/GU089hZiYGJ3tN27cwB133CGtd+7cGVevXm3yeTp06IDS0tJa26tPmQAA27dvR1FREbZu3QpPT09p+/Hjx5t03qo+iA2VXa1W46effsKSJUvwyiuvSNtLS0uRm5urE2tnZ4elS5di6dKlyMzMlGrwwsPDcfbsWamv4IIFCzB+/Pg6z1fVr/J2xyKielhZtXqfv/nz52Pq1Kno3bs3iouLkZaWhv/97394+eWXERISgtLSUnh6eiI0NBQWFhbo1KkTDh48iDVr1qCgoACenp5YvXo1Ro0aBQCYPn06Dhw4gAEDBuDmzZuIj4/HsGHDDFJWJoBEdFsymQwKhUJn286dO3Ht2jXcfffd0rZRo0Zh8eLF2L9/P4YPH17nsaqOU1ctXbdu3ZCVlYXMzEy4uroCAMrKyrB79+5a5al+LAAQQuDjjz9uwtUBgwcPhlKpxPr16/HEE0/U2Qwsk8kghKh1Hz755JMGm2hcXV0RGRmJEydOYM2aNbh16xZ69uwJb29vnDhxolZS3ZC6jmVra9v4CyWiFtWjRw8kJSXV2r5169Y643v16oW4uLh6j9e5c2fs2bPHYOWrjgkgEd1WWFgYNm7ciHvuuQd9+vRBSkoK3nrrrVpNpnPmzME333yDhx9+GK+88goGDhyI4uJiJCQkICwsDIGBgbC3t4enpyd++OEHjBgxAo6OjnB2dka3bt3w+OOPY/HixXjiiSfwr3/9CyUlJXjvvfdqJVhBQUGQy+WYNGkSXnrpJZSUlOCjjz5CXl5ek66vY8eOWL16NZ599lmMHDkS06dPh6urK/7880+cOHEC69atQ6dOnTBkyBC89dZbUnkTEhLw6aef6tSCAoCfnx/CwsLQp08fODg44MyZM/jyyy/h7+8vJWwbNmzAqFGjEBISgsjISNx5553Izc3FmTNncOzYMXz77beNPhYRkd6aNLSEiMxKXl6emDZtmnBxcRG2trYiICBA/PLLL2Lo0KFi6NChtWL/+c9/iq5duwpra2vh4uIixowZI86ePSvF7Nu3T9x3331CoVAIAGLq1KnSvl27dol+/foJGxsbcdddd4l169bVOQp4x44dom/fvqJDhw7izjvvFP/617/Ezz//LACI+Ph4Ka4xo4Crn3vo0KHCzs5O2Nrait69e4s333xT2n/16lUxYcIE4eDgIOzt7UVoaKhITU0Vnp6eOtfwyiuviAEDBggHBwehUCjEXXfdJV588UVx48YNnfOdOHFCTJw4Ubi4uAhra2uhUqnE8OHDxfr16/U+FlF7YbBRwGbudvdRJoQQRs5BiYiIiAAAJSUlSEtLkyY/p6a53X3kKGAiIiIiM8MEkIiIiNocNlA2z+3uHxNAIiIiajOsra0BaN+DTU1XVlYGALC0tKxzP0cBExERUZthaWmJO+64A1lZWQAAW1vbBt/QQ7VVVlYiOzsbtra2sLKqO9VjAkhERERtikqlAgApCST9WVhYoGvXrvUmzxwFTERERG1SRUWFQV9/Zk7kcjksLOrv6ccEkIiIiMjMcBAIERERkZlhAkhERERkZpgAEhEREZkZJoBEREREZub/ACdBPB2WOlCpAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "X_train, X_test, y_train, y_test = split_sc(train, test, feature_cols, label_cols, MinMaxScaler())\n",
    "plotPredAct(X_train, X_test, y_train, y_test, log_reg, 'Logistic Regression (mm_sc)', None, range(1, 2), 1, 1, False)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7bc674cd",
   "metadata": {},
   "source": [
    "# Searching Hyperparameters for SGD Regression"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "121d72fb",
   "metadata": {},
   "source": [
    "### Ridge Regression"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "296b322e",
   "metadata": {},
   "source": [
    "Calculate the hyperparameters and RMSE with RidgeCV"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "106252ca",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "alpha: 80.0, rmse: 904.0065894850176\n"
     ]
    }
   ],
   "source": [
    "X_train, X_test, y_train, y_test = split_sc(train, test, feature_cols, label_cols, None)\n",
    "alphas = [0.005, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 80]\n",
    "ridge_cv = RidgeCV(alphas=alphas, cv=4).fit(X_train, y_train)\n",
    "ridge_cv_rmse = rmse(y_test, ridge_cv.predict(X_test))\n",
    "print('alpha: ' + str(ridge_cv.alpha_) + ', rmse: ' + str(ridge_cv_rmse))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "41060820",
   "metadata": {},
   "source": [
    "### Lasso Regression"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8b17237f",
   "metadata": {},
   "source": [
    "Calculate the hyperparameters and RMSE with LassoCV"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "3b5743cb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "alpha: 5e-05, rmse: 875.9696483224009\n"
     ]
    }
   ],
   "source": [
    "X_train, X_test, y_train, y_test = split_sc(train, test, feature_cols, label_cols, None)\n",
    "alphas = [1e-5, 5e-5, 0.0001, 0.0005]\n",
    "lasso_cv = LassoCV(alphas=alphas, cv=4).fit(X_train, y_train)\n",
    "lasso_cv_rmse = rmse(y_test, lasso_cv.predict(X_test))\n",
    "print('alpha: ' + str(lasso_cv.alpha_) + ', rmse: ' + str(lasso_cv_rmse))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d0cb4fc7",
   "metadata": {},
   "source": [
    "### Elastic Net"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "351a8ea8",
   "metadata": {},
   "source": [
    "Calculate the hyperparameters and RMSE with ElasticNetCV"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "ac095305",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "alpha: 0.0005, l1_ratio: 0.1, rmse: 901.8572333792299\n"
     ]
    }
   ],
   "source": [
    "X_train, X_test, y_train, y_test = split_sc(train, test, feature_cols, label_cols, None)\n",
    "l1_ratios = np.linspace(0.1, 0.9, 9)\n",
    "alphas = [1e-5, 5e-5, 0.0001, 0.0005]\n",
    "elastic_cv = ElasticNetCV(alphas=alphas, \n",
    "                          l1_ratio=l1_ratios,\n",
    "                          max_iter=int(1e4)).fit(X_train, y_train)\n",
    "elastic_cv_rmse = rmse(y_test, elastic_cv.predict(X_test))\n",
    "print('alpha: ' + str(elastic_cv.alpha_) + \n",
    "      ', l1_ratio: ' + str(elastic_cv.l1_ratio_) + \n",
    "      ', rmse: ' + str(elastic_cv_rmse))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "85731c3f",
   "metadata": {},
   "source": [
    "Summary of RMSE of different regularization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "042907a8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>RMSE</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Linear</th>\n",
       "      <td>937.603908</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Ridge</th>\n",
       "      <td>904.006589</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Lasso</th>\n",
       "      <td>875.969648</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>ElasticNet</th>\n",
       "      <td>901.857233</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                  RMSE\n",
       "Linear      937.603908\n",
       "Ridge       904.006589\n",
       "Lasso       875.969648\n",
       "ElasticNet  901.857233"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "rmse_df = pd.DataFrame()\n",
    "labels = ['Linear', 'Ridge', 'Lasso', 'ElasticNet']\n",
    "rmses = [Linear_rmse, ridge_cv_rmse, lasso_cv_rmse, elastic_cv_rmse]\n",
    "\n",
    "rmse_df['RMSE'] = pd.Series(rmses, index=labels)\n",
    "display(rmse_df)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b220fcce",
   "metadata": {},
   "source": [
    "## SGD Regression"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "301115fe",
   "metadata": {},
   "source": [
    "Apply hyperparameters searched from previous code to SGD algorithm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "26b8d0e9",
   "metadata": {},
   "outputs": [],
   "source": [
    "def sgd_reg(parameters, X_train, X_test, y_train, y_test):\n",
    "    sgd = SGDRegressor(**parameters).fit(X_train, y_train)\n",
    "    y_train_pred = sgd.predict(X_train)\n",
    "    y_test_pred = sgd.predict(X_test)\n",
    "    return (y_train_pred, y_test_pred) "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8a994829",
   "metadata": {},
   "source": [
    "Table of RMSE with SGD Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "903c544e",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "ename": "InvalidParameterError",
     "evalue": "The 'penalty' parameter of SGDRegressor must be a str among {'l2', 'l1', 'elasticnet'} or None. Got 'none' instead.",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mInvalidParameterError\u001b[0m                     Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[30], line 13\u001b[0m\n\u001b[0;32m      1\u001b[0m parameters \u001b[38;5;241m=\u001b[39m {\n\u001b[0;32m      2\u001b[0m     \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mLinear\u001b[39m\u001b[38;5;124m'\u001b[39m: {\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mpenalty\u001b[39m\u001b[38;5;124m'\u001b[39m: \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mnone\u001b[39m\u001b[38;5;124m'\u001b[39m },\n\u001b[0;32m      3\u001b[0m     \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mLasso\u001b[39m\u001b[38;5;124m'\u001b[39m: {  \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mpenalty\u001b[39m\u001b[38;5;124m'\u001b[39m: \u001b[38;5;124m'\u001b[39m\u001b[38;5;124ml2\u001b[39m\u001b[38;5;124m'\u001b[39m,\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m      9\u001b[0m                     \u001b[38;5;124m'\u001b[39m\u001b[38;5;124ml1_ratio\u001b[39m\u001b[38;5;124m'\u001b[39m: elastic_cv\u001b[38;5;241m.\u001b[39ml1_ratio_ }\n\u001b[0;32m     10\u001b[0m }\n\u001b[0;32m     12\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m model_lbl, parameter \u001b[38;5;129;01min\u001b[39;00m parameters\u001b[38;5;241m.\u001b[39mitems():  \n\u001b[1;32m---> 13\u001b[0m     display(errorMetrics(sgd_reg, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mRMSE: SGD + \u001b[39m\u001b[38;5;124m'\u001b[39m \u001b[38;5;241m+\u001b[39m \u001b[38;5;28mstr\u001b[39m(model_lbl), parameter))\n",
      "Cell \u001b[1;32mIn[7], line 22\u001b[0m, in \u001b[0;36merrorMetrics\u001b[1;34m(fit_model, title, coefficient)\u001b[0m\n\u001b[0;32m     20\u001b[0m     y_train_pred, y_test_pred \u001b[38;5;241m=\u001b[39m  fit_model(X_train, X_test, y_train, y_test)\n\u001b[0;32m     21\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m---> 22\u001b[0m     y_train_pred, y_test_pred \u001b[38;5;241m=\u001b[39m  fit_model(coefficient, X_train, X_test, y_train, y_test)\n\u001b[0;32m     24\u001b[0m \u001b[38;5;66;03m#append error metric\u001b[39;00m\n\u001b[0;32m     25\u001b[0m new_rmses \u001b[38;5;241m=\u001b[39m pd\u001b[38;5;241m.\u001b[39mSeries()\n",
      "Cell \u001b[1;32mIn[29], line 2\u001b[0m, in \u001b[0;36msgd_reg\u001b[1;34m(parameters, X_train, X_test, y_train, y_test)\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21msgd_reg\u001b[39m(parameters, X_train, X_test, y_train, y_test):\n\u001b[1;32m----> 2\u001b[0m     sgd \u001b[38;5;241m=\u001b[39m SGDRegressor(\u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mparameters)\u001b[38;5;241m.\u001b[39mfit(X_train, y_train)\n\u001b[0;32m      3\u001b[0m     y_train_pred \u001b[38;5;241m=\u001b[39m sgd\u001b[38;5;241m.\u001b[39mpredict(X_train)\n\u001b[0;32m      4\u001b[0m     y_test_pred \u001b[38;5;241m=\u001b[39m sgd\u001b[38;5;241m.\u001b[39mpredict(X_test)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\linear_model\\_stochastic_gradient.py:1582\u001b[0m, in \u001b[0;36mBaseSGDRegressor.fit\u001b[1;34m(self, X, y, coef_init, intercept_init, sample_weight)\u001b[0m\n\u001b[0;32m   1557\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mfit\u001b[39m(\u001b[38;5;28mself\u001b[39m, X, y, coef_init\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m, intercept_init\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m, sample_weight\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m):\n\u001b[0;32m   1558\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"Fit linear model with Stochastic Gradient Descent.\u001b[39;00m\n\u001b[0;32m   1559\u001b[0m \n\u001b[0;32m   1560\u001b[0m \u001b[38;5;124;03m    Parameters\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m   1580\u001b[0m \u001b[38;5;124;03m        Fitted `SGDRegressor` estimator.\u001b[39;00m\n\u001b[0;32m   1581\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[1;32m-> 1582\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_validate_params()\n\u001b[0;32m   1583\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_more_validate_params()\n\u001b[0;32m   1585\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_fit(\n\u001b[0;32m   1586\u001b[0m         X,\n\u001b[0;32m   1587\u001b[0m         y,\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m   1594\u001b[0m         sample_weight\u001b[38;5;241m=\u001b[39msample_weight,\n\u001b[0;32m   1595\u001b[0m     )\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:600\u001b[0m, in \u001b[0;36mBaseEstimator._validate_params\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    592\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m_validate_params\u001b[39m(\u001b[38;5;28mself\u001b[39m):\n\u001b[0;32m    593\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"Validate types and values of constructor parameters\u001b[39;00m\n\u001b[0;32m    594\u001b[0m \n\u001b[0;32m    595\u001b[0m \u001b[38;5;124;03m    The expected type and values must be defined in the `_parameter_constraints`\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    598\u001b[0m \u001b[38;5;124;03m    accepted constraints.\u001b[39;00m\n\u001b[0;32m    599\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[1;32m--> 600\u001b[0m     validate_parameter_constraints(\n\u001b[0;32m    601\u001b[0m         \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_parameter_constraints,\n\u001b[0;32m    602\u001b[0m         \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mget_params(deep\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mFalse\u001b[39;00m),\n\u001b[0;32m    603\u001b[0m         caller_name\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m\u001b[38;5;18m__class__\u001b[39m\u001b[38;5;241m.\u001b[39m\u001b[38;5;18m__name__\u001b[39m,\n\u001b[0;32m    604\u001b[0m     )\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\_param_validation.py:97\u001b[0m, in \u001b[0;36mvalidate_parameter_constraints\u001b[1;34m(parameter_constraints, params, caller_name)\u001b[0m\n\u001b[0;32m     91\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m     92\u001b[0m     constraints_str \u001b[38;5;241m=\u001b[39m (\n\u001b[0;32m     93\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;132;01m{\u001b[39;00m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m, \u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;241m.\u001b[39mjoin([\u001b[38;5;28mstr\u001b[39m(c)\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mfor\u001b[39;00m\u001b[38;5;250m \u001b[39mc\u001b[38;5;250m \u001b[39m\u001b[38;5;129;01min\u001b[39;00m\u001b[38;5;250m \u001b[39mconstraints[:\u001b[38;5;241m-\u001b[39m\u001b[38;5;241m1\u001b[39m]])\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m or\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m     94\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mconstraints[\u001b[38;5;241m-\u001b[39m\u001b[38;5;241m1\u001b[39m]\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m     95\u001b[0m     )\n\u001b[1;32m---> 97\u001b[0m \u001b[38;5;28;01mraise\u001b[39;00m InvalidParameterError(\n\u001b[0;32m     98\u001b[0m     \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mThe \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mparam_name\u001b[38;5;132;01m!r}\u001b[39;00m\u001b[38;5;124m parameter of \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mcaller_name\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m must be\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m     99\u001b[0m     \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mconstraints_str\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m. Got \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mparam_val\u001b[38;5;132;01m!r}\u001b[39;00m\u001b[38;5;124m instead.\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    100\u001b[0m )\n",
      "\u001b[1;31mInvalidParameterError\u001b[0m: The 'penalty' parameter of SGDRegressor must be a str among {'l2', 'l1', 'elasticnet'} or None. Got 'none' instead."
     ]
    }
   ],
   "source": [
    "parameters = {\n",
    "    'Linear': {'penalty': 'none' },\n",
    "    'Lasso': {  'penalty': 'l2',\n",
    "                'alpha': lasso_cv.alpha_ },\n",
    "    'Ridge': {  'penalty': 'l1',\n",
    "                'alpha': ridge_cv.alpha_ },\n",
    "    'ElasticNet': { 'penalty': 'elasticnet',\n",
    "                    'alpha': elastic_cv.alpha_,\n",
    "                    'l1_ratio': elastic_cv.l1_ratio_ }\n",
    "}\n",
    "\n",
    "for model_lbl, parameter in parameters.items():  \n",
    "    display(errorMetrics(sgd_reg, 'RMSE: SGD + ' + str(model_lbl), parameter))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "54526745",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1bd462ae",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "af618cfe",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
