{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "        <script type=\"text/javascript\">\n",
       "        window.PlotlyConfig = {MathJaxConfig: 'local'};\n",
       "        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: \"STIX-Web\"}});}\n",
       "        if (typeof require !== 'undefined') {\n",
       "        require.undef(\"plotly\");\n",
       "        requirejs.config({\n",
       "            paths: {\n",
       "                'plotly': ['https://cdn.plot.ly/plotly-latest.min']\n",
       "            }\n",
       "        });\n",
       "        require(['plotly'], function(Plotly) {\n",
       "            window._Plotly = Plotly;\n",
       "        });\n",
       "        }\n",
       "        </script>\n",
       "        "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Populating the interactive namespace from numpy and matplotlib\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.7/site-packages/IPython/core/magics/pylab.py:160: UserWarning:\n",
      "\n",
      "pylab import has clobbered these variables: ['plot']\n",
      "`%matplotlib` prevents importing * from pylab and numpy\n",
      "\n"
     ]
    }
   ],
   "source": [
    "#'''Importing Data Manipulation Modules'''\n",
    "import numpy as np                 # Linear Algebra\n",
    "import pandas as pd                # Data Processing, CSV file I/O (e.g. pd.read_csv)\n",
    "\n",
    "#'''Seaborn and Matplotlib Visualization'''\n",
    "import matplotlib                  # 2D Plotting Library\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns              # Python Data Visualization Library based on matplotlib\n",
    "plt.style.use('fivethirtyeight')\n",
    "%matplotlib inline\n",
    "\n",
    "#'''Plotly Visualizations'''\n",
    "import plotly as plotly                # Interactive Graphing Library for Python\n",
    "import plotly.express as px\n",
    "import plotly.graph_objects as go\n",
    "from plotly.offline import init_notebook_mode, iplot, plot\n",
    "import plotly.offline as py\n",
    "init_notebook_mode(connected=True)\n",
    "import os\n",
    "%pylab inline\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
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
       "      <th>id</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Driving_License</th>\n",
       "      <th>Region_Code</th>\n",
       "      <th>Previously_Insured</th>\n",
       "      <th>Vehicle_Age</th>\n",
       "      <th>Vehicle_Damage</th>\n",
       "      <th>Annual_Premium</th>\n",
       "      <th>Policy_Sales_Channel</th>\n",
       "      <th>Vintage</th>\n",
       "      <th>Response</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Male</td>\n",
       "      <td>44</td>\n",
       "      <td>1</td>\n",
       "      <td>28.0</td>\n",
       "      <td>0</td>\n",
       "      <td>&gt; 2 Years</td>\n",
       "      <td>Yes</td>\n",
       "      <td>40454.0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>217</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>Male</td>\n",
       "      <td>76</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1-2 Year</td>\n",
       "      <td>No</td>\n",
       "      <td>33536.0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>183</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>Male</td>\n",
       "      <td>47</td>\n",
       "      <td>1</td>\n",
       "      <td>28.0</td>\n",
       "      <td>0</td>\n",
       "      <td>&gt; 2 Years</td>\n",
       "      <td>Yes</td>\n",
       "      <td>38294.0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>27</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>Male</td>\n",
       "      <td>21</td>\n",
       "      <td>1</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>&lt; 1 Year</td>\n",
       "      <td>No</td>\n",
       "      <td>28619.0</td>\n",
       "      <td>152.0</td>\n",
       "      <td>203</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>Female</td>\n",
       "      <td>29</td>\n",
       "      <td>1</td>\n",
       "      <td>41.0</td>\n",
       "      <td>1</td>\n",
       "      <td>&lt; 1 Year</td>\n",
       "      <td>No</td>\n",
       "      <td>27496.0</td>\n",
       "      <td>152.0</td>\n",
       "      <td>39</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   id  Gender  Age  Driving_License  Region_Code  Previously_Insured  \\\n",
       "0   1    Male   44                1         28.0                   0   \n",
       "1   2    Male   76                1          3.0                   0   \n",
       "2   3    Male   47                1         28.0                   0   \n",
       "3   4    Male   21                1         11.0                   1   \n",
       "4   5  Female   29                1         41.0                   1   \n",
       "\n",
       "  Vehicle_Age Vehicle_Damage  Annual_Premium  Policy_Sales_Channel  Vintage  \\\n",
       "0   > 2 Years            Yes         40454.0                  26.0      217   \n",
       "1    1-2 Year             No         33536.0                  26.0      183   \n",
       "2   > 2 Years            Yes         38294.0                  26.0       27   \n",
       "3    < 1 Year             No         28619.0                 152.0      203   \n",
       "4    < 1 Year             No         27496.0                 152.0       39   \n",
       "\n",
       "   Response  \n",
       "0         1  \n",
       "1         0  \n",
       "2         1  \n",
       "3         0  \n",
       "4         0  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv(\"../input/health-insurance-cross-sell-prediction/train.csv\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(381109, 12)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
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
       "      <th>id</th>\n",
       "      <th>Age</th>\n",
       "      <th>Driving_License</th>\n",
       "      <th>Region_Code</th>\n",
       "      <th>Previously_Insured</th>\n",
       "      <th>Annual_Premium</th>\n",
       "      <th>Policy_Sales_Channel</th>\n",
       "      <th>Vintage</th>\n",
       "      <th>Response</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>381109.000000</td>\n",
       "      <td>381109.000000</td>\n",
       "      <td>381109.000000</td>\n",
       "      <td>381109.000000</td>\n",
       "      <td>381109.000000</td>\n",
       "      <td>381109.000000</td>\n",
       "      <td>381109.000000</td>\n",
       "      <td>381109.000000</td>\n",
       "      <td>381109.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>190555.000000</td>\n",
       "      <td>38.822584</td>\n",
       "      <td>0.997869</td>\n",
       "      <td>26.388807</td>\n",
       "      <td>0.458210</td>\n",
       "      <td>30564.389581</td>\n",
       "      <td>112.034295</td>\n",
       "      <td>154.347397</td>\n",
       "      <td>0.122563</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>110016.836208</td>\n",
       "      <td>15.511611</td>\n",
       "      <td>0.046110</td>\n",
       "      <td>13.229888</td>\n",
       "      <td>0.498251</td>\n",
       "      <td>17213.155057</td>\n",
       "      <td>54.203995</td>\n",
       "      <td>83.671304</td>\n",
       "      <td>0.327936</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>20.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2630.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>10.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>95278.000000</td>\n",
       "      <td>25.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>15.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>24405.000000</td>\n",
       "      <td>29.000000</td>\n",
       "      <td>82.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>190555.000000</td>\n",
       "      <td>36.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31669.000000</td>\n",
       "      <td>133.000000</td>\n",
       "      <td>154.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>285832.000000</td>\n",
       "      <td>49.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>35.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>39400.000000</td>\n",
       "      <td>152.000000</td>\n",
       "      <td>227.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>381109.000000</td>\n",
       "      <td>85.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>52.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>540165.000000</td>\n",
       "      <td>163.000000</td>\n",
       "      <td>299.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                  id            Age  Driving_License    Region_Code  \\\n",
       "count  381109.000000  381109.000000    381109.000000  381109.000000   \n",
       "mean   190555.000000      38.822584         0.997869      26.388807   \n",
       "std    110016.836208      15.511611         0.046110      13.229888   \n",
       "min         1.000000      20.000000         0.000000       0.000000   \n",
       "25%     95278.000000      25.000000         1.000000      15.000000   \n",
       "50%    190555.000000      36.000000         1.000000      28.000000   \n",
       "75%    285832.000000      49.000000         1.000000      35.000000   \n",
       "max    381109.000000      85.000000         1.000000      52.000000   \n",
       "\n",
       "       Previously_Insured  Annual_Premium  Policy_Sales_Channel  \\\n",
       "count       381109.000000   381109.000000         381109.000000   \n",
       "mean             0.458210    30564.389581            112.034295   \n",
       "std              0.498251    17213.155057             54.203995   \n",
       "min              0.000000     2630.000000              1.000000   \n",
       "25%              0.000000    24405.000000             29.000000   \n",
       "50%              0.000000    31669.000000            133.000000   \n",
       "75%              1.000000    39400.000000            152.000000   \n",
       "max              1.000000   540165.000000            163.000000   \n",
       "\n",
       "             Vintage       Response  \n",
       "count  381109.000000  381109.000000  \n",
       "mean      154.347397       0.122563  \n",
       "std        83.671304       0.327936  \n",
       "min        10.000000       0.000000  \n",
       "25%        82.000000       0.000000  \n",
       "50%       154.000000       0.000000  \n",
       "75%       227.000000       0.000000  \n",
       "max       299.000000       1.000000  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop('id', axis = 1, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 381109 entries, 0 to 381108\n",
      "Data columns (total 11 columns):\n",
      " #   Column                Non-Null Count   Dtype  \n",
      "---  ------                --------------   -----  \n",
      " 0   Gender                381109 non-null  object \n",
      " 1   Age                   381109 non-null  int64  \n",
      " 2   Driving_License       381109 non-null  int64  \n",
      " 3   Region_Code           381109 non-null  float64\n",
      " 4   Previously_Insured    381109 non-null  int64  \n",
      " 5   Vehicle_Age           381109 non-null  object \n",
      " 6   Vehicle_Damage        381109 non-null  object \n",
      " 7   Annual_Premium        381109 non-null  float64\n",
      " 8   Policy_Sales_Channel  381109 non-null  float64\n",
      " 9   Vintage               381109 non-null  int64  \n",
      " 10  Response              381109 non-null  int64  \n",
      "dtypes: float64(3), int64(5), object(3)\n",
      "memory usage: 32.0+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Gender                  0\n",
       "Age                     0\n",
       "Driving_License         0\n",
       "Region_Code             0\n",
       "Previously_Insured      0\n",
       "Vehicle_Age             0\n",
       "Vehicle_Damage          0\n",
       "Annual_Premium          0\n",
       "Policy_Sales_Channel    0\n",
       "Vintage                 0\n",
       "Response                0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# EDA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "response_0 = df[df['Response'] == 0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "response_1 = df[df['Response'] == 1]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Response Analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Males are more likely to have car accidents than females."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x7f944e029710>"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAb0AAAFgCAYAAAAvjqe1AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAgAElEQVR4nOzdd5hU5fXA8e+ZsrOdZekdBARUFAQ79l5BY9TYsOsv1mg0Jhpb1BgTe0liYlTsXWyIIBZUQEEEqQLSYSnb+045vz/u3d2Z3QV2Yfucz/PMs3PfuffOmdndOfO+9y2iqhhjjDHxwNPSARhjjDHNxZKeMcaYuGFJzxhjTNywpGeMMSZu+Fo6gNYgPz+/qjdPhw4dpCVjMcYY03Qs6dUQnQCNMaaSfSFuH6x50xhjTNywpGeMMSZuWNIzxhgTNyzpGWOMiRuW9IwxxsQNS3rGGGPihiU9Y4wxccOSnjHGmLhhSc8YY0zcsKRnjDEmbljSM/GpqIBgRQU5ZeFt7lJQEaEkFGnGoOJHTlmYYMRm/DPNz+beNPGluJDEp+/h7c1erht8Mdn+VA7tnsALR2aSmegFIBRRrvsmj9dXlJDgEa4fnsqtI9NbOPD2YWtZmAun5fDtpgo6BTw8dFAG4wYktXRYJo5YTc/ElYQPX6ZoyUIuG3IF2f5UAKZnVfDAj4VV+7y2ooRXlpcQVigNKw/8WMjsLRUtFXK7cv8PhXy7yXkvs8sjXPN1LgUVVps2zceSnokrnrUrWJ7UjVJvIKZ8QU6wzvuVFtZRZhpuYW7s+1gUUlYVhlooGhOPLOmZuBLeaz/2LlpD9/LcmPKjeyXWeR/AJ3BYj9gkaXbOkT1j38eeyR6GdfS3UDQmHnnvuuuulo6hxZWXl9/V0jGY5hHZbSjeSJijlk1lRXJPSE3nomFp3DIiDY84y6UNTPfRJcnDmqIwfVK9/OOgDPbvakmvMRzQLYGysLKpJMzIzgk8fWgmPZK9LR1WvSQmJt7d0jGYXSeq1oPKFo41xuyILSLbPljzpjHGmLhhSc8YY0zcsKRnjDEmbljSM8YYEzcs6RljjIkblvSMMcbEDUt6xhhj4oYlPWOMMXHDkp4xxpi4YUnPGGNM3LCkZ+JPcSG+GVPxLvoBbBo+Y+KKLSJr4opkrSX5L1cjRQUAhEYfRtm197RwVMaY5mI1PRNXEj55syrhAfhmf4Vn9bIWjMgY05ws6Zn4UlZSu6y0uPnjMMa0iGZLeiKySkR+EpEfRWS2W5YpIlNEZJn7s2PU/n8UkeUislREjo8qH+WeZ7mIPC7iLIImIgERed0tnyUi/aOOGe8+xzIRGd9cr9m0PsEjTkE91X/24d67Edl9eAtGZIxpTs22np6IrAJGq+rWqLIHgRxVfUBEbgU6quofRGQP4FVgf6AnMBXYXVXDIvIdcD0wE/gYeFxVJ4nIb4G9VfUqETkHOF1VzxaRTGA2MBpQYA4wSlWrls629fTii2fFYnwzpqIdMgkeeSqkprd0SKYNsPX02oeW7sgyFjjCvf8C8AXwB7f8NVUtB1aKyHJgfzdxpqvqDAARmQCMAya5x9zlnust4Em3Fng8MEVVc9xjpgAn4CRVE4ciA4dRMXBYS4dhjGkBzZn0FPhURBT4t6o+A3RT1Y0AqrpRRLq6+/bCqclVWueWBd37Ncsrj1nrniskIvlAp+jyOo6pZdky69RgTFOpiMBbG30sK/ZwYMcwx3cJt3RI9TZ69OiWDsE0guZMeoeo6gY3sU0RkSXb2beuZgTdTvnOHlPL4MGDtxOWMWZXXPpFDm+vLAXgw80+wmnpXD88rYWjMvGk2TqyqOoG9+dm4F2c63WbRKQHgPtzs7v7OqBP1OG9gQ1uee86ymOOEREf0AHI2c65jDHNqKAiwrurSmPKXlhqPWdN82qWpCciKSKSVnkfOA5YALwPVPamHA9MdO+/D5zj9sgcAAwGvnObQgtF5ED3et2FNY6pPNeZwDR1eulMBo4TkY5u79Dj3DJjYizMCXLhtGxO+GgLzy2xD+PGFvAKyd7YhpeOARs1ZZpXczVvdgPedUcX+IBXVPUTEfkeeENELgXWAL8GUNWFIvIGsAgIAVeramXj//8BzwNJOB1YJrnlzwIvup1ecoBz3HPliMhfgO/d/e6p7NRiTKWSUISxk7eytSwCwMzNFaT4hbMGJrdwZO1HwCv8cd90bvsu392G2/a1nrOmeTXbkIXWzIYsmGnryzjj0+yYstP6JTLhqE4tFFH79XNekIW5QQ7uFqBbsrelw6k3G7LQPrT0kAVjWoUBaT6E2B5OA9Pt36Mp7J7hZ/cMf0uHYeKUNagbAwxI93Hbvun43f+IUZ39XLtXassGZYxpdNa8iTVvmmrZZWFyyiMM7mA1ERPLmjfbB2u/MSZKp0QvnRLbznUmY0zDWPOmMcaYuGFJzxhjTNyw5k0Tl5bmBXl+aTFeES4ZmsJu2+ipOT+7gnt/KGB9cZgTeidy4z5ppPjtu6IxbZV1ZME6ssSb1YUhxkzcTGHQ+bVnJAizTu9Wa8zYwpwgh7+/mVDUX8fhPRKYeEKX5gzXtBLWkaV9sK+sJu68vbK0KuEB5FUoE2vMCQnw+oqSmIQH8OXGCtYWhZo6RGNME7GkZ+JOh4TaX9g71DEHZIeE2mVegRSffeE3pq2ypGfizlkDk9mjY/U1vH07+zmtX1Kt/cYPSaZHcuy/yHV7pZBpQxqMabPsmh52TS8eBSPKtPXleAWO7BnA66m79lYWUt5dVcLy/BCn9ktkROdAM0dqWgu7ptc+WNLDkp4xZscs6bUP1rxpjDEmbljSM8YYEzcs6RljjIkblvSMMcbEDUt6xhhj4oYlvTZufXGYNTZDiDHG1ItNON1GRVT57fRcXl9RigKn9Uvk2SMy8W9jvJkxxhir6bVZk9eW8Zqb8ADeX13Guytrzx9pjDGmmiW9NmpFQe0mzbrKjDHGVLOk10ad0CeR6GXdPAIn9U1suYCMMaYNsGt6bdSgDn7ePLYTTywoIhSB/9szhX06JbR0WMYY06rZ3JvY3JvGmB2zuTfbB2veNMYYEzcs6RljjIkblvSMMcbEDUt6xhhj4oYlPWOMMXHDkp4xxpi4YUnPGGNM3LCkZ4wxJm5Y0jPGGBM3LOkZY4yJG5b0jDHGxA1LesYYY+KGJT1jjDFxw5KeMcaYuGFJzxhjTNywpGeMMSZuWNIzxhgTN3wtHYAxrUVeeYTnlhazsSTMrwYkcUC3AACSsxkqytHufVo4QmPMrmrWmp6IeEVkroh86G5nisgUEVnm/uwYte8fRWS5iCwVkeOjykeJyE/uY4+LiLjlARF53S2fJSL9o44Z7z7HMhEZ33yv2LQVEVVO+2Qrd88p4JnFxZw4aStfbCgj8PzDJN94Nil/uIDEv90I5WUtHaoxZhc0d/Pm9cDiqO1bgc9UdTDwmbuNiOwBnAPsCZwAPC0iXveYfwJXAIPd2wlu+aVArqoOAh4B/uaeKxO4EzgA2B+4Mzq5mvgzfWM5N8/M46mFRRQHIwDM3lLB/Jxg1T4RhVlffof/8/cRVQB8i37A/+VHLRKzMaZxNFvSE5HewMnAf6OKxwIvuPdfAMZFlb+mquWquhJYDuwvIj2AdFWdoaoKTKhxTOW53gKOdmuBxwNTVDVHVXOBKVQnShNnPlhdymmfbOU/i4u57bt8zpmaDUCqv/a/woBf5tQqk80bmjxGY0zTac5reo8CtwBpUWXdVHUjgKpuFJGubnkvYGbUfuvcsqB7v2Z55TFr3XOFRCQf6BRdXscxtSxbtqxhr8q0KU8uCKB4q7anZ1UwZd5y+icrv85bwpsZ+wKQGSzkkLXfEfYn4A1WAKAIq7sPoMj+RuLS6NGjWzoE0wiaJemJyCnAZlWdIyJH1OeQOsp0O+U7e0wtgwcP3mFwpu3qsiYb8qqvywkwbGB/eqf6eCnv71y56mM2JnTkhJwfyfBGKL3pQRI+fRPKywgeNZYeow9rueCNMbusuWp6hwCnichJQCKQLiIvAZtEpIdby+sBbHb3XwdEd5XrDWxwy3vXUR59zDoR8QEdgBy3/Igax3zReC/NtCU37p3GFxvKKQo533suGpJM71Tn36Bi3EUc/sSfkXAYgPJxlxMZNoKyYSNaLF5jTOMS1W1WeprmCZ2a3u9V9RQR+TuQraoPiMitQKaq3iIiewKv4HQ86YnTyWWwqoZF5HvgWmAW8DHwhKp+LCJXA8NV9SoROQc4Q1XPcjuyzAH2dUP4ARilqjmVMeXn5zfvm2Ba1ObSMJ+tL6d/mpeD3GEJlWTLRryLfyTSbxCRflbrN9U6dOhQV6uRaWNaepzeA8AbInIpsAb4NYCqLhSRN4BFQAi4WlXD7jH/BzwPJAGT3BvAs8CLIrIcp4Z3jnuuHBH5C/C9u9890QnPxJ+uSV5+Myi5zse0Sw9CXXo0c0TGmObS7DW91shqesaYHbGaXvtg05AZY4yJG5b0jDHGxA1LesYYY+KGJT1jjDFxw5KeMcaYuGFJzxhjTNywpGeMMSZuWNIzxhgTNyzpGWOMiRstPQ2ZMa3WM4uKeH5pMekJHm4dmcYRPRNbOqQ2xffFh/invguBRCpOu5A5vUZy95wC1heHOWNAEreMSMMjtSc58fyyhIQ3n8GTu5XgQccQPO0CqGM/Y3aGTUOGTUNmavtgdSkXTKueojXRCz+e2Z3uyd7tHGUqeefPIumhP1Rtl/iT2O3I/7K1onqf+/fvwG/3TI09sLyUlN+djRQXVBWVXXgDoaPH0dJsGrL2wZo3janD1HVlMdtlYZi+sbyFoml7vPO/i9n+IblPTMKD2u8xgPeXJTEJD8A3f1ajx2filyU9Y+owrKO/XmWmbpHeA2K2dy/JIkFiG1Tqej8jPfqi3tjadKTXgFr7GbOzLOkZU4eLdk9hbP9EBEjyCrfvm85emZb06is05gSCBx2Digf1+8k4cSwPHdyR9ASnhfDQ7gn8fp+0WsdpRifKL7geTXSWfgrtOYqKk3/TrLGb9q1B1/RE5FBgJBDTEK+q9zdyXM3KrumZbckpC5PgFVL99v1wpxQVgNcHSU4SKwspBcEIXZN2cG20ohwpK0HTOzZDkPVj1/Tah3r33hSRJ4CzgOlAadRDljBMu5WZaB1Xdklqesxmok9I9NXjPU0IoAmBHe9nTAM1ZMjCecBeqrqhqYIxxhhjmlJD2mzWAtZ9zRhjTJvVkJrepcB/RORVYFP0A6r6VaNGZYwxxjSBhiS9UcCJwGHUvqbXtzGDMqa1ktyt+GZ+hiYkEjr4GEhKaemQWqeiAjZrgHemzCbxl0Wc41tP8unnEek3uKUjM3Gu3r03RSQbOFtVpzZtSM3Pem+a+pCtWSTfeQVS5AyejvToQ8k9/wXrcFGtqIDEp+9m87IVjB59P5sTOgAwsDSL75f8Dc/fJ0CgbU7nZr0324eGXNMrBqwZ08Qt/1eTqhIegGfjWnxzv2nBiFqfhPdfxLdwDhO6H1aV8ABWJHXnvcQheJcvaMHojGlY0rsDeFREuouIJ/rWVMEZ0xSKghGKgxHKw0peeaT+B9b5Pd++/EfzrF0BgNQ1kkkg0r1PM0dkTKyGJKz/AVcB64Ggewu5P41p9VSVm2fmMeCVjfR9eSN9XtpA/1c2cs7UbIqCO05+wcNOJpKWUbUd7tmf0MiDmzLkNie812gALsyaTvfy3KrywSVZnJZZjnf+LKiwTuCm5TTkml6/bT2mqqsbLaIWYNf04sOHq0s5P2rlhGi/3zuV20d1qFUu61fh/+pjSAgQPPJU8PrwzfocTQgQOuCoqplGjCscIuHt/+H77nM2dRnAawdfSiB/K+e9ezcdQiUAhPYcTdkt/2jhQBvOrum1D7a0EK0v6a0rCnHNN3l8k1XOvp0TeHJMBoM7VM/7OGNTOTd9m8eKwhAn9UnisUMySE+wVuYdeWBuAQ/8WFjnYxkJwlNjMrj9e2e9N79HGFyygW++u41AyKmZbEjI4KzjHuGZE/vQN9WWotwR3+cfkPDe80hBHhIJxzxW/NcX0J7b/B7dKlnSax8aMg3Zi2xjyjFVvbDRIjLcOCOPLzY4H7SzNldwxVe5fH5qVwAqwsr4z3PYXOo0x727qpQuSR4ePDBjm+czjiN7BraZ9PIqlIu/yKXCbeWsiCi/Wje9KuEB9KzIY9DyWVw9PZUPTuzSHCG3WZ41K0h8/qFt7+CzybtNy2hI9WA5sCLqVowzbq/u9iKz02Zujl14bO7WIBVh5/vGysJQVcKrNKvG/qZuB3QL8PSYDPbI8JHur/2lvaLGZb1Cb+2u9YXeRL6z93uHPNvppRk8+Fi0a89mjMaYavWu6anq3TXLRORZ4M5Gjciwd0c/X2+q/mAd0clPgtf5kB6Q5qNLooctZdWf0Pt3TWj2GNuqcwencO7gFBbnBjn2wy0UhZwvE3t29LE8P0R0Z85nexzJpRs/Z2DZZgBmpA/mo04jOdDe7x2KDNqzVlnF4ScTHn0Y4eH7t0BExjh26ZqeiPiAHFVN3+HOrVhruqb3TVY5Z0zeWvXh2zXRw4cndmb3jOrmoG+zyrlpRh7LC0Kc1DeRxw/pSAe7ptdgqwpDvLuylI4BD2fulsSXG8q57ft81heF8btfMi7u76X70ll8m+fl047DGdopwCtHd6Jfml3T2xHftInONb2KCoLHnkHFry5t6ZB2iV3Tax8a0nvzqBpFycA5wCBVPbCxA2tOrSnp/erTrXy2vvo6kkdg8Vnd6ZZsS9wY05Is6bUPDfm6+myN7WLgR8CWNW5EJaHY/BtRKI+0mpxsjDFtWkOu6Q1oykCM44phKcyIup53fJ9E6x5vjDGNZLufpiIi6rZ/bm+6MVVtwFxOZntOH5BMlyQvH68pZVC6n3MH2eBnY4xpLNu9piciBZWdVEQkQu1xegKoqrbpC06t6ZqeMe2F96fv8X/6Fng8BE84i/CwkS0d0i6xa3rtw47azaL7HVvzpmn3Iqq8+HMJX20sp3+alxuGp5FmPWMbzLNmBYkP/wGJOI1A3gWzKbnvObR77xaOzMS77SY9VV0bdb9Nz69pzPZIXjayaT3Xb+zOCyur51B/YkERrx7TiaO7+8gLeViYG2TvTn7S/JYIt8f7w9dVCQ9AQkF8c78heOLZhCKKz7OdSpMqaAQ8bboBybRSDZmGrANwHTASSI1+TFWPa+S4jGk2vs/eI/DSE0gkzL0JGczZ+w8sSO0LONO+hR+/m+RNs8hJyOCLnsdxZb+jeer4Phzes20uhtoctEuPWmWTAgO58c0s1hWHOblvIk+N6Vhrzljf5+8TePtZKCslePjJVJx3jSU/06ga8nX1TeAIYBrweo2baSFrikJsKgnveEdTt9ISAq/9q2pC5B4Vefxl5ZtVD6dEyhmXNROPRuhbnsNfV77Ggq+u4aP3v2ihgNuG0AFHEdr3kKrt3AOO5ZINPVlTFCai8MHqMh6sMQ+qbFhN4IVHkMJ8JFhBwtR38U3/pLlDN+1cQ/rCHwh0UlVbP68VKAsp47/IYfLaMjwClw5J4e8H2aTTDSXFBUhFWUxZ7/Lsqvu/Xfcpnhr9t1Ij5dw473nghGaIsI3y+Si7/j5k0zoQDws9nSn4YEvMLnO2xs5h6l25FKnRsc77yxJCh5/c5OGa+NGQmt7XwLCmCsQ0zCvLS5i81vmwjij8Z0kx32TZ4pwNpZ27E64xT+TrXQ+quv9LUrc6j+tbYfOs14d264127cmQDD+ZgdiPm0O6BWK2w4P3Qj2x+4SH7tPkMZr40pCa3kXAxyIyC9gU/YCq3tOYQZkdW15Qu8K9oiDEId0Ddexttqf0+vtI+OAlNq9YyQOe4fyr5zFVj83odyBlnVcQ+GYyEg5VlUcOOqauU5ltSPIJLx2Vya2z8lldFGJs/yRu2ictZh/t2pPyq24n4e1nobSE0JGnETrw6BaK2LRXDZl78z/AacB0oDTqIW3r6+m1xXF6X2eVc8qkrVXbiV6YfUY3etvsLTstrzzCoFc3Ej0T3H5d/Ew5pStUlOOf9DreZQsID96L4EnngN9WW2hMvs/ewz/1PQgEqBg7nvDIg1s6pBg2Tq99aEjSKwR2V9WNDX4SkUTgKyCAU7t8S1XvFJFMnI4w/YFVwFmqmuse80fgUiAMXKeqk93yUcDzQBLwMXC9qqqIBIAJwCggGzhbVVe5x4wHbnfDuVdVX4iOry0mPYB3V5bwn8XFJPuE3+2dZrW8RnDLzDyeWVwMgFdgwpGZnNwvqerxomCEm2fm89GaUgan+3jwwAxGdbHkt6u882aS9PCtVdvq9VLywIutat09S3rtQ0Ou6f0C7GwnlnLgKFXdBxgBnCAiBwK3Ap+p6mDgM3cbEdkDZwWHPXF6CzwtIpX9lv8JXAEMdm+VvQkuBXJVdRDwCPA391yZOGv+HQDsD9wpIh138nU0OVXl/rkFDH1tIwe+u4mPVpduc9/TByTz8UldeOu4zpbwGsnf9ktjbspXLF39EKtkIid3DsU8ft8PBby6vISCCmXO1iDnT8smZBOC7zLvT9/HbEs4jHfRDy0UjWnPGpL0XgTeF5HfiMhR0bcdHaiOInfT794UGAtU1rpeAMa598cCr6lquaquxFm1fX8R6QGkq+oMd07QCTWOqTzXW8DRIiLA8cAUVc1xa5FTaMXd7l5dXsKDPxaSVRphSV6Ii77IYUOxDUloLFklYSb8XMy09WWoKkTCeH/8Ft9Xk6Aon8DECQz/6N8MXPkDPaa9Qf4/7qA4WD3I+ttNsT0ON5ZEWFEQqvk0poEifXaro2xgC0Ri2ruGXAC62v15f41yBWr/xdbg1tTmAIOAp1R1loh0q2wuVdWNItLV3b0XMDPq8HVuWdC9X7O88pi17rlCIpIPdIour+OYWpYtW7ajl9KkPvo5gehfSzAC781bzbFdqhNfROHLbC+rS4UxmWEGpVhNoz4WFwlXzk+kNOK0Uh3fOcircx4kdeUiAEKvpBJOiG2q7PXLXPZ9ZQmP7O8n3QeD/H7m4S7oqxHSfUJo0yqWxfbGNw3VfSD9h+5LxpK5qM/PpoOPJyvigxb+f4w2evTolg7BNIJmW1pIVcPACBHJAN4Vkb22s3tdbee6nfKdPaaWwYMHbyespnd4sIiPNudXbQtwwl592S29+lf12+m5vLK8BIB/rYGXj87khD5JNU9lavjrFzmURqqbiwtX/EK6m/AAfKVFeBK7xByT60tmSSiZ2dqZKwen8rfeYWa+spz1nlQQD5FQBf7OfRjcyWZn2WllJSQ+fge+JT+gPh/BE88i7czLSNvxkcY0WIMmEBQRv4gcKiJnu9spIpLSkHOoah7wBU4T4ya3yRL352Z3t3VAn6jDegMb3PLedZTHHCMiPqADkLOdc7VKFw9N4YLByfg9kBnw8MjBGTEJb1NJmFfdhAcQVmd+SLNjpW63zIxgMRdv/IKTtv5Ya59In934tOsonux1HPOTevNK10MYn/UV3oIcluUHeWDmJifhuYokgeen/tRsr6E98k95F9/C2QBIKETgg5fwrFvZwlGZ9qohc28OB97H6ZTSG6fX5eHAeODsHRzbBQiqap6IJAHH4HQ0ed89/gH350T3kPeBV0TkYaAnToeV71Q1LCKFbieYWcCFwBNRx4wHZgBnAtPcXp2TgfujOq8cB/yxvq+7ufk9whNjOvLQQRn4POCRHXcYq2cH3Lh32bAU5q/YyLez/0yvitw693k6fX9uzDwQVJk8769cvWEKAEWrXufgEXexKLmO3oQFNlB9V3g2rqlVJhvXQG9b2MU0vobU9P4J3KGqQ6nuxfklMKYex/YAPheR+cD3OB1LPsRJdseKyDLgWHcbVV0IvAEsAj4BrnabRwH+D/gvTueWFcAkt/xZoJOILAduxO0Jqqo5wF/c5/0euMcta9USvFJnwuuW7OXsgdVNmR6Ba/ZKrbWfqe3oXol8mfrdNhNesSeBW337A3Bwwc8cnbew6rHUimKuWje51jFpoVIu6WPfOnZFqMZ4PE1MbvNr75nWqyEdWfYEXnLvK4CqFrs1t+1S1fk4qzPULM8G6pxyQVXvA+6ro3w2UOt6oKqWAb/exrn+B/xvR3G2FU+N6ciJfZNYlh/i+D6JDM/0t3RIbUbPQGSbj1WkZhAWDygkRGr3yAxElYlGuGP9R5w7MEC349r03AwtLrzf4ZRddBP+Lz9CU9OpGDceUtNbOizTTjVkcPpc4HJVnS0iOaqaKSL7A0+q6v5NGmUTa6uD003Dyab1JN95BVLqDECPZHYleMhxiMdD8NATuXppIi8tK8GjEWbO+TP7Fq0CoEz8HD7yDuakOx2Vzx2UzNOHttrhnqYJ2OD09qEhSe8UnCbEfwE34dTCrsJJhJ82WYTNwJJefJEtG/F98ykkBAgeegKkVa9OEY4ob68sZcq6Mj75OZsLs6bTNVjAa10PIqnfAA7rEWBYRz9nDkjCu72FUE27Y0mvfah30gMQkX2By4B+OGPf/qOqc5ootmZjSc/UVBKKsNcbm8gpr24OvXe/dK7ZyzrSxytLeu1DvZKeO7D8Z2APVW1369dY0jN1mb2lgjtn57O+OMyZA5L548g0q93FMUt67UNDmjd/BvZT1fwd7tzGWNIzxuxIYyc9EekPrAT8qlqr55SI/AnYTVUv28F5ngfWqert29vPOBoyZOFR4A0ROVxEBorIbpW3pgrOGGNaOxGZLCK11hQVkbEikuVOltFgqnr/jhJeYxGRVSJS6o6DzhORb0XkKhFp0AQmbUFDfhlPuj+PrVGugBdjjIlPz+NMgHGnxjadXQC8XFctrpU6VVWnikgHnIlHHsNZnebilg2rcdU7i6uqZxs3S3jGmHj2HpAJHFpZ4M4AdQowQURuFZEVIpItIm+4y51FO09E1ojIVhG5Leocd4nIS1HbY9waWJ6IrBWRi+oKRkROEZEfo2psezfkxahqvqq+jzPT1vjKeZJF5GQRmSsiBe7z3xX1nP1FREXkYvexXLemuJ+IzHdjeTJq/4EiMs19T7aKyMvuvMyVj+/rPlehiLwpIq+LyL2N8Rp3mPREJFlE7heR991fgi3cZtoFVWVBTjB26aaifDwrl1X4DosAACAASURBVEK4rXw5Ny1NVUtxZpCKnqXgLGAJcCTO8meH40ypmAs8VeMUY4AhOBN13CEiw2o+h4j0xZl96gmgC866pLUmj3V72P8PuBJnlZl/4ywJ1+DPbVX9Dmfu4spkXuy+xgzgZOD/RGRcjcMOwJk28mycS2K34Uw7uSdwlogcXhkq8Fec92QYzvzId7mvIQF4F6cGnQm8CpzeWK+xPjW9J4FTcX6BZwL/qM+JjWnNssvCHP7+FsZM3Mxeb2Zxz5x8fNMmknLDmSTfdSXJt5yPZK3d8YnM9pWW4JvxGd4fZ0CkXa8L+QLw66gZqi50y64EblPVdW7P97uAM2tc57tbVUtVdR4wD9injvOfB0xV1VdVNaiq2apae8Z0uBz4t6rOUtWwqr6AM1/ygTv5ujbgJB5U9QtV/UlVI+4sW6/iJPNof1HVMnfsdjHwqqpuVtX1wHTcmblUdbmqTnHXTN0CPBx1rgNxLr097r7Wd4DvGus11ifpnQgcp6q3uPdPqc+JjWnNnlxQxPwcZwrZiMJ/f9hMwitPI0GnzLM1i4S3283MdS1CcreS/KfxJP7rLyQ98keSHrix3SY+Vf0a2AKMdTv37Qe8gjOm+V23GS4PWAyEgW5Rh2dF3S8B6ppMtw/OXMM70g+4qfL53Ofsg1Oj2hm9cFarQUQOEJHPRWSLOOuVXgV0rrH/pqj7pXVsp7rn6ioir4nIehEpwJnisvJcPYH1Na6PRn8D3aXXWJ+klxK10OtanCV7jGnT1hTFfvh2rSjAE4wdgurZsrE5Q2p3/J+/jyenenVd79J5eBfMbsGImtwEnBreBcCnqroJ58P6RFXNiLolujWfhlgL1Gcp+bXAfTWeL1lVX23g8yEi++Ekva/doldwVrPpo6odcGbn2tlhHH/F6QS5t6qmA+dHnWsj0EskZsb96OXhduk11ifp+UTkSBE5SkSOqrntlhnTppzWP3ae9PzMngT7xH6mhPar2XJjGqSstHZZeR1l7ccEnOtXl+M0bYKTGO4TkX7gLLMmImN34twvA8eIyFki4hORTiIyoo79/gNc5dbKRJw1T08WkXpPJSQi6eJMO/ka8JKqVi4YmQbkqGqZOPMun7sTr6NSGlAE5IlIL+DmqMdm4NSGr3Ff61ggen7nXXqN9RmysJnYFQqya2wrYGP1TJsytn8S/z6sI68sK6Frkoeb9kmj4vgHYOIEPFlrCO07huCxv2rpMNu00KEn4p82EQlWABDp3I3w3jt7aan1U9VVIvItzjW5993ix3BqMJ+KSE+cz9PXqV47tL7nXiMiJ+H0qfgvkA/cTo3OLO6CAJfj9MUYjNOk+DXwVT2e5gMRCQERnGXdHsZJ2pV+Czzk9sL8EqfzTkats9TP3ThfEvJxlol7Efid+xoqROQMnNf5V5wOPB/iXLfb1dfYsLk3d3gykd6quq7RTthMbEYWY5qGZ90v+L6ejCalEDriFLRDzd76bYdNQ9ZyRGQW8C9VfW6Xz9XISa/AbZ9tUyzpGWN2xJJe83GHNiwFtuL0XP0XzpRsu3yhfaemx9kO+6Mwrd7aohD/mFfImqIwpw9I4sLdU6oem5ddwdR15QzJ8HFS38SY1euLghEqwkpmos3HYNoWd5zfom08vIeqrmnOeOphCE7zaSpOr9UzGyPhgdX0AKvpxZNwRNnvnSx+KaxeMuiSIck8fHBHJq4q5aLPc6j8Y9ijo4/pp3XF6xEe/LGAh+cXUh6Gcf2T+NdhHQl47TtegwQrSHj1afyzphHp1J3yc68mMrSuIWmtk9X02od2N5moMdszPycYk/AAnltawobiMA/PLyT628+i3BBPLCjkx60V3D+3kLKw02vr3VWlTPi5uFnjbg/8H71KwmfvIUUFeFf/TNLjt0N5WUuHZeKMJT0TV3oke2u1wSvw/ZYKioORWvt/tznI4rzaU5ItzrVpyhrKu3RezLYUF+JZW5/x1sY0nsZOelb9N61a92QvJ/RJjCnzACM7+7m2jlXRD+qewGE9AiRE/aeIRjjDnwWlJU0cbfsSGTAkZlsDiUR69muhaEy8auykt0cjn8+YRvfiUZmcPTAJvwe6JXl46tCO9E31MX5ICnfsm0ayz/nHOLF3gCuHpdIrxctrx3TiwK4JnODfwub5t3DCk1eRcv0Z+L75tKVfTptRceoFhEYfhoqHSGYXyq66HZLrmnHLmKaz3Y4sIrIW2GEnD1Xt25hBNTfryBKfIqpVvTPf/qWEiatKKQoq0zZUT0d27V4p3D26Q9V+iU/cgW929RhYTUqh+LG3IRBbezTbEQqC1wfSthqG2kpHFnl0UU/gVqA3zioJD+gNe2zY6fM504FNx5n6a5JbdhZwiaqe0AghN6sdJb16zcOkql82WkQtwJJefHt5WTFXf523zcd3S/PyzvGd6Z/mI+n2S/HWuA5V/I9X0S49mjrMNsuzYhGBFx/Ds2kdoX3HUH7hDRBI2vGBrUxbSHry6KLLgDtxEl6ldcBdesMez+70eZ019d7EWSXBizMTzAmq2uYuyjbqkIW2ypJefDvtk618tbF8u/v8ZlAy/zy0IwnvPEfCxBeqysN9B1H6l/82dYhtVyhE8k1n48nLriqqOOEsKn7z2xYMaue09qTn1vBmEZvwKq0DDtjFGt+DOMsFpbg/+wHDccZ736WqE0VkT+A5IAHnKsGvVHXZzj5nU2jQ4HR3gtNDcZaAqPoDUNU7GjkuY5pNt6QdX9peVej01qwYewHq9eL7cQaRnn2p+NWlTR1emxKKKG+vLGVJbpBjeicyJrwxJuEBeJfUXgauNKT87ttc+Ol7TitezJgDh5N+6FFtrgm0hVU2adalN3ALcMMunP9u4AegAmcuzGmqeom74vl3IjIVZ7mhx1T1ZXcx2FY3k0O9k56IXAE8AnyKs67eJOA4GjhxqjGtzc37pPHVxnI2lUYQYGz/RL7YUE5ehdMAMKx4HVcv/QFfaj9CBx5NcOyFBMdeuP2Txqlrv8nj1eVOr9ZHfirinwdncllqOlJUULVPZLdaC4Mz+p0sxi79hMeXu7Xohe+Rt2IBvouvb5a424ltJbxKfXbw+HaparGIvI6zOsJZwKki8nv34USgL84KCbeJSG/gndZWy4OG1fRuwWnDnS4iuap6uoicCJzTRLEZ0yx2z/Az78zuzNxcTr9UHwPSfSzPD3LoxC2MzFnClB/vJ6Ah+BFC306h7NZHWjrkVimnLMzrK2KHcfz7m9VcVlJc3RvO4yVSY9LpRblB1hdHuG7dJzHlqdM/oOy8qyAh0IRRtys7mux/7Q4er4+IexOcpsulNR5f7E4OfTIwWUQuU9VpjfC8jaYhQxa6qup0935ERDxuT55TmyAuY5rErE3l3DU7nwk/F1MRVjwrl+BZ9wuJPuGInokMSHe+B2aVRigNK1ev+9RJeC7f4rl4Vv3cUuG3al6PUHNmtkBBNhIJIzifkhIJE3j3OTwrFgPgWTqffh/8B28kRLkn9jt4xOMDj82f0QAPsO3Etw54sBGfazJwbeVCryIy0v25G/CLqj6Os7zS3o34nI2iITW9dSLSX1VXAT8DY0VkK077rjGt3vurShkfNbfmlKnf8e6suwEIjTqUsmvuAo9zCaJfqhePQKSua0qeVneZolXokODhymGpPLmwCACfhrl1Td1XP7wrl+DZvJ7Ef91LMnBPnxLu7zeOFxc/jcf9DZWeeA5en7+5wm/z9IY9Nsiji+4C7qLu3ps73YmlDn8BHgXmu4lvFXAKcDZwvogEgSzgnkZ8zkZR796bInIRsElVJ7nNmm/h9NC5TlX/2XQhNj3rvRkfTpm0ha+zYr+jrZhxPf3KtwJQ+ru/Eh5xEAA3zcjjf0uKGVWwgmk/3ktyxDnuhz6j2P3eh5o38DbEs3oZs57+Nwu9mRyT8xNDS2tPjK8ilN77LIH//R2vW+MD+CJjKP8deSFjg79w5MHDSdyzdU1G3dp7b1Zye3HegnMNby3wYCMnvDat3jU9VX0+6v4kEekIJKhqUVMEZkxjq7kqgkcjJEQ1XUruFgC+3FDOs0ucCaVnpw9kr/0fZNyW2awLZJJ20GE81nwhtzkJ7zzHkVmzOdLdVo8HTU0HBCJhSEmj4rQLifTeDXwJMcceXriM0WfuCakHN3vc7Ymb4Hall2a71pDemzUb10NAyL22V3umXmNamRuGp/F1VjnlYWf7kqwv6FHhDErXxCTCI5wP2+UFwZjj1iR24fE+J5LuFz4Z3qFZY25rJD8ndjsSoeSOf9Y5eL/i1PNIXL4QCTtfPIJHj4NUe39N02pI82aEbUxJpqpt+iKHNW/Gj3UbtlI48TV6lmyh84D+eDasBr+f4AlnEek3GIDVhSH2e2cTFe5XOQEuHpJMSUjZXBphbP8kxg9J2eZzxDP/1HcJvFhdFw7vPpzS257Y5v6yaT3eBd+jPfsRHjayOULcaW2ledNsX0OSXs3p0HvgDIb8QFV3enqb1sCSXpxQJenPl+Jd+0tV0YrzbuHh9IMJeIVLhqbQP81p/PhyQzmP/lRIWVi5YmgKD84rjFli6KGDOnDpUJssuS6+bz7FN/cbIt37UHHi2ZBSe/UKz+plaFIK2rVnC0S4cyzptQ+7NA2ZiHQAvlfV3RsvpOZnSS8+eOZMJ/nxP8eUfZ65J8fu/ScAOgU8fHdGVzolxjZcLMgJMmbi5piyg7sl8PFJXZo24HZiyroy3ltVSt9UL1cOEHo89ge8KxYBEDzsJMovvaWFI6wfS3rtQ4OmIatDOmD/+a1ERJVP1paxODfE0b0CjOicsOOD4oh/1ue1ykqj/gWyyyN8uLqsVtNl92QPfg9ErzHbO7VNt+g3m/dWlnLRF9XX+T6dm8MsN+EB+L/6mOChJxLZfXj1PmvL+HJjOft08nPmbklVK1wY0xga0pHlRWKv6SUDhwEvNXZQZufcPDO/qtfhvT/Ac0dkMm5A25vNvsnU0arxSrdDYrY7JNQeDN050cuf903n7jkFhBX6pHq5dUR6k4XZnry4rDhmew6ZzE/pw97F1ZODeLZmVSW9pxcW8afv8qse+35zBX8/KKN5gm0nwuNG1lpayPve3F0asiAiCjysqje5278HUlX1rl0Mt9k1ZLqD5cCKqNtM4FxVvbYpAjOOrWVhbp6ZxymTtvD4T4WEI3W3xOaVR3h+afUHjAKPLyhspijbhtARp6BRnZALJYFhxetJDDtj8A7omsBJfeteF++64WnMO7Mblw1Npm+Kh38uKmJrWbhZ4m5LZm4q5/zPsjlnajZfbCgjo8aXCNEIGaHqqco0OYXQ3vtXbf97UewIqBd+LqYsZFcf6is8buRlOCstXAuc7v6cFR43cldnRi8HzhCRzrt4nhbXkHF6dzdlIKZuF0zLYcYm50P566wKikPKH0daLWNnhPccxcabH6XoifvZvTSLNK3gtjUTGSIFJFz5Bw7vEcDr2XZT2gs/l/DfJc4H9jebgizICTLJrutVWVUYYuzkrVVDQqauK+PZwzOZtqGM3HIncV2z/lP6ljurLmhCIqW3PhozTCHJF/v+J3gEr81EVi9uDa/mWnq423eFx42ctAs1vhDwDPA74LboB9xOjv/DudS1BbhYVdfs5PM0uQb9OYnIcSJyi4jcE31rquDiXVZJuCrhVXpvZWmd+2YEPIzfvfpalADX7mW9C2tKH7YXA8tiO6WMzZrJUb0St5vwAN5bFfvez9hUwaYSq+1V+nhNWVXCAwgpLMoLMu/M7kw4MpNvfNN4ZPmLVY+Hh+6N1ph8+pZ90oj+Ndy4Txr+HfxeTJX6LC20K54CznM7MEZ7EpigqnsDLwOP7+LzNKmGXNN7Emc5ic+B6KnUre2hiXRI8JCeIBRUVL/FfbfTgeKhgzpwdK8AS/KsI8s2eTxI526wpXp6LE89Vz3vm+plWX71sIUOCVLnNcB4VdffZt9UL+kJHk7rnwR9z6OsRyqBV59GQkF887/Dc+eVlNz/XNWwhjN2S2avTD/Ts8oZ0SmBUV3sb7gBmnppoQIRmQBcB0R/AzwIOMO9/yKNO7F1o2vIf+xvgFGqeraqXhx1u6Spgot3ST7hgf07UNmDvkeyhztHb3vGChHh5H5J3LRPmiW87Si/4AY0MRkATUmj4vz6XZa+a3QHursLziZ64YEDMkj0WS2k0ol9EhnXv7rj1LG9Apy5W3L1Dh4PUlqMhKpnvPHkbcX3/Zcx59k9w8+lQ1Mt4TVccywt9ChwKc7q6dvSqitCDRmykA3k7cyTiEgfYALQHWctpmdU9TERyQReB/rjzNJ9lqrmusf8EefNDeNMaj3ZLR8FPA8kAR8D16uqikjAfY5RbqxnuytCICLjgdvdcO5V1Rd25nW0hHMHp3Bi3yRWFoTYK9NPQs21W0yDhfc5gOJH38KzYTWR3gMgUHfnlZqGZ/qZ/+vuLMgJMiDdR8eA1fKieT3C80dmsiI/REiVIRl1rJDgr2NtPH9scvs5L8jffixkQ0mYM3dLskkA6u8BnM4rddX4GmVpIVXNEZE3cD6b/+cWf4uzruqLwHnA17v6PE2pITOyXImzMOBfgU3Rj6nqL3UeVH1sD6CHqv4gImnAHGAccBGQo6oPiMitQEdV/YOI7AG8CuwP9ASmAruralhEvgOux+k9+jHwuDsB9m+BvVX1KhE5BzhdVc92E+tsYDTON5A5ODXW3Mr4bHB6/PB/8gb+SW+AR5h+7JXcWL4nK/MrOJX1jN+7E3/a0JGleSGO7Z3IIwdnVDdfVpTjmzEV748z8K5cApEwwWPOIHjaBS37gtqaonyS77gCT7bzERLu0ZfSu5+p+uJRHlZGvJXFxpLqQZFPjsng/MEtP+1bWxic7vbSvIs6lhbyvjd3p2fOEpEiVU1173cDVgIPqupdItIfJwF2pg10ZGlITa9y+aBTapQrsN2Ruqq6Edjo3i8UkcVAL2AscIS72wvAF8Af3PLXVLUcWCkiy4H9RWQVkK6qMwDc9uVxwCT3mLvcc70FPOmu83Q8MEVVc9xjpgAn4CRVE0e8C+cQePVpACrEy3lrupMViAA+XqYfH8wpocDnNL29s7KUdL/w6CEdAUh66Ba8S+bFnC/w9rNEevUnPOrQZn0dbZo39iPHU5iPFBegbtKbvaUiJuEBfLCqtFUkvbbA+97cZ8PjRk6ixtJCuzpOrzLhufc34YzTrtxeBRy1K+dvTg0ZstAobTnut4KROGNJurkJEVXdKCJd3d164dTkKq1zy4LEtltXllces9Y9V0hE8oFO0eV1HGPiiHfp/Kr7Pyf3ICvQMebxAl9yzPa3G0qAjnhWLq2V8KLPaUmv/nyzv6qq5QFIUT6+rydX1ZirFu+NanupXM3e1I+b4GxpoW1o1r8mEUkF3gZucHsCbXPXOsp0O+U7e0wty5Yt29ZDpo1LT+rAQPf+oNJNdAoWku13eg2mhkpID5exIVDdhX5k9hKWLSshKWsdQ7dxzvXJHcmzv5l667h5C/1rlG3NzWVz1Hv4234+/rXaT0iF3VMijEvdwrJlW5o1zrqMHj26pUMwjaAhQxYGAPcBI4CYK8uq2rcex/txEt7LqvqOW7xJRHq4tbweQOUAqnXEdq/tDWxwy3vXUR59zDoR8QEdgBy3/Igax3yxrTgHDx68o5di2qrBg6kozcc/+Q0CHg8vdF7B+NwhZHuS6FeWzdM//48rh1zG0uQeHJfzEw+VTSN18D9g8GBC30/BN89pfFAg5E1AjzudLmN/QxebG7L++vYmMvszZ0knINKxMxnjzqdD1Hi9ewfD78rCbCmLMLSuzjDG7IKGdGSZgTP92MvEjtNDVb+s86DqYwXnml2Oqt4QVf53IDuqI0umqt4iInsCr1DdkeUzYLDbkeV73Kl1cDqyPKGqH4vI1cDwqI4sZ6jqWW5HljnAvu7T/oDTkaVqFlzryBJn3L/5sMLn68s4c2oOohEWz/o9g8o2EUbwopT/6lKCp12AhkNUhJXbnv6YpMKtvNdpFOsSO/HgQR25fJj1LGywshJnmEIoSGi/IyC1bcww1BY6spgda0jSKwAydmaVdBEZA0wHfsIZsgDwJ5zE9QbQF1gD/Dqqw8ltwCU409/coKqT3PLRVA9ZmARc6w5ZSMTpMjsSp4Z3TmWvUhG5xH0+gPtU9bno+CzpxZfFuUGump7LvOwgB3Tx0zPVx3srSxlUspHnFv+TX5K6MaPznpy2V3f2m/wMafmb+ajLKMYPuZLCqOt+R/QM8N7xbX4qwtanqICEj17Bs34VoX0OJHTUWGgFtWlLeu1DQ5Leh8CdqjqnaUNqfpb04sthEzczPyfIsOJ1DCrNIjJ0BI8FFuJ967881es4/tH3VJLC5ayZcQ0doyZHfqz3Cdw0qHqIwuVDU2wFgAYoDSkbisPslu6l5vV8ycsGjwdN70jSX6+P6ThUfublBE89r7nDrcWSXvvQkI4sq4DJIvIOkBX9gKre0ZhBmZ0XUWXSmjJ+zg9xTO9EhmfaNZFowYjyU3Y5ry58gl9v/Q6AvCXJzD/0HL7tNoaneh0HwJCSjTEJD2D/ghX4xJlTcp9Ofn6/T+0VweOdZK0jYeILSO5WQgceTegIZ4TT+6tKufabXPIrlEHpPl47JpNBHfwQDhF45q/4Zk0DhNCBR9fqKeubMaVVJD3TPjQk6aUAHwB+YjuZ2GqarciN3+bx/M/Oh/VffihgwpGZnNLP1tSr5PcIH694mmPdhAeQESohd/48bh9+Y1XZ4uSebPan0zVYUFU2vcNQ/n1YR0Z0SmBgB+tGX0soSNLffocnx+lp6Vs8lzKfj+KDjueGb/PId+eQXV4Q4o7ZBbxydCd8sz7HP/Mz9wSKf8YU1J+ABKsnWtdMW8nCNJ56j72rMd/mxcAjONN9HdNk0ZkGyS4LM2FZde0kovDEgqLtHBF/JC+bY9bNqFXeORi79mC5N4Gz9ryBVZ0GUOhL4rVehxM6/UJ+tVuyJbxt8KxYXJXwKvm+/5Lssgg55bFdAZZtdv4uPRtrT9wR2ucg1Ot8l9a0DlSceXkTRWziUYP+e0WkC3AuMB7YB6dzyvVNEJdpJPW8ZBs/VJ1OETXemOd6HF5rn1GHjqLzaGeiiZrTEJnatFNXVDxIVF+3SJce9EzxMjK0ibm+blXlp679CiLnERpxMP4PXkLc34d6vVSceSkVF1yHZK0lstswSKhjvk5jdtIOa3oi4heRX4nIB8B64ErgXSAfZ4LoN5s4RlNPnRK9nDeounehR+AaW1MvhnbsTOigY6u3gRcOvpznexxRVfb3FS9z1qYZ7GsrVTSIdu5OxRkXV9XSwr0HEDzFuRb31tKnOHfT1+xTtJo/rJ7IvYueh7ISIgOHUXb1XYR3H05o6AjKbrgf7dEXzehEZOgIS3im0e2w96aI5OAMM3geeEVVf3DLNwL7qOrm7RzeJrSn3psRVT5c7XZksTX16hYJM/nzH/h0kzKkXxcuPKA/X7/8JgtWbOCo3IUcXLCMcm8CP938L1KX/kCfDgH0wKNYFw7wyE9FrC0KMa5/EufafJB1kvwcJD+HSJ+BVUMNAs8+yKx5y3mq1/H4NMy1LGWPW/9cdYxn9TI0kIh236Ul35qU9d5sH+qT9L4AxgAzgJeAN1Q115KeaaueW1LM72ZUr5I11rOBt6bdXGu+ui3+NLq41/qKO/XksAPuZ15JdW/Yp8ZkcJ4lvnpZnFXIYZNyCbr93pK9MPOMbvT1VZD091vwLl8AQPCgYyi/8rZWMS6vJkt67cMOmzdV9QhgIPAp8Hsgy23qTMHpyWlMm/K/pcUx2++Hu7PFHzsrSIknoSrhAaRkb+D8+a/z4qIneXjZBPqWbeHtX0ox9TNxo1YlPICSMHy0ugz/lx9VJTwA/4ypeBf90BIhmjhRr96bqrpaVf+iqoOBo3GWCYoA80SkVS8N35YFI0ppyCqhjS3NH/uFPUFDJEaqV/PekpDBv3vUXinl+nWT+M3mGVy3fjJfzr2HfoFwk8faphUX4vl5PpSV0DOl9simzd/NwLP4x1rlkt3mG49MK9bg5YJU9WtVvQJnFfRrgeGNHpXhqYVFDHxlI31e2sA1X+cSiljyayx/GJFOIOoz+OY1H5IedmptL3Ybw5ADH+bmQedz2l43UexxOlIosf8sfcpzuC1xRfMF3cZ4Z39Fyg1nknzfdaT87teMyKv9Xr1W3h3/3G/QqKZMTUwmPOLA5gzVxJmdXiNPVctU9VVVPbExAzKwKDfIbd/lUxBUQgovLSvhxZ9LdnygqZfDewb4TW/ng1YiYf7Z8xguGnIlPyX35oohl1PgCYAIH3felwuH/R9Z/vQ616fq2t3m3ayTKoEXH0cqygGQkmIy3v0vAIfkLWXm7NvJ+vpKblv9LgCRTt0J7XsIwYOOofRPj6HpHbd5amN2lY2ybYXmZwdrlc3LrsC5jGoaw+x8LxBCPV6yE9J5qcdhfJ6xJ0FP7L9EVkpXHup7CkfnLuD4kmVImVMjDB56IpH+u7dA5G1AOITkVy1iQlZCByYk7kFaqJR3FjxMp5AzMP2qDdOcHTI7U3b9fdXHByvwfTUJz8Y1hEccRHgvW8fONB5Leq3Qwd0TquZ4rHREz8SWC6gd6pbiZUFeKKZsaOl68v3JFPmqp23rV7ieR/qczCN9TuZv+6VxlfzM3PIUcroN4LCI4vNYh75ayssI73MAvh9nEEY4Zp/bWJLSi/0KVlQlvEoqQsW4i2LKEp++B98PXzsbU96m7Io/ETrkuGYK3rR3lvRaob6pPl44MpP75xZQGFQuGZLCuAE2f2Zj+vVuSXy2vjymbGlKb4YXrSHPn0KeL4UxeUvoGNWD859LSpiY3I9vN1UA2ezZ0cekk7qQnrDTVwnanYTX/41/8psQDqGBRGYG+rIkpRcAS5J7UOhNJC1cVrV/8IhTCe85qmrbM29GdcJz+T9715KeaTSWR8iuoQAAIABJREFU9Fqpk/slcbJNFN1knl5YXKtsXSCTx39+jtOyq7vMP9q7+pJ1KIKb8BwLc0O8uryEK/ewWW8APMsWkPDxq9UF5WV09lbX7Ap9yVww7Lc8u3ICmSU5hPc9hIrTL8KzdD7arRcJbzyD/5vJdZzYPqZM47G/plbuteUlfLSmlIHpPq7bK5XMRFvUojEszqt93TQpXM4p2XNjys7aPIPfDzofj8CRPQO8uCy2Q1HNiZTjmWf9qlplu43YkysC63iutDuXb5jGwWWrKf7V5QQOPwrP+lUk334pnoJc1ONBInW/l6F9D2niyE08saTXij27pIibZuRXbX+dVc7UU7q2YETtwzdZ5QRrfL76BMo9fjb70+kerH7P1wY6AfCbgUn8edT/t3fe8VFV2QP/nmmZSe+F0EPvVQQRsWEvqCjoqii2XXVdu65rXV113V111VV/dl3sHRuCLFIF6UVKaCEJ6XXSpt7fHzMpk0kAMQkDud/PZz5577777tz75uWdd84995xo5mTVUu5PkWMzChf11tp4PZ6hY1EmM+JufKFwTzyNvw8ew8P//itJO/wphN5eiMOeh2nVYgyVZQCtCjwVZsU1WYf71rQdejIihPlgR2DEj1VFLnZWuFuprTlYPtoZvPzj5CSweN38qe8V1Bh8gYY8CHdl+AImz95Ry/ycOuafncT1AyOYkRHOB6fE0zdGByWqRyWkUHfb43gGDMfTqz91196DZ/AYqKshcc3/AupavngHY3br6xwV4EnpSt2tj0O4Nh9r2g6t6YUwybbAdxKLAeLCtLfgb8HtVawodAaVzy2EGwsWcXH+Ut8TF1gS05+lsf0b6jywqpJtl6RS6VK8v7OGD3fVMLN/BP84NgYJwViRhwPP4DHUDm62xMBogjAr1DW+bIh3/9FsBHBPOhPPwJHt0EtNZ0ZreiHMvSOjSbT6fiIB7hkZref0fiNf7KllS3nL2vJr6ScxrC6XcOWixBLN3RmXBhyvdnn5Yk8t7+2o8WkiCl7bWs38Zl6gmmaYLTgvuKphVxkO7rEjxfnt1SNNJ0ZreiHM4HgzG6alsKLASa9oEz2j9M/1W9lR2bp52CsGKp/4L+aNS4ia/SLbw9MCjtd54NM9wUGmt5a7OLWrXke5P1ynTcM99BiMe3fgTU7H9o+7kOpKAMqNNmI9gddVAc4zLjkMPdUc7einaIgTbjJwYrp+oLYVNmPrZkiXF3p8WUOcdxBTe8zgv5v+zQXD7sDtd5lXwKJ9DoQGCyhGQQu8/SClRRhy9+DpMwjVpQfuuESMW9az5ORreK4klh2WRJLLc3lt839Ic5ZTZbTy54zp7Og/gSesyfQ53APQHHUcMJ9eZ0Dn0+s8jPmkIEjbCzNAiysP6v83Wpiv6xFpJC3cyC1DIzmju/bgbIohZzemFQuQwn2YVi5EvB6ULYLai66h6JP3Sa8pxITCC/yl1yU81f1sFq19mPGVO6gxWBg69kmybMkMijWxbGrK4R5OAzqf3tGBntM7ylBK8fQGO8d/Ucj0+SVsLg1ej9aZKXcEO1C0ttRucHUO5xavbvFYXrWbWHHTLVIbS5pi2PkLtgevw/LlO5h/+qHBYWW9MZHh27pTKWZMfj3ZADy8+yPi3NV8Gz+CdZE9GDjun2TZfMtyfil3U1Cj0zdp2hYt9I4y/m9LNQ+vrmRjqYvvsuu48PtinB6tyAIYdm/jml1ft3ywucVDKb5Z9wQfb36G17a8hMkbqB06lfBdgZcL5uRSp3MeNmBe8EXAOr16buo7kzhPLUOrcwLr4yXRZWdgzT7uyLiM3LD4hmMGINailStN26KF3hGM3eXlXxvs3LikjG/3+hwBvsuuC6iTX+tlbXGwi35nxPLxq/w1czbHlW8NEHIzCpbyrx3vYPX4rlOUu4YYdzUPZUzj0R7n82nSMYy2726xzUKvmTV5VS0e65QYW163OKV0A2XGcDzNkjQVmyIZW7mLaUU/scsWaMr0ArVa0dO0MVroHcHMmF/CI6srmZ1Zw4wfSnljaxX9YgLNbWYD9IrWJjgAKStiUcwAlsYOCJinO656N3/MnUvV4qtYtup+6gxmKsyRvJE2mUd6XcTXiaNYEdOX7nXFzMz/MaBNk9dNhlkvWajHNeVClDH4fnsg6zP+lPMt/+zWGF3FC5i9Lt7a+iIm5eV3eYsDzplcvZNYd3CMVI3mt6CF3hHKrko3S/IDNbg7fqrg2BQLY5N8b9sRJuHJcbEk2/TaPgD3+FPZEZ4aVL518OSG7a8TR+IytKytJDorcGAi2v8gDvfU8Y8dsykyRbVLf49EVEQUtLLw/IqipTzSbzrHjf878ybPwj1mEtHexheGh7M+5sHdH9G3Oo++NXmMLdlMzZIFHdV1TSdBqwBHKE5PsPeFR8Fja+ysujCFvVVu4sMMRJr1e009rrMv5QTLt1hz3dRJ461/Yt7P5I47k+hVP5DmLG/1/G3h6ayJzmjYf2TXh9y073t2cBMQ1p5dP3LwepFWPMLF48HsqGVFWDrnelNYv3cpTdPwCpDuKCMzwrc+8qnu57K8qJyv2r/Xmk6EfiIeoXyRVddi+d4qn8NF90iTFnjNESH11NN43raZ8RXbmVT+C+9vfpbz1n3Mlq17eCb9dKweF2MqdwBwaukG1q28m6Il1/Ls9jdxNEtxMydxDLmJvUhNjD4cowlJVEIyrrGTWzx2b8Z07P4EvS6DiR8NwVr3S11OCdhf4opl934CCmg0vxat6R2hlNS17Gd/fk+9Zmx/hL36d2YunctM/75bDFw4+E98kTimYZ5vYvkWlq6+nzH23Rj97vU37pvHloh0Xko/taEtq3JjvfepDh5B6OPNGIhatwxczgC3lS8SG2NyXpq/hEsLlwWc92NMf+qamZbNBojWHpyaNkSrAkco0zPCMTV5FpgNcOOgCJ6eEHv4OhXimJbNC0pS+nz6aXyRNDbAsWVJTH8SnPYGgQe+jAtzEkcHnDs3fhifloe3b6ePMCQ/G8sHLyHNBB7ApQVLAV/ewucy3yTC2zgnrYATKraxaO3DJDsaTcx3Fi0g0V7YAT3XdBa0pneEMirJwrdnJvFOZjVRZgPXD4qgu14ovX+++aBh8/OEUcyNH857KcEJSp/b/gZ9HEUBZaujegWsIatnZX4tM/powVdPXdZu/t3tHFKd5VyVvyjg2GO73ifNWc7m8HRimsXarBeQsZ5adv70J+bFD6NPbT6DanJxzd6C409/66ARaI529FPyCGZssoWxyZbD3Y0jhjInxIuRnLB4zi1Zy/kla+hXk8edfS8HoE9NPu9vfpYR1XsDzlPAdluab21fE43Q7HUR9dMimDijI4cR0swq7sVXvTP4asOTQceMwC053x2wDZtycW5JYyQc05a16EUhmrZCmzePYBwexbd7a1mU50DHUD0waydM46luZxPjrsWA4u2U47knwyewRCle2fZ/jKjey5bwLryQfiqLY3y59F5PmcTFhct5YM+nGJRvLrVnbSEbVt7NCftW4vbqaw9Q7vDydYHvpcDTyqPlkK6US4fS07QdWtM7Atljd3Pn8nIW5jlw+f1ZJncJ45NTEzAa9KR/a4w481RuKO7BHdnf4BYDd/a5DE99BgURBlfn8FniGKYP/iMe8a1t/NPer3kjbTLhXhcPZH3KzPwfyQ5LYJx9ByblxWg0YNLXHACrUYjy1FFpsvFp4hhi3TUcV7mdOjFhVT4PTAGqDWH8u+tpbIroxpSyjVzZzAzaHE//YR3Qe01nQQu9EOP9HdXc8VMFVS5FpEl4anwMM/pEBNS5dH4JvzRLhLpwn4M/r6wg2Wbgwt7hOvdeC0RbDEwdmsrrO0/gyvzFlJkCr+vimP78rcf5eMTIqaUb+H3uPOoMZpbEDuDxnuczrWgF3R0ldHeUNJzj9Xopq3ESF67NzFaT8GjhN9ySNpW3upzIW11OJL2uhDJzBB9veoYpZRsB+N2gGxucgj5ImUCBOZq7sltfjee47KYO6b+mc6DNmyHEHrub3y8up8rlMwJVuRV/WFwesE4pt9oTJPDqeXlLNX9dY2fExwXMWljSYp3OzvBEM/f1upjVkT0ZXbkz4NjDPS+kzBTBmMqdfLXh75xbsoaLi1Ywf91jlJvCebjHBZQZA5eELI4ZwNJCvY4MwLh5FX/Y/ik35n7fUJZrTaDGaGVDZHcA9lrig7xg306d1GqbyhqOSk5vnw5rOiVa6IUQq4ucQXMeCvi/LfaGfZMQsFShNT7ZXcfnu4OzfHdmat2Kr75exvYVt3LNwOtADKxYdR8Vi65m9ubn2GNLZo8tmQuLVgYsV4j0Onh01wesiskgbcKLXNvvGj5KGsfHicdwd+9LGBintTwAQ5ZvUf9x5du4Nftr5q17jGe3v0mSo4xTSzdSazBz1vC7g85LdDXe3y6g0mhtuPpSV4P1X/d0QO81nQVtAwshRidZArJy1/PiLzWk2Iz8kOtgSX6wYGyNH3LrOL+XXqxez3ubS3h1/TPsDUtgT1gS6Y4yThpxP2HKxR17v+KxXR9wc7+ryLImBp3bvzqXiwp/4pSS9fyt5wW80eVEAN7f9Ax9qm9CxfTo6OGEHJ5Bo/Ag5IXF8dTOdwE4sfwXppRtoG9tAe8nj2dLRNeAcyweF4/s/qhh//2U47jcv56vHuP2Te3feU2nQWt6IYTNKPSLaTk49EOr7Sz+FQIP4Ku9tZTU6dws9axcl8nbqcdz+rB78IqBxbEDqTZZKTVH8eeMGQ3pht5KnUShOTCI9Liq3czK/5EhNblcnzufFavuo3dtARl1hdTN+fBwDCfk8Pbsx8a+x3FS2eaA8r61BQBMqNjOVXkLA9I6Xb9vPpMqtgKQY4klzREc+1SZWw4ArtEcClrohRD3/VzBtoq2E1JlDsX7O7WJs575hnTu6HM5xWExAevt6vk6cSQmr5t4l514V8spbU4r28jK6AxSHWU8m/kWDoOZ3FKdT68eY9deQZpytcFn/u3uKOGVba/w+K73SHDaiXfZuTXn24Z6Fq+b1BYCfnuT0tq305pOhTZvhhArCg59CW5LZlFAryFrQg0+LdrqcVBnDMyKYPK6qTJY2bv8ZuLc1diNVuI8NS224zSamTDmUZatup8UVznPnnQ117d7748MHpWhbOs9iBFVWaQ7y6gTM7UGc0DIsTuzv+a27G+oMloDIrMku6tIdge/QKiYuA7pu6Zz0CGanoi8LiKFIrKpSVm8iMwTkUz/37gmx+4VkR0isk1ETmtSPlpENvqP/VvE97ouImEi8oG/fIWI9GxyzpX+78gUkSs7YryHitV06D+HrRXvlksydIiseiKMvmt0U/Z3ZNTkN5T3qi3g2tz5JLsqmTL8XpInvMT0wX+k3Oi7du4m/yYrozJYFt2P3LB4PkiZQL4llrSRIzt2ICHMKmcEmyK7k3HsM4wf9QjdJzzXsOaxKUZUUCiy1igurmjrbmo6MR2l6b0JPA+83aTsHuAHpdQTInKPf/9uERkETAcGA12A+SLSTynlAV4ErgN+Ar4BTge+BWYBZUqpPiIyHXgSuERE4oEHgTH4FKHVIvKlUqqs3Uf8K/EqRXbVobu+17hb1uhSbNqCXc+geDOrsiuJ8DrZ2SSZrEJYHDeQTZGNzig/xA+l+4TnGFCTxx5LArfsm0ueJZZ3Uo4HEQzKi1eEeHcNZ3bV3pv1jLfvICtuDG6DiZ+jMxhQnUuyqxIFQQGoD5aEgt04m4WA02gOlQ55IiqlFgGlzYrPA97yb78FnN+k/H2llEMptRvYARwjImlAtFJqufLF3Hq72Tn1bX0MnOzXAk8D5imlSv2Cbh4+QRlyZFd5aGufkySr4dDCPh2llDu8DLHv5e20wHVhe2zJAQKvnhqjlTVRvSgNi+bBXtN4Kf1Uqk1Wflp1H3U/XsFNOXPJscTx0ndrO2oIIc/v93zNxYXLiXVVM7F8Kx9sfhYBHBh5ots5+z13T1giS2L6BZXXmcK0wNO0GYdTDUhRSuUB+P8m+8vTgewm9XL8Zen+7eblAecopdxABZCwn7ZCjvQI4yG/CbdGUZ2XW5e1ngm8s2EUUAZhty0lsFx5MHgP7o0jwl1HviUWAwqrctPFWU7ezuwDn9gJMGzbQO+aAh7Y8wnFS69j4bq/MrgmFy9gxcOw6r3clvE7lkf3ZYc1Oej8no5iTht6N9/HDQkobzlzpEZzaISiI0tLz/7WrCMNa1gP4ZwWyczM3G/n2pMTEywsKGnbn+S/26u5Mr4YHZUMumzfylcJowILlcIjRsZU7qCXo4j5cUOxG624xdiidlFtsnL+sDtJqyvlo83PEOeuIb5832G9b0KFHp+9g8tgYtiYJxhr382Eiu3khcVyfPlWrs9bwJml6zEpL7tsyYyvDL5em8PT2b3iVpJdlTjFyLbwNIZW5xDtqmF9CFzfMWPGHLiSJuQ5nI/CAhFJU0rl+U2X9Zkic4BuTep1Bfb5y7u2UN70nBwRMQEx+MypOcDkZucs3F+n+vbteyhjaRMKNhcAbRvSymAQ+mRkEG3Rc3t3Zz3JgthB1BitjYUiPJ35Fjf7Q2eVG8M5ZcR9rIvq2VDFoLwYlMJtaHTIyLPGc/yoh7k9aw4bYjK4/TDeN6GCCg/n1bTJeA0mVsT0RYnQo66In6N6c33eAgD6Vudysj8GJzRqcSuj+xDtriHZVQmARXnoWVfMbRmX8c+dsw/r/6Xm6OJwPgm/BOq9Ka8EvmhSPt3vkdkL6Aus9JtA7SJyrH++7opm59S3dRGwwD/vNxeYIiJxfu/QKf6ykMPtVWwua/sYjqd1s2qB5yejpoC7s75sSA8U66rmw01Pc2PuvIY6sZ4a7tn7RcN+uKeOn1fdh7RgIFAivJp+IoNHDwk61hmZG5bBd/EjAHh2+5ssW/Mg7/3yPP/cObtBuPVyltLUl9MA7LPEccKI+/k6YSQ5TRL1RnnqmLXvf2wKD8kZCc0RSodoeiLyHj6NK1FEcvB5VD4BfCgis4C9wDQApdRmEfkQ+AWf2nOj33MT4Pf4PEFt+Lw261e2vga8IyI78Gl40/1tlYrIX4Gf/fUeUUo1d6gJCbKr2idyyrUDIg5cqZOwKaIbr6WdyPy1j1JrtDChMpMoT11QvVh34/q86QXLGV69l3RHKXtswfNQKc5KzozVEUMAKirs1IZb6FZXzO/3zW8oP9DShK7OMiI9Du7NuJS/9ZjKorUPM7TaN086uHYfyyJ6ByXw1WgOlQ4Rekqp1lJLn9xK/ceAx1ooXwUEvVYrperwC80Wjr0OvH7QnT1MpNra5x/6piVlrJ+WikE/MPhbj3MZXJPLxMrt7M+v9b8pxzVsD6jJBWD5mgfYZU1mcUx/sq2JvNDVt3z0gsKVzF3Zl7FDM9q380cAmyN7MCvnf7yRNnm/17c5WWEJVJp8MWLtJhvPp0/h5e2vNRwfWpOjBZ6mzdDuDSGCzWzELOBq4zUG2dVefsit49SuOvD0uojubFh9L6ujejHWvquhfGN4V95LmcDl+UvoV5vH/217lWMrMrml70w+SB7PrTnfkuSyk+SyM87uS0dk8boAIcuWyCnbF+NbHdO5CRs8jJmrXiDRZWdTeDpD/C8M0LpXmUNMzBpwPUoaTfBeCTTHhysXOpiepq3Qkz0hxH0jow5c6RAortVBpwGOse9iTWQvzh16Bz9FZaCATFsqMwb/kR22VAbW7sOIwqI83JC3gEsLlrI6OoOt4V2C2npy13uYlJt3U4/nraSJHT+YEOT8ktXEemq5tHApg5sIPA8wP3ZwQN08Syxvp0zkm/gRVDcJCWf1OLmhyRwr+ASjRtNW6LspRFBK8cqW9glc3CVC/8wAV+f/yB/6XU2RJYaJox9hUHUOd2V9ydaIdC4pXB5U/7zin3knbRJjRz/GGSVreWfLf7AqN04xYlQe0py+8Fi96wqDzu2MVO/cgSL4TdoInFremHlhU0RXJo18kEqTL8zbSaWbeGnbqxRYYri4cHlDVoaG+uFdGeT1gKHlDCQaza9Ba3ohwrYKN7m17RM/ZUOpq13aPdJYGtOPfdaEhv1fIrpyZ4Zvuvnb+OHMSRjJWn9kFg9wV8ZlANQZLXyWPI4hx/ydLxNGccLIB7kr4zLcYsTodTO2aneHjyUU2ZY8kDLTgR2nnks/vUHgASyIH8Kwqizuy/o8SOCtiehBvNuuBZ6mzdAqQIiQbG2/948ekfqBATDX707flKKwOM4vXMmiuIFcMvgWBlXnUmW08NrWVyiyRAfU3WNL4YKhtwOw05ZM4dIbOK1kXYtJZzsjMQU7yDXHEO9uOS1TPc4WBJjT0PgocmJgfuwQ/tX9LBbGD+HNzS80xBvUaH4rWuiFCDnV7Tfvdno37cQC0NVRwvaI4Pm5z5PGcox9J59t/CcprkoqjDau7X8tqc4K7KaWs1SkOH2LqAfX7qPSENZinc7GCncsUx1FAWVFpgj+2O9qdtmSOb/oZ+7aO4cbcufzQfJ4nAbfUo9RlbtIaZI81oKXR3pfxKroDJKcFQFr9zSa34oWeiFCajtmQ7AYtbs3QP/qXJbEDqBnbVGg8BPh6cy3SfFHAwn3Ork1+xv+0O+qoDbMXjeTyrdwd9aXDWW9HcXt3vcjgVhXFTZvoCk92lPHlwmjcBgtrI7qTZkpgr/veo9Vq+7jg+RjSXVWcHnBYu7pcQnP72pMwlLlj5pzYtlmzi1e1aHj0BzdaKEXIiSHH/xPIV6Pz8X7INcu2V1eosx6+nZe3FB+WPsoK2L6cEefyxvKU+rK+CZhBG+nTqJHbSHPdj+TAkusb0F0E/rU5DNv/WN0c5QGrEJbEdWbkzpoDKGMw2gNWJqwx5qIUXmZWLGdH+J9y2uXxfSj2hDGgJp9PLznEwBqxMRzTQTewpj+bA3vwu1Zc7CbbES3EEBAozlUtNALEX5VhnORX7VY94PMaq4Z1D7LIY4kzHj5c+/pLI4biEF5UAhKDBSExfKPbmejAKexSW68Ztf4L1mf0s3hC+gjgBdhXXg37sz4Has7bhghy5TC1eRY4kh1VXLpoJv4LOkYAKLcjavs9loTiZn0Ol3rSnhh++ucVbqOcBUYfu+4ikwuz1/MlLKNnFy+mQUxAzmmQ0eiOZrRr/8hQqnj4BOoKPl1P9uHu2oOXKkTEOuqZnHcQAC8YkSJwReHUwSH0RIo8Fog3RGYe9iA4ppBN7DXltRufT6SMHrdWLxuPkge3yDwwBdlBSDeZWeff34ux5rAlQP/QI0h+Jqb8fLGtpc52b/MYXSTQAIazW9FC70QwduOScP0lJ6Pbs7gsKvNo3+Yva0H/X63SXgygI0RXdkQ3g33r3wJOVrZFtGVOE8NO5rlKwS4O+tzetcGrmcsN0ewydY4t1pptLUoBCO8jrbvrKbTov9bQ4RP97TPwnQAq+iILADnFAcbIY0q8G1jhL3Zmrsm83pvpE3mygE3MCdhFM+mn84Zw+4Bg/4Xqqd3TR6VRhvnFq8OnA9VilNLNzGpfEtDUZjHSbe6YlwGE24xMKv/dSRNfJk3U44PaldfYU1bouf0QgS3p/3UMQ96nR6ASwy8sO11buo3s8FEHOZ1+bQLESaV/cI/Mt/h0d4XMidhFDHuWo4v38KcpMbkofPjhjA7NfDBnOC0d+g4QpXFsQPJjExnjH13wHzokOpstoR3YWt4F3rUFvDErg+YWrQShZAdFs+ayJ6cUrYRlxi4wZ93rynaUKFpS7TQCxHO6WnjgdXt8/Csbvs0fUckd/W+jKvzFwbMidY0ift4df6PeA0Gzi5ewxtbXsLmdbLbmowBRaYtlZ51hXybMNJXWSlMysM5xavJM0cDg+nsFJqj6eos5aUujclTphUu57nMN+kz7hmqTDZu2/s104pW+I8qejuK6e0o5hj7LqYWrmwxO0P7xCnSdFa05SBEaOvsCk3J1VIPgApzBE93P5uHd33Iw7s+DDo+O3kClcZwIjwOYjy1bA3vwugxf+OLpLH8EtmNbxNGNgpMEYZWZ3Nc+bYWE8x2Rm7NmoNHjLyVNrmh7Kac7/khbghVfmeWUfsJ2WYl2AzvrtfzlL7GmrZBC70QYW9F+wmm9nSSOZJwI9g8Dl7oehpvp05Cmj1I5yUMZ8rI+/gycTSLYvpzwsgHqGvi0dncazY3LI4TKrfi0o4sAISLm/nxwwLK6gxm+tTkN+z/L3bQfttwNTFmLo/qQ4Elhgqx6Xx6mjZD/7eGCFsqnO3WtllP6QGgDEYqzRGUmcJJcVZwTvEqutSVYlQeDF4PYR4nKMUHyeM5acT9RHj27zUY6a4l3VFKmqOig0YQ2iS67AzxZzwHMHo9RHjqGF21h9v2foVReXg9bTKzkyfgBTwtzNaZUXiB72OHctqwu7m+3zUsi+nTcYPQHPXoOb0QIawdXz+Sw/W7TT1mjwuL8rAstj8mr5snd76LXSw8knExjmbu8oOrs1EivugswPiK7Yyt2E6/ukLiXNW8kXoCt2ZcTq455nAMJeQwobgt+2ueT59ChSmcl7a/yrH+pLt/3/Uet2d/jd1ko48/k4IvDa80zOO9ljqZR3tO5fp9P3DP3i+ZVLmd9VE9SXeW+cybWtvTtAFa6IUINa72s0GW1Wr7JoBBeYn21FLiz57gNph4tOcFXJM7P2i9HkCpOYpr9/1AmqOCZFcFZxWvwWUwEeH1aeXTilZwyvA/syGqV4eOI1QpxUqVOYpycyQP7/6IeHc1j/SYyuX5i+npKCbFVUm0p5Y5CSNJctk5tnIHAAXmaE4ffg8b/Wmd/tL7EnrX5vOH3O+ZOvR2Btfuw+H1glGbLDS/HS30QoSsqvab09urA7IAYFLBBrVyU3irEW7WRvVibVQvzixZy5cb/wGAxdtohjagmFG4jEUxA9qry0cU85JGMKh6H9+tfZTd1hQGVudwTvFq3ko9ga8SRtLFUcaHyeMpsfhC4p1UYZ2lAAAYT0lEQVRUtonv1z/O3PjhDQKvnkd6Xsho+27mrn0Mk/Li0OshNW2EFnohQkFN+2ljWs/zcd2+H4jx1PJYzwsayk4rXc/HyccGVmxmSvsmYSQF5uiGLAxNybfEYlXaOxbgrZSJXJu3kGsHXk/OshuJ9Dp5puvpAcG9m7IgbgifJY5u0aNza0RXtkZ0pW9tPuPsO7V5U9Nm6NenEMGl3d7bnZn5i3hwz6c8lflfJpRvY1rBcnLNcVQawphRsJRjy7e36Bpv8ziI8Dr4KHEsS6P7NpSXmiJ4IX0KRqUj3gDckjOXm/rPYox9N5F+jXh2ysT9nvNd/Ag8CCMqd2PwBl/HBbGDMaDwuFwtnK3R/Hq0phciKLfWx9obu8mGAcX3CcPYFt6FiRXbOL5iKxcVrSDGXUvP8f9uUZtIclZyXd9r+DB1AgBDqvZyYdFPfJJ0LMWWaLrWFHT0UEKSgdV7qTCFk2+OxYvvjTqpBe24KSeXbWJ4dTZX5y/ktj6XB1klxlXu8IWKayHbukZzKGihFyKYjIKOPdG+fJw4ljRHOYtiBrD+53voU9corHZYkymytOyFudeWxF5rYsP+psjuZIansnzV/Vwx6EbqdJg3AErNMSQ67dQZTKwP70bvuiJyLPGgFFfl/8iJZZvZHJHO093Owmkwc0bJWi4sWgnAH/bNx6w8/KHf1Q1zrNMKl/PQ7o8xojAatGlT0zZooRcieOuc6J+jfckzx3DZgN9zZunaAIEH0NVRSrjHERCWrClRnlrsxsZF0grhpn5XcU/W59zf86J27/uRQJXRyrnFK3lm52wsfpNv39p8Li5awV+yPmuod2nBUpJcdpJdgWH3phcuZ27cUBbHDkRQZIUl8lHKeC4rWIK7thZTZGSHjkdzdKLn9EIFgxZ47c2XSWP5duPfObcoONvC/PghrQo8ALspnEf8ocui3TW8v+nf2Dwu7s24FJtbp74BiHXauT7vfw0Cb01kT+YkjmZm3sKAeoNr9gUJPICb+1zB58njKLFEU2yJAYTX0ibzRfxIDDrCgqaN0E/aEGG7DurR7pxTtIoF8UN4I20yveqKmFi5HYAicxS3ZlxxwPPPKFvPuLWZnFCxDRNezi1dw4tdTmFJtI4YAtDVXYHN2ejJGuFxYPM4KTZH0dVZtp8zffxj57tsjOzB+qieAKyM6UPW0hs5bfg9LKyuwRpma6+uazoRWuiFCNnaOa3dGVqTw4xBN6PEwMkxf+GM0nXEuGuYkziaKqM1qL7N46DWr/2dU7yakVVZQXWu3/cDm8OCk6Z2RrIt8YyozWnY71+bxx3ZX3F/70v4aNMzWJWrwcGlJRLdVfx3ywsMPeYpADJq80l3lZPuKMWoEwxp2ght3gwRErX1pt1ZG9WrwUnCYzDyVeJoZqceT6UpnHOLVxHfzOQ2wr6HF7e9yhcbnuLjTU+32KYBhVevHwNgX1hskGg6p3g1c+OH0efYp3kp7WS87N9dq68/OHWqo4yXt71KtVgYU7kLY3R0e3Vb08nQml6IEGmDgvZLnq4BSo3hQWVD7FnMKFzObTnf4BYD/+lyKnf3uQyA5bH9uS/rc04v29BqmzmWODyi31gAzMobpMmNqN5L7rI/ICgSXYE3eEtaX3ZYPKK8OA1mdlmTmVy+hfOLf8bjnIXBpB9Xmt+O1vRChO6R+qdobypbMGFuiurBEz3O492U47B5XdyW8w1X5P3oO6gUD/S+uMVsAAAbw9P5c6+L6V1b1J7dPmKoEzP39p7OnrDEgPIklz1I4CmCHz6VBiuTRj2IEgOl5kh+338WOWHxjKrOQrWwcF2jORT0kzZEOLN7xOHuwlFPv9q8FsvtJhs39Z1JmSkcAR7f9T43Zn9LuKeOPEssxlYMcj/EDaVPXSED61put7OhlId/dTubk0b+hWpD656wQIuvEQ/2uoi8sPiGfY8Y2RjRDYUgpSVt3FtNZ0ULvRBhQ0nd4e7CUc/pJesQ1XLkm1pjGEui+gGQ4qrk2Z3/pauzjCJzFPktpA4qMUXyYvqpPNHjvMbs3p2cbo5SFLDXmsTkkffzdfyI/dZv+irRkhYe6a5lfGUmLjHitWnPTU3boIVeiLCmpP3CkIXpZzIAu6wpzFv7KNGual/C2CaI8vJi11PxNhFgD+/+GCXCtQOuo8Dsc6TItsSzLLoPOZY4bs/+mkhPHe8mje/QcYQqFhqDQq+N6sUDvS5q1WnFi3Br79+xNLovqyJ7kmlLpYuzHIvXBUqR6Kzko03PEOuuQZQXg1ULPU3boGeGQ4TpGVbuX13dLm1Hm9ul2SOOXo5iujnLqDTZoFk6ISUGvk8YwVnD7uLSgqWUmiK4qGglm1feyeKYARSYozF5PSyJ6c+MouUADK/JprujmMfTzzocwwk5+tXlEeZxMqV0A89mvkU3Z2mLOrACbup7BWPseziuMhOA9RHduTdjRkOdYks0e2xJUO4L7B2t9JoeTdughV6IMD7NBrSP0Du3Z7DpqDOSGxbHxsjuQQKvKfPihzEvfhgA/+h+Dtft+4Hrc+cTplzEeOq4uOingPpTSjfyeJcz27XfRwo5xlhm5i3k6vwf6e4sbbWe4PP0vLxgSUPZ+mb59ADWRfbEKUae63oGd4dH64eVpk3Q5s0QYXSipd3avm24XuME0Kc6n4FV2VibmTbraZjv86cXyguLY3VULzJtKYwb9VcGHPNPtoWnBZzjFCPHVWxv134fKSyNGcDTO/9L8gEyKwAcV7GdWkOjCeKE8i2YvIF5CU8p24jdEIZZuYOOaTSHihZ6IYKzHTMLLSvQpiGAZE8VOWHxjK7YyQj7HuJc9sYcbko1LFyPd1bSpbaEGKedZGc5N/e7ih0RXdgRnsqqyF7UiU/n8ALbbWk4jNp+DJDmLqPUGEE3RynF5khu6XMFJ474C2+mHB80txftrqWqSazTLs4yTi9Zj/hfOLrVFTO1eBVbItK5Jn8hytR+L4WazoW2GIQIYUbhtK5hzM1p++DFvaL1zwzwesrxvL39ZV7w2FkV3Ztj7TvJCotnjzUZDI3vf4Nr9mFWbhbEDeb1LicHtDFr4A0si+nPWSVryQuL4byi1Zxasq6jhxKSDKzJJcHtM9FfPPgWFsUOAmBx7EAiPA4uKl7ZMMdXv+DfDdQaLIwd8xg7wrs0tDW8Kgsv0L8mj0R3VTsZ/jWdEa3phRCvnBDP9QMjSLYa2sQJXoDrBkYwJkm/JQMsTR5Ov2Oe5qacudy+dw7PZr7BrLyFAQIPYLh9N7tsjfE0w7yNmrIAXZ2lTCnbwHV5C9kU2Z3t0T07ZgAhTqS7DhOKInNUg8Cr55oB1/Ns2ilB55iAKK+TywuWNpiX41xVPLDnUwz44nGWGsKDfiON5lARpXTi0oqKipC8CM9vsjM7s4ZKp5d4q4Er+0Xw0KoKqppNbxgFRiSY2VDiIsEqPDEulsHxZiJMBrpE6BBZ9fzzxz38daepwa1+culGrsn7H/f3ns5uWzIAoyp38e2GJ0hwV1NqiqDcFI7F6+LFLlP4IW4Qq2L6clbxGs4tXk1meCrLovtw8dhezJyYcTiHFhLk3v17euVvAxG6TniBUnNUw7ExlTtZtuaBoLfsHWFJ9HH4Itpk2lLYbktjUsVWwj0O6jBixc2S6P6Mfu7lDhxJy8TExOjFP0cBWugRukKvJb7YU8stS8sodyq6RRiYNTCSGwdHYtaZpQ9ITpWbMR/lkVxbwr8z32RCZSbfxQ1j5sAbGFizDwcmyi0RFFtimFawjJPLNrPbmsAuWyo/xg2msD6zutdLjKcGg1JMLVrJSZdP4+weeh3ZymXrOP7lWzEAHyaP57r+11BrDCPNUcabv7xApNfBOPuuhvrZljhQim6uctwIHgzUGcysjexBVlgCkV4HPRwl9DvjdAxnTz98A/Ojhd7RgRZ6hJbQS34jF6fLb04zm8HVihOKAZ8nhRFoKSyhyafRlF+V3j4dPUIZeOWdmEZfTJjXSYqjgnXRvYl12Tm+Ygvd6kpxGM2sjM6gZ3UBBWFxVJgjGGnfSRdnJVttXSg3h1NqjCDS6yTPHE1OUgabpqUQb9UaNUDBQ7fzU7GL3IhU+tQW8FNkb2JcdkqtsbiBnjVF1JrDUUoxN2E4k8u2YBIvFcpMlHjoWleEEiHHEk+Cq5Lqop3c/Nbswz0sQAu9o4VOI/RE5HTgWXxi4lWl1BP1x0JF6MW+kQ1KGsxvvxmltOBrwsR/LGBLXB88Br+AUirgWtvctdQarb4ypTB63by2/RV+V7DUVx34Pm4olw26mXKzL1aq0etm+L51LLj/vI4eTsjx8swreWbsn5i38Un61+Yj+K6ZAPnmGM4bchu7bSmUWnxmT1FeHtjzKQp4rOdUPGJkQHUuP659hDh3Fa+kncSXCaOYUvgzs556+DCOzIcWekcHnWJ2WESMwAvAGcAgYIaIDNr/WR3LA0ty21bgQUNbsW/ktl2bRzCFkSmNAg+CrnWtydZYJsKZZesbBB74Ht7fJoxoEHgAHoOJLalD2rPbRwxV3UYyzr6LAX6BB42BpVNdFXR1ljUIPPBFwXmk5wU80vOChvRMWyPSebrbGRiA6/MWUGuy4jYHp4TSaA6VTiH0gGOAHUqpXUopJ/A+EFKv5m/upG0FniaIKtOvi0zTt6YgqCzWFew8791PhJfOxMbI7oyx7271eEkTx5Z6lBiCIuTssKU2bGfUFlAYFkvutp1t11FNp6az/LemA9lN9nP8ZSHDthnJDZFA2pokPd0EQFpd66GxWuKrxJE4myWILTNHBtULd+sMGQAzchfxZuoJrQaZ7uYoDipLdFaS4AyM4DK1+GcAagwW5sYNoVdNPun9tXespm3oLKuWW4t7G0RmZmY7d2U/iBFUG6+pEw/fjHce3nGFCCmr5+MYezq51kRUwC2hCPO6GFuxg20R6dhNNuJdVVg9DqYNvoU7sr+mX80+7EYbU4tWsiGyO8ui++EVId5ZyXcnW/T1BXrffCNT3/6OP2b8jqd2vU+YcuMUI/vMcSyOG0CRMZI/7v2aOYljKLJEE+Wp5ckd75LkquTOPpexx5rM1KKfObl0Ez9FZfCvbmdwZ9aXOPNzQuL6jhkz5nB3QdMGdApHFhEZDzyklDrNv38vgFLqcQgdRxaNRhO6aEeWo4POYt78GegrIr1ExAJMB748zH3SaDQaTQfTKcybSim3iNwEzMW3ZOF1pdTmw9wtjUaj0XQwncK8eSC0eVOj0RwIbd48Ougs5k2NRqPRaLTQ02g0Gk3nQQs9jUaj0XQatNDTaDQaTadBCz2NRqPRdBq00NNoNBpNp0ELPY1GozkIKioqlF7edOSjhZ5Go9FoOg1a6Gk0Go2m06Ajsmg0Go2m06A1PY1Go9F0GrTQ02g0Gk2nQQu9EEJEnhKRrSKyQUQ+E5HYZset/uNDm5TdJSIvdXxvQw8ReV1ECkVk037qdBOR/4nIFhHZLCK3tFBniogsFxHx7xtFZJ2ITGjP/h9NiMg0//X1ikhQ9lV9L2sOF1rotTMiEvcrqs8DhiilhgHbgXubHlRK1QF/Av4jPtKB65vX+5X9O5rSS70JnH6AOm7gdqXUQOBY4EYRGdS0glLqeyALmOUvuhn4WSm17FA7dpRd5wZExCIiES0c2gRcACxq6Tx9L2sOF1rotT+rRORdETmpXnNoDaXU90opt3/3J6BrC3W+A/KAK4CngYcAk4h8IiI/+z/HAYjIMSKyTETW+v/295fPFJGPRGQO8L2IpInIIr82s0lEjm+z0XcgSqlFQOkB6uQppdb4t+3AFiC9haq3AveKyGDgJuDuJhrgGv/1iwQQkQf8132TiPxfEw1xoYj8TUR+BII0yiMZERkoIv8EtgH9mh9XSm1RSm3bXxvtcS+36SA1RydKKf1pxw++pLVnA5/ie8D+GehyEOfNAX7XyrEuQA7wP//+u8BE/3Z3YIt/Oxow+bdPAT7xb8/0nx/v378duK9Jf6MO93X7Dde7J7DpV9TdC0S3cvxmoNJ/vRLxaS0R/mN3Aw/4t+ObnPMOcI5/eyHwn8N9Tdrw2kYAVwFLgKXANQe6V/zXYMx+jrfpvaw/+nOgjzYHtDNKKQ/wFfCViCQBjwN7RWSCUmplS+eIyH34zHCzW2lzn4gs8LcLvofAoCaKZLSIRAExwFsi0hdQgLlJM/OUUvVa0c/A6yJiBj5XSq07xOEeMfi1tE+APymlKlup9gLwhFLqTRE5GxgELPVfZwuw3F/vRBG5CwgH4oHN+F5aAD5opyEcDvKADcA1SqmtbdFgO9zLGs1+0UKvAxCRGOASfG/JLnxzRRtaqXslPs3wZKXU/hZRev0f8Jmpxyulapu19Ry+N+ipItIT31t3PdX1G0qpRSIyCTgLeEdEnlJKvX3QAwxhRKQbjQLoJaXUS37h/gkwWyn1aWvnKqW8IlL/Gwi+h+uMZu1bgf/g02ayReQhwNqkSjVHDxfhu3c/E5H3gLeUUllt0G6b3csazYHQc3rtjIj8F1gD9AauUEpNUkq9pXwT+c3rno7PbHauUqrmV3zN9/jmnerbGeHfjAFy/dsz99PHHkChUuoV4DVg1K/47pBGKZWtlBrh/7zkn297DZ/Z7F+/oqmfgONEpA+AiISLSD8aBVyxX3u8qE0HEEIo35zzJcBEoAL4QkTm+4VQW/Gb7mWN5kBoodf+fAj0V0rdo5TKPEDd54EoYJ7fqeRg3bf/CIwR31KHX4Ab/OV/Bx4XkaX45upaYzKwTkTWAhcCzx7k94YUfu1jOdBfRHJEZFYL1Y4DLgdO8l/jdSJy5oHaVkoV4XvYviciG/AJwQFKqXLgFWAj8Dk+U/FRjVKqRCn1rFJqBL45ak/zOiIyVURygPHA1yIy9yCb/633skazX3QYMo1Go9F0GrSmp9FoNJpOgxZ6Go1Go+k0aKGn0Wg0mk6DFnoajUaj6TRooafRaDSaToMWepojGhHpKSKqtWDDIvJnEXn1INp5U0QebfseajSaUEILPU1IICJzReSRFsrPE5H8Q42gr5T6m1Lqmt/ew4PHH2i6TETCOvJ7NRrNgdFCTxMqvAlc3kImisvxhQtzB58SevijkxyPLz7kuYe1MxqNJggt9DShwuf4gjU3pDUSXy7Cs4G3ReQeEdkpIiUi8qGIxDc7/zIR2Ssixf6A3fVtPOQPBVe/P9GfmqZcRLJFZGZLnRGRs/3RWsr99Ycd5DiuwBet5U3gymZtJojIHBGp9KfNeVREljQ5PkBE5olIqYhsE5GLD/I7NRrNQaKFniYk8AcY/hCf0KjnYmArcCJwPnACvlQ0ZfgyIDRlItAfOBl4QEQGNv8OEekOfAs8ByQBI4CgjBIiMgp4HV9S0wTgZeDLgzRXXoEvO8Zs4DQRSWly7AV8wZFT8QnEBqEovkSs8/Cl1kkGZuBLsDr4IL5To9EcJFroaUKJt4BpImLz71/hL7seX76/HKWUA1+y0YuazfM9rJSqVUqtB9YDw1to/zJgvlLqPaWUyx9DsqU0StcCLyulViilPEqptwAHvkzrrSIiE4EewIdKqdXATuBS/zEjvrimDyqlapRSv/jHVs/ZwB6l1BtKKbfyJbr9hKM4gLVGczjQQk8TMiillgBFwHki0hsYi0/z6YEvnU25iJTjS8brAZpqUflNtmuAyBa+ohs+QXQgegC313+f/zu74dMy98eVwPdKqWL//rs0anNJ+FJ5ZTep33S7BzCu2Xdehk8r1Gg0bYTOp6cJNd7Gp+H1xydACkQkG7haKbW0eeVfmdYmGzjmIOs9ppR67GAb9munFwNGEakXwGFArIgMBzbhSwzcFdjuP96t2Xf+qJQ69WC/U6PR/Hq0pqcJNd7Glz37WhrNfy8Bj/nz/iEiSSJy3iG0PRs4RUQuFhGT37FkRAv1XgFuEJFx4iNCRM4SXwbv1jgfn/Y5CN9c4QhgILAYXx5FD/Ap8JA/F98AAucvvwL6icjlImL2f8a2NDep0WgOHS30NCGFUmoPsAyIAL70Fz/r3/5eROz4vCPHHULbe4EzgduBUnxOLEFzf0qpVfiE7vP4nGZ2cODEpVcCbyil9iql8us//jYu888/3oQvGWo+8A7wHr65QpRSdmAKMB3Y56/zJD5tUaPRtBE6n55Gc5gQkSeBVKXUlQesrNFo2gSt6Wk0HYR/Hd4wv8n0GGAW8Nnh7pdG05nQjiwazUHiX+f3SyuHB/nNp/sjCp9JswtQCPwT+KLteqjRaA6ENm9qNBqNptOgzZsajUaj6TRooafRaDSaToMWehqNRqPpNGihp9FoNJpOgxZ6Go1Go+k0/D84WsxCzJhzbQAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 468x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.catplot(x = 'Vehicle_Age', y = 'Annual_Premium', hue = 'Vehicle_Damage',data = df)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Vehicle Age between 0 - 2 has a higher and stable Annual Premium, people tend to be more taking care of their cars, wherears when Age > 2, customers are more focosed on lower price services where can enoughly cover the basic requirments."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## To conclude: Males at the age between 35~55 who live in region 28 & 8, have not purchased inssurance for their 1-2 year(s) old car yet that had accidents before would be MORE interested in purchasing inssurance."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Group Segmentation & Categorical Treatment"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Gender Segmentation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    206089\n",
       "1    175020\n",
       "Name: Gender, dtype: int64"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#function to group gender into 0,1\n",
    "def gender(dataframe):\n",
    "    dataframe.loc[dataframe['Gender'] == 'Male', 'Gender'] = 0\n",
    "    dataframe.loc[dataframe['Gender'] == 'Female', 'Gender'] = 1\n",
    "    \n",
    "    return dataframe\n",
    "\n",
    "gender(df);\n",
    "\n",
    "df['Gender'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Age Segmentation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1    176981\n",
       "2    127089\n",
       "3     51695\n",
       "4     25344\n",
       "Name: Age, dtype: int64"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#function to devide Age into 4 groups.\n",
    "def age(dataframe):\n",
    "    dataframe.loc[dataframe['Age'] <= 33, 'Age'] = 1\n",
    "    dataframe.loc[(dataframe['Age'] > 33) & (dataframe['Age'] <= 52), 'Age'] = 2\n",
    "    dataframe.loc[(dataframe['Age'] > 52) & (dataframe['Age'] <= 66), 'Age'] = 3\n",
    "    dataframe.loc[(dataframe['Age'] > 66) & (dataframe['Age'] <= 85), 'Age'] = 4\n",
    "           \n",
    "    return dataframe\n",
    "\n",
    "age(df)\n",
    "\n",
    "df['Age'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Premium Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x7f944dfcb150>"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 842.4x595.44 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "rcParams['figure.figsize'] = 11.7,8.27\n",
    "sns.distplot(df['Annual_Premium'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1º Quartile:  24405.0\n",
      "2º Quartile:  31669.0\n",
      "3º Quartile:  39400.0\n",
      "4º Quartile:  540165.0\n",
      "Annual Premium above:  61892.5 are outliers\n"
     ]
    }
   ],
   "source": [
    "print('1º Quartile: ', df['Annual_Premium'].quantile(q = 0.25))\n",
    "print('2º Quartile: ', df['Annual_Premium'].quantile(q = 0.50))\n",
    "print('3º Quartile: ', df['Annual_Premium'].quantile(q = 0.75))\n",
    "print('4º Quartile: ', df['Annual_Premium'].quantile(q = 1.00))\n",
    "#Calculate the outliers:\n",
    "  # Interquartile range, IQR = Q3 - Q1\n",
    "  # lower 1.5*IQR whisker = Q1 - 1.5 * IQR \n",
    "  # Upper 1.5*IQR whisker = Q3 + 1.5 * IQR\n",
    "    \n",
    "print('Annual Premium above: ', df['Annual_Premium'].quantile(q = 0.75) + \n",
    "                      1.5*(df['Annual_Premium'].quantile(q = 0.75) - df['Annual_Premium'].quantile(q = 0.25)), 'are outliers')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Numerber of outliers:  10320\n",
      "Number of clients:  381109\n",
      "Outliers are: 2.71 %\n"
     ]
    }
   ],
   "source": [
    "print('Numerber of outliers: ', df[df['Annual_Premium'] >= 61892.5]['Annual_Premium'].count())\n",
    "print('Number of clients: ', len(df))\n",
    "#Outliers in %\n",
    "print('Outliers are:', round(df[df['Annual_Premium'] >= 61892.5]['Annual_Premium'].count()*100/len(df),2), '%')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2.0    190549\n",
       "1.0     95283\n",
       "3.0     75883\n",
       "4.0     19394\n",
       "Name: Annual_Premium, dtype: int64"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#function to devide Annual Premium into 4 groups.\n",
    "def Premium(dataframe):\n",
    "    dataframe.loc[dataframe['Annual_Premium'] <= 24405.0, 'Annual_Premium'] = 1\n",
    "    dataframe.loc[(dataframe['Annual_Premium'] > 24405.0) & (dataframe['Annual_Premium'] <= 39400.0), 'Annual_Premium'] = 2\n",
    "    dataframe.loc[(dataframe['Annual_Premium'] > 39400.0) & (dataframe['Annual_Premium'] <= 55000), 'Annual_Premium'] = 3\n",
    "    dataframe.loc[(dataframe['Annual_Premium'] > 55000) & (dataframe['Annual_Premium'] <= 540165.0), 'Annual_Premium'] = 4\n",
    "           \n",
    "    return dataframe\n",
    "\n",
    "Premium(df)\n",
    "\n",
    "df['Annual_Premium'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Vintage Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1º Quartile:  82.0\n",
      "2º Quartile:  154.0\n",
      "3º Quartile:  227.0\n",
      "4º Quartile:  299.0\n",
      "Vintage above:  444.5 are outliers\n"
     ]
    }
   ],
   "source": [
    "print('1º Quartile: ', df['Vintage'].quantile(q = 0.25))\n",
    "print('2º Quartile: ', df['Vintage'].quantile(q = 0.50))\n",
    "print('3º Quartile: ', df['Vintage'].quantile(q = 0.75))\n",
    "print('4º Quartile: ', df['Vintage'].quantile(q = 1.00))\n",
    "#Calculate the outliers:\n",
    "  # Interquartile range, IQR = Q3 - Q1\n",
    "  # lower 1.5*IQR whisker = Q1 - 1.5 * IQR \n",
    "  # Upper 1.5*IQR whisker = Q3 + 1.5 * IQR\n",
    "    \n",
    "print('Vintage above: ', df['Vintage'].quantile(q = 0.75) + \n",
    "                      1.5*(df['Vintage'].quantile(q = 0.75) - df['Vintage'].quantile(q = 0.25)), 'are outliers')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1    96174\n",
       "3    95695\n",
       "2    94786\n",
       "4    94454\n",
       "Name: Vintage, dtype: int64"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#function to devide Annual Premium into 4 groups.\n",
    "def Vintage(dataframe):\n",
    "    dataframe.loc[dataframe['Vintage'] <= 82.0, 'Vintage'] = 1\n",
    "    dataframe.loc[(dataframe['Vintage'] > 82.0) & (dataframe['Vintage'] <= 154.0), 'Vintage'] = 2\n",
    "    dataframe.loc[(dataframe['Vintage'] > 154.0) & (dataframe['Vintage'] <= 227.0), 'Vintage'] = 3\n",
    "    dataframe.loc[(dataframe['Vintage'] > 227.0) & (dataframe['Vintage'] <= 450), 'Vintage'] = 4\n",
    "           \n",
    "    return dataframe\n",
    "\n",
    "Vintage(df)\n",
    "\n",
    "df['Vintage'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Label Encoder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "labelencoder_X = LabelEncoder()\n",
    "\n",
    "df['Vehicle_Age']  = labelencoder_X.fit_transform(df['Vehicle_Age']) \n",
    "df['Vehicle_Damage']  = labelencoder_X.fit_transform(df['Vehicle_Damage']) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Gender'] = pd.to_numeric(df['Gender'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Gender                    int64\n",
       "Age                       int64\n",
       "Driving_License           int64\n",
       "Region_Code             float64\n",
       "Previously_Insured        int64\n",
       "Vehicle_Age               int64\n",
       "Vehicle_Damage            int64\n",
       "Annual_Premium          float64\n",
       "Policy_Sales_Channel    float64\n",
       "Vintage                   int64\n",
       "Response                  int64\n",
       "dtype: object"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
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
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Driving_License</th>\n",
       "      <th>Region_Code</th>\n",
       "      <th>Previously_Insured</th>\n",
       "      <th>Vehicle_Age</th>\n",
       "      <th>Vehicle_Damage</th>\n",
       "      <th>Annual_Premium</th>\n",
       "      <th>Policy_Sales_Channel</th>\n",
       "      <th>Vintage</th>\n",
       "      <th>Response</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>28.0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>28.0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2.0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>11.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>152.0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>41.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>152.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Gender  Age  Driving_License  Region_Code  Previously_Insured  Vehicle_Age  \\\n",
       "0       0    2                1         28.0                   0            2   \n",
       "1       0    4                1          3.0                   0            0   \n",
       "2       0    2                1         28.0                   0            2   \n",
       "3       0    1                1         11.0                   1            1   \n",
       "4       1    1                1         41.0                   1            1   \n",
       "\n",
       "   Vehicle_Damage  Annual_Premium  Policy_Sales_Channel  Vintage  Response  \n",
       "0               1             3.0                  26.0        3         1  \n",
       "1               0             2.0                  26.0        3         0  \n",
       "2               1             2.0                  26.0        1         1  \n",
       "3               0             2.0                 152.0        3         0  \n",
       "4               0             2.0                 152.0        1         0  "
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Shrink our dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "381109"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_no_response = df[df['Response'] == 0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_response = df[df['Response'] == 1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2500"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.utils import resample\n",
    "\n",
    "df_no_response_downsampled = resample(df_no_response,\n",
    "                                      replace = False,\n",
    "                                      n_samples=2500,\n",
    "                                      random_state = 42)\n",
    "len(df_no_response_downsampled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2500"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_response_downsampled = resample(df_response,\n",
    "                                   replace = False,\n",
    "                                   n_samples=2500,\n",
    "                                   random_state = 42)\n",
    "len(df_response_downsampled)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Let's merge these 2 downsampled datasets into a single Dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_downsample = pd.concat([df_no_response_downsampled,df_response_downsampled])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5000"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(df_downsample)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## We are off to go, Features & Target Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = df_downsample.drop('Response', axis = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "y = df_downsample['Response']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x7f9446ceebd0>"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAs4AAAKfCAYAAAB+Pk5+AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAgAElEQVR4nOzdeXxU5dn/8c9lyL5BAmpIWMK+KLuyaaV2Q7RlV6hPn2q11iJaba1dfq3to120i1VbW2urUq0VpQqiIOJSQGXfRLYERHYUkpCFLBDC/ftjhpAhJzCQkDNJv+/XKy/mnHPPmeueOffhmmvuOWPOOURERERE5NTO8zsAEREREZGmQImziIiIiEgYlDiLiIiIiIRBibOIiIiISBiUOIuIiIiIhEGJs4iIiIhIGJQ4i4iIiEjEMbOnzGy/ma2vY7uZ2aNmttXM1pnZgBrbRppZTnDbDxsqJiXOIiIiIhKJpgEjT7H9KqBr8O8W4C8AZhYFPBbc3guYbGa9GiIgJc4iIiIiEnGcc4uAglM0GQ084wKWAi3NLAO4FNjqnNvmnDsCTA+2rbcWDbETiTj6OUgREZHmwfx88BF2b6PlFAvcfWfa10xgV43l3cF1XusH1y+6ACXOzdQIu9fvEOptgbuPefNy/Q6j3kaO7MYLz6/1O4x6u25yv2bTj6eeWOp3GPX2jVuGAPCPp1b4HEn9fP0blwAwd+5mnyOpn1GjegDw4vSmPUaundQPgKf/tsznSOrnxm8GcqR/v/iBz5HU34Rr+/odQiTzSrTdKdbXmxJnEREREWmKdgPtaixnAXuBmDrW15vmOIuIiIiIJzNrtL+zMBv43+DVNYYARc65fcAKoKuZZZtZDDAp2LbeVHEWERERkYhjZs8DI4DWZrYb+BkQDeCcexyYC4wCtgJlwI3BbUfNbCrwBhAFPOWc29AQMSlxFhERERFvPn410Tk3+TTbHXBbHdvmEkisG5SmaoiIiIiIhEEVZxERERHxZOf5ejW8iKOKs4iIiIhIGFRxFhERERFPZ3exi+ZLFWcRERERkTCo4iwiIiIi3lRyDqGKs4iIiIhIGFRxFhERERFPKjiHUsVZRERERCQMSpxFRERERMKgqRoiIiIi4kk/gBJKFWcRERERkTCo4iwiIiIi3vTtwBCqOIuIiIiIhEEVZxERERHxpIJzKFWcRURERETCoIqziIiIiHgylZxDKHGWOt3z5BiGXtONwv2l3HjxY55tbn9kFENGdaWirJIHbpjJljX7GjnK8DjnePnlJ9i4cRXR0bFcf/13aNeuS612ixa9xsKFs8nL28cvf/lPkpJSfYi2bhdmJNN/QCZmxraP8tm8aX+tNv0HZJLRNoWqqmMsX7qTgwfLfYj01JpLPzKzUhkyrCNmRu7m/az7YG/I9tTUOC4f0Zn01omsWrGL9esic3y0zUzh0iHtMTO25B5g/bpPQranpMYx/PJs0tMTWLNqDxvWf1LHnvznnGPmzL+xaVNgrE+e/B3atetcq927785h0aLZ5OV9wv33P0tSUooP0Z5a/wGZXJgRHAPLdlLoMQYSE2MYMqwDMTEtOHiwjOVLd3LsmPMhWm+ZWakMHtohMEZy9vPhB6FjIDU1jsuu6ER660RWr9jF+g8j89hyzjFn7tPk5K4hOjqW8eOmkNm2U612BQf388KLD1Nedoi2bbOZMP52WrRQqtWcaKrGWTKzC8zsX2a2zcxWmdkSMxvbAPsdYWavNUSM9TVv2hruGflsndsHX9WVrK7pXN/1EX5/y2zu+suXGzG6M7Nx4yoOHNjLT37yVyZNuo0ZM/7i2a5Tp55MmXI/aWnnN3KEp2cGAwdmsWjBNubN3UyHDq1ISYkNaZORkUxycixzX9vEyuW7GDgoy6do69ac+jH0smzmv76Zl2d8QKcu6bRsGR/S5vDhoyxdvD1iE2YI9GPI0A68NX8Lr7y8nuxO6aS2jAtpc+TwUZYv3RnRCfNxmzat4sCBffz4x49z7bW38e9/e4/17OyefPvb99GqVeSNdQi8uUxKiuX1OZtYuaLuMdCnbwa5OQd4fc4mKo9Ukd0prZEjrZsZDBnekfnzcpj573V06pxOqscYWbZ4R0SPEYDcLWvIy/+E7975KGNG38LsV//u2e6NN/7J8KFX8927HiUuPpFVq99p5EjPAWvEvyZAifNZsMDnFrOARc65Ts65gcAkoNH/dzezc/ZWdt27OygpqLvKN3x0D954Zi0AG5ftJqllHGkXJp2rcOpl/fqlXHLJlZgZHTv2oLy8lKKiglrtsrI6k55+gQ8Rnl5aWgIlhw5TWnqEY8ccO3ceJDMrtCKemZXK9u2BfuXnlxEdE0VcXGRVO5pLP1q3SaK4qIKSksMcO+bY9lE+7Tu2CmlTUXGUvAOlEVUBPFnr1okUFx/mULAfH28roF372v3Iz4vsfhy3fv1yLrnks8Gx3v0UY70TaWmROdYBMjNPjIGC/DKio73HwPkXJLN7VyEA2z8uIDMzcj4la90miZLiiupja9tHBbTv4DFGmsCxtWnTSvr3+wxmRvt23agoL6W45GBIG+cc2z7eQO/eQwAY0G8EGzet8CNcOYeUOJ+dK4EjzrnHj69wzu1wzv3RzKLM7LdmtsLM1pnZt6C6krzAzP5tZpvN7LlgAo6ZjQyuew8Yd3yfZpZoZk8F97XGzEYH199gZjPM7FVgfqP2vIY2mSkc2FVUvXxgdzFtMiPv406AwsJ8WrZsXb2cmppOUVG+jxGdufiEaMrLKquXy8oqiY+PDm0TH01Z6Yk25WWVxCeEtvFbc+lHYmIMpaVHqpdLS4+QkBjjY0RnJ+GkfpSVHiExwp7rM1FUFDrWW7Zs3eTGOgTGQM1xUl5ee5zExERx5EgVLphzlnm08VNCYgylh046thIjJ74zUVxcQGrqieMqJTWd4uLQN2RlZSXExSUQFRUVbJNWq01TZOdZo/01BZFVwmk6egOr69h2E1DknLvEzGKB983seHLbP3jfvcD7wHAzWwn8jUAyvhV4oca+/h/wjnPuG2bWElhuZm8Ftw0F+jjn/BuVHl8YcC6yqwY1/dd84aHpvCSn1hT60RRiDENT7obXOahJjvUwQvbqViS9dl5diKT4zoTziNzCeJGa4JEnp6HEuQGY2WPAZcARYAfQx8wmBDenAl2D25Y753YH77MW6AgcAj52zm0Jrv8ncEvwvl8EvmJmdweX44D2wdtv+po0Awd2F9Gm3YmPBdtkpZC3t8THiEK9++4clix5A4D27btSWJhXva2oKJ+UlMiZCxiOk6uuCQnRlJdXhrYpryQhMRqCXY33aOO35tKP0tIjJNaoMCcmxlBWduQU94hMZSf1IyExhrKyyHquT+e99+awZMmbALRv3yVkrBcW5jWZsd6lS2uyO6cDcLCgLGScxMfXHgOHD1cRExOFGTgHCfHRVETQOCktPUJi0knHVmnkxHc6S5fNY8XKtwHIyuxMUdGJ46q4KJ/klNBpJwkJyVRUlFFVVUVUVBTFRQUkN5Fj71Sa4vvOc0mJ89nZAIw/vuCcu83MWgMrgZ3A7c65N2rewcxGAIdrrKrixPNf15twA8Y753JO2tdgoLQ+HWgIi2fnMHbqYN6Z/iG9BmdRWlRBwSeH/A6r2uWXX83ll18NwIYNK3j33dcYMOAz7NiRQ1xcAqmpTeuEVlBQRnJyLImJMZSXV9K+fSuWLN4R0mbPnmK6dm3Nzh2FpKcnUFlZRUXFUZ8i9tZc+pF34BCpqXEkJcdSVnqETp3TWfDOVr/DOmN5eaWkpMaSlBRImLM7pfHugo/8DuuMXHbZ1Vx22fGxvpL33ptD//6Xs2NHLvHxiU1mrG/dmsfWrYHkLCMjhS5dW7NrZyFppxgD+z89RFa7luzaWUjH7DT27Cmq1cYveQcOkZJSc4yksfA/TefYGjJ4JEMGjwRgc85qli6bR5+Lh7Nr9xZi4xJISQ5NnM2MTtm92bBhKX36DGf12gX07DHIj9DlHFLifHbeAX5lZt92zh3/ynZC8N83gG+b2TvOuUoz6wbsOcW+NgPZZtbZOfcRMLnGtjeA283sduecM7P+zrk1Dd2Zuvz0XxPoNyKb1NYJzNj1PZ7+2X9oER2YFj/7rytZOjeXwaO68tzWOzlcVsmDN85srNDOWK9eg9i4cSX3338LMTGxfPWr36ne9vjjP2fy5NtJTU1n4cLZvP32y5SUHOTBB++gV6+BTJ58h4+Rn+AcrF65mytGdApcxm1bAcXFFXTuEqhQfbQ1n317i8nISObqa3pyNHgJq0jTnPqx5P3tfOmqHth5xpac/RQeLKd7z8BVGnI27Sc+PpqvjL2I6JgonIPeF13IyzPWUVlZ5XP0JzgHy5bs5PNf6s55Blu25FFYWEG37m0AyM05QFx8C675Sm+io6PAOXr2voBXXv6QyspjPkdfW69eA9m0aSW//OWtxMTEMmnS7dXbnnjiPq677jZSU9NZtOhV3nlnJiUlB/ntb++gZ8+BIW39tm9fMRltkxl1TU+OHj3Gihpj4PLPdGLF8p1UVBxl3Qd7GTKsAxddnEHhwXI+3hY5c2qdg6WLt/PFq7oHLnWYc8BzjHx5zPEx4uh1UQYz/x1ZYwSge7f+5Oau5qE/3EF0dAzjxk2p3vaPZ37N2DHfIiUljS998Xqmv/gwb749nbYZ2QwaeKWPUTcQlZxDWFOakxpJzCwD+AMwGDhAoAL8ODAD+AXwZQIV4wPAGALzm+92zl0TvP+fgJXOuWlmNhJ4mMAH0+8BFznnrjGz+OD6YcF9bQ+uvwEY5JybWkd4boTdew563bgWuPuYNy/X7zDqbeTIbrzw/Fq/w6i36yb3azb9eOqJpX6HUW/fuCXwzf1/PNW0v7X/9W9cAsDcuZt9jqR+Ro3qAcCL05v2GLl2Uj8Anv7bMp8jqZ8bvzkYgH+/+IHPkdTfhGv7+pq5XpX2q0ZLFF8v+HHEZ+mqOJ8l59w+Apeg8/Lj4F9NC4J/x+8/tcbteUAPj8coB77lsX4aMO3MIhYRERGR+lDiLCIiIiKeNFMjlK7jLCIiIiISBlWcRURERMRTU/lhksaiirOIiIiISBhUcRYRERERb5rkHEIVZxERERGRMKjiLCIiIiKeVHAOpYqziIiIiEgYVHEWEREREU+mknMIVZxFRERERMKgirOIiIiIeFPBOYQqziIiIiIiYVDFWUREREQ86ZcDQ6niLCIiIiISBlWcRURERMSbCs4hVHEWEREREQmDEmcRERERkTBoqoaIiIiIeNIPoIRSxVlEREREJAyqOIuIiIiIJ1WcQ6niLCIiIiISBnPO+R2DNDy9qCIiIs2DryXfMR1/12g5xaztd0d8eVtTNZqpefNy/Q6h3kaO7MYIu9fvMOptgbuPV2Zt8DuMehs9pjfTnlzudxj1dsNNl7J4yU6/w6i3YUPbA7Bp036fI6mfnj3PB+A3v/6Pz5HUzz0/+iwAj/9psc+R1M+tU4cBsGd3kc+R1E9mVioAM174wOdI6m/idX39DkFqUOIsIiIiIp40xzmU5jiLiIiIiIRBFWcRERER8aSCcyhVnEVEREREwqCKs4iIiIh4U8k5hCrOIiIiIiJhUMVZRERERDyp4BxKFWcRERERkTCo4iwiIiIinuw8lZxrUsVZRERERCQMSpxFRERERMKgqRoiIiIi4k3fDgyhirOIiIiISBhUcRYRERERTyo4h1LFWUREREQkDKo4i4iIiIgn87nkbGYjgUeAKODvzrkHTtr+feD64GILoCfQxjlXYGbbgRKgCjjqnBtU33iUOIuIiIhIxDGzKOAx4AvAbmCFmc12zm083sY591vgt8H2Xwbucs4V1NjNZ51zeQ0VkxJnEREREfHm76TeS4GtzrltAGY2HRgNbKyj/WTg+XMZkOY4i4iIiEgkygR21VjeHVxXi5klACOBl2qsdsB8M1tlZrc0RECqOIuIiIiIJ5/nOHs9uKuj7ZeB90+apjHcObfXzM4H3jSzzc65RfUJSBVnEREREYlEu4F2NZazgL11tJ3ESdM0nHN7g//uB2YSmPpRL0qcRURERMSTmTXan4cVQFczyzazGALJ8WyPGFOBK4BXaqxLNLPk47eBLwLr6/t8aKqGnJJzjpdffoKNG1cRHR3L9dd/h3btutRqt2jRayxcOJu8vH388pf/JCkp1Ydovd3z5BiGXtONwv2l3HjxY55tbn9kFENGdaWirJIHbpjJljX7GjnK8DjnmD37STbnrCY6OpZrr51KVmbnWu3eXzyX9957jfz8T/jZvdNITEzxIdq6ZWamcumQ9th5xpacA3y4LvT5Tk2NY/hnOpGensDqlbvZsP4TnyI9Necc/3ruz6xbt5yYmFhuuvn7dOzYtVa7vz7+a7ZvzyUqqgXZnbrz9a/fSYsWkXP6dc7x978/wqpVS4mNjeWOO35M587d62z/xBN/4J13Xmf69PmNGGV4sjul8bnPd8XOg3Vr97Fs6c6Q7b16X8ClQ9oDUHmkivlv5HBgf6kfoZ5Su/YtGX55NmawaeN+1q7eE7K9Zct4Rny+C23aJLJ86U4+WFNXEc5fzjn+9NjvWbZsMXGxcdxzz71069ajVrsHH/w/Pli3msTEJAB+cM/P6NKlW2OHWyfnHHPmPk3uljVER8cyfuwU2rbtVKtdwcH9vPjiw5SXHyKjbTYTxt0eUWO9qXHOHTWzqcAbBC5H95RzboOZ3Rrc/niw6VhgvnOu5mC+AJgZTMhbAP9yzs2rb0yqOPvMzMaamTOz2meSCLBx4yoOHNjLT37yVyZNuo0ZM/7i2a5Tp55MmXI/aWnnN3KEpzdv2hruGflsndsHX9WVrK7pXN/1EX5/y2zu+suXGzG6M7M5ZzV5efu45/uPMX7crcyc+YRnu44devDNm39Oq1ZtGjfAMJjB4GEdeHN+LrNe+pDsTumktowLaXP48FGWLdnB+g8jM2E+bt265Xz66R4eeHAaN9xwJ88+86hnuyFDr+RXv36K+3/xBJVHDrNo0euNHOmprVq1lH37dvOXvzzPlCn38Pjjv6+z7datmyktPdSI0YXPDD7/xW7MePEDnnxiOT17XUB6ekJIm8LCcp5/bg3TnlzB4ve386WrIu/UawaXXdGJOa9u5IV/raVLt9a0ahUf0qbi8FHeX/RxxCbMxy1bvpg9u3fx7DMv8d3v/oiHH3mwzrbfuuUO/vbEc/ztieciKmkGyN2yhvz8T7jrO48y5iu3MPvVv3u2mz//nwwbdjV33fko8XGJrFr9TiNH2vDsvMb78+Kcm+uc6+ac6+yc+2Vw3eM1kmacc9Occ5NOut8251zf4F/v4/etLyXO/psMvEfg44eIs379Ui655ErMjI4de1BeXkpRUUGtdllZnUlPv8CHCE9v3bs7KCkor3P78NE9eOOZtQBsXLabpJZxpF2Y1FjhnZGNG5YzYOAIzIwOHbpTXl5KcXHt1yMzs1NEvokBaN0miZLiwxwqOcyxY46Pt+XTvn2rkDYVFUfJzyvFHavrOyCRYc2aJQwb/nnMjM5delFWdojCwvxa7fr2HVz9UWR2px4cLDjgQ7R1W778PUaMGImZ0b17b0pLD1FQUPuyp1VVVUyb9me+/vVv+xDl6WW0TaHwYDlFhRUcO+bYtOlTunRrHdJm755iDlccDdzeW0xycqwfoZ7S+RckUVxUTklxYIx8tCWPjp3SQtpUlFdyYP8hjkX4GFn8/iK+8MVRmBm9el3MoUMl5Oc32CV1G82mzSvp1+8zmBnt2nWjoqKUkpKDIW2cc2z7eAO9ew0BoH+/EWzatMKPcOUcUuLsIzNLAoYDNxFMnM3sPDP7s5ltMLPXzGyumU0IbhtoZguDl1V5w8wyznWMhYX5tGx54j+e1NR0iopqJwZNWZvMFA7sKqpePrC7mDaZkTW14bii4gJapp54PVqmplPkkThHsoSEaEpLD1cvl5YdISExxseIzl7hwbyQNyitWrXm4MG6k4KjR4+yePFbXHzxJY0RXtgKCg7QuvWJfqSnt/FMnOfOfZlLLx1OWlrrWtsiQVJSLCXFFdXLJSWHT5kY9+mTwccfRd75LDExlkMlR6qXDx06QmITHSN5efs5v82JokqbNueTl7ffs+2TT/2Fm2/+Ko/9+SGOHDni2cYvJcUFpNY496akpNcqWpSVlRAXl0BUVFSgTWoaxSVN6/zsyazx/poAJc7+GgPMc87lAgVmNgAYB3QELgZuBoYCmFk08EdggnNuIPAU0CAfO5wpv39+s8F59Me5SK3i1I7LPK/W08RE6tN9Gl7HyanGx7PPPEr3bhfTrfvF5zKsM+Z9vIf2o6Agj8WL/8PVV49vnKDOgtdTX9dQbt++JX36ZrBgwUfnNqgG0kSHiGfcXmPk5ptv4x/TZvDnP0+jpLiY6dOfOffBnQHPMRLG/4XN4OwsJ9GMdX9NBh4O3p4eXI4GZjjnjgGfmNl/gtu7AxcRuA4hBCbJn5NvsL377hyWLHkDgPbtu1JYeKLyVFSUT0pKWl13bZIO7C6iTbsTX2Zsk5VC3t4SHyMKtXjx6yxb/iYA7bK6UFh04vUoLMonJaVVXXeNSGVllSQmnqgCJibEUFYWWdWlU3n7rVdYuHAuANnZ3SkoOFE9O3gwj5Yt0z3vN2vWs5SUFPH1G+5slDhPZ+7cl5k//1UAunbtEVIFzM8/QFpaaD+2bctl37493HrrZAAOH67g1lsn8fjj0xsv6NMoKTlMcsqJ+fLJybEcOnS4Vrs2bRL50qge/PvFD6goP9qYIYaltPQwScknKsxJSTGUlTadMTJr1gzmzJ0FQPfuvdh/4NPqbQcO7Cc9vfZ3L9LTA9XcmJgYRo78Mi+++M/GCfYUli6bx8pVbwOQmdmZohrn3uLifFKSQ8+9CQnJVFSUUVVVRVRUFMVFBSQnN6//L0WJs2/MLB24ErjIzByBRNgRuM6g512ADc65oec6tssvv5rLL78agA0bVvDuu68xYMBn2LEjh7i4BFJTm9eJYPHsHMZOHcw70z+k1+AsSosqKPgkcr78NGzYVQwbdhUAmzatZPHi1+nX9zJ27swlPi6hyb2RyTtwiJSU2EAyUFZJdqd0FjWRqh/A5z4/ms99fjQAH6xdxttvv8LgwZ9l20ebiI9P9EycFy6cy/r1K7nnnt9w3nmR8UHfqFHjGDVqHAArVy5m7tyXufzyz5Gbu5HExKRa0zEGDRrGtGnVV3pi0qQvRlTSDLBvbwmtWsWTmhpHSclheva8gFdnbwhpk5wSy5jxFzHn1Y0cPMV3H/y0/9NDpKbGk5wcS2npETp3bc3b83P9DitsY8ZMZMyYiQAsXfoes2bN4MrPfpFNm9aTmJhUnSTXlJ+fR3p6a5xzvPf+Qjpm175aUGMbMngkQwaPBCAnZzVLl82jz8XD2b17C7FxCSSflDibGdnZvdmwcSl9Lh7OmrUL6NlzkB+hN6jm9iFzfSlx9s8E4Bnn3LeOrzCzhUAeMN7M/gG0AUYA/wJygDZmNtQ5tyQ4daObc25D7V03nF69BrFx40ruv/8WYmJi+epXv1O97fHHf87kybeTmprOwoWzefvtlykpOciDD95Br14DmTz5jnMZWth++q8J9BuRTWrrBGbs+h5P/+w/tIgOJC+z/7qSpXNzGTyqK89tvZPDZZU8eGNd713816PHQDbnrObB30whJiaWiROnVm978qlfMGHCFFJT0njv/TksXDCTkkOFPPSHu+jRYwATJ9zmY+QnOAdLl+zgCyN7YAZbcw9QWFhO9x6BKlTO5gPEx0dzzejeREdHgXP0uuhCZr20jsrKYz5HH6pP30tZt24ZP7jn68TExnLTTXdXb3vooR9z443fpVWr1jzzj0dIT7+AX9wfGBMDB13G6NFf8yvsWgYOHMqqVUu59dZJxMbGcccdP6redt9932fq1B9E7LzmmpxzvPVmLhMn9cXM+HDdPvLzyujXvy0Aa9fsZfjwjsTHRfOFLwWu2uCOOZ6ZtsrPsGtxDt5btI2rR/fCzMjZ+CkHC8rp1TswV3jjhk+JT4hm/LV9iImJwjm4uG8GLzy3lsrKKp+jDzV48HCWLVvM/3xtHHFxcdzz/Z9Wb/vhj+7k7u/9P1q3bsMvf/VTiooKcc7RpXM37rrrhz5GXVu3bv3J3bKahx6+g5joGMaNnVK97Zlnf82Y0d8iJSWNL33hel6Y8TBvvT2djIxsBg640seo5VywyJ3L2byZ2QLggZrXFDSzO4CeBKrLnwFygVjgIefcm2bWD3gUSCXwpudh59zfPHbv5s1rOtWJuowc2Y0Rdq/fYdTbAncfr8w6p+9vGsXoMb2Z9uRyv8OotxtuupTFS3aevmGEGzY0cC3iTZu8v2jVVPTsGfhS4m9+/Z/TtIxs9/zoswA8/qfFPkdSP7dOHQbAnt1Fp2kZ2TKzAtPvZrzwgc+R1N/E6/r6WvO9fuBjjZYoPrfqtoivb6vi7BPn3AiPdY9C4GobzrlDwekcy4EPg9vXEkioRURERKSRKXGOTK+ZWUsgBrjfORfZvwIhIiIizZMmOYdQ4hyBvKrRIiIiIuIvJc4iIiIi4kkF51CRcV0kEREREZEIp4qziIiIiHiy81RyrkkVZxERERGRMKjiLCIiIiLeNMk5hCrOIiIiIiJhUMVZRERERDyp4BxKFWcRERERkTCo4iwiIiIinnRVjVCqOIuIiIiIhEEVZxERERHxpoJzCFWcRURERETCoMRZRERERCQMmqohIiIiIp5M16MLoYqziIiIiEgYVHEWEREREU+6HF0oVZxFRERERMKgirOIiIiIeNIU51DmnPM7Bml4elFFRESaB19T12+M+Fuj5RRPLfhmxKfpqjiLiIiIiDeVnEMocW6mXnh+rd8h1Nt1k/vxyqwNfodRb6PH9GaE3et3GPW2wN3HjBc+8DuMept4XV9enN70x8e1k/oBTX+sXze5efXjpRnrfI6kfsZP7APAc8+s8jmS+rn+fwcCNKuxLpFBibOIiIiIeNJVNULpqhoiIiIiImFQxVlEREREPGmKcyhVnEVEREREwqCKs4iIiIh4U8k5hCrOIiIiIiJhUMVZRERERDyZKs4hVHEWEREREfmS6p0AACAASURBVAmDEmcRERERkTBoqoaIiIiIeDKVWEPo6RARERERCYMqziIiIiLiTV8ODKGKs4iIiIhIGFRxFhERERFPKjiHUsVZRERERCQMqjiLiIiIiCc7TyXnmlRxFhEREREJgyrOIiIiIuJNk5xDqOIsIiIiIhIGVZxFRERExJMKzqFUcRYRERERCYMqznJKF2Yk039AJmbGto/y2bxpf602/QdkktE2haqqYyxfupODB8t9iPTUnHPMnv0km3NWEx0dy7XXTiUrs3Otdu8vnst7771Gfv4n/OzeaSQmpvgQbd3ueXIMQ6/pRuH+Um68+DHPNrc/Mooho7pSUVbJAzfMZMuafY0c5ek555gz92lyt6whOjqW8WOn0LZtp1rtCg7u58UXH6a8/BAZbbOZMO52WrSIrNNW/wGZXJgRPP6X7aTQ4/hPTIxhyLAOxMS04ODBMpYv3cmxY86HaL01l3EOzacvzjlem/M0ObmriYmOZfz428j0GiMFnzI9OEbaZmQzccLttGgR7UPE3jLapjDoknaYwdateWxc/2mtNgMvaUdmZgpHq46x5P3tHCyIvNcDmsdYPxu6qkYoVZylTmYwcGAWixZsY97czXTo0IqUlNiQNhkZySQnxzL3tU2sXL6LgYOyfIr21DbnrCYvbx/3fP8xxo+7lZkzn/Bs17FDD755889p1apN4wYYpnnT1nDPyGfr3D74qq5kdU3n+q6P8PtbZnPXX77ciNGFL3fLGvLzP+Gu7zzKmK/cwuxX/+7Zbv78fzJs2NXcdeejxMclsmr1O40c6aldmJFMUlIsr8/ZxMoVdR//ffpmkJtzgNfnbKLySBXZndIaOdK6Nadx3pz6kpu7hvz8fXzvrj8yZsy3eGX23zzbzZv/HMOHXcP37voj8fFJrFwVOWPEDC4Z3J7/vL2F12ZvpGPHNFJS40LatM1MISUlltmzNrBsyU4uHdzBp2hPrTmMdWkYEZc4m1mVma01sw1m9oGZfdfMPOM0s7Zm9u/T7O8rZvbDBo5xgZkNOmndIDN7tCEfx29paQmUHDpMaekRjh1z7Nx5kMys1JA2mVmpbN9eAEB+fhnRMVHExUVWRRBg44blDBg4AjOjQ4fulJeXUlxcUKtdZmYn0tLO9yHC8Kx7dwclp6jGDB/dgzeeWQvAxmW7SWoZR9qFSY0VXtg2bV5Jv36fwcxo164bFRWllJQcDGnjnGPbxxvo3WsIAP37jWDTphV+hFunzMwTx39BfhnR0d7H//kXJLN7VyEA2z8uIDMztVYbvzSncd6c+rJx0wr697sCM6N9cIwUe42Rbeu5qHdgjAzof0VEjZH09ERKSio4dCjweuzYfpB27VqGtMlq15JtH+UDkJ9XSkxMFHHxkfd6NIexftbMGu+vCYi4xBkod871c871Br4AjAJ+dnIjM2vhnNvrnJtwqp0552Y75x44R7HWfJyVzrk7zvXjNKb4hGjKyyqrl8vKKomPD/0IMD4+mrLSE23KyyqJT4icjwmPKyouoGVq6+rllqnpFHkkzk1dm8wUDuwqql4+sLuYNpmRNd0EoKS4gNQar0dKSnqtNzJlZSXExSUQFRUVaJOaRnFJZL1m8fGhY6S8vPYYiYmJ4siRKlzw09oyjzZ+ak7jvDn1pbikgNTU9OrlsMZISmSd1+ITQp/rsrIjtZ7rhIRoysqOhLRJSIhptBjD1RzGujSMSEycqznn9gO3AFMt4AYzm2FmrwLzzayjma0HMLNlZtb7+H2DVeGBwfv8Kbhumpk9amaLzWybmU0Irj/PzP4crHK/ZmZzj28Ll5mNMLPXgreTzOxpM/vQzNaZ2fjg+i+a2RIzWx3sR1Jw/XYz+7/g+g/NrEdw/RXB6vtaM1tjZsnB9d83sxXBff9fPZ/mhheR07lqB2U0jXe3Z8TjHbtzkfeCeMYURrUh4l6xMALy6lbkvSJnoVl0IigS++IxRk4+lJzXeS2CBkl4sTSRAfLfPNYlROR9HnIS59y24FSN45+fDwX6OOcKzKxjjabTgWuBn5lZBtDWObfKzC4+aZcZwGVAD2A28G9gHNARuDj4OJuAp+oR9k+BIufcxQBm1srMWgM/AT7vnCs1sx8A3wXuC94nzzk3wMymAHcDNwf/vc05934wya4wsy8CXYFLCQzl2Wb2GefconrE6+nkSkxCQjTl5ZWhbcorSUiMhrzAcrxHG78sXvw6y5a/CUC7rC4UFuVVbyssyiclpZVfoZ0zB3YX0abdiY8G22SlkLe3xMeITli6bB4rV70NQGZmZ4pqvB7FxfmkJIe+HgkJyVRUlFFVVUVUVBTFRQUkJ/s/X7BLl9Zkdw5UAg8WlIWMkfj42sf/4cNVxMREYRbIhRLio6mIkDECTX+c19TU+7Jk6TxWrnwLgMzMLhQV5VdvKy7OJzkl9PhPTEgJHSPF+aREwBg5rqw0+FwHJSTEhFRtoWaFubS6TVn5ESJBcxvrZyuS3oxFgoiuONdQ82V70znn9VnUi8DE4O1rgRl17GuWc+6Yc24jcEFw3WXAjOD6T4D/1DPezwPVlzxwzh0EhgC9gPfNbC3wdaDmtyBeDv67ikASD/A+8JCZ3QG0dM4dBb4Y/FsDrCbwBqBrPeP1VFBQRnJyLImJMZx3ntG+fSv27C4OabNnTzEdOwZO1OnpCVRWVlFRcfRchHPGhg27irvufIi77nyI3r0vZfWqBTjn2LEjh/i4BFJSIuc/mIayeHYOX/rffgD0GpxFaVEFBZ8c8jmqgCGDRzJ1ym+ZOuW39OpxKWvXLsI5x65ducTGJZB8UuJsZmRn92bDxqUArFm7gJ49B3ntulFt3ZrHm2/k8OYbOezZXVR9/Ked4vjf/+khsoJzOztmp7FnT1GtNn5p6uO8pqbel6FDRnL71N9x+9Tf0avXJaxZuxDnHDt35RIXm1DrzaWZ0Sm7N+s3BMbI6jUL6dnzEj9C95SfX0pychyJSYHXo0PHVtXzf4/bvauQTsHkNL11Ikcqq6goj4zXo7mNdWkYEV9xNrNOQBVw/JpCpV7tnHN7zCzfzPoA1wHfqmOXh2vu/qR/G4pR+xMaI5D0Tz5NXFUEXxfn3ANmNofAPO+lZvb54H5+7Zz7awPHXItzsHrlbq4Y0SlwaadtBRQXV9C5S+Ak99HWfPbtLSYjI5mrr+nJ0eAleiJRjx4D2Zyzmgd/M4WYmFgmTpxave3Jp37BhAlTSE1J473357BwwUxKDhXy0B/uokePAUyccJuPkYf66b8m0G9ENqmtE5ix63s8/bP/0CI68P539l9XsnRuLoNHdeW5rXdyuKySB2+c6XPE3rp160/ultU89PAdxETHMG7slOptzzz7a8aM/hYpKWl86QvX88KMh3nr7elkZGQzcMCVPkZd2759xWS0TWbUNT05evQYK2oc/5d/phMrlu+kouIo6z7Yy5BhHbjo4gwKD5bz8bbImYfanMZ5c+pL924DyMldw+8fup3omBjGjztxHpr2zK8YN+ZWUlLSGPml/2H6C3/gzbeep21GNoMGRs4YcQ5WLt/JlZ/vipnx0dY8iooq6Not8P2GLbl57N1TTGZmKl8ZexFVR4+xZPF2f4OuQ3MY62dLl6MLFdGJs5m1AR4H/uScc3b6zwumA/cAqc65D8/god4Dvm5m/wDaACOAf515xNXmA1OBOyEwVQNYCjxmZl2cc1vNLAHIcs7l1rUTM+sc7MeHZjaUQHX5DeB+M3vOOXfIzDKByuB88Aa3b18J++ZsDln30db8kOXVq/YAe87FwzcYM2PsmFs8t930jZ9U375s+NVcNvzqxgrrjN3/1VNeRAaAR6bOaYRI6sfM+PI1N3tu+9+v/aj6dlraBXz7W79urLDOSl3H/7uLtlXfLi09wttvbmnEqM5Mcxnn0Hz6YmaM/rL3GLnhf39cfTst7QKmfPucf//9rO3dU8zePRtC1m3JzQtZXrF8F7CrEaM6O81hrEv9RWLiHB+cyhANHAWeBR4K877/Bh4B7j/Dx3wJ+BywHsgFlgGn+3xljpkdn7y0hBpTM4BfEEiS1xOoIP+fc+5lM7sBeN7Mjl9Y9CfBx6vLnWb22eA+NgKvO+cOm1lPYEnwjcQh4H84UZEXERERaRBhFC3/q0Rc4uycizrFtmnAtBrL24GLaix/ykl9qnkf59wNJ21LCv57zMzuDlZw04HlQJ0Va+fciDo2LQhuP0RgDvPJ93sHqDUBzTnXscbtlQQq3jjnbq/j8R8h8AZBRERERBpJxCXOPnrNzFoCMcD9wS8JioiIiPz3UsE5hBLnIK8qspnNBLJPWv0D59wbjRKUiIiIiEQMJc6n4Jwb63cMIiIiIn7x+6oaZjaSwPTUKODvJ/8atJmNAF4BPg6uetk5d1849z0bSpxFREREJOKYWRSBiy98AdgNrDCz2cHf4qjpXefcNWd53zOixFlEREREPPl8VY1Lga3OuW3BWKYDowlcaexc3rdOTeWXA0VERETkv0smoRf53h1cd7KhZvaBmb1uZr3P8L5nRBVnEREREfHm7xxnrwc/+ZeZVwMdgpcUHgXMArqGed8zpoqziIiIiESi3UC7GstZwN6aDZxzxcHfz8A5NxeINrPW4dz3bKjiLCIiIiKefP7hwBVAVzPLJvB755OAr9ZsYGYXAp8655yZXUqgKJwPFJ7uvmdDibOIiIiIRBzn3FEzmwq8QeCSck855zaY2a3B7Y8DE4Bvm9lRoByY5JxzgOd96xuTEmcRERERiUjB6RdzT1r3eI3bfwL+FO5960uJs4iIiIh48vlydBFHXw4UEREREQmDKs4iIiIi4s3nn9yONKo4i4iIiIiEQRVnEREREfGkKc6hVHEWEREREQmDKs4iIiIi4sk0xzmEKs4iIiIiImFQxVlEREREvGmScwgL/CqhNDN6UUVERJoHXzPX7930UqPlFL9/cnzEZ+mqODdTLzy/1u8Q6u26yf2Y9uRyv8OotxtuupQZL3zgdxj1NvG6voywe/0Oo94WuPuYNXOD32HU25ixvQGY+fJ6nyOpn7HjLgJo8mP9hpsuBWD6c2t8jqR+Jl3fH4BXZjXtMTJ6TGB8vDRjnc+R1N/4iX18fXz9cmAozXEWEREREQmDKs4iIiIi4slUYg2hp0NEREREJAyqOIuIiIiIJ81xDqWKs4iIiIhIGJQ4i4iIiIiEQVM1RERERMSbpmqEUMVZRERERCQMqjiLiIiIiCddji6Ung4RERERkTCo4iwiIiIinnQ5ulCqOIuIiIiIhEEVZxERERHxdp4qzjWp4iwiIiIiEgZVnEVERETEk+Y4h1LFWUREREQkDKo4i4iIiIgnFZxDqeIsIiIiIhIGVZxFRERExJuuqhFCFWcRERERkTCo4iwiIiIinnRVjVBKnOWULsxIpv+ATMyMbR/ls3nT/lpt+g/IJKNtClVVx1i+dCcHD5b7EOmpZWamcumQ9th5xpacA3y4bl/I9tTUOIZ/phPp6QmsXrmbDes/8SnSU3POMWfu0+RuWUN0dCzjx06hbdtOtdoVHNzPiy8+THn5ITLaZjNh3O20aBE5w/2eJ8cw9JpuFO4v5caLH/Nsc/sjoxgyqisVZZU8cMNMtqzZ59nOT845Zr/6JDk5q4mOjuXaiVPJzOxcq93ixXN57/3XyM//hHt/Oo3ExBQfoq2bc45XX30q0I+YGCZOuJ3MTI/jquBTnn/+D5SVl5DZthPXXnsHLVpE+xBx3ZrLWL8wI5kBg7IC596t+Wza+GmtNgMGZpKRmUrV0WMsW7IjIs+9zjlmz36SzcfHyLVTyfIYI+8vnst77wXGyM/ujcwx8tqcp8nJXU1MdCzjx99Gpte5t+BTpgfPvW0zspk44faIGyNSP5qqIXUyg4EDs1i0YBvz5m6mQ4dWpKTEhrTJyEgmOTmWua9tYuXyXQwclOVTtHUzg8HDOvDm/FxmvfQh2Z3SSW0ZF9Lm8OGjLFuyg/UfRuZ/osflbllDfv4n3PWdRxnzlVuY/erfPdvNn/9Phg27mrvufJT4uERWrX6nkSM9tXnT1nDPyGfr3D74qq5kdU3n+q6P8PtbZnPXX77ciNGFLydnNXl5+/j+3Y8xbtytzJz1hGe7Dh16cPNNP6dVyzaNHGF4cnJWk5e/j7vv/hPjxn6bWXX04/V5z3LZZdfw/bsfIz4+iZUr327kSE+tuYx1Mxh0STsW/ucjXn9tE+07tiIlJbQfGW1TSEqJY87sjaxYtpNBl7bzKdpT2xwcI/d8/zHGj7uVmTO9j62OHXrwzZt/TqtWkTlGcnPXkJ+/j+/d9UfGjPkWr8z+m2e7efOfY/iwa/jeXX8MjJFVkXXulfprNomzmVWZ2VozW29mr5pZy7PcT1sz+3cDx5ZkZn81s4/MbIOZLTKzwWdw/5+b2d0NGVM40tISKDl0mNLSIxw75ti58yCZWakhbTKzUtm+vQCA/PwyomOiiIuLnMomQOs2SZQUH+ZQyWGOHXN8vC2f9u1bhbSpqDhKfl4p7pjzKcrwbNq8kn79PoOZ0a5dNyoqSikpORjSxjnHto830LvXEAD69xvBpk0r/Ai3Tuve3UFJQd3VseGje/DGM2sB2LhsN0kt40i7MKmxwgvbho3LGThgBGZGh/bdKS8vpbi4oFa7zMxOpKWd70OE4dm4aQUD+l+BmdG+fTfKK0opLq59XH300XouumgoAAMGjGDDxuV+hFun5jLW09ITKCk5TOmh4Ll3x0Ey23mce7dF9rkXYOOG5QwYGBwjHZr2GOnfLzhGgufeYq9z77b1XNQ7cO4d0P+KiDv3ng2zxvtrCppN4gyUO+f6OecuAgqA285mJ865vc65CQ0bGn8nEFNX51xv4AagdQM/RoOLT4imvKyyermsrJL4+NCPnOLjoykrPdGmvKyS+ITI+lgqISGa0tLD1culZUdISIzxMaKzV1JcQGrqiUMnJSW91n9CZWUlxMUlEBUVFWiTmkZxSe3/qCJZm8wUDuwqql4+sLuYNpmR9dEtQHFxAaktT7weqam1X4+moLiogJa1+pEf0qasrIT4uMTq4yoS+9pcxnp8fAxlZUeql8vLjtQ+9yZEn9Qm8s69AEXFBbSscc5qmZpOUYQdN+EoLikgNTW9ejmsc29K0+yrnFpzSpxrWgJkAphZZzObZ2arzOxdM+tRY/1SM1thZveZ2aHg+o5mtj54O87MnjazD81sjZl9Nrj+BjN7ObjfLWb2m7oCMbPOwGDgJ865YwDOuW3OuTnB7d8NVsnXm9mdNe73/8wsx8zeArrX3J9XfyJK5BZyTmgKMXpwziPwMN6mN5E38id49Mmz737zjKnJPds4rwFx0mvg/fw3gb5G4GFzOuFU3prAMx9U+wWwJhR9NY/j/+ReeI2jplJFPaXzrPH+moDI+1ynnswsCvgc8GRw1RPArc65LcHpEX8GrgQeAR5xzj1vZrfWsbvbAJxzFwcT1Plm1i24rR/QHzgM5JjZH51zuzz20RtY65yr8oh1IHAjgcTagGVmtpDAG5pJwf23AFYDq07TnwZ3cgUjISGa8vLK0DbllSQkRkNeYDneo43fysoqSUw8MTc7MSG0mhPpli6bx8pVgbmkmZmdKSrKq95WXJxPSnLoR9EJCclUVJRRVVVFVFQUxUUFJCenNWrM9XVgdxFtanw03SYrhby9JT5GdMLiJa+zfPmbAGRldaGo8MTrUVSUT0pKq7ruGlGWLHmd5SveAgL9KDy5HycdM4mJKZRXlFYfV5HY16Y+1o8rKztCQsKJSnl8Qkyt82pZWWWwTWmwTegnhH5avPh1lgXHSLusLhTWOGcVRuBxU5clS+excmVgjGRmdqGo6MSnMMXF+SSnnDRGElJCz73FtceRNH3NKXGON7O1QEcCSeabZpYEDANm1LicyvGz6lBgTPD2v4DfeezzMuCPAM65zWa2AzieOL/tnCsCMLONQAfAK3E+lcuAmc650uB+XgYuJ5A4z3TOlQXXzw7+e6r+NLiCgjKSk2NJTAyctNu3b8WSxTtC2uzZU0zXrq3ZuaOQ9PQEKiurqKg4eq5COit5Bw6RkhJLUlIMZWWVZHdKZ9GCj/wOK2xDBo9kyOCRQOBLXEuXzaPPxcPZvXsLsXEJJJ+UOJsZ2dm92bBxKX0uHs6atQvo2XOQH6GftcWzcxg7dTDvTP+QXoOzKC2qoOCTQ36HBcCwoVcxbOhVQGDO+eLFr9O372Xs3JVLXFwCKSlN4z/KoUOvYmiwH5s3r2LxkkA/du3aEuxH7eOqc6eLWL9+CX37Xsbq1Qvo1fNSP0KvU1Mf68cV5J907u3QiiXvbw9ps2d3EV27t2HnjoOBc++RyDn3Dht2FcOGBcfIpsAY6df3MnbuzCW+KY2RISMZOiRw7t2cs4qlS+fRp89wdu3eQlxsQq2ihZnRKbs36zcspW+f4axes5CePS/xI/QGpcvRhWpOiXO5c66fmaUCrxGoFk8DCp1z/c5yn6c6Wg7XuF1F3c/lBqCvmZ13fKpGmPv3+oDxPOrXnzPiHKxeuZsrRnQKXBJpWwHFxRV07hKY5/XR1nz27S0mIyOZq6/pydGqYyxftrMxQjsjzsHSJTv4wsgemMHW3AMUFpbTvUfg29s5mw8QHx/NNaN7Ex0dBc7R66ILmfXSOiorT37J/NWtW39yt6zmoYfvICY6hnFjp1Rve+bZXzNm9LdISUnjS1+4nhdmPMxbb08nIyObgQPOyYcSZ+2n/5pAvxHZpLZOYMau7/H0z/5Di+jAzLHZf13J0rm5DB7Vlee23snhskoevHGmzxF769F9IDmbV/Ob304hJjqWiROnVm976ulfMGH8FFJS0nj//TksWDiTQ4cK+cPDd9Gj+wAmTDirr2GcE927D2Bzzmp++7vbiI6OZWKN2J5++heMD/Zj5FX/w/PP/4H585+nbdtsLrnkcz5GXVtzGevOwaqVu7niys6cF7wUaHFRBZ27Bs+9WwLn3raZKVzzlV4crQpcji4S9egxkM05q3nwN1OIiQkdI08+9QsmTJhCakoa770/h4ULZlJyqJCH/nAXPXoMCDkO/da92wByctfw+4duJzomhvHjTsQ27ZlfMW7MrYEx8qX/YfoLf+DNt56nbUY2gwZG1rlX6s8ict7gWTCzQ865pODt/sArQGdgIfAH59wMC7xt6uOc+8DM5gDPOOdeMLNbgIecc0lm1hF4zTl3kZl9F+jtnLspOEXjTQIV58nAIOfc1ODjvQb8zjm3oI7YXgRygHudc87MugK9CFSopwFDCE7VAL4WvD2NwBSO41M1/uqc+52ZLfbqz0kP6V54fm09ns3IcN3kfkx7MrK+tX82brjpUma8cPJL1PRMvK4vI+xev8OotwXuPmbN3OB3GPU2ZmxvAGa+vN7nSOpn7LiLAJr8WL/hpkD1ffpza3yOpH4mXd8fgFdmNe0xMnpMYHy8NGOdz5HU3/iJfXwt+d77w3mNlije98DIiC9vN8svBzrn1gAfEJgnfD1wk5l9QKD6OzrY7E7gu2a2HMgAijx29Wcgysw+BF4AbnDOHfZodzo3AxcCW4P7+huw1zm3mkCCvJxA0vx359ya4PoXgLXAS8C7NfZVV39ERERE5BxqNlM1jlebayzX/MWEkR532QMMCVaAJwErg/fbDlwUvF1B4NJxJz/WNAIJ7/Hla04TWzHwzTq2PQQ85LH+l8AvPdZ/XEd/RERERBpWxNeAG1ezSZzPwkDgT8HpDoXAN3yOR0REREQi2H9t4uycexfo25D7NLNl1L7Kxdeccx825OOIiIiINAZdVSPUf23ifC4458L+GW0RERERaVqUOIuIiIiIJ2siv+jXWJrlVTVERERERBqaKs4iIiIi4klznEOp4iwiIiIiEgYlziIiIiIiYdBUDRERERHxppkaIVRxFhEREREJgyrOIiIiIuJJXw4MpYqziIiIiEgYVHEWEREREU8qOIdSxVlEREREJAyqOIuIiIiIJ1WcQ6niLCIiIiISBlWcRURERMSTrqoRShVnEREREZEwqOIsIiIiIp5UcA6lirOIiIiISBjMOed3DNLw9KKKiIg0D77WfH/9i3caLaf40U+ujPj6tqZqNFNPPbHU7xDq7Ru3DGHxkp1+h1Fvw4a258Xpa/0Oo96undSPWTM3+B1GvY0Z25sRdq/fYdTbAncfAH9+9H2fI6mfKXcMB6Dq6DGfI6mfqBaBD3Bnvrze50jqZ+y4iwD43QML/A2knu7+4QgApj+3xt9AGsCk6/v7HYKvzGwk8AgQBfzdOffASduvB34QXDwEfNs590Fw23agBKgCjjrnBtU3HiXOIiIiIuLJzznOZhYFPAZ8AdgNrDCz2c65jTWafQxc4Zw7aGZXAU8Ag2ts/6xzLq+hYtIcZxERERGJRJcCW51z25xzR4DpwOiaDZxzi51zB4OLS4GscxmQEmcRERERiUSZwK4ay7uD6+pyE/B6jWUHzDezVWZ2S0MEpKkaIiIiIuLJ5x9A8Xpwzy8rmtlnCSTOl9VYPdw5t9fMzgfeNLPNzrlF9QlIFWcRERERiUS7gXY1lrOAvSc3MrM+wN+B0c65/OPrnXN7g//uB2YSmPpRL0qcRURERMSTWeP9eVgBdDWzbDOLASYBs0Pjs/bAy8DXnHO5NdYnmlny8dvAF4F6X/ZGUzVEREREJOI4546a2VTgDQKXo3vKObfBzG4Nbn8cuBdIB/4cnFZy/LJzFwAzg+taAP9yzs2rb0xKnEVERETEk/n7+ys45+YCc09a9/j/Z+++46Qqz/6Pfy6W7X0XkGUXWMrSpYmAJQZLoqAJYkksiSUaNYk1MT5P8qQoxhI1xURji71g1IiiYkdEaQvSi/QOwvY22/f+/TED7OzOwsDCziy/75vXvDjlOmeumTPnzD3Xuc/ZRsPXAtcGWG4jMOxI56OuGiIiIiIiQVDFWUREREQCCu1Nr6BLtQAAIABJREFUNcKPKs4iIiIiIkFQxVlEREREAlLF2Z8qziIiIiIiQVDFWUREREQCCvFfDgw7qjiLiIiIiARBFWcRERERCUgFZ3+qOIuIiIiIBEEVZxEREREJTCVnP6o4i4iIiIgEQQ1nEREREZEgqKuGiIiIiASknhr+VHEWEREREQmCKs5yQJlZyYw9ORszY+3Xe1i2dKff/OTkGL41rg/pneL5asE2VizbFaJMD8w5xysv/4tly3KJiormmmt/TXZ2TrO4Jx6/j82b1xIR0ZFevftz5ZW30rFjeO0mI0Zm0jUjifr6BnLnb6W4qLJZTHx8FGNP7klUVEeKijzkzttKQ4MLQbaBOeeY9s7TrFmziMjIaH5w8Y1kZvZpFjdnznS+nP0uBQXf8IffP0d8fFIIsm3ZHU+fz0nn9aN4TwVXH/9owJibHp7A2Ak5VHlquf+qqaxbHH77SPeeKZx6Wm86GKxauZvFX+3wm5+SGssZZ/Wlc5cE5s/ZwpLFO1tYU+g557j3vnuZNWsWsbEx3HvPvQwaNLhZ3Nx5c3nooQdpaHDEx8Vxzz330rNnzxBkHJhzjnfeeca7j0RFcfFFN5GZ2btZ3Jw505k9+z0KCr/h9797Nuz2kexeaZxxVl+sg7F86S5y5231mz9wUBdGj+0BQE1NPZ98tJa8PRWhSPWAumYkMnJUFmbGxvUFrF61u1nMyBMyychMpr6ugflzt1AU4NjcHukPoPhTxVlaZAYnndqLj97/mjdfX0rvvumkpMT6xVRX1zFvzuawbTDvtWxZLrt37+D+Pz/HVVfdyosv/CNg3NiTzuDe+57h7j89SW1NNbNmvd/GmR5Y14xEEhKief+91SxcsI0TRmUFjBs6LIO1a/J4/73V1NbU06t3WhtnemBr1iwiP38Xv779US644AamvvVkwLiePQdw7TV3kprSuY0zDM4Hzy3mjnNebHH+mPE5ZOWkc3nOw/zlumnc9tj32jC74JjBaeN6897bK5ny0mJy+nUmNa3Jfl5Vx5efb2LJoh0trCV8zPpiFlu2bOGD9z/grjvv4q7JkwPGTZ58Fw/8+UGmvjmVc889lyeeeLyNMz2wNWsWkV+wi9tvf4QLJv2Mt1raR7IHcM21fyQlDPcRMzjruzn897VlPPtULgMGdSE9Pc4vpqSkildfXsLzzyxk3pwtfPec/iHKtmVmMOrE7nz+2Qbef3c1PbJTSUqK8YvJ6JZEQlIM701bxYL5Wxk1unuIspWj7ag0nM2s3syWmNkKM3vdzOIOvtRB1znKzAK3dg5/nTPNbNRhLFd+JPM4XGb2nJlddLTW36lzAqUlVZSVVdPQ4Ni4oYAe2al+MVVVdeTnVYRVNTOQxYvncvIpZ2Fm9Ok7CI+nnOLigmZxw4aNwcwwM3r1HkBRYV4Ism1ZZmYymzcXAlBY4CEyMoKYmOYV8S7HJbJ9WzEAmzcVkpmZ3KZ5HszKVbmcMHIcZkbPHv2prKygtLSwWVxmZm/S0rqEIMPgLPtiC2WFLVeVTpk4gA9fWALAqvnbSUiJIa1rQlulF5QuxyVSUlxFaal3P1+/Lq/ZD63Kylr27CkP+/0cYMaMGUz8/kTMjGHDhlNWVkpe3p5mcWZGeYX3UF5WXk7nLuH1OVu1egEjR3wbM6NHj35UVlVQWlrULC6zW2/SUsMr9726ZiRRVFRJSUkVDQ2Or1ftoU9OJ7+YnTtKqa6u2zeckBgdilQPKC09jrKyairKa2hocGzdUkRmd/9jamZWMps3eo9hBQUeIqMCH5vbI7O2e7QHR6viXOmcG+6cGwLUADc0nmlmEYe6QufcQufczUcqwXB1OO/N0RIfH0VFRc2+8YqKGuLio0KY0eErLsr3a4ClpnaiqCi/xfi6ujrmzPmE448/sS3SC1psbCSVntp945WVtcTGRvrFREVFUFNTj/O1cTwBYkKttLSQ5JT9X6DJyekBG87tXefMJPK2lewbz9teSufM8DqVHp8QRXn5/v28vLyG+Pjwa7wEa8+e3XTt2nXf+HHHdWX37uYN58mT7+aGG67n9DPGMW3aNH567U/bMMuDKy0pJKXZPtL8x344S0yMpqyset94eVk1iQdoGB8/LINNG8PvOBAbG4XHs38fqfTUNDumxsZFNompJTYuvI67cmS0RVeNL4C+ZjbOzD4zs1eA5WYWYWYPmtkCM1tmZtcDmNl/zGzC3oV9VdULfcu/65uWZmZv+ZabZ2ZDfdPvNLPbGy27wsyyzSzezN4zs6W+aT9snKCZXWNmf2s0/lMz++vBXpgvp5lm9oaZfW1mL5uvM5CZ3W9mq3w5PtTotVzUaPnyRusJ5r0xM3vEt973gLYvM4R/wSkg55onfqB+Wy++8A/69zuefv2PP5ppHbogfpEHellht9kCbI+gXlx7E2BjBPoshlKgd92F3ycmaMHu6y+88DyPP/4En82YyaRJk/jzA/e3RXpBC7gN2ktJ7gBa+mx175HC8UO7MuuzDW2c0cEF87a3/y3Tsr1nYdvi0R4c1fMIZtYRGA984Js0GhjinNtkZtcBJc65E80sGphtZh8BrwI/BKabWRRwJvAzYEyjVd8FLHbOnW9mZwAvAMMPkMo5wE7n3Lm+vJqet34VWGZmdzjnaoGrgeuDfJkjgMHATmA2cIqZrQImAQOcc87MUoJYTzDvzQigP3A8cBywCngmyDwPWUVFDfGNKszx8f6/usPdp5+8zeefTwegV6/+FBburzoVFeWTkpIecLm33nqRsrISrrzq1jbJ82D69u1Erz7eXIsKPX5VjNjYSCora/3iq6vriYqKwMzbPo2LjaSqSUwozJn7Prm5HwOQldWXkuL9Ff+SkgKSklJbWrTdytteQudGp3Q7ZyWRv7MshBk1V15eQ0LC/v08ISEKT0X72c8BXnnlZV5/4w0Ajh8yhG+++WbfvN27v6FLF//+v4WFhaxZs4ZhQ4cBMP6c8Vx3/XVtl3AL5s59n9wFnwDefaS46T6SGF7XKhxMWZMKc0JiNOVlzT9bnTrHc/b4/vz3tWVUVdW1ZYpB8XhqiIvbv4/ExkU1O+56PLW+mApfjP/ZQTl2HK2Gc6yZLfENfwE8DZwM5DrnNvmmfxcY2qgCmwzkAO8D//A1GM8BZjnnKpv8EjkVuBDAOTfDzNIDNIYbWw48ZGZ/Bt51zn3ReKZzrsLMZgDnmdlqINI5tzzI15rrnNsO4HvN2cA8oAr4t68y/G6Q6znYe3MaMMU5Vw/s9OV81OTnlZOcHENCYjSeihp690ln5oz1R/Mpj6gzz5rImWdNBGDpkvl8+unbjBlzOhs3rCY2Nj5gw/nzz6ezYsVC7rjjATp0CI9rZ9evz2f9eu8XaEZGEn1zOrFtazFp6XHU1tYH/KLZs7ucrO4pbNtaTHavNHbsKGkW09ZOPmk8J580HoDVXy9kzpz3GTbsVLZuW0tMTBxJSe2rURCMOdPWMOnGMcx4dTmDxmRRUVJF4TdhcYnEPnt2l5GcEktiUjQV5TX0zenMxx+uCXVah+Syyy7nsssuB+Dzz2fy8iuvMGHCBJYtW0piQiKdO/ufnEtKSqKsrIzNmzeRnd2LuXPn0Kd38ztWtLWTThrPSb595Ouvv2LOXO8+sm3bOt8+0r5+XH6zq4zUtFiSk2MoK6tmwKAuvDdtlV9MYlI0Ey8YwvR3V4ftXSgKCzwkJkYTH+9tMPfomcrc2Zv9YnZsLyGnf2e2bikiPT2O2prAx+b2qJ0UgtvM0Wo4Vzrn/CrAvoZv43vMGHCTc+7Dpgub2UzgbLyV5ykB1h/47CLU4d/9JAbAObfWzE4AJgD3mdlHzrmml1r/G/gt8DXwbIuvrLnqRsP1QEfnXJ2ZjcZbLb8EuBE4o3F+vi4djTsMH/S98XVhabNzqM7B3NmbOXv8AKyDsW7NHoqLKuk/0PsltGb1HmJjI/n+pCFERkXgHAwe0pU3X19GbW19W6UZlKHDRrNs2Xz+544riYqO5ppr9vXo4a9//S1XX/1LUlM78cLzD5Oefhx/utvbnf6EUacyceKPQ5V2M7t2lZLRLZEJ5w2krq6BBfP339rpW6f1ZkHuVqqq6li2dCdjT+7JkOMzKC6qDLt+gwP6n8CarxfxwIM/JyoymosvvnHfvGee/RMXXfhzkpLSmD37PWZ+PpXy8mL+9vfbGNB/JBdd9IsQZu7v969cxPBxvUjuFMfr237Fs3/8jI6R3kPQtCcWMm/6WsZMyOHl9bdS7anlz1dPDXHGzTkHX8zcyPcmDsY6wNcr91BUWMngId5+witXfENsXCQXXzKMKN9+PnREN6a8tJjamvDazwFOO+3bzJo1i3PGn01MTAz3/OneffOuv+E67p78J7p06cLkuyZzy6230ME6kJScxJ/uvieEWTfXv/9Ivl6ziAcf+gWRkdFc3Ohz/+yzf+LCRvvI57Peory8mL8//Ev69x/JRRf+PISZ7+ec49OP1nHhD4fSwYzly3ZRkO9h2PBuACxdspOTTskmNrYjZ323HwANDY6Xnv8qlGk34xx8tXA73z6jDx3M2LihgNKSKvrkeIsvG9YVsGtnKd0ykzjv+4Ooq/fejk6OTXY0+tuZWblzLqHJtHHA7c6583zj1+FtyF7snKs1s37ADl/191zgWmAU0Mc5V9N4efPeXSPPOXe3b/rfnHMjzOxHwHnOuUvMbCSwAOiD9wLFQudclZmdD1zl6+Yx07fOhb6cFgGdgaHOueaXLzd5fQFe0yPAQuANIM45t8fM0oD1zrk0M/sdkOic+x9fHlO9PTmCe2/w/pi43jevC96uGj91zr3RJEX3zJPzgthS4e0n141lztytBw8Mcyef1IPXXl1y8MAw94NLhvPW1JWhTqPVzp80mHH2h1Cn0Wozfb/9//WP2SHOpHV+fvMpANTXNYQ4k9aJ6Oj9wTT1zRUhzqR1Jl0wBICH7p8Z2kRa6fb/HQfAqy8vDm0iR8All48Iac330Ye/bLOC3S9uOTXs69uhvFfKv/F2a1jkq77mAef75n2Et9/yNOdcoM52dwLPmtkywANc6Zv+X+AKX5eJBcBa3/TjgQfNrAGoxdtnOpDXgOEHajQHKRF428xi8FaPb/NNf8o3PRf4FP8qc2MtvTdT8Vaul+N9bZ+3Mk8RERERCdJRaTg3rTb7ps0EZjYab8DbNeK3AWJrgfQm0/Yt75wrBCYGWK4Sb//gpjYDzbqEOOfGNZl0KvC3pnEBlktompNv/MZGYaMDLLcbGNto0m9aWE+L7w3ebh8iIiIiR117udtFWwmPq59CzMxSzGwt3r7Zn4Y6HxEREREJP8fGn7VpJedcMdCv8TQzS8fbnaKpM51z7esu9CIiIiKHQQVnf2o4t8DXOD7QvaFFRERE5P8j6qohIiIiIhIEVZxFREREJCBdHOhPFWcRERERkSCo4iwiIiIiAang7E8VZxERERGRIKjiLCIiIiIBqY+zP1WcRURERESCoIqziIiIiASkgrM/VZxFRERERIKgirOIiIiIBKSKsz9VnEVEREREgqCKs4iIiIgEpLtq+FPFWUREREQkCKo4i4iIiEhAKjj7U8VZRERERCQIqjiLiIiISEDq4+zPnHOhzkGOPG1UERGRY0NIW67PPjW/zdoUV/90TNi30lVxFhEREZHAwr4p27bUcD5GPf/MglCn0GpX/uREVq/eE+o0Wm3gwC78Z8qSUKfRaj+8dDhT31wR6jRabdIFQ/jXP2aHOo1W+/nNpwAwzv4Q4kxaZ6abDMCMzzaEOJPWOeP0PgC89PzCEGfSOj+6chQAD9z3WYgzaZ07fnM6AFNeWhTiTFrv0h+NDHUK0oguDhQRERERCYIqziIiIiISkC4O9KeKs4iIiIhIEFRxFhEREZGAVHH2p4qziIiIiEgQVHEWERERkYBUcPanirOIiIiISBBUcRYRERGRgNTH2Z8qziIiIiIiQVDFWUREREQCUsHZnyrOIiIiIiJBUMVZRERERAJSH2d/qjiLiIiIiARBFWcRERERCUgVZ3+qOIuIiIiIBEEVZxEREREJSAVnf6o4i4iIiIgEQQ1nEREREZEgqOEsIiIiIgGZWZs9Wnj+c8xsjZmtN7P/DTDfzOwfvvnLzGxksMseDjWcRURERCTsmFkE8CgwHhgEXGpmg5qEjQdyfI/rgMcOYdlDposDRURERCQg6xDSqwNHA+udcxsBzOxVYCKwqlHMROAF55wD5plZipllANlBLHvI1HCWA+qWmcTosT0wM9atzWPFsm/85iclx3DKt3qRnh7H4q92sHLFNy2sKbScc/z73w/z1VfziI6O5uabf0ufPv1bjH/yyb8xY8b7vPrqR22Y5cF1zUhkxMhMzIyNGwr4evWeZjEjRmaS0S2J+voGcudtpaioMgSZHphzjnfeeYY1axYRGRXFxRfdRGZm72ZxhYW7mTLlb3gqy8js1psf/OBmOnaMDEHGgXXvmcKpp/Wmg8GqlbtZ/NUOv/kpqbGccVZfOndJYP6cLSxZvDNEmR7YHU+fz0nn9aN4TwVXH/9owJibHp7A2Ak5VHlquf+qqaxbvKuNswyOc47XXnuClSsWEBUVzRVX/pIePfo2i3vm6QfYsnUdEREdyc7ux+WX30RERPh8JWZ0S+LE0T0wg/Xr8gMeW0eN7k5mZjJ1dQ3Mnb2ZwkJPCDI9sF690zjzrBysAyxbsov587b6zR80+DhGj+0BQG1NPR99uIa8PRWhSPWAMjKSGHliFmawYX0Bq1fubhYzclQW3TKTqK9zzJu7maLC8Dv2tkOZwLZG49uBMUHEZAa57CFTVw1pkRmMPaknn3y0jrffXEGv3ukkp8T4xdRU15E7b2vYNpj3+uqreezatZ3HHpvCz39+B48//pcWY9ev/5qKivI2zC44ZnDCCVnMmrmRD6Z/Tc+eqSQlRfvFZGQkkpgYzfR3V7MwdxsnjMoKUbYHtmbNIvILdnH77Y9wwaSf8dZbTwaMe/+DFzn11PP49e2PEhubwMKFn7Zxpi0zg9PG9ea9t1cy5aXF5PTrTGparF9MdVUdX36+iSWLdrSwlvDwwXOLueOcF1ucP2Z8Dlk56Vye8zB/uW4atz32vTbM7tCsXLGQPXt2cNfkf3PZ5Tcz5ZVHAsaNHn06d975JL///b+oranhyy8/bONMW2YGo8f2YMYna3nn7ZVk90ojOdn/2NstM5nExBjenrqC+XO37Gt8hhMzOOu7/Xj9taU8/WQuAwcdR3p6nF9McXElU15ezHNPL2DO7M2cPX5AiLJtmRmcMLo7M2esZ/o7q+mZnUpSk+2R0S2JxMRo3n17FbnztzBqdPhtj8Nl1naPQE8fYJoLMiaYZQ/ZMdlwNrOZZnZ2k2m3mtm/WojfbGadAkz//sE6k5vZYbewzGySmTkzC78jBdCpUzylpdWUl1XT0ODYtLGQ7j1S/WKqquooyK+goaHVn8WjKjf3S8aNOwczo3//wVRUlFNYmN8srr6+nuee+xdXXvmzEGR5YGlpcZSVV1NRUUNDg2Pr1iIys5L9YjKzktm8uRCAggIPkVERxMSETxVtr1WrFzByxLcxM3r06EdlVQWlpUV+Mc45NmxYwZAhJwEwcuQ4Vq7KDUW6AXU5LpGS4ipKS737x/p1efTqneYXU1lZy5495WG/fyz7YgtlB6iOnTJxAB++sASAVfO3k5ASQ1rXhLZK75AsXTaPsWPPxMzo3XsAnsoKSkoKm8UNOf7EfRckZWf3o7io+fEgVNI7xVNWWk15uXdf37ypkKzuKX4x3bunsGljAQD5+RVERXUkNjZ8zsaAtzFZXFRJSXEVDQ2O1at307ef/1ftzh2lVFfVeYd3lpKYGB1oVSGVlh5PeVk1Fb7tsXVzEVlNjr1Z3ZPZvMl37M33EBUVQUxs+B1726HtQPdG41lA01N3LcUEs+whOyYbzsAU4JIm0y7xTQ+ac26ac+7+I5ZVc5cCX9I817AQFx9FRUXNvnFPRQ3xceF1YA5WYWEenTp12Teent45YMN5+vQ3GT36FNLSmv2OCrnYuEgqPbX7xj2e2mZflLGxkXgq9sdUemqJDcNtVlpSSErK/vc4OTmd0tICvxiPp4zYmHgiIiIaxTRvAIVKfEIU5eX794/y8hri48PvS/9I6JyZRN62kn3jedtL6ZyZFMKMWlZcnE9qaud946kpnSgubrlRXF9fx/z5Mxg0+IS2SC8ocXFReBofez01xMVH+cXExkX6HZ8rPDVht68nJERTVlq1b7ysrPqADeOhQzPYtKGgxfmhEhcXicfTeHs0P67Gxjb/voyL9d9m7VWI76qxAMgxs15mFoW3vTStScw04Arf3TXGAiXOuV1BLnvIjtWG8xvAeWYWDWBm2UA3IM7M5prZIjN73cwal0xu8k1fvrcCbGZXmdkjvuHjzGyqmS31PU5u+qRm9mszW+C7HcpdB0rQ99ynANfQqOFsZh3M7F9mttLM3jWz6WZ2kW/eCWb2uZl9ZWYf+jq/t6nwrpu1zHvNQFP+O2lhYT5z5nzGuede2DZJtZUw3GguUFJNDprBbLNQCnwOMAzf7CMhwBda4O0TBgKm1fLnZsorj9I3Zwg5OUOOWkpHQtP3O2AbI8w2SaAcW/rY9OiRwtBhGcycueHoJnWkBNFZ4Jg9HrQh51wdcCPwIbAaeM05t9LMbjCzG3xh04GNwHrgKeDnB1q2tTkdk+cRnHMFZpYLnAO8jbdh+inwf8BZzrkKM/sf4JfAZN9i+c65kWb2c+B24Nomq/0H8LlzbpLvFid+5ynN7Lt4b4UyGu8uNM3MTnPOzWohzfOBD5xza82s0MxGOucWARfgvRL0eKAL3o39jJlFAv8EJjrn8szsh8A9wE8O600KgqeihvhGVY64+Cg8jSqe4W769Df56KN3AMjJGUB+/v4L6QoK8khLS/eL37hxLbt27eCGGy4FoLq6ihtuuITHH3+17ZI+gKbV47i4SCor/bdHZWUtcfGR4CuwxQaICZW5c98nd8EnAGRl9fWrApaUFJCU6N/NIT4+icqqCurr64mIiPDGJPl3FQql8vIaEhL27x8JCf5VwmNJ3vYSOnfff2q6c1YS+TvLQpiRv5kz32G2r49yz545FBXl7ZtXVJxPSkp6wOXeffdlystLuO7ym9okz2A1rTDHxUX5nW0C8FTUEh8fxd5XGh8XFTb7+l5lZdUkJu3vC5yYGE15eXWzuM6d4zl7wgDeeG0pVZV1bZliUDyeWuLiGm+PAMdej/f7Mj/Pe2FjXHz4bY/DFeo/ue2cm463cdx42uONhh3wi2CXba1jteIM/t01LgE24b2P32wzWwJcCfRsFP+m7/+v8DZcmzoD370BnXP1zrmSJvO/63ssBhYBA/A2pFtyKbC3RfaqbxzgVOB151yDc+4b4DPf9P7AEOBjX/6/w9tf56jJz68gKTmahIQoOnQwevVOY/vWooMvGCYmTLiAv//9Wf7+92cZM+ZbzJz5Ac451qxZSXx8QrPuGKNGncxzz73NU0+9zlNPvU50dEzYNJoBCgs9JCZGEx/v3R49eqSyY3upX8yOHaVkZ3sboOnpcdTW1lNVFR5fRCedNJ5bbv4Lt9z8FwYPGs2ixZ/jnGPr1rXExMQ1axSbGX16D2HFirkALFo0k0EDR4ci9YD27C4jOSWWxKRoOnQw+uZ0ZtPG8OlKciTNmbaGs68YDsCgMVlUlFRR+E34XEA7btz3+L/fPcL//e4Rhg0/iXnzPsU5x8aNXxMbE09yclqzZb788gNWr1rET675Hzp0CK+vwoL8ChKTYoj3HXuze6WxfXuxX8z2bcX06u39QdCpUzw1tfVh11DbtbOM1NRYkpNj6NDBGDjwONav8+82k5gUzfkXDuG9d1aF7V0oCgsq/I+92als3+7fBNixvYTsXr5jb6c4amvqw/JHgLTeMVlx9nkL+KvvL8jE4m3Qfuycu7SF+L0/g+s5vPfFgPucc08cNNAsHW9DfIiZOSACcGZ2By2fUzRgpXPupMPI7bA4B/PnbuWss/vTwWDdunyKi6vo19/bf3DtmjxiYjty3vcHExkZAc4xcPBxvP3mcmprG9oqzaCccMJJfPXVPG644RKio2O4+ebf7Js3efKvufHG/wnLfs2NOQeLFm7n2+N6e29Ht7GQ0tIq+vT1fnluWF/Arp2lZGQkcu55A6mrbyB3/taDrDU0+vcfyddrFvHgQ78gMjKaiy/aXyx49tk/ceGFPycpKY1zxv+IKVP+xkcfTaFbt16ceOKZIczan3PwxcyNfG/iYKwDfL1yD0WFlQwe0hWAlSu+ITYukosvGUZUVATOwdAR3Zjy0mJqa+pDnL2/379yEcPH9SK5Uxyvb/sVz/7xMzpGehuT055YyLzpaxkzIYeX199KtaeWP189NcQZt2zIkBNZsWIBf/j9Nb7b0d22b94j//wDP/rxLaSkpDPllUdIS+vCgw/8CoDhI07m3HMvC1XafpyDBfO3cuZZ/bAOsGFdASXFVeT08x57163NY8eOErplJTPxgiH7bkcXbpxzfPLxWi6+ZBhmxvJluyjI9zB8RDcAlizeySmnZBMbE8l3zu7nXabB8cJzX4Uy7Wacg4ULtjHuzL77bgVaWlJF3xzvd8b6dfns3FFKRrdkzps4mPq6BubP3RLirI+cFvoe/3/Lwraf2hFgZq8B/fA2oh/FW00+wzm33szigCxfV4nNwCjnXL6ZjQIecs6NM7OrfNNv9N04e55z7u++rhrxzrlSMyt3ziX4umrcDZzpnCs3s0yg1jnX7Ea7ZnY9MNI5d32jaZ/jrSJ3xVsN/z7QGW9XjevwdmhfBfzYOTfX13WjXwv9ddzzzyxo9fsXalf+5ERWB7hPcXszcGAX/jNlSajTaLUfXjqcqW+uCHUarTbpgiH86x+zQ51Gq/385lMAGGd/CHEmrTPTeXvLzfisnfRtbcEZp/dQtWMSAAAgAElEQVQB4KXnF4Y4k9b50ZWjAHjgvs8OEhne7vjN6QBMeWlRiDNpvUt/NDKkLdf/vr6szRqKF148NOxb6eF1furImwIMA151zuUBVwFTzGwZMA9vd4pg3QKcbmbL8TbABzee6Zz7CHgFmOuLeQNIbGFdlwJNSzb/BS7z/b8dWAE8AczHe4VoDXAR8GczWwosAZpdoCgiIiJypIT4rhph51juqoFzbiqNuj4452YAJwaIy240vBAY5xt+DnjON7wb759qbLpsQqPhh4GHg8hrXIBp/9g7bGa3+6rW6UAusNwXswQ47WDrFxEREZEj75huOLdj75pZChAF3O27SFBERESkTbWTQnCbUcP5KPJVjAP9jeAznXMt3uU9UEVaREREREJLDeejyNc4Hh7qPERERESk9dRwFhEREZHA1FfDz7F+Vw0RERERkSNCFWcRERERCai93CaurajiLCIiIiISBFWcRURERCQgFZz9qeIsIiIiIhIEVZxFREREJCDroJJzY6o4i4iIiIgEQRVnEREREQlIfZz9qeIsIiIiIhIEVZxFREREJCDdx9mfKs4iIiIiIkFQxVlEREREAlLF2Z8qziIiIiIiQVDFWUREREQCUsHZnyrOIiIiIiJBUMNZRERERCQI5pwLdQ5y5GmjioiIHBtC2lni/ffXtFmbYvz4/mHfMUR9nI9R06d/HeoUWm3ChAE8cN9noU6j1e74zen8Z8qSUKfRaj+8dDjPPZ0b6jRa7aprRlNf1xDqNFotoqP3hOGMzzaEOJPWOeP0PgCMsz+EOJPWmekmA7T7feSqa0YDkJu7LcSZtM7o0d0BeO3V9n/s/cElw0OdgjSihrOIiIiIBKTb0flTH2cRERERkSCo4iwiIiIiAang7E8VZxERERGRIKjiLCIiIiIBqY+zP1WcRURERESCoIqziIiIiASkirM/VZxFRERERIKgirOIiIiIBKSCsz9VnEVEREREgqCKs4iIiIgEZB1Ucm5MFWcRERERkSCo4iwiIiIiAamPsz9VnEVEREREgqCGs4iIiIhIENRVQ0REREQCMtRXozFVnEVEREREgqCKs4iIiIgEpoKzH1WcRURERESCoIqziIiIiARkuh+dH1WcRURERESCoIqziIiIiASkgrM/NZzlgJxzTJ36FKtXf0VkZDSXXnoL3bv3aRb3xRfvMWvWNPLzv+Huu18kISEpBNm2rFfvNM48KwfrAMuW7GL+vK1+8wcNPo7RY3sAUFtTz0cfriFvT0UoUj2grhmJjBiZiZmxcUMBX6/e0yxmxMhMMrolUV/fQO68rRQVVYYg0wPLzExm9NgeWAdj3Zo8li/b5Tc/OTmGU07rTXp6HIsWbmflim9ClOmBOee49757mTVrFrGxMdx7z70MGjS4WdzceXN56KEHaWhwxMfFcc8999KzZ88QZByYc47XXnuClSsWEBUVzRVX/pIePfo2i3vm6QfYsnUdEREdyc7ux+WX30RERPh8jdzx9PmcdF4/ivdUcPXxjwaMuenhCYydkEOVp5b7r5rKusW7AsaF2rG0j7z44qMsXZpLdHQ01113B9nZOc3i/vWve9m0aS0RER3p06c/V199Gx07hs9nC7zH1q4ZvmPr/K0UBzi2xsdHMfbknkRFdaSoyEPuvK00NLgQZCtHS0i7apjZTDM7u8m0W83sXy3EbzazTgGmf9/M/vcgz1V+mDnWm9kSM1tpZkvN7Jdm9v9NF5fVq78iL28Xv/3t4/zgB7/gjTceCxjXq9dAfvazyaSmdmnjDA/ODM76bj9ef20pTz+Zy8BBx5GeHucXU1xcyZSXF/Pc0wuYM3szZ48fEKJsW2YGJ5yQxayZG/lg+tf07JlKUlK0X0xGRiKJidFMf3c1C3O3ccKorBBl2zIzGHNyTz7+aC1v/Xc5vXqnk5wS4xdTXV3H/LlbWLE8PBsDe836YhZbtmzhg/c/4K477+KuyZMDxk2efBcP/PlBpr45lXPPPZcnnni8jTM9sJUrFrJnzw7umvxvLrv8Zqa88kjAuNGjT+fOO5/k97//F7U1NXz55YdtnOmBffDcYu4458UW548Zn0NWTjqX5zzMX66bxm2Pfa8NswvesbSPLF2ay+7dO3jooef5yU9u49lnHw4Yd/LJZ/LAA89y331PUVNTw8yZ09s40wPrmpFIQkI077+3moULWj62Dh2Wwdo1ebz/3mpqa+rp1TutjTM98syszR7tQagbgFOAS5pMu8Q3PWjOuWnOufuPWFb+Kp1zw51zg4HvABOAPx6l5wo7K1bkcuKJp2NmZGf3p7KygpKSwmZxWVm9SUs7LgQZHlxGtySKiyopKa6iocGxevVu+vbz//21c0cp1VV13uGdpSQmRgdaVUilpcVRVl5NRUUNDQ2OrVuLyMxK9ovJzEpm82bv9iko8BAZFUFMTHhVbTp1TqCstJrysmoaGhybNhbQo0eqX0xVVR0F+RW4MK/UzJgxg4nfn4iZMWzYcMrKSsnLa34WwMwor/D+di8rL6dzl/D6gbl02TzGjj0TM6N37wF4WtjPhxx/4r4vuOzsfhQX5Ycg25Yt+2ILZYUtn2E5ZeIAPnxhCQCr5m8nISWGtK4JbZVe0I6lfWTRojmceup3MDP69h2Ex1NOcXFBs7jhw8fs+2z17t2fojD7bGVm7j+2FhZ4iIwMfGztclwi27cVA7B5UyGZmcnNYqR9C3XD+Q3gPDOLBjCzbKAbEGdmc81skZm9bmaNj2w3+aYvN7MBvuWuMrNHfMPHmdlUX3V4qZmd3PRJzezXZrbAzJaZ2V3BJuuc2wNcB9xoXtlm9oUvn0V7n8vMxpnZ52b2mpmtNbP7zexyM8v15d3HF/c9M5tvZovN7BMzO843vbOZfexb5xNmtmVvpd3MfuRbzxLfvIhDfdMPRUlJASkp+xuZKSmdKClpftALZwkJ0ZSVVu0bLyurPmDDeOjQDDZtCL/XGBsXSaWndt+4x1NLbGykf0xsJJ6K/TGVnlpi4/xjQi0uLpKKiup94xWeGuLio0KY0eHbs2c3Xbt23Td+3HFd2b27ecN58uS7ueGG6zn9jHFMmzaNn1770zbM8uCKi/NJTe28bzw1pRPFxS03XOrr65g/fwaDBp/QFukdMZ0zk8jbVrJvPG97KZ0zw6tbGRxb+0hRUT5pafs/W2lpnSksbPmzVVdXx+zZnzB06IltkV7QYmP9j7+Vlc2Pv1FREdTU1ON8v2U8AWLaI7O2e7QHIW04O+cKgFzgHN+kS4BPgf8DznLOjQQWAr9stFi+b/pjwO0BVvsP4HPn3DBgJLCy8Uwz+y6QA4wGhgMnmNlph5DzRrzvWxdgD/AdXz4/9D33XsOAW4DjgR8D/Zxzo4F/Azf5Yr4ExjrnRgCvAnf4pv8RmOFb71Sghy/3gb7nOcU5NxyoBy4PNvfD4VzzakZ7OZ2yV6B0A7wsAHr0SGHosAxmztxwdJNqS+FdkPJqDzkGEOz+8cILz/P440/w2YyZTJo0iT8/cLROkB2mgO9/y/v5lFcepW/OEHJyhhy1lI6KANsm0DYMS+0kzaYO9Tvk+ecfZsCAofTvf/zRTOvQBfG1F/C75shnIiEWDudw93bXeNv3/5vA94DZvp0rCpjbKP5N3/9fARcEWN8ZwBUAzrl6oKTJ/O/6Hot94wl4G9KzDiHnvbtHJPCIme1txPZrFLPAObcLwMw2AB/5pi8HTvcNZwH/MbMMvK9zk2/6qcAk32v4wMyKfNPPBE4AFvjem1i8jfcj6ssv32Pu3I8B6NGjr1/lqbg4n6Sk9tVnq6ysmsSk/f0DExOjKS+vbhbXuXM8Z08YwBuvLaWqsq4tUwxK0+pxXFwklZW1/jGVtcTFR4Jvk8UGiAk1j6eW+Pj9Ff/4uCg8npoQZnRoXnnlZV5/4w0Ajh8yhG++2d/HdPfub+jSpbNffGFhIWvWrGHY0GEAjD9nPNddf13bJdyCmTPfYbavj3LPnjkUFeXtm1dUnE9KSnrA5d5992XKy0u47vKbAs4PZ3nbS+jcff+p885ZSeTvLAthRoG1933k44/f3tdHuXfvfhQW7v9sFRbmkZoa+LP15psvUFpawi233NYmeR5M376d6NXHm2tRocfv+Bsb2/zYWl1dT1RUBGbe4kxcbCRVYXb8PRztrVh2tIVDw/kt4K9mNhJvQ3Ax8LFz7tIW4ve2eOo5vPwNuM8598RhLIuZ9fY99x68leHdeKvLHYCqRqGNW2YNjcYb2J/3P4G/Ouemmdk44M5GObaU+/POud8cTu7BOvXUczn11HMBWLlyIV9++R4jRnyLLVvWEhsbT3Jy+2o479pZRmpqLMnJMZSVVTNw4HG8M83vRASJSdGcf+EQ3ntnFUUH6CMZSoWFHhITo4mPj6KyspYePVKZO2eLX8yOHaXk5HRi65Zi0tPjqK2tp6oqvH4E5OeVk5QUTUJCFB5PLb16pzOrHVX4L7vsci67zHui5/PPZ/LyK68wYcIEli1bSmJCIp07+/dfTkpKoqysjM2bN5Gd3Yu5c+fQp3fvUKTuZ9y47zFunPfiuOXLc5k58x1Gjfo2mzatITYm8H7+5ZcfsHrVIm659V46dAh1T79DN2faGibdOIYZry5n0JgsKkqqKPzmsK4bP6ra+z7yne9M5DvfmQjAkiXz+Pjjtxk79nQ2bFhNXFx8wB9lM2dOZ/nyhfzmNw+GzWdr/fp81q/3ViEyMpLom9OJbVuLSTvAsXXP7nKyuqewbWsx2b3S2LGjae1O2ruQN5ydc+VmNhN4Bm/1eR7wqJn1dc6tN7M4IMs5tzbIVX4K/Az4u6//b7xzrrTR/A+Bu83sZd9zZwK1vv7LB2RmnYHHgUecc87MkoHtzrkGM7sSONT+xsnADt/wlY2mfwn8APizr2vJ3qtCPgXeNrO/Oef2mFkakOic8289HUGDBp3A6tULueeeG4iKiuaSS/ZXmZ58cjI//OEvSE5OZ9asd5gxYyplZUU8+ODNDBx4gl9sKDnn+OTjtVx8yTDMjOXLdlGQ72H4iG4ALFm8k1NOySY2JpLvnO09aeAaHC8891Uo027GOVi0cDvfHtfbezu6jYWUllbRp6/3S2jD+gJ27SwlIyORc88bSJ3vlknhxjmYN3cL3zlnAGawfm0excWV9B/grdSu+TqP2NhIzps4mMjICHCOQUO68tZ/l1Fb2xDi7P2ddtq3mTVrFueMP5uYmBju+dO9++Zdf8N13D35T3Tp0oXJd03mlltvoYN1ICk5iT/dfU8Is25uyJATWbFiAX/4/TW+29Htr/g98s8/8KMf30JKSjpTXnmEtLQuPPjArwAYPuJkzj33slCl3czvX7mI4eN6kdwpjte3/Ypn//gZHSO9jbBpTyxk3vS1jJmQw8vrb6XaU8ufr54a4owDO5b2kWHDxrBkSS63334FUVHR/PSnv94378EHf8u11/6S1NROPPvs3+nU6TjuuutmAEaNOpVJk34cqrSb2bWrlIxuiUw4byB1dQ0saHRs/dZpvVmQu5WqqjqWLd3J2JN7MuT4DIqLKtm0sflFtu2NCs7+LBz6d5nZJLxdMAY65742szOAPwN7z1X9zleV3QyMcs7lm9ko4CHn3Dgzu8o3/UbfBXZPAnsrwz9zzs01s3LnXILv+W4BrvWtuxz4kXMu4M95M6vH270iEqgDXsRbJW4wsxzgv4AH+Ay4yTmX4Kse3+6cO8+3jpm+8YWN55nZROBveBvP84ATfa+nC94fEanA53j7NfdyzlWb2Q+B3+CtcNcCv3DOzWuStps+/evgN0CYmjBhAA/c91mo02i1O35zOv+ZsiTUabTaDy8dznNP54Y6jVa76prR1NeFV+PicER09DYIZ3zWfiqRgZxxuve+8OPsDyHOpHVmOu+tCNv7PnLVNaMByM3dFuJMWmf06O4AvPZq+z/2/uCS4SFtus6es6XNGoqnnNwz7JvpIa84AzjnptKoe4JzbgbQ7JJa51x2o+GFwDjf8HPAc77h3cDEAMsmNBp+GAh8M8nmy7VYRXbOrQOGNpr0G9/0mcDMRnHjGg3vm+ecextv3+6mSoCznXN1ZnYScLpzrtq3zH+A/wSTu4iIiIgcOWHRcJZmegCvmfcPrdQA4XXvKhEREfn/gi4O9KeGM2Bm6Xj7Dzd1pu+WeW3KV8ke0dbPKyIiIiItU8OZffeTHh7qPERERETCiQrO/sLjni8iIiIiImFOFWcRERERCUgVZ3+qOIuIiIiIBEENZxEREREJyNrw3yHlZZZmZh+b2Trf/6kBYrqb2WdmttrMVvr+jsfeeXea2Q4zW+J7TAjmedVwFhEREZH25n+BT51zOXjvjPa/AWLqgF855wYCY4FfmNmgRvP/5pwb7ntMD+ZJ1XAWERERkYDM2u5xiCYCz/uGnwfObxrgnNvlnFvkGy4DVgOZh/9uqOEsIiIiIu3Pcc65XeBtIANdDhRsZtl4/0bG/EaTbzSzZWb2TKCuHoHorhoiIiIiElAo/3KgmX0CdA0w6/8OcT0JwH+BW51zpb7JjwF3A873/1+AnxxsXWo4i4iIiEjYcc6d1dI8M9ttZhnOuV1mlgHsaSEuEm+j+WXn3JuN1r27UcxTwLvB5KSuGiIiIiISUBj3cZ4GXOkbvhJ4u3nuZsDTwGrn3F+bzMtoNDoJWBHMk6rhLCIiIiLtzf3Ad8xsHfAd3zhm1s3M9t4h4xTgx8AZAW4794CZLTezZcDpwG3BPKm6aoiIiIhIQKHs43wgzrkC4MwA03cCE3zDX0LgG0Q75358OM+rirOIiIiISBDUcBYRERERCYK6aoiIiIhIQGHaUyNkVHEWEREREQmCKs4iIiIiElC4XhwYKuacC3UOcuRpo4qIiBwbQtpyXbx4Z5u1KUaM6Bb2rXRVnI9Rr726JNQptNoPLhnO44/MCXUarXbDjSfz39eXhTqNVrvw4qG8+vLiUKfRapdcPoKpbwZ1n/uwNumCIQC89PzCEGfSOj+6chQAzz2dG+JMWueqa0YDMM7+EOJMWmemmwzAF19uDm0irfStU7MBmPLSotAmcgRc+qORoU0g7JuybUt9nEVEREREgqCKs4iIiIgEpD7O/lRxFhEREREJgirOIiIiIhKQCs7+VHEWEREREQmCKs4iIiIiEpD6OPtTxVlEREREJAiqOIuIiIhIQKo3+1PFWUREREQkCKo4i4iIiEhA6uPsTxVnEREREZEgqOEsIiIiIhIEddUQERERkYDUU8OfKs4iIiIiIkFQxVlEREREAtLFgf5UcRYRERERCYIqziIiIiISkArO/lRxFhEREREJgirOIiIiIhKQKs7+VHEWEREREQmCKs4iIiIiEpDuquFPFWcRERERkSCo4iwHNWJkJl0zkqivbyB3/laKiyqbxcTHRzH25J5ERXWkqMhD7rytNDS4EGQbWPceKZzyrV6YwepVe1iyaIff/JSUWMad1ZfOnePJnbeVpYt3hijTA3PO8e57z7Jm7SKiIqO58MJfkNmtd7O4wsLdvPra36msLKdbRi8uvugmOnaMDEHGgXXNSGTkqCzMjI3rC1i9anezmJEnZJKRmUx9XQPz526hKMDnLtScc7zzzjOsWbOIyKgoLr7oJjIzm2+POXOmM3v2exQUfsPvf/cs8fFJIci2ZRndkjhxdA/MYP26fFau+KZZzKjR3cnMTKauroG5szdTWOgJQaYHl5mZzOixPbAOxro1eSxftstvfnJyDKec1pv09DgWLdwe8LWG2h1Pn89J5/WjeE8FVx//aMCYmx6ewNgJOVR5arn/qqmsW7wrYFyoOeeYMuUxli/PJSoqhp/85Ff07JnTLO6pJ+9n8+Z1RERE0KtXf358xS107Bg+TZSMjCRGnpiFGWxYX8DqlQGOWaOy6JaZRH2dY97czRQVht8x63Co4OxPFWc5oK4ZiSQkRPP+e6tZuGAbJ4zKChg3dFgGa9fk8f57q6mtqadX77Q2zrRlZnDqt3vz3jur+M8rS+jbrxOpqbF+MVXVdcyetSlsG8x7rV27mIKCXfzqtn9y/vnX8/a0pwLGffDRy5xy8nn86rZ/EhubwMKvZrRxpi0zg1Endufzzzbw/rur6ZGdSlJSjF9MRrckEpJieG/aKhbM38qo0d1DlO2BrVmziPyCXdx++yNcMOlnvPXWkwHjemYP4Jpr/0hKSuc2zvDgzGD02B7M+GQt77y9kuxeaSQn+2+PbpnJJCbG8PbUFcyfu4XRY3uEKNsDM4MxJ/fk44/W8tZ/l9OrdzrJKf6vpbq6jvlzt7Biefg1mPf64LnF3HHOiy3OHzM+h6ycdC7PeZi/XDeN2x77Xhtmd2iWL1/Ant07uPfeZ7niilt46cV/BowbM/YM/nTPv7lr8hPU1tbwxRfvt3GmLTODE0Z3Z+aM9Ux/ZzU9s1NJSm5+zEpMjObdt1eRO38Lo0aH5z4irRfShrOZTTIzZ2YD2vh5N5tZpwPMrzezJWa2wsxeN7O4o5DDDWZ2xZFe75GWmZnM5s2FABQWeIiMjCAmpnkVoMtxiWzfVgzA5k2FZGYmt2meB9LluARKSyopK62mocGxYV0+2U0a9lWVteTtKQ+rKnkgq1YvYMTwb2Nm9Ojej6qqCkrLivxinHNs3LiCIYPHAjByxLdZvXpBKNINKC09jrKyairKa2hocGzdUkRmd//PS2ZWMps3ej93BQUeIqMCf+5CbdXqBYwc4dsePfpRWVVBaWlRs7jMbr1JS+0SggwPLr1TPGWl1ZT7tsfmTYVkdU/xi+nePYVNGwsAyM+vICqqI7Gx4XMGY69OnRO8r6XMu69v2lhAjx6pfjFVVXUU5FfgwnhfX/bFFsoOUK08ZeIAPnxhCQCr5m8nISWGtK4JbZXeIVmyZC4nnXwWZkafPgPxeCooLi5oFjd06GjMDDMju1d/ioryQ5BtYGnp8ZQ3PmZtLiIry/+YldU9mc2bfMesfA9RURHExIbfMetw7N0ubfFoD0Jdcb4U+BK4JMR5NFXpnBvunBsC1AA3NJ5pZhGtfQLn3OPOuRdau56jLTY2kkpP7b7xysraZl+YUVER1NTU43zfQ54AMaEUHx9NeVnNvvHy8hri46NCmNHhKy0rJDk5fd94UlI6paWFfjEeTxkxMXFERETsiylpEhNKsbFReDz7t0elp6bZ5yU2LrJJTC2xceHzmdqrtKSQlJT9v8GTk9MpLW3eKAhncXFReCr2v9ceTw1xTfaP2LhIKhrFVHhqwnJ7xMVFUlFRvW+8IsBrORZ0zkwib1vJvvG87aV0zgyv7j97FRflk5a2/0xLamqngA3nverq6pg391OGDBnVFukFJa7J8cgT4HgUGxvlt494KmqIiz32PnsSwoazmSUApwDX4Gs4m9k4M5tpZm+Y2ddm9rL5foL4qsR3mdkiM1u+t0ptZnea2e2N1rvCzLJ9w2+Z2VdmttLMrjvMVL8A+vpy+8zMXgGWm1mEmT1oZgvMbJmZXd/oNXxuZq+Z2Vozu9/MLjezXF/efZrm7XvNo3zDncxss2/4Kt9reMfMNpnZjWb2SzNbbGbzzOzo94cI4gdgoB+J4VvL8Qr3/Frkmmfe9O13AV5dOP2QDyaXMEr3gAK912H1Zh8m1+RzFvAltZedqL3keSgCbJCm2yxcBE6r5X3k5Zf+Sb9+Q+jX7/ijltMR0fR1BfweDM9tIq0TyvMI5wMfOOfWmlmhmY30TR8BDAZ2ArPxNq6/9M3Ld86NNLOfA7cD1x7kOX7inCs0s1hggZn91zkXdDnIzDoC44EPfJNGA0Occ5t8DfES59yJZhYNzDazj3xxw4CBQCGwEfi3c260md0C3ATcGmwOwBC870kMsB74H+fcCDP7G3AF8PdDWFdQ+vbtRK8+3qpmUaHH75d1bGwklZW1fvHV1fVERUVg5j1IxsVGUtUkJpQqKqpJSNz/yz8hwb/CFu7mzvuAhQs/ASAzsy8lJfs/wqWlBSQm+f9+io9LoqrKQ319PREREZSWFpCUGD59zj2eGuLi9m+P2LioZp8pj6fWF1Phi/E/8xFKc+e+T+4C7/bIyupLcfH+U8olJeH1XgejaYU5Li6q2XvtqaglPj6KPN94fIBtFg48nlri46P3jcfH+Z/dOFbkbS+hc6PuTZ2zksjfWRbCjPzNmDGNL2Z5+yhnZ/ejsDBv37yionxSUgLvI9PefomyshJ+fMUtbZJnsPYfj7zi4pp/D1Z6vGcy8/O8x6y4+PDcR6T1QtlV41LgVd/wq75xgFzn3HbnXAOwBMhutMybvv+/ajK9JTeb2VJgHtAdaH4pb2CxZrYEWAhsBZ5ulNsm3/B3gSt8cfOB9EbrX+Cc2+WcqwY2AHsb1MuDzLuxz5xzZc65PKAEeKcV6wrK+vX5fPzhGj7+cA07tpeQne09yKWlx1FbW09VVV2zZfbsLt/XLzK7Vxo7dpQ0iwmVPbvLSU6OJTExmg4djD45nfb1RWsPThp7Djfd+BA33fgQgwadyOIln+OcY+u2tcREx5GU6N+H08zo3WswK1bOA2DR4s8ZOPDEUKQeUGGBh8TEaOLjo+jQwejRM5Ud2/0/Lzu2l+zrh56eHkdtTeDPXSicdNJ4brn5L9xy818YPGg0ixb7tsfWtcTExJGUlHrwlYSRgvwKEpNiiE/wbo/sXmls317sF7N9WzG9ent/THfqFE9NbX1YNgry88pJSoomwfdaevVOZ9vW4oMv2M7MmbaGs68YDsCgMVlUlFRR+E15iLPa74wzvs8f73yMP975GCNGnMzcOZ/gnGPDhtX8P/buOz6r+vz/+OsNEkggYathD5kqW3G1xT3qwK3Vuuro141af7W7dqm1raNqXcW9UFHcG1EREJmyFREZlZ0AYXP9/jgn4b6TO8mdBHPuE64nDx6573M+d3J9cu5z53Ou8xnZOTk0a9ayzGvGjHmDGTMmcullN2PvZmwAACAASURBVFGvXtS9SJOtWrk++TOrU3MWpfrM6hx+ZrUKP7M2ZMZnltu5Isk4S2oJHAbsI8mA+gQ3Pl4HNiUU3UZyjJtSbN9K8gVAo/BnDAGOAA40syJJo4v3pWGDmfUrFTMUp7/CTcBVZvZWqXJDStVhe8Lz7aT+nSfWoXSMVf1eO9XSpYXkt8nluON7sXXrdj4bv7Bk3w9+2IXPJixk48atTJu6hAMO6sg+++azZvUGvp6fOQ1TM/h4zHx+fFJvJDFn5nesXrWB3nvvAcDMGd+RndOAU8/oQ1ZWfcxg3775PPvkFLZs2RZx9Ml6dB/AnLmT+cc/r6JBVhannnJFyb5HHvsrpwz9OXl5LTjm6HN55tl/8c67T9MmvzODBh4WYdTJzODziYv40WFdqScx/6uVFBZspGu34I/pV/NWsnRJIW3a5nH8ib3Zui2Yji4T9egxgNlzJvH326+gQYOGnH7ajuMxfPifOfXUy8nLa8Enn7zGh2NeYt26Ndxx53X06DGA0069PMLIdzCDz8Yv5PAjuqN6we+/YM1GunUP+qXOm7ucxYsLaNOuKSedsk/JdHSZyAzGffoNRx7TM5hab+5y1qzZQI+eQV3mzF5OdnYDjj9pbxo0qA9m9N5nT156YRpbtmyPOPodfvvUafQb0pmmrXIY8e31DP/9B+zWIPgTMer+iYx7fS6Dj+vGk19ey6aiLdx64ciIIy7fvn32Z/r0z/jVTReSldWQCy+6vmTfHXf8hgvOH0az5i154vG7aNlyD/721+CG7IABB3PCiedGFXYSM5j42bcMOXyvYArN8DNrr27B+IYv561gyeJC8ts05fiT9i6ZQrOuiMugvdoSVVeN04DHzOyy4g2SPgQOqcb3WgAcH36PAUDncHtTYHXYaO4JHFCjiMt6C/g/Se+b2RZJ3YHFlb2oHAuAgcAEgt9NRpn0+WJSVe2jMfNLHq9fv5n33plXi1FVzcJv1rDwm8lJ22YmzMO5oWgLTzzyeW2HVWWSOOmE1D2ULjjvVyWPW7TYg8v/75baCqvKli4pZOmSwqRtX81L7kX1+WeLajOkapHE0JMuSbnvwgt/U/L44IN/zMEH/7i2wqqyJYsLGFXqLtG8ucuTnideNGeyxYsKGPn8tKRtc2bvqMuGDVsY8cyU2g6rSv70k+crLXPnla/VQiQ1J4lzzr0y5b5rr/1zyeMHHsyc6edSWbqkkNdGzUza9uW85Jk/Pv/s29oMyUUkqvshZwOlL5FfAH5Sje/1AtAi7DLxf8DccPubwG6SpgF/IuiusTM9BMwEJkn6Arif6l+I3E7QCB8LlDtNnnPOOedcbZJq738cRJJxNrMhKbbdBdxVatuVCY87JTyeCAwJH28g6G+cyrHl/PxOqbYn7C8zIaaZjQZGJzzfDvwq/J+odLkhqb6Hmf0hYftsoE/C9/hNuP0R4JFUcZfe55xzzjnnvl+Z1QPfOeecc865DFU3lrWphnCA4nspdh1elSnrnHPOOefcrmGXbTiHjeN+lRZ0zjnnnNtFxaXvcW3xrhrOOeecc86lYZfNODvnnHPOuYqpgiXSd0WecXbOOeeccy4NnnF2zjnnnHOpecI5iWecnXPOOeecS4NnnJ1zzjnnXEo+q0Yyzzg755xzzjmXBs84O+ecc865lHxWjWSecXbOOeeccy4N3nB2zjnnnHMuDd5VwznnnHPOpeY9NZJ4xtk555xzzrk0eMbZOeecc86l5AnnZJ5xds4555xzLg2ecXbOOeeccynJV0BJ4hln55xzzjnn0iAzizoGt/P5QXXOOefqhkhTvt9+u6bW2hTt2zfL+PS2Z5ydc84555xLg/dxrqOGPzg+6hBq7MJLBrN4UUHUYdRY23ZNefKxz6MOo8bOOW8gL780I+owauykoXtz+y2jow6jxm745RAAbvvbB9EGUkM33nQoABMmfBtxJDWz//7tAfjo4wXRBlJDPzikEwBD9LtoA6mh0XYzAM8+PSXiSGruzLP7RfrzMz4FXMs84+ycc84551waPOPsnHPOOedS8lk1knnG2TnnnHPOxYqkFpLekTQv/Nq8nHILJE2XNEXSxKq+vjRvODvnnHPOubj5JfCemXUD3gufl+dQM+tnZoOq+foS3nB2zjnnnHNxcxLwaPj4UWBobbzeG87OOeeccy4lqfb+V9EeZrYUIPy6eznlDHhb0ueSLq3G65P44EDnnHPOOZdxJL0L7Jli16+r8G0ONrMlknYH3pE028zGVDcmbzg755xzzrmUopxVw8yOKG+fpO8k5ZvZUkn5wLJyvseS8OsySSOB/YExQFqvL827ajjnnHPOubgZBZwfPj4feLl0AUmNJeUWPwaOAr5I9/WpeMPZOeecc87FzS3AkZLmAUeGz5HURtLrYZk9gI8lTQUmAK+Z2ZsVvb4y3lXDOeecc87FipmtBA5PsX0JcFz4eD7Qtyqvr4xnnJ1zzjnnnEuDZ5ydc84551xKvuJ2Ms84O+ecc845lwbPODvnnHPOuZSEp5wTecbZOeecc865NHjG2TnnnHPOpeYJ5ySecXbOOeeccy4NnnF2zjnnnHMp+awayTzj7JxzzjnnXBo84+ycc84551LyhHMybzi7CrVt15TBB3ZEEnPnLGP61KVJ+5s2bcQhP+pCy1aNmfTZt3wx/X8RRVoxM+Pf9/yD8ePH0qhhI2688Xd0796zTLlbb/0jU6dNonHjJgD8vxt/z157da/tcMuV3yaPQfu1R4Ivv1zBzC++K1Nm4H7tads2j63btvPpJwtYvWpDBJFWzMwYNephZs+ZRIMGDTnjjCtp17ZrmXKfjH2djz9+lZUr/8fvf/cIjRvnRRBt+Tp1bsFhR+yF6onpU5cyYdzCpP29eu/O/gd0AGDz5m28+/Zcli9bH0WoFercpQWHH9EN1YNpU5YyvlQ9eu+9R0k9tmzexttvzcnIekDw3nr88XuYOnUCDRs25NJLb6RTp25lyt1771/5+uu51K+/G1279uDCC4ex226Z8yfRzHj66fuYPn0CWVmNuOii6+nYsWw9HnzgFhYsmEf9+vXp3LkHPz3vmoypx40PD+XA47uzZtl6Ltz3npRlrrrzOA44rhsbi7ZwywUjmTd5acpyUdszP5f+A9oiiflfrWT2rGVlyvQf0Jb8Nnls27adCeMWsnp15n32uprzrhquXBIccHAn3n5zDiOfn0aXri1p2iw7qcymTVsZP/YbvpiWmR92xcZPGMviRd/y+GMvcN11N3HHnbeWW/ayS6/mwQee5MEHnsyoRrME+w3uwAfvzePVUTPp1KkFeU0bJZVp0zaPvLyGjHppBuM/Xcj+gztGFG3FZs+ZxIoVS7nxF/dw6ik/Z+TIB1KW69SxJ5dc/AeaN29duwGmQYIjjurGC89NY/iDE+jZe3datsxJKlNQsJFnnpzCo/+dyLix33DUMT0iirZ8QT26M+K5qTz8wAR69d6jTD3WrNnA009O5pGHP2PsJws4+tiyF52ZYurUCXz33WJuv/1RLrpoGMOH35my3EEHHc5ttw3nb397kM2bNzN69Ou1HGnFpk//jGXfLeavfx3OeeddwxOP352y3OADDuPPf3mIP958P1u2bOajj96o5UjL9+Yjk7nxmMfL3T/42G6069aSc7rdyT8uHcWw+06oxejSJ8HAge0YM3o+b74+m44dm5OX1zCpTH5+Lrm5DXn91VlMnPAtAwe1iyja74FUe/9joNKGs6RtkqZI+kLSCEk5FZS9QNK/w8c/l3TezghSUo6kJyVND+P4WFKTSl6zQFKrnfHzw+93g6TZ4c+fWly3nf1zqhhTye/7+9CqdRPWFm5k3dpNbN9uzP9qFR06Nk8qs3HjVlasWM/27fZ9hbFTjP1kDEcedRyS6N17X9atW8vKlSuiDqtKWrZszNq1G1m3bjPbtxvfLFhN+/bNksq0a9+M+V+tBGDlivVkZdWnUXZmZJ8SzZwxgQEDhyCJjh17sGHDegoLV5Up17ZtF1q02D2CCCu3Z34eq1dvoKBgI9u3G7NnLqNrt+SPgiWLC9m0aWvJ4ya5DVN9q0jlt8ljzeoNFKwJ6jFr1nfs1T1FPTaG9VhSSG4G1qPYpEljOeSQI5HEXnv1pqhoHWvWrCxTrl+/wUhCEl269GD16sz6PJgy5VMOPOgIJNG1ay+KitanrEefPvuX1KNT58yqx7SPvmFtBXe8Dj6pJ289NgWAmeMX0aRZI1rsWeGf9ki0aJHD2nWbWL8++OxduHA1bds1TSrTtl1TFiwIPsNWriyiQVZ9GjXKvM9eV3PpZJw3mFk/M9sH2Az8PJ1vbGb/MbPHahTdDtcA35nZvmEcPwO27KTvXSlJPweOBPYPf/4P2QW6/eQ0zmL9us0lz4vWb6Zx4wYRRlR9K1YsY/fWe5Q8b916d1asKHurDeDh/97HxRf/hHvu/SebN29OWSYK2TkNKFq/421fVLSZ7Jzk45GT04Cios1JZXJysmotxnQVFK6iWdMdjbNmTVtSkKLhnMlycxuydu2mkufr1m6qsEG5b998vp6feXVs0qQhaws3ljxfW0k9+vTJ5+uvyjbgMsXq1Sto0WLHHYoWLVqzalX5jcmtW7fyySfv0qfPfrURXtrWlKpH8+atUjaci23dupVxn77HPvsMqo3wdorWbfNY/m1ByfPliwpp3TazumNB8Nm7oSjxs3cL2dnJn73Z2cmfzxuKtpT5fI4r1eL/OKhqV42PgL0ktZD0kqRpksZJ6lO6oKQ/SLohfLyXpHfDTO0kSV0lPS7ppITyT0o6sZyfmw8sLn5iZnPMbFP4upckfS5phqRLU71Y0rmSJoSZ8/sl1Q//PxJmkKdLGlZBvX8FXG5mheHPLzCzRxP2XxXWa7qknuHP3F/SWEmTw689wu0XSHpR0puS5km6LSHOdZL+Ev6exknaI9zeWtILkj4L/x9cQaw7Tao3cWbnlcuXKm6luC108cVX8OgjI7j33kdYW1jIM8/srGu/mkvvLlaKQhl50MoGVReWdbVyftntOzRj3z57MuaDr2o5osqlel9ZOe+ZDh2a0advPqNHZ149ilmK4FOd68UeffROevbsQ48e+36fYVVZ6mNQfj2efOJuunffh+7dM6seFUpxXFIdv9iqQ1VxO6R9H0HSbsCxwJvAH4HJZjZU0mHAY0C/Cl7+JHCLmY2U1Iigwf4QMAx4WVJT4CDg/HJe/1/gbUmnAe8Bj5rZvHDfRWa2SlI28JmkF8ys5LJcUi/gTOBgM9si6V7gHGAG0DbMICMp+Z73jtfnArlmVtFfihVmNkDS5cANwMXAbOCHZrZV0hHAX4FTw/L9gP7AJmCOpLvN7FugMTDOzH4dNqgvAf4M3An8y8w+ltQBeAvoVUE8O8X69Ztp3GRHtjKncVbSFXWme+mlEbz2+ksA9OjRm2XLdwykW758GS1blu0327JlkAXNysrimGNO4LnnnqidYNNQtH4LOQkZ/5ycrKQsCCRmmNeXlCnakBlZ87Fj32D8hHcAaN9uL9YU7MgCrilYSV5e8/JempFKZ2ab5DZk3dqyv+tWrRtz9LE9eOG5aWwMuztkkrVrN5Gbt6OvfG5uQ9at21SmXOvWjTn6uJ48/9xUNm7IrHq8887LJX2Uu3TpzqpVy0v2rVq1nObNW6Z83YsvPkZhYQHXXFNR3qT2vP/+KD4aE/RR7tQpuR6rV6+gWbMWKV836uUnWLu2gJ+ed02txLmzLF9UQOv2O7o8tG6Xx4olayOMKLXS2eOcnAZs2JD82bthQ/j5HH6sZaco4+qGdBrO2ZKmhI8/Ah4GxhM2As3sfUktw8ZvGWHDs62ZjQzLF98T/FDSPZJ2B04BXjCzlJ/GZjZFUhfgKOAIggbygWY2C7ha0slh0fZANyDxftbhwMDwNQDZwDLgFaCLpLuB14C3y6m/qPy68cXw6+dhXQCaAo9K6ha+PvGezXtmVgAgaSbQEfiWoCvMqwnf68jw8RFA74SsSV74e/1erVi+jry8RjTJbUjR+s106dqCDzMwY1aeoUNPZ+jQ0wEYN+5jXnppBIcdehSzZn1B48ZNShrJiVauXEHLlq0wMz7+5EM6dS4700NUVq5cT25uIxo3CRrMHTs155OPvk4qs+jbNfTouTvfLFhNy1aN2bxlW8Y0cg466FgOOuhYAGbNmsjYsW/Qr+8hLFw4l+xGOeTlpW4UZKr/LV1L8xbZNG3aiLVrN9Gz9+68NmpmUpncvIacdMo+vP7qrIwdYb90yVqaN99Rj1699uCVUTOSyuTmNWToqfvw2iszM3KWliOPPIkjjwxuYE6ZMo533nmZAw44lK++mkVOTmOaNSvbcB49+nWmT5/ITTf9nXr1MmOc/GGHnchhhwU3XqdNHc/7749i//2HMH/+bLJzclLWY8yYN5gxYyLX33BrxtQjXWNHzeHkKwfz/jPT6T24HesLNrLqf+uiDquMVauKyM1tSOPGWWzYsIUOHZrz6dhvksosXlxIt26tWPjNGlq2zGHLlm0ZeaFcHTEZs1dr0mk4bzCzpGyyUt/3Kq9xWdGv/HGC7O9ZwEUVBWFm6wgaqC9K2g4cF3ZlOAI40MyKJI0GGpV6qQgy1DeVCUzqCxwNXAGckSoGMyuUtF5SFzObX054xemZbez4nf4J+MDMTpbUCRidonzp12yxHfepErfXC+uY9BerotuPO4MZjBu7gKOO7YEk5s1ZzprVG+jRKxisNWfWMrKzG3DC0H1okFUfM6P3PvmMfH4aW7Zs+15jq6rBgw9m/PixnPvTU2jUqBE3/uK3Jft+edO13HD9r2nVqjV/+etvKShYg5mxV9fuDBv2ywijTmYGEycs5LAjuiGJr75cQUHBRrqFA7nmzV3BksWFtG3blBNP3odtW7fz6dgF0QZdjp49BzJ7ziRuve1ysrIacvrpV5bse/i/f+a00y6naV4LPv7kNT4cPZK169bwz38No2fPAZx+2hURRr6DmfHe2/M49cw+1JOYPm0pK1cU0bdfGwCmTlnCgQd3Ijt7N444KpidZft244lHP48y7DLMjHffmcvpZ/VFCfXo1z+ox5TJSzj44E5kN2rAkUcH9bDtxmOPZFY9ivXtO5gpUyZwww3nkZXVkEsu+UXJvr///VdcfPF1NG/eiuHD76BVqz344x+vBmDQoEM4+eSfRhV2Gfv22Z/p0z/jVzddSFZWQy686PqSfXfc8RsuOH8YzZq35InH76Jlyz3421+vBWDAgIM54cRzowo7yW+fOo1+QzrTtFUOI769nuG//4DdGgSN+1H3T2Tc63MZfFw3nvzyWjYVbeHWC0dGHHFqZjBp4iJ+NKRLMB3d/FUUFm6k617BhcxXX65k6ZJC8vNz+fHxvdi6bTsTxi+s5Lu6uFJl/YkkrTOzJqW23QUsN7M/SRpC0I2gv6QLgEFmdqWkPwDrzOx2SeMIumq8JKkhUD9s6O4BTAD+Z2aDK4jhYGCmma2WlEXQXeReggGCF5vZCWHf4inAMWY2WtICYBCwO/AyQVeNZZJaALkE97I3hw3jfsAjpS8QEn7+5cAJwJlh+TzgLDN7oPjnmNkKSYOA281siKSRwBNm9kL4u7jAzDol/o7C7/1q+JrRib/rsFvK8WZ2gaSnCLrG/D3c1y/Mwid9rwQ2/MHx5f06Y+PCSwazeFFB5QUzXNt2TXnyscxsZFTFOecN5OWXZlReMMOdNHRvbr9ldNRh1NgNvxwCwG1/+yDaQGroxpsOBWDChG8jjqRm9t+/PQAffbwg2kBq6AeHdAJgiH4XbSA1NNpuBuDZp6dUUjLznXl2v0hzvitXrK+13totWzXO+Px2dedK+QMwXNI0oIjy+yYX+ylwv6SbCRq7pwPzzew7SbOAlyp5fVfgvjDTXY+ga8ULQBbw8zCOOcC40i80s5mSfkPQR7pe+POvADaEdSi+t1UmI53gPqAJQXePLeH3+EclMd9G0FXjOuD9SspW5mrgnrCeuwFjSHN2E+ecc845t3NU2nAunW0Ot60CTkqx/RHgkfDxHxK2zwMOK11ewZzQ3YCnK4nhMYIBiKVtIhiwmOo1nRIePws8m6LYgIp+bsLrjaAhfFuKfYk/ZyIwJHz8KZC4esZvw+2PEP6OwufHJzxukvD4eeD58PEKggGOpX920vdyzjnnnNuZMj4FXMsiG0kQzjQxG7i7eKCcc84555xzmSqyZW3M7F2gQ+I2SUcDpddC/trMTqYWSLoHKD1H8p1mNrw2fr5zzjnnXCbxWTWSZdR6kGb2FsEcxVH9/MwYru+cc8455zJORjWcnXPOOedcJvGUc6J4zZbunHPOOedcRDzj7JxzzjnnUvI+zsk84+ycc84551wavOHsnHPOOedcGrzh7JxzzjnnXBq8j7NzzjnnnEvJ+zgn84yzc84555xzafCGs3POOeecc2nwrhrOOeecc64c3lcjkWecnXPOOeecS4NnnJ1zzjnnXEo+ODCZZ5ydc84555xLgzecnXPOOeecS4M3nJ1zzjnnnEuD93F2zjnnnHOpeR/nJJ5xds4555xzLg0ys6hjcDufH1TnnHOubog057u2YGOttSlymzbK+Py2d9Woo55/bmrUIdTYaWf0ZcSz8a/H6Wf25blnpkQdRo2dcVY/XhgxLeowauzU0/vwzJOTow6jxs46pz8ATz8xKeJIaubscwcAxP4cOeOsfkDdOR7PPh3v43Hm2cHxGKLfRRxJzY22m6MOwSXwrhrOOeecc86lwRvOzjnnnHPOpcG7ajjnnHPOuZR85cBknnF2zjnnnHMuDd5wds4555xzLg3ecHbOOeeccy4N3nB2zjnnnHMuDT440DnnnHPOpeajA5N4xtk555xzzrk0eMbZOeecc86l5PnmZJ5xds4555xzLg2ecXbOOeecc6l5yjmJZ5ydc84555xLg2ecnXPOOedcSp5wTuYZZ+ecc84559LgGWfnnHPOOZeaz+OcxDPOzjnnnHPOpcEbzs4555xzzqXBG87OOeecc86lwfs4O+ecc865lLyHczLPODvnnHPOOZcGzzg755xzzrnUPOWcxBvOrkJmxmuvD2fO3Mk0aNCQU0+5nLZtupQpt2r1Mp597g42FK2jTZvOnHbqVey2W+a8vYrrMXdeWI+TL6dNOfV47rk72LBhHfltOnPaKZlVD4D+A9qyZ34e27ZtZ8L4haxZvaFMmcaNszjgoI5kZe3G6tVFTBi3kO3bLYJoUzMzXn1tOHPmTiKrQUNOPfWK1O+rVd/xTHg82uR35vTTrmK33RpEEHFqe+bnMmBQOyQx/8uVzJr5XZkyAwa2Jb9tU7Zt3c74T79hdYrjFbX8/DwG7NcOCb76ciWzZqSox6B2tGmbx7atxrhPF7B6VebVo1hdOEfqyjHZMz+X/gPaBufIVyuZPWtZmTL9B7Qlv014vMYtzLhz5MaHh3Lg8d1Zs2w9F+57T8oyV915HAcc142NRVu45YKRzJu8tJaj3PVIagE8C3QCFgBnmNnqUmV6hGWKdQF+Z2Z3SPoDcAmwPNz3KzN7vbKf6101qkDSaElHl9p2raT5kn5ZyWs7SfrJ9xvhzjd33mRWrPwf1117F0NPupRRrzyUstxbbz3BwQf+mOuG3UWj7MZ8Pun9Wo60YnPnTWblyv8x7Jq7GHpi+fV4++0nOOigHzPs2rvIbpR59dgzP5cmTRryxmuzmPjZtwwc1C5luT5985k7ZzlvvDaLLZu30blLi1qOtGJz505m5cqlXD/sboYOvYyXRz2Ystybbz/JwQcdz/XD7iY7uwkTP8+c4yHBoP3a8+EHX/HGq7Po0Kk5eXmNksrkt8mjSV4jXhs1k8/GL2TQ/u0jirZ8Egzcvz2j3/+S11+ZRcdOzclrWrYeubkNefXlmUwY/w2D9u8QUbSVqwvnSF05JhIMHNiOMaPn8+brs+nYsTl5eQ2TyuTn55Kb25DXX53FxAnlH68ovfnIZG485vFy9w8+thvturXknG538o9LRzHsvhNqMbpd2i+B98ysG/Be+DyJmc0xs35m1g8YCBQBIxOK/Kt4fzqNZvCGc1U9DZxVattZwPlmdkslr+0ExK7hPGvWRPr3+yGS6NC+Oxs3rKdwbdIFHWbG/K9nsPfeBwAwoN8QZs76LIpwyzVr9kT6hfVo3747GzeuZ2159egd1KN/vyHMyrB6tG3blAULVgGwamURDRrUp1Gjshnx3ffIZdG3awBY8PUq2rZtWqtxVmbmrM/o3+9HO95XG8t5X83/gn2K31f9f5RRx6NFyxzWrt3E+nWb2b7dWPjNatq2T/49t23XlAXzg+O1cmURDbJSH68otWjZmHWJ9ViwmnbtkuvRrn1TFnwd1mNFEVlZ9WmUnVn1KFYXzpG6ckxatMhh7bpNrF8f1mPhatq2S3GOLMjsc2TaR9+wtoJs/sEn9eStx6YAMHP8Ipo0a0SLPZvUVnjfO9Xivyo6CXg0fPwoMLSS8ocDX5nZN1X9QYm84Vw1zwPHS2oIQRYZaAPsJenf4bZHJN0laWyYiT4tfO0twA8kTZE0LMxAfyRpUvj/oPD19STdK2mGpFclvV78PSQNlPShpM8lvSUp//uucGHhKpo2bVXyPK9pSwoLVyWVKSpaS6NGOdSvXz8s06JMmaitLV2PvDTrsTaz6pGd3YANRVtKnm/YsIXs7OSuC1lZ9dm8eRsW3nUuSlEmaoVrV9G0acuS52kdj7yWFGTQ+yo7O4uios0lzzcUbS7ze87OaVCqzBayczLrWOSUirEoRYzZ2VmsX59QZv1mcrKzai3GqqgL50hdOSbZOcnHoqio7O85O7sBResTjlcGniOVad02j+XfFpQ8X76okNZt8yKMaJexh5ktBQi/7l5J+bMIEqCJrpQ0TdJ/JTVP54d6w7kKzGwlMAE4Jtx0FkHfmdId4/KBQ4DjCRrMENxC+Ci8HfAvYBlwHxk4LAAAIABJREFUpJkNAM4E7grLnUKQnd4XuBg4EEBSA+Bu4DQzGwj8F/jLTq5iGVamaqR1VZhpYwnMUvRdTGMZ0UyrRzoBpapW5vTcDKU4HqXDTvney6ADkk4sGRRu1ZT+1ad8T2XcuypQV86R0uJ8TKoqbtVI8YZK+TcnrlSL/0v/aOldSV+k+H9SlaogZQEnAiMSNt8HdAX6AUuBf6TzvTLrfkg8FHfXeDn8ehHQp1SZl8xsOzBT0h7lfJ8GwL8l9QO2Ad3D7YcAI8LX/0/SB+H2HsA+wDsKTtL6BAd6pxs3/k0+m/geAO3adqWgYEXJvsKCleTmJV+U5eTksnFjEdu2baN+/foUFqwiNy/6/oLjxr/JxM+DerQtXY/CleTlplGP3OjrsdderejcNcjOrl5VlJSNyc5uwIYNW5LKb9q0jays+khB+zQnuwEbS5WJwqfj3mTixHcBaNt2LwoKVpbsKyxcWeY90zgnL/l4FK4kLwOOR7Gios3k5OzI8GXnZJU5FkVFW8Iy68MyyRm4TLAjxkBOTtn31IaizTRunMWK5UE9chqXrWuU6so5UqwuHBMomz1OWY8NW8hp3ADCj+fsFGUy3fJFBbRO6KbVul0eK5asjTCiusPMjihvn6TvJOWb2dLwDnzZkac7HAtMMrOSUbaJjyU9CLyaTkyeca66l4DDJQ0Ass1sUooymxIel5f/GAZ8B/QFBgFZlZQXMCOhE/u+ZnZU1cOv3AGDj+GqK/7OVVf8nV699mfylDGYGQu/nUvDRjllGpyS6NJ5b2bMGAfApCmj6dVz0PcRWpUcMPgYrrz871x5+d/p3XN/poT1+DasR26KenTuvDczZgb1mDxlNL16RV+PL79cwTtvzeGdt+aweFEBnToFjccWLXPYsmUbGzduLfOaZd+to137ZgB06tyCxYsLypSpbQcecAxXXXk7V115O71778fkKR+WvK8aNSz/ffVF8ftq8of06rVfFKGntGplEbm5DWncOIt69USHjs1ZvCj597x4UQGdwkFnLVvmsGVz6uMVpVUr1yfXo1NzFqWqR+ewHq3CemzInHrUlXOkWF04JgCrVpU6Rzo0Z/GiwqQyixcXlhyvlhUcr0w2dtQcjj6vHwC9B7djfcFGVv1vXcRR7TwRJpwrMwo4P3x8PkFCszxnU6qbRqnuricDX6TzQz3jXEVmtk7SaIKuEqX7ylRkLZCb8LwpsMjMtks6nyCDDPAxcL6kR4HWwBDgKWAO0FrSgWb2adh1o7uZzahRhSrRo3t/5s6dxD//dTUNGmRxyimXl+x79LG/cfLQy8jLa8HRR53DM8/dwTvvPUOb/M4MGnjY9xlWlXXv3p+58ybxzzuuJqtBFqecvKMejz3+N4aeFNbjyHN4dsQdvPveM+Tnd2bggMyqx9KlheS3yeW443uxdet2Phu/sGTfD37Yhc8mLGTjxq1Mm7qEAw7qyD775rNm9Qa+np85fYMBenQfwJy5k/nHP6+iQVYWp55yRcm+Rx77K6cM/Tl5eS045uhzeebZf/HOu09n3PvKDD6fuIgfHdaVeuFUW4UFG+naLch8fjVvJUuXFNKmbR7Hn9ibrduC6egyjRlM/Oxbhhy+V8mUYYUFG9mrWzAm4Mt5K1iyuJD8Nk05/qS9S6bVy1R14RypK8fEDCZNXMSPhnQJ6jF/FYWFG+m6V3iOfBmcI/n5ufz4+F5sDacPzDS/feo0+g3pTNNWOYz49nqG//4DdmsQ5B1H3T+Rca/PZfBx3Xjyy2vZVLSFWy8cWcl3dDvJLcBzkn4GLAROB5DUBnjIzI4Ln+cARwKXlXr9beFdfyOYzq70/pRUp/rh1BJJJwMvAr3MbLakC4BBZnalpEeAV83s+bDsOjNrEjZ03wRaAY8Q3BJ4gWBqlA+Aq8Jy9YB7gR8Cc4GGwD/N7J3wAN9F0OjeDbjDzFLN42XPPzf1e6p97TntjL6MeDb+9Tj9zL4898yUqMOosTPO6scLI6ZFHUaNnXp6H555cnLUYdTYWef0B+DpJ1Ld9IqPs88dABD7c+SMs4KMY105Hs8+He/jcebZwfEYot9FHEnNjbabIx0usWnjllprKDZs1CDjh4Z4xrkazGwkCXcVzOwRgsYwZnZBqbJNwq9bCKZCSZTYN/qmsNx2STeEme2WBIMRp4f7phA0qJ1zzjnnXC3zhnNmelVSM4J+z38ys/9FHZBzzjnndkUZnwSuVd5wzkBmNiTqGJxzzjnnXDJvODvnnHPOuZQ835zMp6NzzjnnnHMuDZ5xds4555xzqXnKOYlnnJ1zzjnnnEuDZ5ydc84551xKnnBO5hln55xzzjnn0uANZ+ecc84559LgXTWcc84551xq8s4aiTzj7JxzzjnnXBq84eycc84551wavOHsnHPOOedcGryPs3POOeecS8m7OCfzjLNzzjnnnHNp8Iazc84555xzafCGs3POOeecc2nwPs7OOeeccy4leSfnJJ5xds4555xzLg3ecHbOOeeccy4NMrOoY3A7nx9U55xzrm6ItK/Etq3ba61NUX+3ehnfL8Qbzs4555xzzqXBu2o455xzzjmXBm84O+ecc845lwZvODvnnHPOOZcGbzg755xzzjmXBm84O+ecc845lwZvOLu0SKov6d2o43BlSTpE0oXh49aSOkcdk3NRU+BcSb8Ln3eQtH/UcTnn4s2X3HZpMbNtkookNTWzgqjjqS5JewB/BdqY2bGSegMHmtnDEYdWLZJ+DwwCegDDgQbAE8DBUcZVVZK6A/cBe5jZPpL6ACea2Z8jDi0tkk6paL+ZvVhbsdRUHTpH7gW2A4cBNwNrgReA/aIMqjokNQPOAzqR8HfbzK6OKqaqknQ3FawxEJe61KHzw1WTN5xdVWwEpkt6B1hfvDEuH3ihRwgamL8On88FngXi+qF3MtAfmARgZksk5UYbUrU8CPwCuB/AzKZJegqIRcMZOCH8ujtwEPB++PxQYDQQm4YzdeccGWxmAyRNBjCz1ZKyog6qml4HxgHTCS4G4mhi1AHsJI9QN84PV03ecHZV8Vr4P85amdlzkm4CMLOtkrZFHVQNbDYzk2QAkhpHHVA15ZjZBClp0aitUQVTVWZW3FXmVaC3mS0Nn+cD90QZWzXUlXNki6T6hFlOSa2Jb6OzkZldF3UQNWFmjyY+l9TYzNaXVz6D1ZXzw1WTN5xd2szsUUnZQAczmxN1PNW0XlJLdvwxPQCIbdcT4DlJ9wPNJF0CXESQvY2bFZK6suO4nAYsjTakaulU3GgOfQd0jyqYaqor58hdwEhgd0l/AU4DfhNtSNX2eHh+vwpsKt5oZquiC6l6JB1IkJ1tAnSQ1Be4zMwujzaytNWV88NVky+57dIm6QTgdiDLzDpL6gfcbGYnRhxa2iQNAO4G9gG+AFoDp5nZtEgDqwFJRwJHAQLeMrN3Ig6pyiR1AR4g6OawGvgaONfMFkQZV1VJ+jfQDXia4A/rWcCXZnZVpIFVQV06RyT1BA4nODfeM7NZEYdULZKuAP4CrGFHP2Ezsy7RRVU9ksYTXMSMMrP+4bYvzGyfaCNLT106P1z1eMPZpU3S5wQDbUYnfOBNN7N9o42saiTtRjCYTsAcM9sScUjVFnbN2BgO3uxBUK834lqnsD71zGxt1LFUl6STgR+GT8eY2cgo46mOunCOSGqRYvPamNblK4I+2yuijqWmJI03s8GSJif8HZlqZn2jji1ddeH8cNXnXTVcVWw1s4JS/VBjdeWVYvaD7pIKgOlmtiyKmGpoDPADSc2BdwkG4JwJnBNpVGmSlLLfZvF7zMz+WasB7RyTCBpo70rKkZQbpwuBOnSOTALaE9zBENAMWCppGXCJmX0eZXBVNAMoijqIneRbSQcBFg7WvBqIzZ2AOnR+uGryhrOrii8k/QSoL6kbwQfe2IhjqqqfAQcCH4TPhxCMVu8u6WYzezyqwKpJZlYk6WfA3WZ2W/EsAjFRPANID4JpwkaFz08guCiIlbAf6qVAC6Ar0Bb4D0F3gbioK+fIm8BIM3sLQNJRwDHAcwRT1Q2OMLaq2gZMkfQByX2c4zSjUbGfA3cSnBuLgLeBKyKNqGrqyvnhqskbzq4qriKYgmcTQR/Ot4A/RRpR1W0HepnZd1AyJ+d9BH9ExwBx+9BTONjmHIIPdIjReW1mfwSQ9DYwoDgzK+kPwIgIQ6uuK4D9gfEAZjZP0u7RhlRldeUcGWRmPy9+YmZvS/qrmV0nqWGUgVXDS+H/2Au7m8Tijlg56sr54aopNn9gXfTMrIig4fzryspmsE7FH3ihZUB3M1slKY791K4BbiLIrM0IB9l9UMlrMlEHYHPC880Eiz3EzSYz21zc1STsCxmr7kzUnXNklaT/BzwTPj8TWB1OUReraelKT+UWZ+G0gJdQdjGXi6KKqYrqyvnhqskbzq5Skl6h4hWfYjOrBvBRONducTbzVGBMOChtTXRhVY+ZjSGhS4OZzSfoQhM3jwMTJBUPpBsKxLGx8KGkXwHZ4WwnlwOvRBxTVZU+R04Lt8XtHPkJ8HuCTK2Aj8Nt9YEzIoyryiR9TYrP4DjOqgG8DHxEMCYjjvMf16m/Ia7qfFYNVylJPwofngLsSbCkM8DZwAIz+1UkgVWDglTgKcAh4aaVQL6ZxamPXQkFS1XfQNnszWFRxVRd4TRPPyBoIHxkZnHqqw2UvL8uJmF6QOAhi9EHbalzRMDHZvZ8tFHt2sJ5g4s1Ak4HWpjZ7yIKqdokTTGzflHHUV3h+XEqcDA7LsheiNM57mrGG84ubZLGmNkPK9uW6cL5p39CkHX6muBD79/RRlU9kqYSDD77nITsTcxmDAAgXAjhh+xoOE+NOKQqkVQPmBaX+WjTJekQ4Oy4XVyGXQJuBPYmaGwC8byoTEXSx2Z2SOUlM4ukPwNjzez1qGNxrjq8q4aritaSuoTdAZDUmWDy94wXZmbPIsiSrwSeJbhwPDTSwGpuq5ndF3UQNSXpGoJ+jy8QZHGekPSAmd0dbWTpM7PtkqZK6mBmC6OOpybCi8uzCfoFfw28GG1E1fIkwXl+PMFMDucDyyONqJrCuzHF6gGD2DEjTdxcA/xK0iZgC8H5bmaWF21Y6QlXCrwb6AVkEXT9WR+X+F3NecbZpU3SMQSru80PN3UiWCr1rciCSpOk7QT96n5mZl+G2+bHtI9giXD2iWUESwvHdileSdOAA81sffi8MfCpmfWJNrKqkfQ+wbR6E4D1xdvjMA6gnIvLG8ysY6SBVZOkz81soKRpxe8jSR+a2Y8qe22mCaehK7YVWADcbmZzoolo1yVpIsF5MoLgAuY8YC8zi/OgeVcFnnF2aTOzN8P5m3uGm2ab2aaKXpNBTiX4sPtA0psEI+1V8Uti4fzw6y8SthkQtwsCkTxQaBvxPD5/jDqAGphNcHF5QsLF5bBoQ6qR4hkOlkr6MbAEaBdhPNVWB+6MJZHUFuhI8riM2MzbbmZfSqpvZtuA4ZLitp6BqwFvOLuqGsiOgWh9JWFmj0UbUuXCZY9HhpnMocAwYA9J9xFM5fZ2pAFWk5l1jjqGnWQ4ML7UrBoPRxhPtZjZh1HHUAN17eLyz5KaAtcT3FrPIzjvY0PSuWb2RHkrbMZxZU1JtxJ0AZrJjotlIz4LHhWFKx5OkXQbsBRoHHFMrhZ5Vw2XNkmPE6yGNoWED7yYrl6FpBYEo9PPjOuAIUk5wHVABzO7NLwj0MPMXo04tCoL+3EWz+QwJqazaqxlx7RhWUADYtb/MeHi8mzgMIJpAWN7cRlnki4zs/sl/T7V/uIFhOJE0hygT4zuViaR1BH4juD8HgY0Be4xs68iDczVGm84u7RJmgX09ml3MoekZwlm1DjPzPaRlE3QNzgW0z1J2g9oZWZvlNp+IrA4jrODJJI0FNg/TlM2Jkp1cSmpuZmtjjayyoWDl6+i7FSNGd/fvC6T9AZwupmtizqW6pB0jZndWdk2V3d5w9mlTdII4GozWxp1LC4gaaKZDZI02cz6h9ummlnfqGNLh6TRwAVmtqDU9r2AB+J6JyCRpHFmdkDUcewskiaZ2YDKS0YrnKrxYWA6CSsFxrE7TV26CJD0AtAXeI/kAc2xuHOZ6v2f+Pnr6j7v4+yqohUwU9IEkj/wYvfhXYdsDrPMBiCpKwnHJgZalm40Q8ngm5Ypymc0SackPC2eNqyuZSfi0u95o5ndFXUQO8lLBBcBrxCz5cJTGBX+jxVJZxPM/99ZUmL8uQSz0LhdhDecXVX8IeoAXBm/B94E2kt6kmA1qwsijahqsivYF8cBNyckPC6eNuykaEL53sTlQuDOsG/w2yRf6E+KLqRqqzMXAWb2aNQxVNNYgoGArYB/JGxfC0yLJCIXCe+q4aokHBjRzczeDQem1TeztVHHtSsLM7MHEGQCx5nZiohDSpuk/xBka36T2Hde0h8JlkK/NLLgXEox6qrxN+CnwFfsyNJaHLv/SPoJ0I06cBEQDmD+G9Cb5BUd4zaFpttFecbZpU3SJcClQAuC2TXaEiz3fHiUce3KJJ0MvG9mr4XPm0kaamYvRRxauq4HHgK+lDQl3NYXmAhcHFlU1RROT/VnYAPBnYC+wLVm9kSkge1ccemqcTLQxcw2Rx3ITrAvwUXAYSRcBITP42Y4wZ2yfwGHAhcSn/dUcXesW4HdCeKO1cqHruY84+zSFjZs9gfGJwxEm25m+0Yb2a5L0pTSM2jEcaCKpC7A3uHTGcXLuifs39vMZtR+ZFVTfDzCC5ri+cI/iMtgzWKSDiG4szRcUmugiZl9He5rEYeVKcMZZ64ys2VRx1JTkmYTTOEW+4uAhBUdS/52SPrIzH4QdWzpkPQlwSJBs6KOxUXDM86uKjaZ2WYpSA5I2o349Hesq+ql2Ba78zpsKM+voMjjQMZ3DyCYtxngOOBpM1tVfL7ERdgveBDQgyA72AB4gqD/fJyWc98DmC3pM+I/mHkq0AyI/UUAsFFSPWCepCuBxQTZ27j4zhvNu7bY/YF1kfpQ0q+AbElHApcTjPJ20Zko6Z/APQQXMVcRzOtc18Sl9flKmB3cAFweZms3RhxTVZ0M9AcmAZjZEkm50YZULSkXDYmpunQRcC2QA1wN/Imgu8n5kUZUNRPDuxkvkXwsXowuJFebvKuGS1uYJfgZcFS46S0zeyjCkHZ54SpvvwWOIGhcvg382czWRxrYThaXAWkQLBACFJrZtnAAbZ6Z/S/quNIlaYKZ7V/8Ow/fY5+aWZ+oY9tVSfpRqu1xnJM67iQNT7HZzOyiWg/GRcIbzq5Skk4C2pnZPeHzCUBrggznjWb2fJTxubovZg3ngyi7UMVjkQVURZJuIJjB4UiC2Q8uAp4ys7sjDayKJB0A3A30IlgeuT4xW/48UV2Z0UhSd+AXQEeSz5E4DnR0uyDvquHScSNwVsLzLGAg0ISgD6Q3nGuZpDvM7FpJr5Cin3lMb+FWJBaDoiQ9TjDjzBRgW7jZgNg0nM3s9rArViFBP+ffmdk7EYdVHf8m+NwaQdBn+zyCC4LYqWMzGo0giP1BdpwjGU/SjWZ2m6S7Sf2ZG4uVD13NecPZpSPLzL5NeP5xOEBoVXgb19W+x8Ovt0caxU4kqS1ls1Bjwq9xWbJ6ENDbYn4rL2wox7GxnCRcgbK+mW0DhksaG3VM1XQF4YxGAGY2T1KcBtQl2mpm90UdRDVcJukTgjEksT6/Xc14w9mlo3niEzO7MuFp61qOxQFm9nn4tUwfx3DgSqz6Pkq6FTgTmElypnZMZEFVzxfAngQrjMWKpLUEv3OR3DCI6zy1RZKygCnh/NpLiedqlFAHZjSS1CJ8+Iqky4GRJA+uy/TZWu4mSFTkA88SzJozpeKXuLrI+zi7SoVLOY82swdLbb8MGGJmZ0cTmUtF0kIz6xB1HFUhaQ7BPLWbKi2cwSR9APQDJhD/2Q9iLewT/B1B17JhQFPgXjP7MtLAqiFs+K8h6G5yFcGMRjPN7NeRBlYFkr5mx4VZaRaXlQPD99VZ4f9GwNMEjeh5kQbmao03nF2lwluCxVPvFC/xOhBoCAw1s++iis2VFdOG8xvA6Wa2LupYaqIuzH4QDqqbUTzwTFITYG8zGx9tZFUXTgeImS2POpaaUJBqvphgRiMBbwEPxb1LUNxJ6g/8l+Civ37U8bja4Q1nlzZJh5G8utv7UcazK5NU3gwTAl41s/zajKemJL1AsDz1eyRnan3ATS2TNBkYUNwoC6ehnBijWU1EMIfzlQTnQz1gK3C3md0cZWzVEf7+p5nZPlHHUhOSziVoczxeavslBLOdPBVNZFUjqQFwDEHG+XCCbnFPm9lLkQbmao33cXZpCxvK3ljODP+oYN/sWoti5xkV/o+lhP7BZXYRv/7BSsxkmtn2sE9tXFxLsMrhfgnLhHcB7pM0zMz+FWl0VRT+/qdK6mBmC6OOpwauB36YYvuzwAdARjecw5lmzgZ+TNAV6xng0ro2Z76rnGecnavDJB0Zl6nEwoFc3cOnc8xsS5TxfB8kNTez1VHHURFJLwKjgeKZDy4HDjWzoZEFVQVhxvxIM1tRantr4G0z6x9NZNUn6X1gP4IGW0lDLU595yVNK28RnYr2ZYpw/MJTwAsxGMjovkdxyiI456ruVmIwrZikIcCjwAKCLG17SecXT0dXh7wHZHqXh58DdwG/Iciiv0cwh3BcNCjdaIagn3N4mz2O/hh1ADtBA0mNS2dow+XcsyKKKW1mdmjUMbjM4A1n5+q2VCPYM9E/gKPMbA6UrC72NMEg1Lok44+HmS0jecGjuKlosZxYLKRTTFIjgguZvYDpwMNmtjXaqKrtYeB5Sf9nZgsAJHUC7gn3ORcL3nB2rm6LS1+sBsWNZgAzmxvj7GBFMvZ41KGV0fpKKkyxXQTTh8XJo8AW4CPgWKA3cE2kEVVTuCLlOuDDcKYWI+h2cktMF0RxuyhvODvnMsFESQ+zY0XEcwhW6HK1Z1b4dWKkUdRQutOCxaG/OcEqlPsChOfHhIjjqREz+w/wn7DhrOIpDxOFXbQerf3onEuPN5ydq9sWRB1Amv6PYFnhqwkyg2OAeyON6PuRsV01zOyV8Ouu0miJQ3/zkgGyZra1eOXAuKtkvvZrCDLtzmUkn1XDuRiTdEqKzQXA9LCvqqtFkm4HhpvZjHL2t8j0Eflh//IbgE4kJFfM7LCoYvo+SJqc6TNsSNrGjlk0BGQDRcRzmsO0xOG4uF2bZ5ydi7efAQcSzIMKMAQYB3SXdHPpxQYyjaTnzOwMSdNJ3a82o6eoSmE28EA47/FwgoURCop3ZnqjOTQC+A/wELAt4li+TxmfNapj3U7SlfHHxe3avOHsXLxtB3oVL3suaQ+C+XcHE3R3yOiGMzsGOh0faRQ7iZk9BDwkqQdwITBN0ifAg2b2QcWvzhhbfbBW7MSh20m66kZ/FFdnecPZuXjrVNxoDi0DupvZKkkZv4CImS0Nv34TdSw7i6T6QM/w/wpgKnCdpMvMLGOneZPUInz4iqTLgZEkL38eh2x5VdSlBlpdqssnUQfgXEW8j7NzMSbpXqADwe11gFOBRcAvgFfjMml/OUtWFxDM8HC9mc2v/aiqTtI/gRMJMoAPm9mEhH1zzKxHZMFVQtLXBMcgVSPMzKxLLYdUY5IOAbqZ2fBw5cAmCctwZ3x/83RJmmRmscg4S7qGoBvTWoLuQP2BX5rZ25EG5lyavOHsXIwpGGZ/KnAwQYPnY4IlYWN1Ykv6I7CEYElbESzAsScwB/g/MxsSXXTpk3QR8IyZFaXY1zSxv7P7fkn6PTAI6GFm3SW1AUaY2cERh7bTxazhPNXM+ko6mmAmnd8SDKiNRfzOecPZORc5SePNbHCpbePM7IDiP7RRxZYOSRX+0TezSbUVS01JygGuAzqY2aWSuhE0Pl+NOLQqkTSFIJs5qXiWBknTYjjgtFJxmomi+BhIuhMYbWYj4xS/c97H2bkYC6ejuxXYnSBTG9dpqrZLOgN4Pnx+WsK+OFzd/6OCfQbEaSq34QSLzxwUPl9E0BUoVg1nYLOZmSQDkNQ46oCqKqHfeUoJXU0Or4VwdpbPJb0NdAZukpRLMMjZuVjwjLNzMSbpS+AEM5tVaeEMJqkLcCfB1HpGMKXeMGAxMNDMPo4wvF2KpIlmNigxCxiHrH9pkm4AugFHAn8DLgKeMrO7Iw2sCupov/N6QD9gvpmtkdQSaGtm0yIOzbm0eMbZuXj7Lu6NZoBw8N8J5eyOTaNZ0unAm2a2VtJvCKYI+5OZTY44tKrYLCmbMNMvqSsJs2vEhZndLulIoBDoAfzOzN6JOKwqMbPOUcfwPTCgN8EUlDcDjYFGkUbkXBV4xtm5GAv7Ce4JvETy1GEvRhZUNYSr1d0H7GFm+0jqA5xoZn+OOLQqSei/eQhBlvN24Fel+29nIkn/Bp4maMj8mqBx8zbBwNMLzGx0dNE5Sc0JMugljUwzGxNdRNUj6T6CrhmHmVmvsF5vm9l+EYfmXFo84+xcvOURLMF7VMI2A2LVcAYeJJhC734AM5sm6SkgVg1ndqy092PgPjN7WdIfIoynKuYRNPTzgfeBd4DJwDVmtiLKwKoiYWpDkdw/Pq79/5F0McFiQe2AKcABwKfEq+98scFmNkDSZAAzWy0pK+qgnEuXN5ydizEzuzDqGHaSHDObEMyuV2JrVMHUwGJJ9wNHALdKagjUizimtJjZncCdkjoSTAd4FnAO8JSkZ81sbqQBpsnMcqOO4XtwDbAfMM7MDpXUE/hjxDFV15ZwkaDirkCt8cGBLka84excDEm60cxuk3Q3KWadMLOrIwirJlaEfWmL/5ieBiyNNqRqOQM4Brg9HPiUT5BJj41wFcdbCRr+/YH/An8A6kcZV1VJOgCYYWZrw+dNgL3NbHy0kVXLRjPbKAlJDc1sdrisexzdRbAq5e6S/kIwg85vog3JufR5w9m5eCoeEDhNdUzfAAAKPklEQVQx0ih2niuAB4CekhYDXxNkO+OmFeExkdQh3DY7unCqTlIDgsb/WQTTnH1IPLOb9xEMzixWlGJbXCyS1IxgLMM7klYTLBgUO2b2pKTPCd5bAobWhQHObtfhgwOdizFJ/WM2Y0OFwrl26wEbgDPN7MmIQ6oSSdPZ0b+2EcFctXPMbO9IA0tDOAPF2QT9sycAzwAvmdn6SAOrJklTzKxfqW2xXwBF0o+ApgSzt2yOOp50VWFOaucymjecnYsxSR8QDOYaQbDU84yIQ6oSSXkE2ea2wMvAu+HzG4CpZnZShOHVWLii4GVmdlnUsVQmfC89RbBke+wbMZJeBEYTZJkBLgcONbOhkQVVTQl3L5KY2cLajqW6ypmTumQQZxznpHa7Jm84OxdzkvYk6Ft7JsEsG8/GZRo3SS8DqwlmCDj8/7d3b7FylWUYx/9PKaQVwVhBrVGOBcJJhBBj0AspgqICGhTBNGARCAERLNELDQp6Y5CoESKKQFMqIiCHiIiACIELY2M5FSEBUyBoMICJQWyEvbePF9+azQQ27Z59mG/WzPNLJnutNbvJkzRN3/Wtb70v8FZgG0onhwdrZpsrku633cbtAa0m6e2U/bTLKQXaXcA5tp+rGmwG2vwkI2LYpHCOGBKS9ge+Rtni0Ir2TpI22N6/Od4KeAHYqfNCV9tIWtV1uoCyn/Zttj9aKVIMoTY9yZjKsPSkjtGUlwMjWkzS3pSV5s8A/6TsSz23aqjejHUObE9IerKtRXOjuxXaOHArcEOlLCNpCDvOvI7t+yW1cmDIkPWkjhGUwjmi3VZTpr0dYbuNb9kfIOnF5ljA4ua8lcMqbF8AIGm7cuqXKkcaRcPWceaNnmQ8XynObA1TT+oYQSmcI1rM9gdqZ5gN263qDbwlkvYD1gJLmvMXgJNsP1I12AixfUvzc03tLHNomJ5kDFNP6hhBKZwjWkjSdbaP63ppaPIrykpnq1tutdhlwCrbdwNI+nBz7ZCaoUaRpD0p3Vl2oev/Otut2xLQeZIxJIamJ3WMprwcGNFCkpbafrYZj/w6zfS36DNJD9k+YEvXYv5Jegj4CbAemOhct72+WqgZGqabgG5t7Ukdoy0rzhEt1BTNWwFX2P5I7TwxaaOk8yjbNQBWUKYgRv+N2750y7/WCtdTbgIup+smoE0kvQkYsz3WnO9F2av9dIrmaJMFtQNExMzYngA2SXpL7Swx6WRgR+BG4KbmeGXVRCNG0pJmSt0tks6QtLRzbUvT6wbYuO1Lba+zvb7zqR2qR7+jrJgjaRmlk8ZuwJmSvlsxV0RPslUjosUkXUdp53QnMDkaeRhabkXMxBtMqOto5YQ6SecDz1Fuxl7uXG/ThMfX9Gz/DrDE9pmStgHWd76LGHTZqhHRbrc2n6hI0g9tnyPpFqbuHXx0hVgjyfautTPMg5Oan1/tumbKim1bdP+7WA58D8D2K5L+VydSRO9SOEe0mO01knZsjtva13UYdPY0X1Q1RUxq9tSuokyiPE3SHsBetn9TOVrPhuRm4GFJFwF/B5YBdwA0HTYiWiOFc0QLSRLwLeBLlEfSCySNAxfb/nbVcCOoa7/pEuC3tl/e3O9HX6ymdNTotAL8G+Ulu9YVzgCSDuH1XTWuqhaod6dShp/sQhnYtKm5vg+54YwWyR7niBaS9BXg48Bptp9sru0GXEpp7fSDmvlGlaTVlMfQ91LGn99ue7xuqtEk6c+2D5b0gO0Dm2utbA0oaS2wO2VEdaerhofxXQZJN9g+tnaOiDeSFeeIdjoRONz2C50LtjdKWkF5BJrCuQLbKyVtDRwJfB74saQ7bZ9SOdooekXSYpq9tZJ2p+vFupY5GNjHo7HS1aZ92zGCUjhHtNPW3UVzh+3nm8ItKrE9Juk2SsG2GDgGSOHcJ5IuAa4Bzqe0QHuPpKuBDwJfqJdsVh4B3gk8WztIH4zCzUG0WArniHba3MCADBOoRNLHgOOBQ4F7KAMrjquZaQQ9QdkzuxT4A6VV4wPA2VPdbLbEDsCjktbx6qq5bR9TMVPESMoe54gWkjRBV9/m7q+ARbaz6lyBpF9S9jbflhcE62rG0R/ffBYBvwCutf141WAz0IymnjwFPgScYHvfSpHmTfee9IhBlMI5ImIONQXbHrZ/3+yxXWj737VzjTJJBwJXAu+1vVXtPDMh6X2UffPHUca432j74rqpeifpk5TOM1P2bpZ0hO07+hwrYtoycjsiYo5IOhX4FfDT5tK7gZvrJRpdkraWdFSzv/k24HGgVd0aJO0p6ZuSHgMuAZ6hLHgd2saiuXE88ISkCyXt/dovUzTHoMuKc0TEHJH0IPB+4E9dLdAmRw3H/JN0OHAC8AlgHWXrzM22p9raNNCaiXr3AV+0/dfm2sY2jg3vJml7yt/RSsrLgKuBa/JkJtogK84REXPnZduTL2dKWki6BPTb14E/AnvbPsr21W0smhvHAv8A7pb0M0mHUfY4t5rtF4EbKDc1S4FPA/dLOqtqsIhpyIpzRMQckXQh8C9Kn+2zgDOAR21/o2qwaDVJ2wKfoqzSLgfWADe1cVuDpKOAkykDXdYCa2w/14xIf8z2zlUDRmxBCueIiDnSjEI/BTiCsjJ4O3D5iAyuiD6QtAT4LPA528tr5+mVpKso/ybuneK7w2zfVSFWxLSlcI6ImAOSFgAP296vdpaIQSVpV+BZ2/9tzhcD77D9VNVgEdOUPc4REXOgaa/1kKSdameJGGDXA92t6CaaaxGtkMmBERFzZynwl2bC2+QLabaPrhcpYqAs7H6B1vYrkrapGSiiFymcIyLmzgW1A0QMuOclHW371wCSjgHaOgo9RlD2OEdEzJKkRcDpwDJgA3CF7fG6qSIGj6TdgauBd1FeoH0GOLHTpzpi0KVwjoiYJUnXAmOUYRVHAk/bPrtuqojBJenNlBokQ0+iVVI4R0TMUvd0wGboyTrbB1WOFTEwJK2w/XNJq6b63vb3+50pYiayxzkiYvbGOge2x0s754josm3zc7uqKSJmKSvOERGzJGmCV7toCFgMbGqObXv7WtkiImLupHCOiIiIeSXpR5v73vaX+5UlYjayVSMiIiLm2/raASLmQlacIyIioq8kbUfZxvRS7SwRvcjI7YiIiOgLSftJegB4BHhU0npJ+9bOFTFdKZwjIiKiXy4DVtne2fZOwLnAzypnipi2FM4RERHRL9vavrtzYvseXm1VFzHw8nJgRERE9MtGSecBa5vzFcCTFfNE9CQrzhEREdEvJwM7Ajc2nx2AlVUTRfQgXTUiIiJiXklaBJwOLAM2AFfaHtv8n4oYPCmcIyIiYl5JupYymv4+4EjgKdvn1E0V0bsUzhERETGvJG2wvX9zvBBYZ/ugyrEiepY9zhERETHfJrdl2B6vGSRiNrLiHBEREfNK0gTwn84psBjY1Bzb9va1skX0IoVzRERERMQ0ZKtGRERERMQ0pHCOiIiIiJiGFM4REREREdOQwjkiIiIiYhr+D553h0dGa3BTAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 720x720 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "f,ax = plt.subplots(figsize=(10, 10))\n",
    "sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax, cmap='Purples')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Modeling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import f1_score,confusion_matrix\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.metrics import classification_report\n",
    "from sklearn.model_selection import KFold\n",
    "k_fold = KFold(n_splits=10, shuffle=True, random_state=0)\n",
    "accuracies = {}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "sc_X = StandardScaler()\n",
    "x_train = sc_X.fit_transform(x_train)\n",
    "x_test = sc_X.transform(x_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Optimal number of features : 4\n",
      "Best features : Index(['Region_Code', 'Previously_Insured', 'Vehicle_Damage',\n",
      "       'Policy_Sales_Channel'],\n",
      "      dtype='object')\n"
     ]
    }
   ],
   "source": [
    "from sklearn.feature_selection import RFECV\n",
    "\n",
    "# The \"accuracy\" scoring is proportional to the number of correct classifications\n",
    "clf_rf_1 = RandomForestClassifier(random_state = 42) \n",
    "rfecv = RFECV(estimator=clf_rf_1, step=1, cv=k_fold,scoring='accuracy')   #10-fold cross-validation\n",
    "rfecv = rfecv.fit(x_train, y_train)\n",
    "\n",
    "print('Optimal number of features :', rfecv.n_features_)\n",
    "print('Best features :', x.columns[rfecv.support_])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_1 = df_downsample[['Region_Code','Previously_Insured','Vehicle_Damage','Policy_Sales_Channel']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train, x_test, y_train, y_test = train_test_split(x_1,y, test_size=0.25, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "sc_X = StandardScaler()\n",
    "x_train = sc_X.fit_transform(x_train)\n",
    "x_test = sc_X.transform(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [],
   "source": [
    "clf_rf_2 = RandomForestClassifier(random_state=43)      \n",
    "clr_rf_2 = clf_rf_2.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy is:  0.7872 \n",
      "\n",
      "RFC Reports\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.86      0.69      0.76       626\n",
      "           1       0.74      0.89      0.81       624\n",
      "\n",
      "    accuracy                           0.79      1250\n",
      "   macro avg       0.80      0.79      0.79      1250\n",
      "weighted avg       0.80      0.79      0.78      1250\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 842.4x595.44 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "ac = accuracy_score(y_test,clf_rf_2.predict(x_test))\n",
    "accuracies['Random_Forest'] = ac\n",
    "\n",
    "print('Accuracy is: ',ac, '\\n')\n",
    "cm = confusion_matrix(y_test,clf_rf_2.predict(x_test))\n",
    "sns.heatmap(cm,annot=True,fmt=\"d\")\n",
    "\n",
    "print('RFC Reports\\n',classification_report(y_test, clf_rf_2.predict(x_test)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 842.4x595.44 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "plt.figure()\n",
    "plt.xlabel(\"Number of features selected\")\n",
    "plt.ylabel(\"Cross validation score of number of selected features\")\n",
    "plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy is:  0.7752 \n",
      "\n",
      "Logistic regression Reports\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.95      0.58      0.72       626\n",
      "           1       0.70      0.97      0.81       624\n",
      "\n",
      "    accuracy                           0.78      1250\n",
      "   macro avg       0.82      0.78      0.77      1250\n",
      "weighted avg       0.82      0.78      0.77      1250\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 842.4x595.44 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "logmodel = LogisticRegression() \n",
    "logmodel.fit(x_train,y_train)\n",
    "\n",
    "ac = accuracy_score(y_test,logmodel.predict(x_test))\n",
    "accuracies['Logistic regression'] = ac\n",
    "\n",
    "print('Accuracy is: ',ac, '\\n')\n",
    "cm = confusion_matrix(y_test,logmodel.predict(x_test))\n",
    "sns.heatmap(cm,annot=True,fmt=\"d\")\n",
    "\n",
    "print('Logistic regression Reports\\n',classification_report(y_test, logmodel.predict(x_test)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# KNN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "k=1 72.80 (+/- 3.73)\n",
      "k=2 67.15 (+/- 4.05)\n",
      "k=3 74.11 (+/- 4.71)\n",
      "k=4 70.59 (+/- 4.67)\n",
      "k=5 77.04 (+/- 2.27)\n",
      "k=6 72.40 (+/- 3.76)\n",
      "k=7 76.77 (+/- 2.37)\n",
      "k=8 74.27 (+/- 4.00)\n",
      "k=9 76.80 (+/- 2.65)\n",
      "k=10 74.77 (+/- 3.98)\n",
      "k=11 75.89 (+/- 4.00)\n",
      "k=12 75.41 (+/- 4.33)\n",
      "k=13 76.96 (+/- 2.05)\n",
      "k=14 76.75 (+/- 2.50)\n",
      "k=15 77.79 (+/- 1.93)\n",
      "k=16 75.57 (+/- 3.80)\n",
      "k=17 77.55 (+/- 1.53)\n",
      "k=18 77.09 (+/- 1.65)\n",
      "k=19 77.41 (+/- 1.96)\n",
      "k=20 77.28 (+/- 2.27)\n",
      "k=21 77.31 (+/- 2.51)\n",
      "k=22 77.15 (+/- 2.54)\n",
      "k=23 77.33 (+/- 2.45)\n",
      "k=24 77.44 (+/- 2.29)\n",
      "k=25 77.95 (+/- 2.27)\n",
      "The optimal number of neighbors is 24 with 77.9%\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 842.4x595.44 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn import model_selection\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "#Neighbors\n",
    "neighbors = np.arange(0,25)\n",
    "\n",
    "#Create empty list that will hold cv scores\n",
    "cv_scores = []\n",
    "\n",
    "#Perform 10-fold cross validation on training set for odd values of k:\n",
    "for k in neighbors:\n",
    "    k_value = k+1\n",
    "    knn = KNeighborsClassifier(n_neighbors = k_value, weights='uniform', p=2, metric='euclidean')\n",
    "    kfold = model_selection.KFold(n_splits=10, random_state=123)\n",
    "    scores = model_selection.cross_val_score(knn, x_train, y_train, cv=k_fold, scoring='accuracy')\n",
    "    cv_scores.append(scores.mean()*100)\n",
    "    print(\"k=%d %0.2f (+/- %0.2f)\" % (k_value, scores.mean()*100, scores.std()*100))\n",
    "\n",
    "optimal_k = neighbors[cv_scores.index(max(cv_scores))]\n",
    "print (\"The optimal number of neighbors is %d with %0.1f%%\" % (optimal_k, cv_scores[optimal_k]))\n",
    "\n",
    "plt.plot(neighbors, cv_scores)\n",
    "plt.xlabel('Number of Neighbors K')\n",
    "plt.ylabel('Train Accuracy')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy is:  0.7776 \n",
      "\n",
      "KNN Reports\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.87      0.66      0.75       626\n",
      "           1       0.72      0.90      0.80       624\n",
      "\n",
      "    accuracy                           0.78      1250\n",
      "   macro avg       0.79      0.78      0.77      1250\n",
      "weighted avg       0.79      0.78      0.77      1250\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 842.4x595.44 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "knn = KNeighborsClassifier(n_neighbors=24)\n",
    "knn.fit(x_train, y_train)\n",
    "\n",
    "ac = accuracy_score(y_test,knn.predict(x_test))\n",
    "accuracies['KNN'] = ac\n",
    "\n",
    "\n",
    "print('Accuracy is: ',ac, '\\n')\n",
    "cm = confusion_matrix(y_test,knn.predict(x_test))\n",
    "sns.heatmap(cm,annot=True,fmt=\"d\")\n",
    "\n",
    "print('KNN Reports\\n',classification_report(y_test, knn.predict(x_test)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Decision Tree"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy is:  0.7848 \n",
      "\n",
      "DecisionTree Reports\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.85      0.69      0.76       626\n",
      "           1       0.74      0.88      0.80       624\n",
      "\n",
      "    accuracy                           0.78      1250\n",
      "   macro avg       0.79      0.78      0.78      1250\n",
      "weighted avg       0.79      0.78      0.78      1250\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAArcAAAHdCAYAAAAHGlHWAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+j8jraAAAazUlEQVR4nO3dfdimZV0n8O+PYQDxFURoAkrQMYPd1A1ZVs1NQBhYE1rf2A5ttmjtUEvNjkPxZXfTYiO1tcWy8vBtSm0cM4PKl1jUxNVETTNBkVEMRmYZRQRMeZl5zv1j7p1mppkHs3me++I8Px+O+3iu+7yv67rP+w/0d3z5nedVrbUAAEAP9pv3BAAAYF9R3AIA0A3FLQAA3VDcAgDQDcUtAADdUNwCANCN/Zfipt9+9c/ZXwyYhGf99k3zngLALtZ95V017zns7s6vf3lJareVhx277L9VcgsAQDeWJLkFAOBuZGHbvGewz0huAQDohuQWAGB0bWHeM9hnJLcAAHRDcgsAMLqFfpJbxS0AwOCatgQAAJgeyS0AwOg6akuQ3AIA0A3JLQDA6PTcAgDA9EhuAQBG19HjdxW3AACj05YAAADTI7kFABidrcAAAGB6JLcAAIPz+F0AAJggyS0AwOg66rlV3AIAjE5bAgAATI/kFgBgdB09oUxyCwBANyS3AACj03MLAADTI7kFABidrcAAAOiGtgQAAJgeyS0AwOg6akuQ3AIA0A3JLQDA4FrzEAcAAJgcyS0AwOg62i1BcQsAMDoLygAAYHoktwAAo+uoLUFyCwBANyS3AACjW7AVGAAATI7iFgBgdG1haV57UFVfqaq/q6rPVNUnZ2OHVtUlVXX17O8hO53/4qraWFVXVdXpd/VTFLcAAKNbWFia1949rrX28NbaCbP35yW5tLW2Osmls/epquOSnJPk+CRrkryuqlYsdmPFLQAA83ZWknWz43VJzt5pfH1r7fbW2jVJNiY5cbEbKW4BAEa3jG0JSVqSv6yqT1XVM2djR7TWNifJ7O/hs/Ejk1y307WbZmN7ZbcEAACW06Nba9dX1eFJLqmqLyxybu1hrC12c8UtAMDolvHxu62162d/t1TVu7O9zeCGqlrVWttcVauSbJmdvinJ0TtdflSS6xe7v7YEAACWRVXds6ru/f+Pk5yW5HNJLk6ydnba2iQXzY4vTnJOVR1YVcckWZ3k8sW+Q3ILADC65Utuj0jy7qpKttehb2+tva+qPpFkQ1Wdm+TaJE9JktbaFVW1IcmVSbYmeU5rbdEnTihuAQAGdxf14j78nvblJA/bw/iNSU7ZyzXnJzn/u/0ObQkAAHRDcgsAMLplXFC21CS3AAB0Q3ILADC6vT9w4W5HcgsAQDcktwAAo+uo51ZxCwAwOm0JAAAwPZJbAIDRddSWILkFAKAbklsAgNHpuQUAgOmR3AIAjK6jnlvFLQDA6DoqbrUlAADQDcktAMDoLCgDAIDpkdwCAIxOzy0AAEyP5BYAYHQd9dwqbgEARqctAQAApkdyCwAwuo7aEiS3AAB0Q3ILADA6PbcAADA9klsAgNFJbgEAYHoktwAAo2tt3jPYZxS3AACj05YAAADTI7kFABid5BYAAKZHcgsAMDqP3wUAgOmR3AIAjK6jnlvFLQDA6Dra51ZbAgAA3ZDcAgCMrqO2BMktAADdkNwCAIxOcgsAANMjuQUAGF1HD3FQ3AIADK4t2AoMAAAmR3ILADA6C8oAAGB6JLcAAKPraEGZ5BYAgG5IbgEARtfRbgmKWwCA0VlQBgAA0yO5BQAYneQWAACmR3ILADC61s+CMsktAADdkNwCAIyuo55bxS0AwOjscwv7SFUOevp/TfvWTbn93a/NykeflRUPfkTSFtK+fWvueO+b0v7h5n88/d6H5qCfeUXu/OjF2frJv5zjxIHenPvKZ+fhJ5+QW268OS89/ZeSJEf/8A/mP5//8znw4IPy9U1fy+89/7dy27e+k+Mf8yN56ouenhUr98+2O7dm/f/4g3z+Y5+b8y8AEj23zNn+/+bULHxj8473d37i/blt3a/ktj94RbZ96bPZ/9/9xC7nr3zc07LtGv8HAux7H/njD+XVa391l7GfveDZ2fAbb83L1rwgn3r/x3PmM89Kktx60615zbm/npeteUFe/8uvzc+/5rnzmDLsO21haV5zcJfFbVU9tKpeVFUXVtX/mh3/8HJMjr7VvQ7JimN/JFs/e9k/Dt5x2z8erzxgl/NXPPjhaTd/Le3G65dphsBIrrr8yvzDzd/aZWzVsd+fqz5+ZZLkio/8bU4446QkybVXXJNvbrkpSfLVL16XlQcekP0P8B9DYQoWLW6r6kVJ1iepJJcn+cTs+I+q6rylnx49W3ny03LHh/84ya59Pisf85M56JmvzP7HnZQ7/8+fzgYPyP4nnpE7P/pnyz9RYFibvnhtHvH4RyZJHnnmo3LoqsP+yTknnHFS/v6Ka7L1jq3LPT3Ydxba0rzm4K6S23OTPLK1dkFr7a2z1wVJTpx9Bt+T/Y79kbRv35p2w9//k8/u/Mi7c9vrX5itV/51Vj7i5CTJykedla2fuiS58/blniowsDe+8HU59Rlr8vI/e2Xuca+Dsu3OXQvYI1cfnaed94y85SW/N6cZAru7q/+GspDk+5PsXoGsmn0G35MVRz44Kx70sKw45l+n9l+ZHHBQDjjz53LHe96w45xtX/h4DvyPz8udH704+606Jise8qNZ+dgnpw48ePtm09vuzNZPf3COvwLo3eYvfTWv+untfbhHHLMqD3vcj+747JDvOzTP/f0X5vUvuDBbrr1hXlOEfaINtBXY85NcWlVXJ7luNvYDSR6c5BeWcmL07c7L/iR3XvYnSZL9jv6hrDzhtNzxnjek7nd42je3JElWPOjhOxab3b7+lTuuXfmoJ6bdcZvCFlhy977/fXLrjbekqnLWLzw5H3jb9l1aDr7PwXnBm1+ad77ybbn6U1fNeZawD4yyFVhr7X1V9ZBsb0M4Mtv7bTcl+URrbdsyzI/BrHzsk7Lfod+XtJZ2y42545I/nPeUgEE868JfykNPOj73OuTeec3HXp93v+YdOfCeB+XUZ6xJknzy/R/PZe/8QJLk1J8+I0f84Pflic99cp743CcnSV71jFfk1htvmdv8ge2qLcGzhL/96p/rp/wH7tae9ds3zXsKALtY95V31bznsLt/+LWnL0ntds+XvXXZf6t9bgEA6IZN+QAARtdRz63kFgCAbkhuAQBGN9BWYAAA9E5bAgAATI/kFgBgdK2ftgTJLQAA3ZDcAgCMTs8tAABMj+QWAGBwzVZgAAB0Q1sCAABMj+QWAGB0klsAAJgeyS0AwOg8xAEAAKZHcgsAMLqOem4VtwAAg2sdFbfaEgAA6IbiFgBgdAttaV57UVUrqurTVfXns/eHVtUlVXX17O8hO5374qraWFVXVdXpd/VTFLcAACy35yX5/E7vz0tyaWttdZJLZ+9TVcclOSfJ8UnWJHldVa1Y7MaKWwCA0S0sLM1rD6rqqCT/Ickbdho+K8m62fG6JGfvNL6+tXZ7a+2aJBuTnLjYT1HcAgCwnH4ryQuT7Fz9HtFa25wks7+Hz8aPTHLdTudtmo3tleIWAGB0y9RzW1VPSLKltfap73JmtYexRbd2sBUYAMDolm8rsEcneWJVnZnkoCT3qaq3Jrmhqla11jZX1aokW2bnb0py9E7XH5Xk+sW+QHILAMCyaK29uLV2VGvtgdm+UOwDrbWnJ7k4ydrZaWuTXDQ7vjjJOVV1YFUdk2R1kssX+w7JLQDA4Fqb+0McLkiyoarOTXJtkqckSWvtiqrakOTKJFuTPKe1tm2xGyluAQBYdq21DyX50Oz4xiSn7OW885Oc/93eV3ELADA6j98FAIDpkdwCAIxOcgsAANMjuQUAGFzrKLlV3AIAjK6j4lZbAgAA3ZDcAgCMbmHeE9h3JLcAAHRDcgsAMLieFpRJbgEA6IbkFgBgdB0lt4pbAIDRWVAGAADTI7kFABicBWUAADBBklsAgNHpuQUAgOmR3AIADK6nnlvFLQDA6LQlAADA9EhuAQAG1yS3AAAwPZJbAIDRSW4BAGB6JLcAAIPrqedWcQsAMLqOilttCQAAdENyCwAwuJ7aEiS3AAB0Q3ILADA4yS0AAEyQ5BYAYHA9JbeKWwCA0bWa9wz2GW0JAAB0Q3ILADC4ntoSJLcAAHRDcgsAMLi2oOcWAAAmR3ILADC4nnpuFbcAAINrtgIDAIDpkdwCAAyup7YEyS0AAN2Q3AIADM5WYAAAMEGSWwCAwbU27xnsO4pbAIDBaUsAAIAJktwCAAxOcgsAABMkuQUAGFxPC8oktwAAdENyCwAwuJ56bhW3AACDa62f4lZbAgAA3ZDcAgAMri3Mewb7juQWAIBuSG4BAAa3oOcWAACmR3ILADC4nnZLUNwCAAyup31utSUAANANyS0AwOBam/cM9h3JLQAA3ZDcAgAMTs8tAABMkOQWAGBwPT3EQXELADC4nva51ZYAAEA3JLcAAIOzFRgAAEyQ5BYAYHA9LSiT3AIA0A3JLQDA4OyWAAAAEyS5BQAYXE+7JShuAQAG19OCsiUpbu/zkvcuxW0B/tm+c/1l854CAMtIcgsAMDgLygAAYIIktwAAg+up51ZyCwBANyS3AACD62gnMMUtAMDotCUAAMAEKW4BAAbXWi3Ja3dVdVBVXV5Vf1tVV1TVy2fjh1bVJVV19ezvITtd8+Kq2lhVV1XV6Xf1WxS3AAAsl9uTnNxae1iShydZU1UnJTkvyaWttdVJLp29T1Udl+ScJMcnWZPkdVW1YrEvUNwCAAxuYYleu2vbfWv2duXs1ZKclWTdbHxdkrNnx2clWd9au721dk2SjUlOXOy3KG4BAFg2VbWiqj6TZEuSS1prH09yRGttc5LM/h4+O/3IJNftdPmm2dhe2S0BAGBwLcu3W0JrbVuSh1fV/ZK8u6r+1SKn72lii+5cprgFABjcwhw2um2tfbOqPpTtvbQ3VNWq1trmqlqV7alusj2pPXqny45Kcv1i99WWAADAsqiqB8wS21TVPZKcmuQLSS5OsnZ22tokF82OL05yTlUdWFXHJFmd5PLFvkNyCwAwuIXla0tYlWTdbMeD/ZJsaK39eVV9LMmGqjo3ybVJnpIkrbUrqmpDkiuTbE3ynFlbw14pbgEAWBattc8mecQexm9Mcsperjk/yfnf7XcobgEABrecC8qWmp5bAAC6IbkFABjcnh64cHeluAUAGJy2BAAAmCDJLQDA4HpqS5DcAgDQDcktAMDgJLcAADBBklsAgMH1tFuC4hYAYHAL/dS22hIAAOiH5BYAYHALHbUlSG4BAOiG5BYAYHBt3hPYhyS3AAB0Q3ILADC4nh7ioLgFABjcQllQBgAAkyO5BQAYnAVlAAAwQZJbAIDB9bSgTHILAEA3JLcAAINb6GezBMUtAMDoFtJPdastAQCAbkhuAQAGZyswAACYIMktAMDgelpQJrkFAKAbklsAgMH19BAHxS0AwOAsKAMAgAmS3AIADM6CMgAAmCDJLQDA4HpaUCa5BQCgG5JbAIDB9ZTcKm4BAAbXLCgDAIDpkdwCAAyup7YEyS0AAN2Q3AIADE5yCwAAEyS5BQAYXJv3BPYhxS0AwOAWbAUGAADTI7kFABicBWUAADBBklsAgMFJbgEAYIIktwAAg+tpKzDJLQAA3ZDcAgAMrqd9bhW3AACDs6AMAAAmSHILADA4C8oAAGCCJLcAAINb6Ci7ldwCANANyS0AwOB62i1BcQsAMLh+mhK0JQAA0BHJLQDA4HpqS5DcAgDQDcktAMDgFmreM9h3JLcAAHRDcgsAMLieHuKguAUAGFw/pa22BAAAOiK5BQAYnK3AAABggiS3AACD62lBmeQWAIBuSG4BAAbXT26ruAUAGJ4FZQAAMEGSWwCAwVlQBgAAEyS5BQAYXD+5reQWAICOSG4BAAbX024JilsAgMG1jhoTtCUAANANyS0AwOB6akuQ3AIA0A3JLQDA4DzEAQAA/pmq6uiq+mBVfb6qrqiq583GD62qS6rq6tnfQ3a65sVVtbGqrqqq0+/qOxS3AACDa0v02oOtSX65tfbDSU5K8pyqOi7JeUkuba2tTnLp7H1mn52T5Pgka5K8rqpWLPZbFLcAAINbSFuS1+5aa5tba38zO741yeeTHJnkrCTrZqetS3L27PisJOtba7e31q5JsjHJiYv9FsUtAADLrqoemOQRST6e5IjW2uZkewGc5PDZaUcmuW6nyzbNxvbKgjIm4SEPeVDe/rbf3fH+2GN+IL/y8lfnpJN+NA95yIOSJPe7733yzZtvyQmPPG1e0wQ6d9qT1uaeBx+c/fbbLytWrMiGN12447M3v/2P85u/88Zc9hfrc8j97ps/f/8H8ua3v2vH51/80jV555tem4fO/jcL7k6WeyuwqrpXkncleX5r7Zaq2uupexhbdPWb4pZJ+OIXv7SjaN1vv/1y7Vc+lT+96L258LVv2HHOq37jv+XmW26Z1xSBQbzptRfkkPvdd5exzTd8LR/7xKez6ojDd4w94fST84TTT06yvbB97nmvUNjCd6GqVmZ7Yfu21tqfzIZvqKpVrbXNVbUqyZbZ+KYkR+90+VFJrl/s/toSmJxTTn5Mvvzlv8+11351l/EnP/knsv4dF81pVsDIXnnh7+cFzz43ewuX3nPJX+WMU//98k4K9qG2RP/srrZHtG9M8vnW2v/c6aOLk6ydHa9NctFO4+dU1YFVdUyS1UkuX+y3KG6ZnKc+9aysf8ef7jL2Y4/5t7lhy9eyceM1c5oVMIKqyjN/6aV56s/+Yt550XuSJB+87K9z+AMOy0NXH7vX69536V/lzMf/+DLNEu7WHp3kGUlOrqrPzF5nJrkgyeOr6uokj5+9T2vtiiQbklyZ5H1JntNa27bYF3zPbQlV9TOttTd/r9fDnqxcuTI/8YTT8tKX/fou40972tl5h9QWWGJ/+Lu/mcMfcP/ceNM381+e/5Ic84NH5/V/sD6vf835e73ms1d8Ifc46KCsPvaByzdR2MeWq+e2tfaR7LmPNklO2cs15yfZ+7+Eu/mXJLcv/xdcC3u0Zs3j8ulP/122bPn6jrEVK1bkJ88+IxveefEcZwaM4PAH3D9Jcv9D7pdTHvuofPLTf5evXv9/86S1z85pT1qbG7729TzlZ38xX7/xGzuuee//1pLA3d9ytSUsh0WT26r67N4+SnLEvp8OozvnaWf/k5aEU0/5sVx11cZ89aub5zQrYATf/s5taQsLuec9D863v3NbPnr53+RZP/NT+fBfrN9xzmlPWpt3vPHCHQvOFhYW8pcfvCxv+Z1XzWvawG7uqi3hiCSnJ7lpt/FK8tElmRHDusc9Dsqppzw2z3r2i3YZ396DqyUBWFo3fuOmPO8lv5ok2bZ1W8487cfzmJNOWPSaT37mczniAYfl6CNXLccUYcks91ZgS6la23tkXFVvTPLmWX/E7p+9vbX2U3u6bv8DjpxPDg2wm+9cf9m8pwCwi5WHHbvXTV3nZe0Dn7Qktdu6r7xr2X/roslta+3cRT7bY2ELAMDdy8IiYefdja3AAADohieUAQAMrp/cVnELADC8hY7KW20JAAB0Q3ILADC4eT1wYSlIbgEA6IbkFgBgcD09xEFyCwBANyS3AACD62m3BMUtAMDgLCgDAIAJktwCAAzOgjIAAJggyS0AwOBa03MLAACTI7kFABicrcAAAOiGBWUAADBBklsAgMF5iAMAAEyQ5BYAYHA9LSiT3AIA0A3JLQDA4DzEAQAAJkhyCwAwuJ72uVXcAgAMzlZgAAAwQZJbAIDB2QoMAAAmSHILADA4W4EBAMAESW4BAAbXU8+t4hYAYHC2AgMAgAmS3AIADG7BgjIAAJgeyS0AwOD6yW0ltwAAdERyCwAwOFuBAQDQjZ6KW20JAAB0Q3ILADC4ZiswAACYHsktAMDg9NwCAMAESW4BAAbXOkpuFbcAAIOzoAwAACZIcgsAMDgLygAAYIIktwAAg9NzCwAAEyS5BQAYXE89t4pbAIDB9bTPrbYEAAC6IbkFABjcggVlAAAwPZJbAIDB6bkFAIAJktwCAAyup55bxS0AwOC0JQAAwARJbgEABtdTW4LkFgCAbkhuAQAGp+cWAAAmSHILADC4nnpuFbcAAIPTlgAAABMkuQUAGFxrC/Oewj4juQUAoBuSWwCAwS3ouQUAgOmR3AIADK7ZCgwAgF5oSwAAgAmS3AIADK6ntgTJLQAA3ZDcAgAMbkFyCwAA0yO5BQAYXOtotwTFLQDA4CwoAwCA70FVvamqtlTV53YaO7SqLqmqq2d/D9npsxdX1caquqqqTr+r+ytuAQAGt5C2JK+9eEuSNbuNnZfk0tba6iSXzt6nqo5Lck6S42fXvK6qViz2WxS3AAAsm9bah5N8Y7fhs5Ksmx2vS3L2TuPrW2u3t9auSbIxyYmL3V/PLQDA4CbQc3tEa23zbC6bq+rw2fiRSf56p/M2zcb2SnILAMBU1R7GFq3EJbcAAIObwEMcbqiqVbPUdlWSLbPxTUmO3um8o5Jcv9iNJLcAAINrrS3J65/h4iRrZ8drk1y00/g5VXVgVR2TZHWSyxe7keQWAIBlU1V/lOTHkxxWVZuS/PckFyTZUFXnJrk2yVOSpLV2RVVtSHJlkq1JntNa27bo/ZeigXj/A46ce7YNkCTfuf6yeU8BYBcrDzt2T32kc3Xfez1oSWq3m7/1pWX/rdoSAADohrYEAIDBTWArsH1GcgsAQDcktwAAg5vAVmD7jOQWAIBuSG4BAAbXFn/o192K4hYAYHDaEgAAYIIktwAAg7MVGAAATJDkFgBgcD0tKJPcAgDQDcktAMDgeuq5VdwCAAyup+JWWwIAAN2Q3AIADK6f3DapnmJoAADGpi0BAIBuKG4BAOiG4hYAgG4obpmkqlpTVVdV1caqOm/e8wHGVVVvqqotVfW5ec8FuGuKWyanqlYk+Z0kZyQ5Lsl/qqrj5jsrYGBvSbJm3pMAvjuKW6boxCQbW2tfbq3dkWR9krPmPCdgUK21Dyf5xrznAXx3FLdM0ZFJrtvp/abZGADAohS3TFHtYcyGzADAXVLcMkWbkhy90/ujklw/p7kAAHcjilum6BNJVlfVMVV1QJJzklw85zkBAHcDilsmp7W2NckvJHl/ks8n2dBau2K+swJGVVV/lORjSX6oqjZV1bnznhOwd9WaVkYAAPoguQUAoBuKWwAAuqG4BQCgG4pbAAC6obgFAKAbilsAALqhuAUAoBuKWwAAuvH/ALN8t/2ymdPbAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 842.4x595.44 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "dtree = DecisionTreeClassifier(criterion='gini') #criterion = entopy, gini\n",
    "dtree.fit(x_train, y_train)\n",
    "\n",
    "ac = accuracy_score(y_test,dtree.predict(x_test))\n",
    "accuracies['decisiontree'] = ac\n",
    "\n",
    "print('Accuracy is: ',ac, '\\n')\n",
    "cm = confusion_matrix(y_test,dtree.predict(x_test))\n",
    "sns.heatmap(cm,annot=True,fmt=\"d\")\n",
    "\n",
    "print('DecisionTree Reports\\n',classification_report(y_test, dtree.predict(x_test)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# SVM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy is:  0.7752 \n",
      "\n",
      "SVM report\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.95      0.58      0.72       626\n",
      "           1       0.70      0.97      0.81       624\n",
      "\n",
      "    accuracy                           0.78      1250\n",
      "   macro avg       0.82      0.78      0.77      1250\n",
      "weighted avg       0.82      0.78      0.77      1250\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 842.4x595.44 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.svm import SVC\n",
    "svc = SVC()\n",
    "\n",
    "svc1= SVC(random_state = 42,kernel = 'rbf')\n",
    "svc1.fit(x_train, y_train)\n",
    "\n",
    "ac = accuracy_score(y_test,svc1.predict(x_test))\n",
    "accuracies['SVM'] = ac\n",
    "\n",
    "\n",
    "print('Accuracy is: ',ac, '\\n')\n",
    "cm = confusion_matrix(y_test,svc1.predict(x_test))\n",
    "sns.heatmap(cm,annot=True,fmt=\"d\")\n",
    "\n",
    "print('SVM report\\n',classification_report(y_test, svc1.predict(x_test)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Our Final Model would be 【Random Forest】 with Accuracy 79%, AUC: 0.83"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
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
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
