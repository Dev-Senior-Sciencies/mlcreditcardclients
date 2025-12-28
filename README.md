# Ml Credit Card Clients

1. [Setup](#setup)
1. [Dataset](#dataset)

## Setup

### Pre-reqs

- Python 3 (I specifically use 3.13, but should work in other versions)
- Git
- Knowledge about Machine Learning Models

### Setting up your working environment

1. Clone this repo

```console
git clone git@github.com:Dev-Senior-Sciencies/mlcreditcardclients.git
```

2. Create a Python venv

   There are many ways to do that. I'll use the simple one, but you can use your preferable.

   Inside of the clonned path run:

```console
py -m venv .env
```

3. Activate venv

> For Linux:

```console
source .env/bin/activate
```

> For Windows:

```console
.env/Scripts/Activate
```

> or deactivate

```console
deactivate
```

3. Install requirements.txt

```console
pip install -r requirements.txt
```

> or new dependencies

```console
pip install tensorflow scikit-learn matplotlib mlflow omegaconf
```

```console
pip freeze > requirements.txt
```

## Dataset

#### Dataset Information

This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

#### Source

https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset?resource=download

Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
