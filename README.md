# Solution for Suspicious Network Event Recognition challenge

## Setup
To setup environment install python 3.9, all necessary packages are listed in requirements.txt. To install them run:
```
pip install -r requirements.txt
```
And put challenge data into data folder.
## Data analysis
Data analysis was done with data reports using `ydata-profiling` which contain count and hist plots for each feature and correlation matrix. To generate those reports run:
```
python generate_data_reports.py
```

## Preprocessing
To preprocess data run:
```
python preprocess_data.py
```
You can modify input and output paths thru command line arguments.

## Training
To train model that was final solution run:
```
python train.py -c config.yaml
```
You can modify config file or pass command line arguments to change training script behavior.
By default it will train models and fine tune the best one. By default it will save tuned model and its predictions on test set in `results` folder. By default every run will be logged into MLFlow.