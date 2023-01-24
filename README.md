# Pseudo-supervised outlier detection

> A highly performant alternative to purely unsupervised approaches.

PSOD uses supervised methods to identify outliers in unsupervised contexts. It offers higher accuracy for outliers
with top scores than other models while keeping comparable performance on the whole dataset.

The usage is simple.

1.) Install the package:
```sh
pip install psod
```

2.) Import the package:
```sh
from psod.outlier_detection.psod import PSOD
```

3.) Instantiate the class:
```sh
iso_class = PSOD()
```
The class has multiple arguments that can be passed. If older labels exist these could be used
for hyperparameter tuning.

4.) Recommended: Normalize the data. PSOD offers preprocessing functions. It can downcast all
columns to reduce memory footprint massively (up to 75%). It can also scale the data. For
convenience both steps can be called together using:
```sh
from psod.preprocessing.full_preprocessing import auto_preprocess

scaled = auto_preprocess(treatment_data)
```
However they can also be called individually on demand.

5.) Fit and predict:
```sh
full_res = iso_class.fit_predict(scaled, return_class=True)
```

6.) Predict on new data:
```sh
full_res = iso_class.predict(scaled, return_class=True, use_trained_stats=True)
```
The param use_trained_stats is a boolean indicating of conversion from outlier scores to outlier class
shall make use of mean and std of prediction errors obtained during training shall be used. 
If False prediction errors of the provided dataset will be treated as new distribution 
with new mean and std as classification thresholds.

Classes and outlier scores can always be accessed from the class instance via:
```sh
iso_class.scores  # getting the outlier scores
iso_class.outlier_classes  # get the classes
```
Many parameters can be optimized. Detailed descriptions on parameters can be found using:
```sh
help(iso_class)
```
By printing class instance current settings can be observed:
```sh
print(iso_class)
```

The repo contains example notebooks. Please note that example notebooks do not always contain the newest version. 
The file psod.py is always the most updated one.
[See the full article](https://medium.com/@thomasmeissnerds)

## Release History

* 1.2.0
    * Added use_trained_stats to predict function
    * Added doc strings to main functions
    * Fixed a bug where PSOD tried to drop categorical data in the absence of categorical data.
* 1.1.0
    * Add correlation based feature selection
* 1.0.0
    * Some bug fixes
    * Added yeo-johnson to numerical transformation options and changed the parameter name and type
    * Added preprocessing functionality (scaling and memory footprint reduction)
    * Added warnings to flag risky input params
    * Changed default of numerical preprocessing to None (previously logarithmic)
    * Suppressed Pandas Future and CopySettings warnings
    * Enhanced Readme
* 0.0.4
    * First version with bare capabilities


## Meta

Creator: Thomas Meißner – [LinkedIn](https://www.linkedin.com/in/thomas-mei%C3%9Fner-m-a-3808b346)

[PSOD GitHub repository](https://github.com/ThomasMeissnerDS/PSOD)