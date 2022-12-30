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

4.) Recommended: Normalize the data
```sh
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(treatment_data[cols])
scaled = scaler.transform(treatment_data[cols])
scaled = pd.DataFrame(scaled, columns=cols)
```

5.) Fit and predict:
```sh
full_res = iso_class.fit_predict(scaled, return_class=True)
```

6.) Predict on new data:
```sh
full_res = iso_class.predict(scaled, return_class=True)
```

Classes and outlier scores can always be accessed from the class instance via:
```sh
iso_class.scores  # getting the outlier scores
iso_class.outlier_classes  # get the classes
```

The repo contains example notebooks. Please note that example notebooks do not always contain the newest version. 
The file psod.py is always the most updated one.
[See the full article](https://medium.com/@thomasmeissnerds)


## Meta

Creator: Thomas Meißner – [LinkedIn](https://www.linkedin.com/in/thomas-mei%C3%9Fner-m-a-3808b346)

[PSOD GitHub repository](https://github.com/ThomasMeissnerDS/PSOD)