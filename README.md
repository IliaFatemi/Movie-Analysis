# CMPT_353_Project

## Predicting Movie Revenue

There are two files for predicting movies reviews. The `revenue_predictions.ipynb` shows the results that is used on the report and `rev_predictions.py` is used for testing which we will demostrate on how to run.<br>

There are two datasets that can be tested, `movie_dataset_cleaned_with_profits.csv` and `movies_dataset_complete.csv`.<br>

The `movie_dataset_cleaned_with_profits.csv` has a faster output because there is less data but `movies_dataset_complete.csv` will take longer but might show more accurate results.

To test run the movie predictions, run the following file with it's argument:
```
python rev_predictions.py movie_dataset_cleaned_with_profits.csv
```

or

```
python rev_predictions.py movies_dataset_complete.csv
```

This will print the training and test scores in the terminal. A plot of <strong><i>Actual Vs Predicted Revenue</i></strong> and <strong><i>Feature Importance</i></strong> will be shown and saved as a `.png` file. 

## Project Requirements

```
pip install scikit-learn;
pip install pandas;
pip install matplotlib;
pip install numpy;
pip install statsmodels;
pip install ipython;
```
