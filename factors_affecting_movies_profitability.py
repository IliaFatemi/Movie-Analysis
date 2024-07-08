import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

class MovieFactors:
    def __init__(self):
        self.ratings = np.linspace(0,10,21)
        self.data = self.get_data()
        self.data_grouped_by_ratings = self.group_data_by_rating()

    def get_data(self):
        return pd.read_csv("movie_dataset_cleaned_with_profits.csv", index_col = "tconst")
    
    def scatter_plot_ratings_vs_profitability(self):
        plt.figure(figsize=(8,5))
        plt.title('Movies Ratings vs Profitability')
        plt.xlabel('Rating')
        plt.ylabel('Profitability in millions')
        plt.scatter(self.data['averageRating'], self.data['profit']/1000000, color = 'blue', alpha = 0.5)
        plt.show()

    def group_data_by_rating(self):
        temp = self.data.copy(deep=True)
        temp['rating_groups'] = pd.cut(temp['averageRating'], self.ratings) 
        # Referenced this webpage for how to use the cut function to give a new ratings column that will be used to group data
        # https://pandas.pydata.org/docs/reference/api/pandas.cut.html

        # take sum so we get the profitability of all entries in the groupings
        return temp.groupby('rating_groups', observed = False).sum().reset_index() # set observed to false remove warning

    def histogram_ratings_vs_profitability(self):
        plt.figure(figsize=(10,7))
        plt.title('Movies Ratings vs Profitability')
        plt.xlabel('Rating')
        plt.xticks(rotation=25)
        plt.ylabel('Profitability in millions')
        plt.bar(self.data_grouped_by_ratings['rating_groups'].astype(str), self.data_grouped_by_ratings['profit']/1000000, width=0.5, edgecolor = "black")
        # Referenced this on how to make a histogram using with pre-grouped data
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html
        plt.show()
    

if __name__ == "__main__":
    movies = MovieFactors()
    movies.scatter_plot_ratings_vs_profitability()
    movies.histogram_ratings_vs_profitability()
