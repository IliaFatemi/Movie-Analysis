import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display
from statsmodels.nonparametric.smoothers_lowess import lowess

class MovieFactors:
    def __init__(self):
        self.ratings = np.linspace(0,10,21)
        self.data = self.get_data()
        self.clean_data_more()
        self.data_grouped_by_ratings = self.group_data_by_rating()
        
    def clean_data_more(self):
        self.data['profit'] = self.data['profit'] / 1000000 # 1 mil
        self.data['runtimeMinutes'] = pd.to_numeric(self.data['runtimeMinutes'], errors='coerce').dropna()
        self.data = self.data[self.data['runtimeMinutes'] >= 0]

    def get_data(self):
        return pd.read_csv("movie_dataset_cleaned_with_profits.csv", index_col = "tconst")
    
    def scatter_plot_ratings_vs_profitability(self):
        plt.figure(figsize=(8,5))
        plt.title('Movies Ratings vs Profitability')
        plt.xlabel('Rating')
        plt.ylabel('Profitability in millions')
        plt.scatter(self.data['averageRating'], self.data['profit'], color = 'blue', alpha = 0.5)
        self.data['linearRating'] = self.data['averageRating']
        fit = stats.linregress(self.data['linearRating'], self.data['profit'])
        plt.plot(self.data['averageRating'], self.data['linearRating'] *fit.slope + fit.intercept, color = 'red', linewidth = 3)
        plt.show()
        # plt.savefig('test2.png')

    def scatter_plot(self, title, xlabel, ylabel, scat_x, scat_y, lin_name, lin_copy, lin_column, show_lin, show_loess, show):
        plt.figure(figsize=(8,5))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.scatter(scat_x, scat_y, color = 'blue', alpha = 0.5)
        if show_lin:
            self.data[lin_name] = self.data[lin_copy]
            fit = stats.linregress(self.data[lin_name], lin_column)
            print(fit)
            # plt.plot(self.data[lin_name], self.data[lin_name] *fit.slope + fit.intercept, color = 'red', linewidth = 3)
        if show_loess:
            loess = lowess(scat_y, scat_x, frac = 0.2, it = 300)
            plt.plot(loess[:, 0], loess[:, 1], 'g-')
        if (show):
            plt.show()
        else:
            plt.savefig('test2.png')

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
        plt.bar(self.data_grouped_by_ratings['rating_groups'].astype(str), self.data_grouped_by_ratings['profit'], width=0.5, edgecolor = "black")
        # Referenced this on how to make a histogram using with pre-grouped data
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html
        plt.show()
        # plt.savefig('test.png')
    
    def RunTime_vs_Profitability(self, show):
        plt.figure(figsize=(8,5))
        plt.title('Movies Runtime vs Median Profitability')
        plt.xlabel('Run Time')
        plt.ylabel('Profitability in millions')
        # plt.scatter(movies.data['runtimeMinutes'], movies.data['profit'], color = 'blue', alpha = 0.5)

        temp = self.data[['runtimeMinutes', 'profit']].copy(deep = True)
        print(self.data['profit'].median())
        print(self.data['runtimeMinutes'].median())

        grouped = temp.groupby('runtimeMinutes')['profit'].median().reset_index() 
        grouped = temp.groupby('runtimeMinutes').agg(
            median_profit = pd.NamedAgg(column='profit', aggfunc='median'),
            entries = pd.NamedAgg(column='profit', aggfunc='count')
        ).reset_index() 
        grouped = grouped[grouped['entries'] >= 20] # drop groupings less than 10 as data may get skewed by outliers

        # plt.plot(grouped['runtimeMinutes'], grouped['median_profit'], color = 'red')

        # median on profit column, reset index so I can access runTime colum
        pd.set_option('display.max_rows', None)
        display(grouped)

        loess = lowess(grouped['median_profit'], grouped['runtimeMinutes'], frac = 0.1)
        plt.plot(loess[:, 0], loess[:, 1], 'g-')

        if (show):
            plt.show()
        else:
            plt.savefig('runtime_vs_profits_median_cleaned.png')

if __name__ == "__main__":
    movies = MovieFactors()
    # movies.scatter_plot_ratings_vs_profitability()
    movies.RunTime_vs_Profitability(True)
    # movies.scatter_plot('Movies Ratings vs Profitability', 'Rating', 'Profitability in millions', 
    #                     movies.data['averageRating'], movies.data['profit'], 
    #                     'linearRating', 'averageRating', movies.data['profit'] , True, False, True )
    
    # movies.data = movies.data[movies.data['runtimeMinutes'] <= 250]
    # print (movies.data['runtimeMinutes'])
    # movies.data = movies.data[movies.data['profit'] <= 500]

    # TODO
    # maybe do loess smoothing or something?

    

    # movies.scatter_plot('Movies Runtime vs Profitability', 'Run Time', 'Profitability in millions', 
    #                     movies.data['runtimeMinutes'], movies.data['profit'], 
    #                     'linearRuntime', 'runtimeMinutes', movies.data['profit'] , False, True, True )
    # movies.data['linearRuntime'] = movies.data['runtimeMinutes']
    # movies.data['linearRating'] = movies.data['averageRating']
    # movies.data['profit_in_millions'] = movies.data['profit'] 
    # fit = stats.linregress(movies.data['linearRuntime'], movies.data['profit_in_millions'])
    # print(fit)

    # movies.histogram_ratings_vs_profitability()
    # display(movies.data.sort_values(by="startYear"))
