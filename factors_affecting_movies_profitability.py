import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display
from sklearn.impute import KNNImputer
from statsmodels.nonparametric.smoothers_lowess import lowess

class MovieFactors:
    def __init__(self):
        self.data = self.get_data()
        self.clean_data_more()
        
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
            plt.plot(self.data[lin_name], self.data[lin_name] *fit.slope + fit.intercept, color = 'red', linewidth = 3)

        if show_loess:
            loess = lowess(scat_y, scat_x, frac = 0.2, it = 300)
            plt.plot(loess[:, 0], loess[:, 1], 'g-')
        if (show):
            plt.show()
        else:
            plt.savefig('test2.png')

    def Histogram_Ratings_vs_Profits(self, show):
        rating_groups = np.linspace(1,10,19)

        plt.figure(figsize=(10,7))
        plt.title('Movies Ratings vs Median Profits', fontsize=16)
        plt.xlabel('Rating', fontsize=14)
        plt.xticks(rating_groups[:-1])
        plt.ylabel('Value', fontsize=14)

        ## GROUP DATA BY RATING
        temp = self.data.copy(deep=True)
        temp['rating_groups'] = pd.cut(temp['averageRating'], rating_groups) 
        # Referenced this webpage for how to use the cut function to give a new ratings column that will be used to group data
        # https://pandas.pydata.org/docs/reference/api/pandas.cut.html

        grouped = temp.groupby('rating_groups', observed = False).agg(
            profit = pd.NamedAgg(column='profit', aggfunc='median'),
            entries = pd.NamedAgg(column='profit', aggfunc='count'),
        ).reset_index() # set observed to false remove warning
        grouped['groups'] = rating_groups[:-1] # create groups column for merging, excludes last index of value 10

        ## INCONCLUSIVE GROUPED DATA
        null_groups = pd.DataFrame(rating_groups[:-1], columns=["groups"])
        less_than_20 = grouped[grouped['entries'] < 40] 
        # rating groups with less than 20 are excluded from bar graph
        null_groups = pd.merge(null_groups, less_than_20, on = "groups", how="outer")
        null_groups.loc[null_groups['entries'].isnull() , 'profit'] = 0
        null_groups.loc[null_groups['entries'].isnull() == False, 'profit'] = 1
        # Referenced this for how to alter data in rows depending on a condition
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html
        
        ## ENTRIES PER RATING GROUP AS A PERCENTAGE SCALED BY 15
        grouped.loc[grouped['entries'] < 40, 'profit'] = 0
        total_entries = self.data.shape[0]
        grouped['entries_perc_times_200'] = (grouped['entries'] / total_entries) * 15

        # # -- Can uncomment to see raw data on group pins --
        # pd.set_option('display.max_rows', None)
        # display(grouped)
        
        plt.bar(grouped['groups'] + 0.25, grouped['profit'], width=0.5, edgecolor = "black") # profits
        plt.bar(rating_groups[:-1]+ 0.25, null_groups['profit'],width=0.5, color='none', hatch='/', edgecolor = "black") # inconclusive profits
        # Referenced this on how to make a histogram using with pre-grouped data
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html
        plt.plot(rating_groups[:-1]+ 0.25, grouped['entries_perc_times_200'], color = 'red') # entries percentage scaled by 200

        plt.legend(['Percentage of Movies Per Group Scaled by 15', 'Profits in Millions', 'Rating Groups With Less Than 40 Entries (Inconclusive)'], fontsize = 12)

        if show:
            plt.show()
        else:
            plt.savefig('Movies_Ratings_vs_Median_Profits.png')
    
    def PrintMedianRunTime(self):
        print(f'Median Run Time: {self.data['runtimeMinutes'].median()}')
    
    def PrintMedianProfit(self):
        print(f'Median Profit: {self.data['profit'].median()}')

    def PrintMedianRating(self):
        print(f'Median Rating: {self.data['averageRating'].median()}')

    def PrintMeanRating(self):
        print(f'Mean Rating: {self.data['averageRating'].mean()}')

    def PrintStandardDeviationRating(self):
        print(f'Standard Deviation Rating: {self.data['averageRating'].std()}')

    def RunTime_vs_Profitability(self, show):
        plt.figure(figsize=(8,5))
        plt.title('Movies Run Time vs Median Profits', fontsize=16)
        plt.xlabel('Run Time', fontsize=14)
        plt.ylabel('Dollars in Millions', fontsize=14)
        
        ## LOESS PROFITS
        temp = self.data[['runtimeMinutes', 'profit']].copy(deep = True)

        grouped = temp.groupby('runtimeMinutes')['profit'].median().reset_index() 
        grouped = temp.groupby('runtimeMinutes').agg(
            median_profit = pd.NamedAgg(column='profit', aggfunc='median'),
            entries = pd.NamedAgg(column='profit', aggfunc='count'),
        ).reset_index() 
        grouped = grouped[grouped['entries'] >= 20] 
        # drop groupings less than 20 as data may get skewed by outliers

        loess = lowess(grouped['median_profit'], grouped['runtimeMinutes'], frac = 0.1, it = 5)
        plt.plot(loess[:, 0], loess[:, 1], 'g-')

        ## LINEAR REGRESSION
        linspaceRunTime = np.linspace(75,155, 81)
        runTime = pd.DataFrame(linspaceRunTime, columns=["runtimeMinutes"])
        runTime = pd.merge(runTime, grouped, on = "runtimeMinutes", how="outer")
        # outer since we want to keep linspace times

        # -- Can uncomment to see which data points are imputed --
        # pd.set_option('display.max_rows', None)
        # display(runTime)

        imputerKNN = KNNImputer(n_neighbors=5) 
        # 5 neighbours since a lot of variance in neighbours of data
        runTime = pd.DataFrame(imputerKNN.fit_transform(runTime), columns=runTime.columns)

        fit = stats.linregress(runTime['runtimeMinutes'], runTime['median_profit'])
        plt.plot(runTime['runtimeMinutes'], runTime['runtimeMinutes'] *fit.slope + fit.intercept, color = 'red', linewidth = 1)

        plt.legend(['Profits', 'Linear Fit'], fontsize = 12)

        if (show):
            plt.show()
        else:
            plt.savefig('Movies_Run_Time_vs_Median_Profits.png')

    def PrintMajorityOfDataInRange(self):
        lower_bound = self.data['averageRating'].mean() - self.data['averageRating'].std()
        upper_bound = self.data['averageRating'].mean() + self.data['averageRating'].std()
        print(f'Approximately 68% of data is in the Rating range of : [{lower_bound:.2f}, {upper_bound:.2f}]')
        # Referenced this for how to limit a number to 2 decimal places
        # https://stackoverflow.com/questions/455612/limiting-floats-to-two-decimal-points

if __name__ == "__main__":
    movies = MovieFactors()
    # movies.scatter_plot_ratings_vs_profitability()

    # movies.RunTime_vs_Profitability(show = True) # False will save png
    movies.Histogram_Ratings_vs_Profits(show = True) # False will save png
    movies.PrintMedianProfit()
    movies.PrintMedianRunTime()
    movies.PrintMedianRating()
    movies.PrintMeanRating()
    movies.PrintStandardDeviationRating()
    movies.PrintMajorityOfDataInRange()
