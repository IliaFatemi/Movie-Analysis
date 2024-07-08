import pandas as pd
from IPython.display import display

class CleanMovies:
    def __init__(self):
        self.data = self.get_data()

    def get_data(self):
        return pd.read_csv("movies_dataset_complete.csv", index_col = "tconst")
    
    def clean_zero_budget_and_revenue(self):
        self.data = self.data[(self.data['budget'] != 0) & (self.data['revenue'] != 0)].dropna()

    def add_profit_column(self):
        self.data['profit'] = self.data['revenue'] - self.data['budget']

    def save_cleaned_data(self):
        self.data.to_csv("movie_dataset_cleaned_with_profits.csv")

if __name__ == "__main__":
    movies = CleanMovies()
    movies.clean_zero_budget_and_revenue()
    movies.add_profit_column()
    movies.save_cleaned_data()
    display(movies.data)