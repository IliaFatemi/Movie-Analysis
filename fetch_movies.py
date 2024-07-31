import requests
import json
import pandas as pd

class FetchMovies:
    def __init__(self) -> None:
        self.headers = {
            "accept": "application/json",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJkNzE3YjY2NDM5YzVkYTRjYTc5M2E1NzZmNDFiYzg5NiIsIm5iZiI6MTcxOTg5MTg2Ny43ODM0ODcsInN1YiI6IjY2ODM3NjQwMmEyZjkzN2Q5ZjA0MTA5MSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.8Mv44K91pqlMhbqayEcr_IWJ3CjZqGBXlwpPB1_h4cQ"
        }
        
    # method to get the details of each movie by its movie ID
    def getDetailsById(self, id):
        print(f'getting id: {id}')
        movie_details_url = f"https://api.themoviedb.org/3/movie/{id}?language=en-US"
        response = requests.get(movie_details_url, headers=self.headers).json()
        return response.get('budget', ''), response.get('revenue', '')
    
if __name__ == "__main__":
    movies = FetchMovies()
    
    # https://datasets.imdbws.com/
    # columns names: https://developer.imdb.com/non-commercial-datasets/
    #Files not added to repository due to large size. Files can be downloaded from the links above
    print("unloading title.basics.tsv.gz...")
    title_basics_df = pd.read_csv('title.basics.tsv.gz', compression='gzip', sep='\t')
    print("unloading title.akas.tsv.gz...")
    title_akas_df = pd.read_csv('title.akas.tsv.gz', compression='gzip', sep='\t')
    print("unloading title.ratings.tsv.gz...")
    title_ratings_df = pd.read_csv('title.ratings.tsv.gz', compression='gzip', sep='\t')
    
    # Merge the DataFrames on 'tconst' and 'titleId'
    merged_df = pd.merge(title_basics_df, title_akas_df, left_on='tconst', right_on='titleId')
    
    # Merge the resulting DataFrame with the title ratings DataFrame on 'tconst'
    merged_df = pd.merge(merged_df, title_ratings_df, on='tconst')

    # Convert 'startYear' to numeric, coercing errors to NaN
    merged_df['startYear'] = pd.to_numeric(merged_df['startYear'], errors='coerce')

    # Filter the DataFrame
    filtered_df = merged_df[(merged_df['region'] == 'US') & 
                            (merged_df['startYear'] >= 1860) & 
                            (merged_df['titleType'] == 'movie')]

    # Remove duplicate rows based on 'primaryTitle'
    filtered_df = filtered_df.drop_duplicates(subset=['primaryTitle'])

    # Drop the 'titleId' column
    filtered_df = filtered_df.drop(columns=['titleId', 'language', 'originalTitle', 'isAdult', 'ordering', 'title', 'attributes', 'types', 'isOriginalTitle', 'region', 'endYear', 'titleType'])
    
    print("getting budget and revenue...")
    # Use Api to get budget and revenue information
    filtered_df[['budget', 'revenue']] = filtered_df.apply(lambda id: movies.getDetailsById(id['tconst']), axis=1, result_type='expand')

    # Save the filtered DataFrame to a CSV file
    filtered_df.to_csv('movies_dataset_complete.csv', index=False)

    ## Create cleaned dataset with profits

    cleaned_profits = filtered_df.copy(deep = True)

    cleaned_profits = cleaned_profits[(cleaned_profits['budget'] != 0) & (cleaned_profits['revenue'] != 0)].dropna()

    cleaned_profits['profit'] = cleaned_profits['revenue'] - cleaned_profits['budget']

    cleaned_profits.to_csv("movie_dataset_cleaned_with_profits.csv")