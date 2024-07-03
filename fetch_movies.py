import requests
import json
import pandas as pd

class FetchMovies:
    def __init__(self) -> None:
        self.headers = {
            "accept": "application/json",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJkNzE3YjY2NDM5YzVkYTRjYTc5M2E1NzZmNDFiYzg5NiIsIm5iZiI6MTcxOTg5MTg2Ny43ODM0ODcsInN1YiI6IjY2ODM3NjQwMmEyZjkzN2Q5ZjA0MTA5MSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.8Mv44K91pqlMhbqayEcr_IWJ3CjZqGBXlwpPB1_h4cQ"
        }

        self.data:pd.DataFrame = pd.DataFrame(columns=[
            'movie_title', 
            'genres', 
            'runtime_minutes', 
            'vote_average',
            'vote_count', 
            'budget',
            'revenue',
            'release_date'
            ])
        
        self.genre_dict = {
            28: "Action",
            12: "Adventure",
            16: "Animation",
            35: "Comedy",
            80: "Crime",
            99: "Documentary",
            18: "Drama",
            10751: "Family",
            14: "Fantasy",
            36: "History",
            27: "Horror",
            10402: "Music",
            9648: "Mystery",
            10749: "Romance",
            878: "Science Fiction",
            10770: "TV Movie",
            53: "Thriller",
            10752: "War",
            37: "Western"
        }
        
    def getMovieTitlesFromFile(self) -> pd.DataFrame:
        df = pd.read_csv('movie_statistic_dataset.csv')
        return df['movie_title']
        
    def __saveData(self):
        self.data.to_csv('movies.csv', index=False)
        
    # Private method to get the details of each movie by its movie ID
    def getDetailsById(self, id):
        movie_details_url = f"https://api.themoviedb.org/3/movie/{id}?language=en-US"
        response = requests.get(movie_details_url, headers=self.headers).json()
        return response.get('budget', ''), response.get('revenue', '')
        
    # def getMovieDetails(self, filename):
    #     checkpoint = 50
    #     count = 0
    #     df = pd.read_csv(filename)
    #     used_movies = set()
        
    #     for id in df['titleId']:
    #         url = f"https://api.themoviedb.org/3/find/{id}?external_source=imdb_id"
    #         response = requests.get(url, headers=self.headers).json()
    #         result = response['movie_results']
    #         genres = [self.genre_dict[genre_id] for genre_id in response.get('genre_ids', [])]
    #         for movie in result:
    #             if movie['original_title'] not in used_movies:
    #                 used_movies.add(movie['original_title'])
    #                 row = self.__getDetailsById(movie['id'])
    #                 if len(genres) == 0:
    #                     row.update({'genres': '-'})
    #                 else:
    #                     row.update({'genres': ', '.join(genres)})
                        
    #                 self.data = self.data.append(row, ignore_index=True)
    #                 count += 1
    #         if count == checkpoint:
    #             print(count)
    #             checkpoint += 50
    #             self.__saveData()
    #     self.__saveData()
    
if __name__ == "__main__":
    movies = FetchMovies()
    
    # https://datasets.imdbws.com/
    # columns names: https://developer.imdb.com/non-commercial-datasets/
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