import requests
import json
import pandas as pd

class FetchMovies:
    def __init__(self) -> None:
        self.url = "https://api.themoviedb.org/3/search/movie?query=Guardians%20of%20the%20Galaxy&include_adult=true&language=en-US&page=1"
        self.headers = {
            "accept": "application/json",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJkNzE3YjY2NDM5YzVkYTRjYTc5M2E1NzZmNDFiYzg5NiIsIm5iZiI6MTcxOTg5MTg2Ny43ODM0ODcsInN1YiI6IjY2ODM3NjQwMmEyZjkzN2Q5ZjA0MTA5MSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.8Mv44K91pqlMhbqayEcr_IWJ3CjZqGBXlwpPB1_h4cQ"
        }

        self.response = requests.get(self.url, headers=self.headers).json()

            
    def getMovies(self):
        # Print the JSON response in a pretty format
        # print(json.dumps(json_response, indent=4))
        results = self.response['results']
        for movies in results:
            print(movies['title'])
    
    
if __name__ == "__main__":
    FetchMovies()