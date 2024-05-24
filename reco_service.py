import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from reco_env import RecommendationEnv
from reco_online_trainer import online_Recommendation_Trainer


class RecommendationService:
    def __init__(self, user_id):
        self.user_id = user_id
        self.agent = PPO.load('path_to_loaded_agent')
        user_movie_data = pd.read_csv('path_to_test_user_csv')
        user_data = user_movie_data[user_movie_data['user_id'] == user_id]
        movies_data = pd.read_csv('path_to_movies_csv')  # Load movies data
        
        self.env = RecommendationEnv(user_data, movies_data)
        self.reco_trainer = online_Recommendation_Trainer(self.agent, self.env)
        
        self.set_recommended_movies()
        self.set_unwatched_movies()
        self.set_watched_movies()
        
    def set_recommended_movies(self):
        recommended_movies = self.reco_trainer.online_update(num_steps=1)
        self.recommended_movies = recommended_movies
        
    def set_watched_movies(self):
        self.watched_movies = self.env.user_data['watched_movies'].values[0]

    def set_unwatched_movies(self):
        self.unwatched_movies = [m for m in self.env.user_data['unwatched_movies'].values[0] if m not in self.get_recommended_movies()]

    def get_recommended_movies(self):
        return self.recommended_movies
    
    def get_watched_movies(self):
        return self.watched_movies
    
    def get_unwatched_movies(self):
        return self.unwatched_movies
    
    def submit_feedback(self, submit_movies):
        watched_movies = self.get_watched_movies() + [m['movie'] for m in submit_movies['recommended_movies']] + [m['movie'] for m in submit_movies['selected_movies']]
        unwatched_movies = submit_movies['unwatched_movies']
        ratings = []
        
        for movie in watched_movies:
            rating = 0
            if movie in [m['movie'] for m in submit_movies['recommended_movies'] if m['like']]:
                rating = 5
            elif movie in [m['movie'] for m in submit_movies['recommended_movies'] if not m['like']]:
                rating = -5
            elif movie in [m['movie'] for m in submit_movies['selected_movies']]:
                rating = 3
            ratings.append(rating)
        
        self.env.set_user_data({
            'user_id': self.user_id,
            'user_indx': self.env.user_data['user_indx'].values[0],
            'watched_movies': watched_movies,
            'unwatched_movies': unwatched_movies,
            'ratings': ratings
        })
        
        feedback = {
            'liked_movies': [m['movie'] for m in submit_movies['recommended_movies'] if m['like']],
            'disliked_movies': [m['movie'] for m in submit_movies['recommended_movies'] if not m['like']],
            'new_select_movies': [m['movie'] for m in submit_movies['selected_movies']]
        }
        self.env.set_current_feedback(feedback)
        self.reco_trainer.online_update(num_steps=1)
        
        self.set_recommended_movies()
        self.set_watched_movies()
        self.set_unwatched_movies()
