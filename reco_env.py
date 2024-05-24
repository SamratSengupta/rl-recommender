import gym
from gym import spaces
import numpy as np

class RecommendationEnv(gym.Env):
    def __init__(self, user_data, movies_data, total_users=4083, movie_embedding_dim=100):
        super(RecommendationEnv, self).__init__()
        
        self.user_data = user_data  # DF of 1 row with user_id, user_indx, watched_movies, unwatched_movies, ratings
        self.movies_data = movies_data  # DF with all movies (watched + unwatched)
        self.total_users = total_users
        self.movie_embedding_dim = movie_embedding_dim
        
        self.max_item_len = len(self.user_data['watched_movies'][0]) + len(self.user_data['unwatched_movies'][0])
        
        self.action_space = spaces.Box(low=0, high=1, shape=(self.max_item_len,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            "user_indices": spaces.Box(low=0, high=total_users, shape=(self.max_item_len,), dtype=int),
            "movie_indices": spaces.Box(low=0, high=self.max_item_len-1, shape=(self.max_item_len,), dtype=float),           
            "movie_embeddings": spaces.Box(low=-float('inf'), high=float('inf'), shape=(self.max_item_len, self.movie_embedding_dim), dtype=float),
            "movie_ratings": spaces.Box(low=0.0, high=5.0, shape=(self.max_item_len,), dtype=float)
        })

        self.feedback = {}

    def set_user_data(self, user_data):
        self.user_data = user_data

    def set_current_feedback(self, feedback_movies):
        self.feedback = feedback_movies

    def calculate_rewards(self):
        reward = 0
        for movie in self.feedback.get('liked_movies', []):
            reward += 5  # Positive reward for liking recommended movies
        for movie in self.feedback.get('disliked_movies', []):
            reward -= 5  # Negative reward for disliking recommended movies
        for movie in self.feedback.get('new_select_movies', []):
            reward += 3  # Positive reward for selecting new movies from unwatched set
        return reward

    def step(self, action):
        self.set_current_feedback(self.get_user_feedback(action))  # Assume get_user_feedback is implemented
        
        reward = self.calculate_rewards()
        
        self.update_user_data()  # Update user data based on feedback
        
        state = self.get_current_state()
        
        done = True  # Assuming each step represents a complete interaction cycle

        return state, reward, done, {}

    def reset(self):
        return self.get_current_state()
    
    def get_current_state(self):
        user_index = self.user_data['user_indx'].values[0]
        movie_indices = np.arange(self.max_item_len)
        movie_embeddings = self.movies_data.loc[movie_indices, 'movie_embedding'].values
        movie_ratings = np.zeros(self.max_item_len)
        
        watched_movies = self.user_data['watched_movies'].values[0]
        ratings = self.user_data['ratings'].values[0]

        for i, movie in enumerate(watched_movies):
            movie_index = self.movies_data.index[self.movies_data['movie_id'] == movie].tolist()[0]
            movie_ratings[movie_index] = ratings[i]

        return {
            "user_indices": np.full(self.max_item_len, user_index),
            "movie_indices": movie_indices,
            "movie_embeddings": movie_embeddings,
            "movie_ratings": movie_ratings
        }
    
    def get_user_feedback(self, action):
        # This method should be implemented to get actual user feedback
        feedback = {
            "liked_movies": np.random.choice(action, size=2).tolist(),
            "disliked_movies": np.random.choice([m for m in self.max_item_len if m not in action], size=2).tolist(),
            "new_select_movies": np.random.choice(self.max_item_len, size=3).tolist()
        }
        return feedback

    def update_user_data(self):
        feedback = self.feedback
        watched_movies = self.user_data['watched_movies'].values[0]
        watched_movies.extend(feedback['liked_movies'])
        watched_movies.extend(feedback['new_select_movies'])
        
        self.user_data['watched_movies'] = [watched_movies]
        
        unwatched_movies = [m for m in self.user_data['unwatched_movies'].values[0] if m not in watched_movies]
        self.user_data['unwatched_movies'] = [unwatched_movies]

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)
