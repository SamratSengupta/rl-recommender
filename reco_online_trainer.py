import numpy as np

class online_Recommendation_Trainer:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def online_update(self, num_steps=1):
        for _ in range(num_steps):
            action, _ = self.agent.predict(self.env.get_current_state())
            next_state, reward, done, _ = self.env.step(action)
            
            self.agent.rollout_buffer.add(self.env.get_current_state(), action, reward, next_state, done, [1.0])
            
            if len(self.agent.rollout_buffer) == self.agent.rollout_buffer.buffer_size:
                self.agent.rollout_buffer.compute_returns_and_advantage(last_values=self.agent.policy.predict_values(next_state))
                self.agent.train()
                self.agent.rollout_buffer.reset()
            
            self.env.state = next_state  # Update state
        return action


# # Example usage
# model_path = "path_to_pretrained_agent"
# all_movies = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]  # Extended set of movies
# user_ids = [1, 2, 3, 4, 5]  # Example user IDs

# # Create instances of the service, agent, and environment
# recommendation_agent = RecommendationAgent(model_path)
# recommendation_service = RecommendationService(recommendation_agent, all_movies)
# env = RecommendationEnv(recommendation_service, recommendation_agent, user_ids, all_movies)

# # Initialize the environment
# env.reset()
# done = False

# # Online update loop
# while not done:
#     online_update(recommendation_agent.agent, env)
#     done = env.current_user_index == 0  # End episode when all users have been processed

# # Show the shift in user preferences over time
# import matplotlib.pyplot as plt

# # Track the shifts in user preferences
# user_preference_shifts = []

# env.reset()
# done = False

# while not done:
#     action, _ = recommendation_agent.predict(env.state)
#     user_preference_shifts.append(action)  # Record the predicted actions
#     online_update(recommendation_agent.agent, env)
#     done = env.current_user_index == 0  # End episode when all users have been processed

# # Plot the shift in user preferences over time
# plt.plot(user_preference_shifts)
# plt.xlabel('Time Steps')
# plt.ylabel('Predicted Actions')
# plt.title('Shift in User Preferences Over Time')
# plt.show()
