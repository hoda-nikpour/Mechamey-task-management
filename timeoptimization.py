import numpy as np

class ProjectManagementEnv:
    def __init__(self):
        self.budget = 100000  
        self.time = 20  
        self.scope = 100  
        self.reward = 0
    
    def step(self, action):
       
        if action == 0:
            self.scope -= 10
            self.reward -= 1
        elif action == 1:
            self.budget += 5000
            self.reward -= 1
        elif action == 2:
            self.time += 1
            self.reward -= 1
        
        if self.scope >= 100 and self.budget >= 100000 and self.time <= 20:
            self.reward += 10
        
        self.budget -= 5000
        self.time -= 1
        
        done = self.time <= 0 or self.budget <= 0 or self.scope <= 0
        
        return self.scope, self.budget, self.time, self.reward, done

    def reset(self):
        self.budget = 100000
        self.time = 20
        self.scope = 100
        self.reward = 0
        return self.scope, self.budget, self.time


class QNetwork:
    def __init__(self, n_actions):
        self.Q = np.zeros((n_actions,))

    def predict(self, state):
        return self.Q[state]

    def update(self, state, action, reward, next_state, alpha, gamma):
        predicted_value = self.predict(state)
        next_max_value = np.max(self.Q[next_state])
        updated_value = predicted_value + alpha * (reward + gamma * next_max_value - predicted_value)
        self.Q[state] = updated_value


n_actions = 3  
n_states = 3  
alpha = 0.1  
gamma = 0.9  
epsilon = 0.1  


env = ProjectManagementEnv()
q_network = QNetwork(n_actions)


num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
   
        if np.random.rand() < epsilon:
            action = np.random.choice(n_actions)
        else:
            action = np.argmax(q_network.predict(state))

        next_state, _, _, reward, done = env.step(action)
        q_network.update(state, action, reward, next_state, alpha, gamma)
        state = next_state

    print(f"Episode {episode + 1}, Reward: {q_network.predict(state)}")

