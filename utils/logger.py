import csv 
import os 

class logger:
    def __init__(self,output_dir,filename = 'results.csv'):
        self.path = os.path.join(output_dir,filename)
        with open(self.path,'w',newline = '') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'avg_reward', 'avg_loss',
                'epsilon', 'steps',
                'J1_reward', 'J2_reward', 'J4_reward', 'J5_reward'
            ])

    def log(self, episode, avg_reward, avg_loss,
            epsilon, steps, junction_rewards):
        with open(self.path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                round(avg_reward, 4),
                round(avg_loss, 6),
                round(epsilon, 4),
                steps,
                round(junction_rewards.get('J1', 0), 4),
                round(junction_rewards.get('J2', 0), 4),
                round(junction_rewards.get('J4', 0), 4),
                round(junction_rewards.get('J5', 0), 4),
            ])
