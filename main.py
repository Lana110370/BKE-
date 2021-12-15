from _ml import MLAgent, train, save, load, train_and_plot
from _core import is_winner, opponent, start
from _agent import EvaluationAgent
import random

class MyRandomAgent(EvaluationAgent):
    def evaluate(self, board, my_symbol, opponent_symbol):
        return random.randint(1, 500)
    
    
my_random_agent = MyRandomAgent()
start(player_o=my_random_agent)