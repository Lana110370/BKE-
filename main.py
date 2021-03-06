from _ml import MLAgent, train, save, load, validate, plot_validation, train_and_plot
from _core import is_winner, opponent, start
from _agent import EvaluationAgent, RandomAgent
import random

class MyAgent(MLAgent):
    def evaluate(self, board):
        if is_winner(board, self.symbol):
            reward = 1
        elif is_winner(board, opponent[self.symbol]):
            reward = -1
        else:
            reward = 0
        return reward
     
random.seed(1)
 
 
#Hoe werken hyperparameters? Hyperparameters zijn variabelen die door middel van bepaalde waardes de manier waarop de agent leert anders samenstelt. Hyperparameters beheersen dus het leerproces van de agent, doordat ze de mogelijkheid bieden om de samenstelling van de agent te veranderen.

#Er wordt, zoals hieronder te zien is, gebruik gemaakt van twee verschillende parameters. 'Alpha' is de leerfactor van de agent en bepaalt hoe snel de agent oude informatie vervangt door nieuwe kennis op te nemen. Door de alpha een hogere waarde te geven die dichter bij de 1 ligt, zal de agent sneller nieuwe informatie willen overnemen. 'Epsilon' is de verkenningsfactor van de agent en bepaalt hoe vaak de agent nieuwe zetten probeert. Door de epsilon een hogere waarde te geven die dichter bij de 1 ligt, zal de agent sneller geneigd zijn om nieuwe zetten te proberen in plaats van alleen sterke zetten te gebruiken die hij al kent. Hoe vaker de agent nieuwe zetten probeert, hoe hoger de kans dat hij een nieuwe goede zet ontdekt.


my_agent = MyAgent(alpha=0.7, epsilon=0.3)

random_agent = RandomAgent()

train(my_agent, 3000)

save(my_agent, 'MyAgent_3000')

my_agent = load('MyAgent_3000')

my_agent.learning = True


while True:
  features = input("Klik op 1 of 2 om te spelen. Klik op 3 of 4 om een grafiek te plotten.")

  # Speel tegen agent
  if features == '1':
    my_agent = load('MyAgent_3000'); 
    start(player_o=my_agent); 

  # Speel tegen iemand anders
  if features == '2':
    start()

  # Plot validatiegrafiek
  if features == '3':
   validation_agent = RandomAgent()

   validation_result = validate(agent_x=my_agent,  agent_o=validation_agent, iterations=150)

   plot_validation(validation_result)

  # Plot lijngrafiek
  if features == '4':
   train_and_plot(
    agent=my_agent,
    validation_agent=random_agent,
    iterations=25,
    trainings=150,
    validations=1500)