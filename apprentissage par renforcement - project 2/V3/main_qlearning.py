import qlearning_a_completer as qlearning
import grid6 as game


# Evaluate if the simulation reached an end state.
print("\n----------------------------")
print("Simple Q-Learning")
print("----------------------------\n")

# Initialize state-action matrice Q to zero
Q= { game.s0 : { a:0.0 for a in game.actions } }
qlearning.printQ("Initial: ", Q)

# Perform n episodes/iterations of Q-learning
for i in range(20000):
  Q, trace= qlearning.playEpisode(Q, 
                                  game.s0,
                                  game.actions,
                                  game.observe_next_state, 
                                  game.observe_reward, 
                                  game.isEnd)
  print("Iteration " + str(i) + " finished.")


  
qlearning.printQ("Result: ", Q)


# plot number of actions per episode
# plot sum rewards per episode

print("\n----------------------------")
print("Question 1 : prendre en mail le projet et le code fourni")
print("Découvrir l’ensemble de fichiers et le code fourni. Lancer le code. Une boucle d’affichage est lancée. Décrire ce qu’elle affiche. Quelle est sa condition d’arrêt ?")
print("----------------------------")
print("La boucle d'affichage affiche l'état, l'action, l'état suivant et la récompense à chaque itération.")
print("La condition d'arrêt est atteinte lorsque l'état final est atteint (dans chaque iteration) et que le nombre d'itérations defini est atteint (1000).")
print("----------------------------")
print("Question 2 : implementer Q-learning")
print("Compléter la fonction playEpisde de Q-learning en respectant l’algorithme donné dans le support du cours.")
print("----------------------------")
print("Question 3 : Tester Q-learning")
print("Tester qlearning sur le problème de Grid6. Est-ce que votre agent a appris la meilleure politique ? Expliquez.")
print("----------------------------")
print("L'agent a appris la meilleure politique car il a maximisé les récompenses en choisissant les actions optimales à chaque état.")
print("En revanche, il a fallu plus que 1000 iterations (environ 20 000) pour que l'algorithme converge vers la politique optimale.")
print("----------------------------")
print("Question 4 : Tester différents paramètres")
print("Tester qlearning avec différentes valeurs d’epsilon. Expliquez les résultats.")
print("----------------------------")
print("En augmentant alpha, l'agent apprend plus rapidement mais peut être moins stable.")
print("En diminuant epsilon, l'agent explore moins et exploite plus, ce qui peut ralentir l'apprentissage.")
print("En augmentant gamma, l'agent valorise davantage les récompenses futures, ce qui peut améliorer la politique à long terme.")
print("----------------------------")
print("Question 5 : Joueur de morpion I")
print("Créer et implémenter morpion1.py. Votre agent Q-learning représente le joueur X. Le joueur O est un joueur aléatoire")
print("joue une action possible aléatoire avec une distribution uniforme) et ses actions sont implémentées via la fonction next_state.")
print("Une récompense de 1 est donnée si le joueur X réalise une action finale permettant de gagner le match, sinon toutes autres actions valent 0.")