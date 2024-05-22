# Problème du Voyageur de Commerce (TSP)

Vous trouverez ci-dessous les instructions et détails pour utiliser le programme de recherche d'une solution au Problème du Voyageur de Commerce (TSP).
Le but du programme est de trouver le chemin le plus court pour visiter un ensemble de villes une seule fois chacune et revenir à la ville de départ

On présente ici une technique classique de résolution adaptée au problème du TSP, qui est NP-difficile. Nous utilisons l'algorithme de Q-Learning, une méthode d'apprentissage par renforcement, pour chercher une solution au TSP.
Contrairement aux méthodes exactes qui peuvent devenir rapidement impraticables pour un grand nombre de villes, le Q-Learning offre une approche heuristique qui peut fournir des solutions de qualité dans des délais raisonnables.
Dans notre approche, nous utilisons une matrice Q pour représenter les valeurs d'utilité de chaque action dans chaque état. L'agent (le voyageur) apprend à choisir les actions (les villes à visiter) en fonction de ces valeurs d'utilité, en explorant l'espace des solutions possibles.

## Installation

Pour installer l'application, commencez par copier le dépot du git,
soit en recupérant l'archive zip depuis github, soit à l'aide de l'outil git:
```
git clone https://github.com/Jboistel/INFOH410-Project.git
```

Puis, accedez au dossier:

```bash
cd INFOH410-Project
```

Après avoir installé python et poetry, rendez vous dans ce dossier et installez les
dépendances du projet:

```bash
poetry install
```

## Utilisation

Vous pouvez ensuite lancer l'application, dans l'environnement virtuel
nouvellement crée, en utilsant la commande:

```bash
poetry run python main.py
```

Plusieurs options sont disponibles lors du lancement de la commande.
Il est par exemple possible de changer les paramètres utilisés pour l'apprentissage avec les options
`--alpha`, `--gamma`, `--epsilon`, `--epsilon_decay`, `--epsilon_min`, `--episodes`.
Il est aussi possible de choisir un fichier d'instance, qui permet de changer 
le graphe à parcourir, avec l'option `--instance` et d'y ajouter le nom du
fichier d'instance à ouvrir.

Par exemple:
```bash
poetry run python main.py --instance datasets/13_nodes.txt --episodes 2000
```
permet de lancer l'instance `13_nodes.txt` avec 2000 episodes.

NB: plusieurs instances sont disponibles dans le dossier `datasets`.
