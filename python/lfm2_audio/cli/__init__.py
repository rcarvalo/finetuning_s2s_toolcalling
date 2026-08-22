"""Points d'entrée en ligne de commande (cf. ``[project.scripts]``).

Chaque module n'y porte QUE son argparse et sa boucle : toute la logique vit
dans les sous-paquets métier, de sorte qu'elle reste testable sans CLI.
"""
