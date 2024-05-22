from Algorithm import Algorithm, HEURISTICS
import argparse
import os
import logging
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Illustration of Q-learning algorithm")
    parser.add_argument(
        "--instance",
        type=str,
        help="Path to instance",
        required=False,
        default=Path("datasets/20_nodes.txt"),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Learning rate",
        required=False,
        default=0.1,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        help="Discount factor",
        required=False,
        default=0.9,
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        help="Exploration rate",
        required=False,
        default=0.1,
    )
    parser.add_argument(
        "--epsilon_min",
        type=float,
        help="Minimum exploration rate",
        required=False,
        default=0.01,
    )
    parser.add_argument(
        "--epsilon_decay",
        type=float,
        help="Exploration decay rate",
        required=False,
        default=0.995,
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="Number of episodes",
        required=False,
        default=10000,
    )
    parser.add_argument(
        "--log",
        dest="logLevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logger level",
    )
 
    args = parser.parse_args()

    logging.basicConfig(format="[%(levelname)s] : %(message)s")
    logger = logging.getLogger("Traveling Salesman Problem")
    logger.setLevel(args.logLevel)

    logger.debug(args)
    instance = args.instance
    if not os.path.isfile(instance):
        logger.warning('Instance "{}" not found'.format(instance))
        exit(1)

    algorithm = Algorithm(
        instance,
        logger=logger,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        episodes=args.episodes,
    )
    algorithm.q_learn()
    if len(algorithm.G.nodes) > 50:
        answer = input(
            f"Many nodes to draw ({len(algorithm.G.nodes)}), confirm drawing? [y/N]\n"
        )
        if answer != "y":
            return
    algorithm.show()


if __name__ == "__main__":
    main()
