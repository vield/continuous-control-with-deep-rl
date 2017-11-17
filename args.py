import argparse


def parse_and_validate_args():
    parser = argparse.ArgumentParser()

    # Output
    parser.add_argument('--logging-level', default='DEBUG', type=lambda x: x.upper(),
                        choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
                        help="""Log level messages starting from this level will
be printed to stdout, unless --quieted"""
    )
    parser.add_argument('--no-render-training', dest='render_training', action='store_false',
                        help="Don't bother rendering the env while training, I want to save time")
    parser.add_argument('--render-training', dest='render_training', action='store_true',
                        help="Yes please, I want to look at my env while training")
    parser.add_argument('--quiet', dest='verbose', action='store_false',
                        help="Don't print output to stdout")
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help="Print logging output to stdout")
    parser.set_defaults(verbose=True, render_training=True)

    # Sizes and stuff
    parser.add_argument('--num-episodes', type=int, default=150,
                        help="Number of training episodes")
    parser.add_argument('--num-timesteps', type=int, default=200,
                        help="Maximum number of exploration/training steps within an episode")
    parser.add_argument('--batch-size', type=int, default=100,
                        help="Minibatch size to sample from replay buffer when training")
    parser.add_argument('--buffer-size', type=int, default=1000,
                        help="Replay buffer size")

    options = parser.parse_args()

    return options