import argparse


def parse_and_validate_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Output
    output_opts = parser.add_argument_group("Output")
    output_opts.add_argument('--logging-level', default='DEBUG', type=lambda x: x.upper(),
                        choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
                        help="""Log level messages starting from this level will
be printed to stdout, unless --quieted"""
    )
    # FIXME The defaults are printed confusingly when using ArgumentDefaultsHelpFormatter
    output_opts.add_argument('--no-render-training', dest='render_training', action='store_false',
                        help="Don't bother rendering the env while training, I want to save time")
    output_opts.add_argument('--render-training', dest='render_training', action='store_true',
                        help="Yes please, I want to look at my env while training")
    output_opts.add_argument('--quiet', dest='verbose', action='store_false',
                        help="Don't print output to stdout")
    output_opts.add_argument('--verbose', dest='verbose', action='store_true',
                        help="Print logging output to stdout")
    output_opts.set_defaults(verbose=True, render_training=True)

    # Sizes and stuff
    training_opts = parser.add_argument_group("Training settings")
    training_opts.add_argument('--num-episodes', type=int, default=150,
                        help="Number of training episodes")
    training_opts.add_argument('--num-timesteps', type=int, default=200,
                        help="Maximum number of exploration/training steps within an episode")
    training_opts.add_argument('--batch-size', type=int, default=64,
                        help="Minibatch size to sample from replay buffer when training")
    training_opts.add_argument('--buffer-size', type=int, default=1000000,
                        help="Replay buffer size")

    # Training details
    hyperparam_opts = parser.add_argument_group("Hyperparameters")
    hyperparam_opts.add_argument('--gamma', type=float, default=0.99,
                                 help="Discount factor for future rewards")
    hyperparam_opts.add_argument('--noise-sigma', type=float, default=0.2,
                                 help="Parameter for noise generation process")

    options = parser.parse_args()

    return options