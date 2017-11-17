import argparse


def parse_and_validate_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--logging-level', default='DEBUG', type=lambda x: x.upper(),
                        choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
                        help="""Log level messages starting from this level will
be printed to stdout, unless --quieted"""
    )

    parser.add_argument('--quiet', dest='verbose', action='store_false',
                        help="Don't print output to stdout")
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help="Print logging output to stdout")
    parser.set_defaults(verbose=True)

    options = parser.parse_args()

    return options