import args
import ddpg
import logs


if __name__ == "__main__":
    options = args.parse_and_validate_args()

    logs.initialize_logging(
        stdout_logging_level=options.logging_level,
        log_to_stdout=options.verbose
    )

    ddpg.run_ddpg(options)