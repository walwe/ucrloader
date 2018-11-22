import logging


def setup_logging(log_level):
    ext_logger = logging.getLogger("py.warnings")
    logging.captureWarnings(True)
    logging.basicConfig(format="[%(asctime)s] %(levelname)s %(filename)s: %(message)s", level=log_level)
    if log_level <= logging.DEBUG:
        ext_logger.setLevel(logging.WARNING)


def main():
    from ucrloader import UCRLoader
    from sys import argv

    try:
        file_name = argv[1]
    except IndexError:
        logging.error("Please provide UCR directory path")
        return

    ucr = UCRLoader(file_name)
    for name in ucr.names:
        data = ucr.load(name)
        print(name, "Labels: ", sorted(set(data.train_labels)))


if __name__ == "__main__":
    setup_logging(logging.DEBUG)
    main()
