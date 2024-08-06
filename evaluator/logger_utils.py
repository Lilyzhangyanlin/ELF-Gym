import logging
def enable_log(level=logging.WARN):
    logging.basicConfig(
        style='{',
        format="[{levelname[0]}][{asctime},{filename}:{lineno}] {message}",
        datefmt='%Y-%m-%d %H:%M:%S',
        level=level,
    )
enable_log()

