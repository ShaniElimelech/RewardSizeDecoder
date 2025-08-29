import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


class SafeExtraFormatter(logging.Formatter):
    """
    Formatter that won't crash if an 'extra' field is missing.
    It fills missing fields with '-'.
    """
    _safe_keys = ("subject_id", "session", "model")

    def format(self, record):
        for k in self._safe_keys:
            if not hasattr(record, k):
                setattr(record, k, "-")
        return super().format(record)


def make_console_logger(name: str, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if verbose else logging.INFO)
        fmt = SafeExtraFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | "
                "subj=%(subject_id)s sess=%(session)s model=%(model)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    logger.propagate = False
    return logger


def attach_file_handler(logger: logging.Logger,
                        log_dir: Path,
                        filename: str = "decoder.log",
                        when: str = "D",
                        interval: int = 1,
                        backupCount: int = 7,
                        level: int = logging.DEBUG) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / filename

    # Don't attach duplicates
    for h in logger.handlers:
        if isinstance(h, TimedRotatingFileHandler) and getattr(h, "baseFilename", None) == str(logfile):
            return logfile

    fh = TimedRotatingFileHandler(
        filename=str(logfile),
        when=when, interval=interval, backupCount=backupCount, encoding="utf-8"
    )
    fh.setLevel(level)
    fh.setFormatter(SafeExtraFormatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | "
            "subj=%(subject_id)s sess=%(session)s model=%(model)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)
    return logfile
