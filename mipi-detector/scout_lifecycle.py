"""SCOUT owns one model group: signal every owner before waiting for cleanup."""


class RestartModels(Exception):
    """Restart the application only after the complete model group is closed."""


def shutdown_all(resources):
    """(name, stop, close) callbacks; one failure must not skip other owners."""
    errors = []
    for operation, index in (('stop', 1), ('close', 2)):
        for resource in resources:
            callback = resource[index]
            if callback is None:
                continue
            try:
                callback()
            except Exception as exc:
                errors.append(f'{resource[0]} {operation}: {exc}')
    return errors
