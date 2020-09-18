#!/usr/bin/env python
# encoding: utf-8

from datetime import datetime
import warnings

TIMESTAMP = datetime.today().strftime('%Y%m%d-%H%M%S')


def warn_deprecated(arguments):
    for argument, name in arguments:
        warnings.warn(f"'{name}' has been deprecated. It will not have any effect "
                      f"(got '{argument}')", DeprecationWarning)