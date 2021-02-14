"""
Python RunStats API
===================

Online statistics and regression.

"""

try:
    from ._core import (
        ExponentialCovariance,
        ExponentialStatistics,
        Regression,
        Statistics,
    )
except ImportError:  # pragma: no cover
    from .core import (
        ExponentialCovariance,
        ExponentialStatistics,
        Regression,
        Statistics,
    )

__all__ = [
    'Statistics',
    'Regression',
    'ExponentialStatistics',
    'ExponentialCovariance',
]
__title__ = 'runstats'
__version__ = '1.8.0'
__author__ = 'Grant Jenks'
__license__ = 'Apache 2.0'
__copyright__ = '2013-2021, Grant Jenks'
