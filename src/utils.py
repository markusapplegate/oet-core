"""Compatibility layer forwarding to :mod:`oet_core.utils`.""""""Compatibility layer forwarding to :mod:`oet_core.utils`."""



from oet_core.utils import *  # noqa: F401,F403from oet_core.utils import *  # noqa: F401,F403

from typing import List, Optional, Tuple, Union, Dict, Any
import logging
import sys


# Module-level configuration for verbose logging
_VERBOSE_LOGGING = False


def set_verbose_logging(enabled: bool) -> None:
	"""Enable or disable verbose logging for this module.
	
	Parameters
	----------
	enabled:
		If True, enables detailed logging for all operations.
		If False (default), only logs important events.
	"""
	global _VERBOSE_LOGGING
	_VERBOSE_LOGGING = enabled


class Matrix:
	"""
	Instance-backed matrix with simple helpers.

	The matrix stores its data in row-major order as a list of lists.
	Elements are restricted to int, float, or str.
	"""

	def __init__(self, rows: int, cols: int, fill: Optional[Union[int, float, str]] = None) -> None:
		if _VERBOSE_LOGGING:
			log(f"Matrix.__init__ called with rows={rows}, cols={cols}, fill={fill}", level="info")
		if not isinstance(rows, int) or not isinstance(cols, int):
			raise TypeError("rows and cols must be integers")
		if rows < 0 or cols < 0:
			raise ValueError("rows and cols must be non-negative")

		if fill is None:
			fill_value: Union[int, float, str] = 0
		elif isinstance(fill, (int, float, str)):
			fill_value = fill
		"""Compatibility layer forwarding to :mod:`oet_core.utils`."""

		from oet_core.utils import *  # noqa: F401,F403
		self.rows = rows

		self.cols = cols

		# store data as list of rows
