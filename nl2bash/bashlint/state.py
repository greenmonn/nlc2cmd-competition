from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..bashlint import flags, butils


def parserstate(): return butils.typedset(flags.parser)
