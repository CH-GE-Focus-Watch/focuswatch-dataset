# Importing each adapter module registers it with base.register.
from . import airpods, ml4scs, moleskine, sensorlogger, zhaw_watchapp

__all__ = ["airpods", "ml4scs", "moleskine", "sensorlogger", "zhaw_watchapp"]
