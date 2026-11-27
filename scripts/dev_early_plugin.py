"""Exercise an explicit early pytest plugin importing the application."""

import server
from runtime_profile import current

if current().mode != "test":
    raise RuntimeError("Unexpected pytest runtime")
