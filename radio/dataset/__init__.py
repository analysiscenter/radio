import sys
import importlib

sys.modules[__package__] = importlib.import_module('.dataset', __package__)
