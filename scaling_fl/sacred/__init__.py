from sacred import SETTINGS

from .mongoobserver import observer_from_env

SETTINGS['CAPTURE_MODE'] = 'sys'
