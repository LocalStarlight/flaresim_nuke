# FlareSim — Nuke node menu registration
#
# Copy this file (and FlareSim_LensBrowser.py) to the same directory as
# FlareSim.dll on your NUKE_PATH.
# If you already have a menu.py in that directory, add these lines to it
# instead of replacing the file.
#
# Nuke loads every menu.py it finds on NUKE_PATH at startup.

import nuke

nuke.menu('Nodes').addCommand(
    'Filter/FlareSim',
    'nuke.createNode("FlareSim")',
)

try:
    import FlareSim_LensBrowser
    FlareSim_LensBrowser.register()
except Exception as e:
    nuke.warning(f'FlareSim: could not load lens browser: {e}')
