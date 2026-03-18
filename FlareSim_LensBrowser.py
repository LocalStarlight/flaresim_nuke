"""
FlareSim_LensBrowser.py — Dockable lens file browser panel for FlareSim.

Place this file in the same directory as FlareSim.dll on your NUKE_PATH.
Open the panel from the Pane menu: "FlareSim Lens Browser".
"""

import os
import re
import nuke
import nukescripts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_lens_meta(path):
    """Read name, focal_length and f-number from the header of a .lens file.

    Stops reading at the first 'surfaces:' line for speed.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    focal = 0.0
    f_num = 0.0
    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                s = line.strip()
                if s.startswith('name:'):
                    v = s[5:].strip()
                    if v:
                        name = v
                elif s.startswith('focal_length:'):
                    try:
                        focal = float(s[13:].strip())
                    except ValueError:
                        pass
                elif s.startswith('#') and f_num == 0.0:
                    # Pick up "f/2.8" written in comment lines by the converter
                    m = re.search(r'\bf/(\d+(?:\.\d+)?)', s)
                    if m:
                        try:
                            f_num = float(m.group(1))
                        except ValueError:
                            pass
                elif s.startswith('surfaces:'):
                    break
    except Exception:
        pass
    return name, focal, f_num


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class FlareLensBrowser(nukescripts.PythonPanel):

    PANEL_ID = 'uk.co.flaresim.lensbrowser'

    def __init__(self):
        super().__init__('FlareSim Lens Browser', self.PANEL_ID)

        # --- Lens folder row ---
        self._dir_knob = nuke.String_Knob('lens_dir', 'Lens Folder')
        self._dir_knob.setTooltip(
            'Directory containing .lens files. '
            'Click Browse to pick any .lens file and the folder will be set automatically.'
        )
        self._browse_knob = nuke.Script_Knob('browse_dir', 'Browse...')
        self._browse_knob.clearFlag(nuke.STARTLINE)
        self._browse_knob.setTooltip('Open a file browser — the folder of the picked file is used.')

        # --- Filter row ---
        self._filter_knob = nuke.String_Knob('filter', 'Filter')
        self._filter_knob.setTooltip('Case-insensitive substring filter. Press Refresh to apply.')
        self._refresh_knob = nuke.Script_Knob('refresh_list', 'Refresh')
        self._refresh_knob.clearFlag(nuke.STARTLINE)

        # --- List ---
        self._list_knob = nuke.Enumeration_Knob('lens_list', 'Lens', ['(choose a folder above)'])
        self._list_knob.setTooltip('Select a lens, then click Load.')

        # --- Load ---
        self._load_knob = nuke.Script_Knob('load_lens', 'Load onto selected FlareSim')
        self._load_knob.setTooltip(
            'Sets the Lens File knob on the selected FlareSim node(s). '
            'If no FlareSim node is selected but exactly one exists in the script, '
            'it is used automatically.'
        )

        for k in (
            self._dir_knob,
            self._browse_knob,
            self._filter_knob,
            self._refresh_knob,
            self._list_knob,
            self._load_knob,
        ):
            self.addKnob(k)

        # label → absolute path, kept in sync with the Enumeration_Knob values
        self._path_map = {}

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    def knobChanged(self, knob):
        if knob is self._browse_knob:
            path = nuke.getFilename('Select any .lens file', '*.lens')
            if path:
                folder = os.path.dirname(os.path.abspath(path))
                self._dir_knob.setValue(folder)
                self._refresh_list()

        elif knob is self._dir_knob:
            self._refresh_list()

        elif knob is self._refresh_knob:
            self._refresh_list()

        elif knob is self._load_knob:
            self._load_selected()

    # ------------------------------------------------------------------
    # List refresh
    # ------------------------------------------------------------------

    def _refresh_list(self):
        folder = self._dir_knob.value().strip()
        if not folder:
            return

        # Accept a full file path — just take the directory
        if os.path.isfile(folder):
            folder = os.path.dirname(folder)

        if not os.path.isdir(folder):
            nuke.message(f'FlareSim Lens Browser: folder not found:\n{folder}')
            return

        filt = self._filter_knob.value().strip().lower()

        entries = []
        try:
            filenames = sorted(os.listdir(folder), key=str.lower)
        except OSError as e:
            nuke.message(f'FlareSim Lens Browser: cannot read folder:\n{e}')
            return

        for fname in filenames:
            if not fname.lower().endswith('.lens'):
                continue
            fpath = os.path.join(folder, fname)
            name, focal, f_num = _read_lens_meta(fpath)

            if focal > 0 and f_num > 0:
                label = f'{name}  [{focal:.0f}mm  f/{f_num:.1f}]'
            elif focal > 0:
                label = f'{name}  [{focal:.0f}mm]'
            else:
                label = name

            if not filt or filt in label.lower():
                entries.append((label, fpath))

        self._path_map = {label: fpath for label, fpath in entries}
        labels = [label for label, _ in entries]

        if labels:
            self._list_knob.setValues(labels)
        else:
            self._list_knob.setValues(['(no matches)'])
            self._path_map = {}

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _load_selected(self):
        label = self._list_knob.value()
        fpath = self._path_map.get(label)
        if not fpath:
            nuke.message('No lens selected, or list needs refreshing.')
            return

        nodes = nuke.selectedNodes('FlareSim')
        if not nodes:
            all_fs = nuke.allNodes('FlareSim')
            if len(all_fs) == 1:
                nodes = all_fs
            elif len(all_fs) > 1:
                nuke.message(
                    'Multiple FlareSim nodes exist but none are selected.\n'
                    'Select the node(s) you want to update and try again.'
                )
                return
            else:
                nuke.message('No FlareSim node found in the script.')
                return

        fpath_nuke = fpath.replace('\\', '/')
        for n in nodes:
            n['lens_file'].setValue(fpath_nuke)


# ---------------------------------------------------------------------------
# Registration — called once from menu.py at startup
# ---------------------------------------------------------------------------

def _show_browser():
    """Create (or raise) the Lens Browser panel in the current pane."""
    panel = FlareLensBrowser()
    return panel.addToPane()


def register():
    """Register the panel with Nuke and add it to the Pane menu."""
    nukescripts.registerPanel(FlareLensBrowser.PANEL_ID, _show_browser)
    nuke.menu('Pane').addCommand('FlareSim Lens Browser', _show_browser)
