# FlareSim — Installation Guide

This guide walks you through installing FlareSim step by step. It assumes you've already downloaded a release zip for your Nuke version from the GitHub releases page.

---

## What's in the release zip

When you unzip the release, you'll find:

```
FlareSim.dll          (Windows) or FlareSim.so (Linux)
menu.py
FlareSim_LensBrowser.py
lenses/
    lens_files/
        doublegauss.lens
        ...
```

All of these files need to end up somewhere that Nuke can find them.

---

## Find your .nuke folder

Every Nuke installation creates a `.nuke` folder in your home directory. This is where Nuke looks for plugins, scripts, and settings.

**Windows:**
```
C:\Users\<YourUsername>\.nuke\
```
You can get there quickly by typing `%USERPROFILE%\.nuke` into the File Explorer address bar.

**Linux:**
```
/home/<yourusername>/.nuke/
```
Note: the folder starts with a dot, so it's hidden by default. In your file manager, enable "Show Hidden Files", or in a terminal use `ls -a ~` to see it.

If the `.nuke` folder doesn't exist, open Nuke once and close it — Nuke creates it on first launch.

---

## Option A: Simple install (one Nuke version)

If you only use one version of Nuke, this is the easiest approach.

**1.** Create a `plugins` folder inside your `.nuke` folder:

```
Windows:  C:\Users\<YourUsername>\.nuke\plugins\
Linux:    /home/<yourusername>/.nuke/plugins/
```

**2.** Unzip the entire contents of the release into that `plugins` folder. You should end up with:

```
.nuke/
    plugins/
        FlareSim.dll  (or .so)
        menu.py
        FlareSim_LensBrowser.py
        lenses/
            lens_files/
                ...
```

**3.** Create or edit the file called `init.py` in your `.nuke` folder (not inside `plugins/` — directly in `.nuke/`). Add this line:

```python
nuke.pluginAddPath('./plugins')
```

If `init.py` doesn't exist yet, create a new text file called `init.py` and add that line. If it already exists, add the line at the end.

**4.** Restart Nuke. FlareSim should appear under the **Filter** menu in the node toolbar, and when you press Tab and type "FlareSim".

---

## Option B: Multi-version install (recommended)

If you have multiple Nuke versions installed (e.g. Nuke 15 and Nuke 16), each version needs its own copy of the plugin because the binary is compiled specifically for that version. This setup keeps them organised in separate folders and automatically loads the right one.

**1.** Create version-specific folders inside `plugins`:

```
.nuke/
    plugins/
        nuke15/
        nuke16/
```

**2.** Unzip each version's release into its matching folder:

```
.nuke/
    plugins/
        nuke15/
            FlareSim.dll  (or .so)
            menu.py
            FlareSim_LensBrowser.py
            lenses/
                lens_files/
                    ...
        nuke16/
            FlareSim.dll  (or .so)
            menu.py
            FlareSim_LensBrowser.py
            lenses/
                lens_files/
                    ...
```

**3.** Create or edit `init.py` in your `.nuke` folder with the following:

```python
import nuke
major = nuke.NUKE_VERSION_MAJOR
nuke.pluginAddPath(f'./plugins/nuke{major}')
```

This automatically detects which Nuke version is running and adds the correct folder. When you open Nuke 15, it adds `plugins/nuke15`. When you open Nuke 16, it adds `plugins/nuke16`. You never need to edit this line again, even when new versions come out — just add a new folder.

**4.** Restart Nuke.

---

## Troubleshooting

### "FlareSim doesn't appear in the menu"

This almost always means Nuke can't find the plugin files. Check the following:

- **Is `init.py` in the right place?** It must be directly in the `.nuke` folder, not inside `plugins/` or any subfolder.
- **Is the path in `init.py` correct?** Open Nuke's Script Editor (or the Python console) and type:
  ```python
  import nuke
  print(nuke.pluginPath())
  ```
  This prints every directory Nuke is searching. Your plugins folder should be in the list. If it isn't, the path in `init.py` is wrong.
- **Are the files in the right folder?** `FlareSim.dll` (or `.so`), `menu.py`, and `FlareSim_LensBrowser.py` must all be in the same folder, and that folder must be the one listed in your `init.py`.
- **Did you put the path in `menu.py` instead of `init.py`?** This is a common mistake. Nuke runs all `init.py` files first (which is where new paths need to be registered), and then scans for `menu.py` files across all known paths. If you add a path in `menu.py`, it's too late — Nuke has already finished looking for `menu.py` files. Always use `init.py` for `pluginAddPath`.

### "It only appears after I go to All Plugins → Update"

This is the same problem as above — the path is being added too late (in `menu.py` instead of `init.py`), or not at all. Move your `pluginAddPath` line to `init.py` and restart Nuke. The plugin should appear immediately without needing to manually update.

### "FlareSim appears but says no compatible GPU detected"

- Make sure you have an NVIDIA GPU (AMD and Intel GPUs are not supported).
- Update your NVIDIA driver to version 525 or newer.

**Windows:** Check your driver version by right-clicking on the desktop → NVIDIA Control Panel → Help → System Information. If you're on a laptop, make sure Nuke is running on the NVIDIA GPU and not the integrated Intel GPU: go to Settings → Display → Graphics, find Nuke in the list, and set it to "High Performance".

**Linux:** Check your driver version by running `nvidia-smi` in a terminal — it shows the installed driver version in the top-right corner. If the command is not found, the NVIDIA driver is not installed; install it via your distribution's package manager (e.g. `sudo apt install nvidia-driver-535` on Ubuntu) or from [nvidia.com](https://www.nvidia.com/en-us/drivers/). After installing or updating the driver, reboot before launching Nuke.

### "I already have an init.py with other plugins"

That's fine — just add the `pluginAddPath` line alongside your existing content. `init.py` is just a Python script and can contain as many lines as you need. For example:

```python
import nuke

# My other plugin
nuke.pluginAddPath('./plugins/some_other_plugin')

# FlareSim (auto-detects Nuke version)
major = nuke.NUKE_VERSION_MAJOR
nuke.pluginAddPath(f'./plugins/nuke{major}')
```

### "Where do I put the lens files?"

The lens files should stay inside the `lenses/lens_files/` folder that came in the zip. As long as that folder is inside your plugin directory, the Lens Browser panel will find them automatically. You can also store lens files anywhere else on your machine — just use the **Lens File** knob on the FlareSim node to browse to the `.lens` file you want.
