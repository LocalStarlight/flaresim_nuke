#!/usr/bin/env python3
"""
convert_p2p.py — Convert photonstophotos.net Optical Bench data to .lens format.

Photons to Photos Optical Bench shows patent lens prescriptions with columns:
    Surface  Type  Radius  Thickness  nd  Vd  Semi-Diameter

Usage:
    1. Go to photonstophotos.net → Encyclopaedia of Encyclopaedias → Optical Bench
    2. Pick a lens (e.g. "Nikon AF-S Nikkor 50mm f/1.4G")
    3. Copy the table data from the browser and paste it into a text file
    4. Run:  python3 convert_p2p.py input.txt -o nikkor50f14.lens

Input format (tab or space separated, one of these styles):

  Style A — with "Surface" and "Type" columns:
    1   L   28.20   7.00   1.8042  46.5  25.0
    2   L  135.70   0.20   AIR           25.0
    3   L   21.49   8.40   1.4970  81.6  18.5
    4   L  -90.00   1.60   1.8340  37.2  18.5
    5   L  -90.00   0.20   AIR           16.0
    6   S    INF    5.40   AIR            8.5
    7   L  -19.50   1.40   1.8042  46.5  12.0
    ...

  Style B — just the 6 data columns (no surface number or type):
    28.20   7.00   1.8042  46.5  25.0
   135.70   0.20   1.000   0.0   25.0
    ...

  Style C — copy-paste from the rendered HTML table (columns may be labeled):
    Radius    Thickness  nd      Vd    Semi-Diameter
    28.20     7.00       1.8042  46.5  25.0
    ...

The script auto-detects the format.  Air gaps are identified by:
  - nd = 1.0 or nd = "AIR" or nd absent with Vd = 0
  - Type column = "A" (air)
Aperture stops are identified by:
  - Type = "S" (stop) or Radius = "INF"/"STOP"/"stop" with nd = AIR

Options:
    -o FILE         Output .lens file (default: stdout)
    -n NAME         Lens name (default: from filename)
    -f FOCAL        Focal length in mm (default: guess from filename)
    --coating N     Coating layers for glass surfaces (default: 1)
    --multi-coat    Use coating=2 for all glass surfaces (modern lenses)
    --uncoated      Use coating=0 for all surfaces (vintage)
"""

import argparse
import re
import sys


def parse_focal_from_name(name):
    """Try to extract focal length from a lens name like 'Nikkor 50mm f/1.4'."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*mm", name, re.IGNORECASE)
    return float(m.group(1)) if m else 0.0


def is_air(nd_str, vd_str=None):
    """Check if a material is air."""
    if nd_str.upper() in ("AIR", "", "-"):
        return True
    try:
        nd = float(nd_str)
        return nd < 1.001
    except ValueError:
        return True


def parse_table(lines):
    """Parse a variety of tab/space-separated lens table formats.

    Returns a list of dicts with keys:
        radius, thickness, nd, vd, semi_ap, is_stop
    """
    surfaces = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Skip header rows
        if any(
            h in line.lower()
            for h in ["radius", "surface", "thickness", "semi-diameter", "refractive"]
        ):
            continue

        # Normalize tabs/multiple spaces to single space
        tokens = line.split()
        if len(tokens) < 3:
            continue

        # Auto-detect format by trying to figure out which tokens are numbers
        # Strategy: walk tokens, identify radius/thickness/nd/vd/semi_ap

        # If first token is an integer (surface number), skip it
        idx = 0
        try:
            int(tokens[0])
            idx = 1  # skip surface number
        except ValueError:
            pass

        # If next token is a single letter (type code: L/A/S/G), record and skip
        surface_type = None
        if (
            idx < len(tokens)
            and len(tokens[idx]) == 1
            and tokens[idx].upper() in ("L", "A", "S", "G", "M")
        ):
            surface_type = tokens[idx].upper()
            idx += 1

        remaining = tokens[idx:]

        if len(remaining) < 2:
            continue

        # Parse radius
        radius_str = remaining[0]
        is_stop = False
        if radius_str.upper() in ("STOP", "S", "INF", "INFINITY", "FLAT"):
            radius = 0.0
            if radius_str.upper() in ("STOP", "S") or surface_type == "S":
                is_stop = True
        else:
            try:
                radius = float(radius_str)
            except ValueError:
                continue

        if surface_type == "S":
            is_stop = True

        # Parse thickness
        try:
            thickness = float(remaining[1])
        except ValueError:
            continue

        # Parse nd (refractive index)
        nd = 1.0
        vd = 0.0
        semi_ap = 0.0

        if len(remaining) >= 3:
            if is_air(remaining[2]):
                nd = 1.0
                # vd and semi_ap follow
                vi = 3
            else:
                try:
                    nd = float(remaining[2])
                except ValueError:
                    nd = 1.0
                vi = 3

            if len(remaining) > vi:
                # Next could be Vd or semi_ap
                try:
                    vd = float(remaining[vi])
                    vi += 1
                except (ValueError, IndexError):
                    pass

            if len(remaining) > vi:
                try:
                    semi_ap = float(remaining[vi])
                except (ValueError, IndexError):
                    pass

        # If nd ≈ 1.0, this is an air gap
        if nd < 1.001:
            nd = 1.0
            vd = 0.0

        # Mark as stop if type says so
        if surface_type == "A" and abs(radius) < 1e-6:
            # Air gap with zero radius could be stop or flat surface
            pass

        surfaces.append(
            {
                "radius": radius,
                "thickness": thickness,
                "nd": nd,
                "vd": vd,
                "semi_ap": semi_ap,
                "is_stop": is_stop,
            }
        )

    return surfaces


def assign_coating(surfaces, coating_layers):
    """Assign coating values: glass surfaces get coating, air/stop get 0."""
    for s in surfaces:
        if s["nd"] > 1.001 and not s["is_stop"]:
            s["coating"] = coating_layers
        else:
            s["coating"] = 0


def infer_stop(surfaces):
    """If no explicit stop was found, insert one at the most likely position
    (between the front and rear groups — roughly the middle)."""
    has_stop = any(s["is_stop"] for s in surfaces)
    if has_stop:
        return surfaces

    # Heuristic: the stop is usually at the narrowest semi-aperture
    if all(s["semi_ap"] > 0 for s in surfaces):
        min_idx = min(range(len(surfaces)), key=lambda i: surfaces[i]["semi_ap"])
        # If the narrowest surface is an air gap, mark it as stop
        if surfaces[min_idx]["nd"] < 1.001:
            surfaces[min_idx]["is_stop"] = True
        else:
            # Insert a stop before this surface with its semi_ap
            stop = {
                "radius": 0.0,
                "thickness": 0.0,
                "nd": 1.0,
                "vd": 0.0,
                "semi_ap": surfaces[min_idx]["semi_ap"],
                "is_stop": True,
                "coating": 0,
            }
            # Steal some thickness from the previous air gap if possible
            for i in range(min_idx - 1, -1, -1):
                if surfaces[i]["nd"] < 1.001:
                    # Split the air gap: put stop in the middle
                    half = surfaces[i]["thickness"] / 2.0
                    surfaces[i]["thickness"] = half
                    stop["thickness"] = half
                    break
            surfaces.insert(min_idx, stop)

    return surfaces


def write_lens(surfaces, name, focal_length, outfile):
    """Write surfaces in .lens format."""
    out = []
    out.append(f"# {name}")
    out.append(f"#")
    out.append(f"# Converted from photonstophotos.net Optical Bench data")
    out.append(
        f"# {len(surfaces)} surfaces, {sum(1 for s in surfaces if s['nd'] > 1.001)} glass elements"
    )
    out.append(f"")
    out.append(f"name: {name}")
    out.append(f"focal_length: {focal_length:.1f}")
    out.append(f"")
    out.append(f"surfaces:")
    out.append(f"# radius    thickness   ior     abbe    semi_ap  coating")

    for s in surfaces:
        if s["is_stop"]:
            radius_str = "stop"
        elif abs(s["radius"]) < 1e-6:
            radius_str = "0"
        else:
            radius_str = f"{s['radius']:.2f}"

        line = f"  {radius_str:<10s} {s['thickness']:<11.2f} {s['nd']:<7.3f} {s['vd']:<7.1f} {s['semi_ap']:<8.1f} {s['coating']}"
        out.append(line)

    text = "\n".join(out) + "\n"

    if outfile == "-":
        sys.stdout.write(text)
    else:
        with open(outfile, "w") as f:
            f.write(text)
        print(f"Wrote {outfile}: {len(surfaces)} surfaces", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Convert photonstophotos.net Optical Bench data to .lens format"
    )
    parser.add_argument("input", help="Input text file with lens table data")
    parser.add_argument(
        "-o", "--output", default="-", help="Output .lens file (default: stdout)"
    )
    parser.add_argument(
        "-n", "--name", default=None, help="Lens name (default: from filename)"
    )
    parser.add_argument(
        "-f", "--focal", type=float, default=0.0, help="Focal length in mm"
    )
    parser.add_argument(
        "--coating",
        type=int,
        default=1,
        help="Coating layers for glass surfaces (default: 1)",
    )
    parser.add_argument(
        "--multi-coat", action="store_true", help="Use coating=2 (broadband multi-coat)"
    )
    parser.add_argument(
        "--uncoated", action="store_true", help="Use coating=0 (vintage uncoated)"
    )

    args = parser.parse_args()

    # Read input
    with open(args.input) as f:
        lines = f.readlines()

    # Parse
    surfaces = parse_table(lines)
    if not surfaces:
        print("ERROR: no surfaces parsed from input file", file=sys.stderr)
        sys.exit(1)

    # Coating
    coating = args.coating
    if args.multi_coat:
        coating = 2
    if args.uncoated:
        coating = 0
    assign_coating(surfaces, coating)

    # Stop inference
    surfaces = infer_stop(surfaces)

    # Name and focal length
    name = args.name
    if not name:
        # Derive from input filename
        name = args.input.rsplit("/", 1)[-1].rsplit(".", 1)[0].replace("_", " ")

    focal = args.focal
    if focal <= 0:
        focal = parse_focal_from_name(name)
    if focal <= 0:
        focal = parse_focal_from_name(args.input)
    if focal <= 0:
        # Last resort: estimate from last thickness (back focal distance)
        focal = surfaces[-1]["thickness"] if surfaces else 50.0
        print(
            f"WARNING: could not determine focal length, using {focal:.1f}mm "
            f"(specify with -f)",
            file=sys.stderr,
        )

    # Validate
    n_glass = sum(1 for s in surfaces if s["nd"] > 1.001)
    n_stop = sum(1 for s in surfaces if s["is_stop"])
    missing_ap = sum(1 for s in surfaces if s["semi_ap"] <= 0)

    print(
        f"Parsed: {len(surfaces)} surfaces, {n_glass} glass, {n_stop} stop(s)",
        file=sys.stderr,
    )
    if missing_ap > 0:
        print(
            f"WARNING: {missing_ap} surface(s) have semi_ap=0 — "
            f"you may need to add semi-diameters manually",
            file=sys.stderr,
        )

    write_lens(surfaces, name, focal, args.output)


if __name__ == "__main__":
    main()
