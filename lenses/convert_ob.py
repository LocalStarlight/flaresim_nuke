#!/usr/bin/env python3
"""
convert_ob.py — Convert photonstophotos.net Optical Bench .txt data to .lens format.

Reads the structured text format served by the Optical Bench at:
  https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/

Each lens prescription is a .txt file with sections:
  [descriptive data]   — title, patent info
  [constants]          — scaling, aspherical conventions
  [variable distances] — focal length, f-number, variable air gaps (d##, Bf)
  [lens data]          — surface table (tab-separated)
  [aspherical data]    — aspherical coefficients (informational)
  [group data]         — group assignments
  [figure]             — diagram references

The [lens data] section has tab-separated columns:
  surface#   radius   thickness   nd   semi_diameter   Vd

Air surfaces omit nd and Vd (empty tab fields).
The aperture stop has 'AS' in the radius column.
Variable thicknesses (d13, d19, Bf, etc.) reference [variable distances].

Usage:
    # From a local file:
    python3 convert_ob.py WO2021-039813_Example01P.txt -o nikkor18f28.lens

    # Directly from a URL:
    python3 convert_ob.py https://www.photonstophotos.net/GeneralTopics/Lenses/OpticalBench/Data/WO2021-039813_Example01P.txt -o nikkor18f28.lens

    # With multi-coat and custom name:
    python3 convert_ob.py US004717245_Example02P.txt --multi-coat -n "Canon EF 50mm f/1.0L" -o canon50f1.lens

Options:
    -o FILE         Output .lens file (default: stdout)
    -n NAME         Override lens name (default: from title in file)
    -f FOCAL        Override focal length in mm
    --coating N     Coating layers for glass surfaces (default: 1)
    --multi-coat    Use coating=2 for all glass surfaces
    --uncoated      Use coating=0 for all surfaces
"""

import argparse
import re
import sys
import urllib.request
import urllib.error


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def fetch_data(source):
    """Read data from a URL or local file path."""
    if source.startswith("http://") or source.startswith("https://"):
        req = urllib.request.Request(
            source, headers={"User-Agent": "convert_ob.py/1.0 (lens converter)"}
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to fetch URL: {e}") from e
    else:
        with open(source) as f:
            return f.read()


# ---------------------------------------------------------------------------
# Section parser
# ---------------------------------------------------------------------------


def parse_sections(text):
    """Split the text into named sections based on [section name] headers.

    Returns {section_name: [lines...]}.
    """
    sections = {}
    current = None
    lines = []

    for raw_line in text.splitlines():
        m = re.match(r"^\[(.+)\]\s*$", raw_line)
        if m:
            if current is not None:
                sections[current] = lines
            current = m.group(1).strip().lower()
            lines = []
        elif current is not None:
            lines.append(raw_line)

    if current is not None:
        sections[current] = lines

    return sections


# ---------------------------------------------------------------------------
# [descriptive data]
# ---------------------------------------------------------------------------


def parse_descriptive(lines):
    """Parse [descriptive data] → dict of key/value pairs."""
    info = {}
    for line in lines:
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[0].strip():
            info[parts[0].strip()] = parts[1].strip()
    return info


# ---------------------------------------------------------------------------
# [variable distances]
# ---------------------------------------------------------------------------


def parse_variable_distances(lines):
    """Parse [variable distances] section.

    Returns:
        variables : dict  — mapping variable names (d19, Bf, …) to float values
                            at the *infinity focus* position (first numeric column).
        meta      : dict  — focal_length, f_number, image_height extracted here.
    """
    variables = {}
    meta = {}

    for line in lines:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue

        name = parts[0].strip()
        values = [p.strip() for p in parts[1:]]

        # Extract metadata from well-known entries
        if name == "Focal Length" and values:
            try:
                meta["focal_length"] = float(values[0])
            except ValueError:
                pass
        elif name == "F-Number" and values:
            try:
                meta["f_number"] = float(values[0])
            except ValueError:
                pass
        elif name == "Image Height" and values:
            try:
                meta["image_height"] = float(values[0])
            except ValueError:
                pass
        elif name == "Aperture Diameter" and values:
            try:
                meta["aperture_diameter"] = float(values[0])
            except ValueError:
                pass

        # Store the infinity-focus value for variable distance resolution
        if values:
            val = values[0]
            if val not in ("undefined", "Infinity", ""):
                try:
                    variables[name] = float(val)
                except ValueError:
                    pass

    return variables, meta


# ---------------------------------------------------------------------------
# [lens data]
# ---------------------------------------------------------------------------


def parse_lens_data(lines, variables):
    """Parse the [lens data] section.

    Tab-separated columns:
        [0] surface#
        [1] radius         — float, 'AS', 'Infinity', '-Infinity'
        [2] thickness       — float or variable name (d13, Bf, …)
        [3] nd              — float or empty (air)
        [4] semi_diameter   — float
        [5] Vd (Abbe)       — float or empty (air)

    Returns list of surface dicts.
    """
    surfaces = []

    for line in lines:
        if not line.strip():
            continue

        fields = line.split("\t")
        if len(fields) < 4:
            continue

        # --- surface number ---
        # Some files use float sub-numbering (3.4, 3.5, 3.6) or append 'AS'
        # to the surface number to mark the aperture stop (e.g. "3.5AS").
        surf_num_raw = fields[0].strip()
        is_stop_from_num = False
        if surf_num_raw.upper().endswith("AS"):
            is_stop_from_num = True
            surf_num_raw = surf_num_raw[:-2]
        try:
            surf_num = float(surf_num_raw)
        except ValueError:
            continue

        # --- radius ---
        radius_str = fields[1].strip()
        is_stop = is_stop_from_num
        if radius_str.upper() in ("AS", "STOP", "APERTURE"):
            radius = 0.0
            is_stop = True
        elif radius_str.upper() in ("INF", "INFINITY"):
            radius = 0.0  # flat surface
        elif radius_str.upper() in ("-INF", "-INFINITY"):
            radius = 0.0  # flat surface (sign doesn't matter for flat)
        elif radius_str.upper() == "FS":
            radius = 0.0  # flat surface (field stop / flat element, not aperture stop)
        elif radius_str.upper() == "CG":
            radius = 0.0  # cover glass — flat surface; nd/vd come from the same row
        else:
            try:
                radius = float(radius_str)
            except ValueError:
                print(
                    f"WARNING: cannot parse radius '{radius_str}' on surface "
                    f"{surf_num}, skipping",
                    file=sys.stderr,
                )
                continue

        # --- thickness ---
        thickness_str = fields[2].strip()
        try:
            thickness = float(thickness_str)
        except ValueError:
            # Variable reference — look up in [variable distances]
            if thickness_str in variables:
                thickness = variables[thickness_str]
            else:
                # Try common variants: "Bf(p)" → look for "Bf"
                base = re.sub(r"\(.*\)", "", thickness_str)
                if base in variables:
                    thickness = variables[base]
                else:
                    print(
                        f"WARNING: unresolved variable '{thickness_str}' on "
                        f"surface {surf_num}, using 0.0",
                        file=sys.stderr,
                    )
                    thickness = 0.0

        # --- nd (refractive index) ---
        nd = 1.0
        if len(fields) > 3:
            nd_str = fields[3].strip()
            if nd_str:
                try:
                    nd = float(nd_str)
                except ValueError:
                    nd = 1.0

        # --- semi_diameter ---
        semi_ap = 0.0
        if len(fields) > 4:
            sa_str = fields[4].strip()
            if sa_str:
                try:
                    semi_ap = float(sa_str)
                except ValueError:
                    semi_ap = 0.0

        # --- Vd (Abbe number) ---
        vd = 0.0
        if len(fields) > 5:
            vd_str = fields[5].strip()
            if vd_str:
                try:
                    vd = float(vd_str)
                except ValueError:
                    vd = 0.0

        # Air normalisation
        if nd < 1.001:
            nd = 1.0
            vd = 0.0

        surfaces.append(
            {
                "num": surf_num,
                "radius": radius,
                "thickness": thickness,
                "nd": nd,
                "vd": vd,
                "semi_ap": semi_ap,
                "is_stop": is_stop,
            }
        )

    return surfaces


# ---------------------------------------------------------------------------
# Coating assignment
# ---------------------------------------------------------------------------


def assign_coating(surfaces, coating_layers):
    """Glass surfaces get the requested coating; air / stop surfaces get 0."""
    for s in surfaces:
        if s["nd"] > 1.001 and not s["is_stop"]:
            s["coating"] = coating_layers
        else:
            s["coating"] = 0


# ---------------------------------------------------------------------------
# .lens writer
# ---------------------------------------------------------------------------


def write_lens(surfaces, name, focal_length, f_number, patent, outfile):
    """Emit the .lens file."""
    n_glass = sum(1 for s in surfaces if s["nd"] > 1.001)

    out = []
    out.append(f"# {name}")
    out.append(f"#")
    if patent:
        out.append(f"# Patent: {patent}")
    out.append(f"# Converted from photonstophotos.net Optical Bench data")
    if f_number > 0:
        out.append(
            f"# {len(surfaces)} surfaces, {n_glass} glass elements, "
            f"f/{f_number:.1f}"
        )
    else:
        out.append(f"# {len(surfaces)} surfaces, {n_glass} glass elements")
    out.append(f"")
    out.append(f"name: {name}")
    out.append(f"focal_length: {focal_length:.2f}")
    out.append(f"")
    out.append(f"surfaces:")
    out.append(f"# radius       thickness    ior      abbe     semi_ap   coating")

    for s in surfaces:
        if s["is_stop"]:
            r_str = "stop"
        elif abs(s["radius"]) < 1e-9:
            r_str = "0"
        else:
            r_str = f"{s['radius']:.4f}"

        line = (
            f"  {r_str:<14s} {s['thickness']:<12.4f} "
            f"{s['nd']:<8.5f} {s['vd']:<8.2f} "
            f"{s['semi_ap']:<9.3f} {s['coating']}"
        )
        out.append(line)

    text = "\n".join(out) + "\n"

    if outfile == "-":
        sys.stdout.write(text)
    else:
        with open(outfile, "w") as f:
            f.write(text)
        print(
            f"Wrote {outfile}: {len(surfaces)} surfaces, "
            f"focal={focal_length:.2f}mm",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Name / patent helpers
# ---------------------------------------------------------------------------


def extract_patent(source):
    """Try to extract a patent number from the filename or URL."""
    m = re.search(r"([A-Z]{2}\d[\w-]+?)_", source)
    if m:
        return m.group(1)
    m = re.search(r"([A-Z]{2}\d[\w-]+)", source)
    return m.group(1) if m else ""


def extract_focal_from_text(text):
    """Try to extract a focal length from a title / name string."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*mm", text, re.IGNORECASE)
    return float(m.group(1)) if m else 0.0


# Regex matching patent-number prefixes in parenthesised title content.
# Covers US, JP, WO, DE, EP, GB, FR, CH, KR, CN, AU followed by digits.
_PATENT_PARENS_RE = re.compile(
    r"^(US|JP|WO|DE|EP|GB|FR|CH|KR|CN|AU)\s*[\d,\s.-]+",
    re.IGNORECASE,
)


def extract_name_from_title(title):
    """Extract the commercial lens name from an Optical Bench title string.

    The Optical Bench uses two title patterns:

      Pattern A — commercial name before parens, patent inside:
        "Nikon AF Micro-Nikkor 60mm f/2.8D (US 5,751,485 Example 1)"
        → "Nikon AF Micro-Nikkor 60mm f/2.8D"

      Pattern B — patent / document before parens, commercial name inside:
        "DE 1 180 159 B Example 1 (Zeiss Rollei-Mutar 0.7)"
        → "Zeiss Rollei-Mutar 0.7"

    Also handles:
      - Leading tilde (~) — variant / unverified marker, stripped silently
      - "3D left" / "3D right" in parens — stereo variant tag, not a brand name
      - No parens at all — return the whole title (e.g. "Standard Lens 1")
    """
    title = title.strip()
    # Strip leading tilde (used for unverified / variant entries)
    if title.startswith("~"):
        title = title[1:].strip()

    m = re.search(r"\((.+?)\)\s*$", title)
    if m:
        parens = m.group(1).strip()
        before = title[: m.start()].strip()

        # Does the paren content look like a patent ref, example tag, or 3D variant?
        is_non_name = (
            bool(_PATENT_PARENS_RE.match(parens))
            or bool(re.match(r"^Example\s+\d", parens, re.IGNORECASE))
            or bool(re.match(r"^3D\s+(left|right)", parens, re.IGNORECASE))
        )

        if is_non_name:
            # Parens contain metadata — the real name is the text before them
            return before if before else title
        else:
            # Parens contain the commercial brand name (Pattern B)
            return parens

    return title


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Convert photonstophotos.net Optical Bench .txt to .lens"
    )
    parser.add_argument("input", help="Input .txt file path or Optical Bench URL")
    parser.add_argument(
        "-o", "--output", default="-", help="Output .lens file (default: stdout)"
    )
    parser.add_argument(
        "-n",
        "--name",
        default=None,
        help="Override lens name (default: title from file)",
    )
    parser.add_argument(
        "-f", "--focal", type=float, default=0.0, help="Override focal length in mm"
    )
    parser.add_argument(
        "--coating",
        type=int,
        default=1,
        help="Coating layers for glass surfaces (default: 1)",
    )
    parser.add_argument(
        "--multi-coat",
        action="store_true",
        help="Use coating=2 for all glass surfaces (broadband multi-coat)",
    )
    parser.add_argument(
        "--uncoated",
        action="store_true",
        help="Use coating=0 for all surfaces (vintage uncoated)",
    )

    args = parser.parse_args()

    # ---- Fetch / read ----
    try:
        text = fetch_data(args.input)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # ---- Parse sections ----
    sections = parse_sections(text)

    if "lens data" not in sections:
        print("ERROR: no [lens data] section found in input", file=sys.stderr)
        sys.exit(1)

    desc = parse_descriptive(sections.get("descriptive data", []))
    title = desc.get("title", "")
    var_dist, meta = parse_variable_distances(sections.get("variable distances", []))
    if var_dist.get("Bf", 1.0) == 0.0:
        print(
            "WARNING: Bf = 0 in [variable distances] — back focal distance is zero. "
            "A cover-glass surface may account for the remaining air gap.",
            file=sys.stderr,
        )

    # ---- Parse lens surfaces ----
    surfaces = parse_lens_data(sections["lens data"], var_dist)
    if not surfaces:
        print("ERROR: no surfaces parsed from [lens data]", file=sys.stderr)
        sys.exit(1)

    # ---- Determine name ----
    name = args.name
    if not name:
        if title:
            name = extract_name_from_title(title)
        else:
            name = args.input.rsplit("/", 1)[-1].rsplit(".", 1)[0]

    # ---- Determine focal length ----
    focal = args.focal
    if focal <= 0:
        focal = meta.get("focal_length", 0.0)
    if focal <= 0:
        focal = extract_focal_from_text(title)
    if focal <= 0:
        focal = extract_focal_from_text(name)
    if focal <= 0:
        focal = 50.0
        print(
            "WARNING: could not determine focal length, using 50mm "
            "(specify with -f)",
            file=sys.stderr,
        )

    f_number = meta.get("f_number", 0.0)

    # ---- Coating ----
    coating = args.coating
    if args.multi_coat:
        coating = 2
    if args.uncoated:
        coating = 0
    assign_coating(surfaces, coating)

    # ---- Patent ----
    patent = extract_patent(args.input)

    # ---- Summary ----
    n_glass = sum(1 for s in surfaces if s["nd"] > 1.001)
    n_stop = sum(1 for s in surfaces if s["is_stop"])
    n_cemented = 0
    for i in range(len(surfaces) - 1):
        if surfaces[i]["nd"] > 1.001 and surfaces[i + 1]["nd"] > 1.001:
            n_cemented += 1

    print(f"Title : {title}", file=sys.stderr)
    print(f"Name  : {name}", file=sys.stderr)
    fstr = f"f/{f_number:.1f}" if f_number > 0 else "?"
    print(f"Focal : {focal:.2f}mm  {fstr}", file=sys.stderr)
    print(
        f"Surfs : {len(surfaces)} total, {n_glass} glass, "
        f"{n_stop} stop(s), {n_cemented} cemented interface(s)",
        file=sys.stderr,
    )
    if patent:
        print(f"Patent: {patent}", file=sys.stderr)

    # ---- Write ----
    write_lens(surfaces, name, focal, f_number, patent, args.output)


if __name__ == "__main__":
    main()
