#!/usr/bin/env python3
"""Convert CSST FITS GT to DOTA format for mmrotate.

FITS convention (1-indexed, bottom-left origin):
    XWIN_IMAGE, YWIN_IMAGE - center coordinates (1-indexed)
    AWIN_IMAGE, BWIN_IMAGE - semi-major and semi-minor axes
    THETAWIN_IMAGE - angle in degrees

Deep Learning convention (0-indexed, top-left origin):
    x_ctr, y_ctr - center coordinates (0-indexed)
    w, h - full width and height (w = 2*a, h = 2*b)
    angle - angle in radians
"""

import os
import glob
import math
import zipfile
import xml.etree.ElementTree as ET

# Image dimensions
IMG_HEIGHT = 9232
IMG_WIDTH = 9216

# Class name for single-class detection
CLASS_NAME = "star"


def convert_fits_to_dl(x_fits, y_fits, a_fits, b_fits, theta_fits):
    """Convert FITS 1-indexed coordinates to deep learning 0-indexed coordinates.

    Args:
        x_fits: X center in FITS 1-indexed coordinates
        y_fits: Y center in FITS 1-indexed coordinates (bottom-left origin)
        a_fits: Semi-major axis in pixels
        b_fits: Semi-minor axis in pixels
        theta_fits: Angle in degrees

    Returns:
        tuple: (x_dl, y_dl, w, h, angle_rad)
    """
    # Convert from 1-indexed to 0-indexed
    x_dl = x_fits - 1
    # Flip Y axis: FITS bottom-left -> image top-left
    y_dl = IMG_HEIGHT - y_fits - 1

    # Semi-axes to full width/height
    w = 2 * a_fits
    h = 2 * b_fits

    # Convert angle from degrees to radians
    angle_rad = theta_fits * math.pi / 180.0

    return x_dl, y_dl, w, h, angle_rad


def process_csv(input_path, output_path):
    """Process a single CSV file.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output DOTA txt file
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()

    # Skip header
    header = lines[0].strip()
    if not header.startswith('XWIN_IMAGE'):
        print(f"Warning: Unexpected header in {input_path}: {header}")
        return

    converted_lines = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split(',')
        if len(parts) != 5:
            print(f"Warning: Skipping malformed line in {input_path}: {line[:50]}...")
            continue

        x_fits = float(parts[0])
        y_fits = float(parts[1])
        a_fits = float(parts[2])
        b_fits = float(parts[3])
        theta_fits = float(parts[4])

        x_dl, y_dl, w, h, angle_rad = convert_fits_to_dl(
            x_fits, y_fits, a_fits, b_fits, theta_fits
        )

        # Format: x_ctr y_ctr w h angle class_name
        out_line = f"{x_dl:.6f} {y_dl:.6f} {w:.6f} {h:.6f} {angle_rad:.6f} {CLASS_NAME}\n"
        converted_lines.append(out_line)

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.writelines(converted_lines)

    print(f"Converted {input_path} -> {output_path} ({len(converted_lines)} objects)")


def process_xlsx(input_path, output_path):
    """Process an Excel file (misnamed as .csv) with CSST GT data.

    Args:
        input_path: Path to input Excel file
        output_path: Path to output DOTA txt file
    """
    ns = {'ns': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

    with zipfile.ZipFile(input_path) as z:
        # Read shared strings
        with z.open('xl/sharedStrings.xml') as f:
            ss_root = ET.parse(f).getroot()
        strings = [si.find('.//ns:t', ns).text or '' for si in ss_root.findall('ns:si', ns)]

        # Read sheet1
        with z.open('xl/worksheets/sheet1.xml') as f:
            sheet_root = ET.parse(f).getroot()

    rows = sheet_root.findall('.//ns:row', ns)
    if len(rows) < 2:
        print(f"Warning: No data in {input_path}")
        return

    # Parse header to find column indices
    header_row = rows[0].findall('ns:c', ns)
    col_idx = {}
    for idx, cell in enumerate(header_row):
        t = cell.get('t', '')
        v = cell.find('ns:v', ns)
        if v is not None and v.text is not None and t == 's':
            col_name = strings[int(v.text)]
            col_idx[col_name] = idx

    # Expected columns: XWIN_IMAGE, YWIN_IMAGE, A_IMAGE or AWIN_IMAGE, B_IMAGE or BWIN_IMAGE, THETA_IMAGE or THETAWIN_IMAGE
    optional_a = ['AWIN_IMAGE', 'A_IMAGE']
    optional_b = ['BWIN_IMAGE', 'B_IMAGE']
    optional_theta = ['THETAWIN_IMAGE', 'THETA_IMAGE']

    a_col = next((c for c in optional_a if c in col_idx), None)
    b_col = next((c for c in optional_b if c in col_idx), None)
    theta_col = next((c for c in optional_theta if c in col_idx), None)

    if a_col is None or b_col is None or theta_col is None:
        print(f"Warning: Missing columns in {input_path}. Found: {list(col_idx.keys())}")
        return

    converted_lines = []
    for row in rows[1:]:
        cells = row.findall('ns:c', ns)
        row_data = {}
        for cell in cells:
            c_r = cell.get('r', 'A1')
            # Extract column letter and convert to index (A=0, B=1, etc.)
            col_letter = ''.join(c for c in c_r if c.isalpha())
            c_idx = ord(col_letter[0]) - ord('A')
            t = cell.get('t', '')
            v = cell.find('ns:v', ns)
            if v is not None and v.text is not None:
                if t == 's':
                    # Shared string - but numeric values may be stored as shared strings too
                    str_val = strings[int(v.text)]
                    try:
                        row_data[c_idx] = float(str_val)
                    except ValueError:
                        row_data[c_idx] = str_val
                else:
                    row_data[c_idx] = float(v.text)

        try:
            x_fits = float(row_data.get(col_idx['XWIN_IMAGE'], 0))
            y_fits = float(row_data.get(col_idx['YWIN_IMAGE'], 0))
            a_fits = float(row_data.get(col_idx[a_col], 0))
            b_fits = float(row_data.get(col_idx[b_col], 0))
            theta_fits = float(row_data.get(col_idx[theta_col], 0))
        except (ValueError, KeyError) as e:
            continue

        x_dl, y_dl, w, h, angle_rad = convert_fits_to_dl(
            x_fits, y_fits, a_fits, b_fits, theta_fits
        )

        out_line = f"{x_dl:.6f} {y_dl:.6f} {w:.6f} {h:.6f} {angle_rad:.6f} {CLASS_NAME}\n"
        converted_lines.append(out_line)

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.writelines(converted_lines)

    print(f"Converted {input_path} -> {output_path} ({len(converted_lines)} objects)")


def main():
    """Main conversion function."""
    gt_dir = "/home/fenglonghan/projects/mmrotate/data/CSST_data/gt"
    output_dir = "/home/fenglonghan/projects/mmrotate/data/CSST_data/gt_converted"

    # Find all CSV files (excluding the xlsx file that has .csv extension)
    csv_files = glob.glob(os.path.join(gt_dir, "*.csv"))

    for csv_path in csv_files:
        basename = os.path.basename(csv_path)

        # Determine output path
        txt_name = os.path.splitext(basename)[0] + ".txt"
        output_path = os.path.join(output_dir, txt_name)

        try:
            if basename == "image_01.csv":
                # This is actually an Excel file
                process_xlsx(csv_path, output_path)
            else:
                process_csv(csv_path, output_path)
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")


if __name__ == "__main__":
    main()
