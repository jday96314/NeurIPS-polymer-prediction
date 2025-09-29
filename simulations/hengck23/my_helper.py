import pandas as pd
import json

def read_lammps_log(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    data_lines = []
    headers = []
    reading_data = False

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Find the header line (starts with "Step" and followed by "Time")
        if line.startswith("Step") and "Time" in line:
            headers = line.split()
            reading_data = True
            continue

        # Start reading numeric data after headers
        if reading_data:
            parts = line.split()
            # If line looks like a full row of numbers
            if all(p.replace('.', '', 1).replace('-', '', 1).isdigit() for p in parts[:2]):
                data_lines.append([float(x) if '.' in x or 'e' in x.lower() else int(x) for x in parts])
            else:
                # Stop reading if another section starts
                break

    # Convert to DataFrame
    df = pd.DataFrame(data_lines, columns=headers)
    return df

def read_lammps_timeavg_profile(filename):
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == '' or line.startswith('#'):
            i += 1
            continue
        # Read block header: timestep and row count
        timestep_info = line.split()
        if len(timestep_info) == 2:
            timestep = int(timestep_info[0])
            n_rows = int(timestep_info[1])
            i += 1
            # Read next n_rows lines of actual data
            for _ in range(n_rows):
                print(lines[i])
                idx, rg = map(float, lines[i].strip().split())
                data.append([timestep, int(idx), rg])
                i += 1
        else:
            i += 1  # Skip any malformed lines

    df = pd.DataFrame(data, columns=['Timestep', 'Index', 'Rg'])
    return df