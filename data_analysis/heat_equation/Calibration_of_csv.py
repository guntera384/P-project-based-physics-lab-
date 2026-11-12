# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -jupyter
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python (Pyodide)
#     language: python
#     name: python
# ---

# %%
import pandas as pd

def calibrate_heat_csv(
    input_csv_path, 
    output_csv_path="heat_measurement_calibrated_6sensors.csv"
):
    """
    Reads a 6-sensor heat measurement CSV, removes any rows containing empty/missing cells,
    and applies 2-point linear calibration so each sensor maps to T=0.0 at its first
    calibration point and T=100.0 at its second.
    """
    # Calibration lookup dictionary: sensor_number: (raw_at_0, raw_at_100)
    calib_map = {
        1: (0.0, 97.5),
        2: (0.1, 98.0),
        3: (0.0, 98.2),
        4: (0.0, 98.0),
        5: (0.4, 98.3),
        6: (0.8, 98.7)
    }

    # Load raw CSV file
    df = pd.read_csv(input_csv_path)
    
    # Remove any row where at least one cell is empty (NaN/missing)
    df = df.dropna()
    
    df_calibrated = df.copy()

    # Iterate through each sensor column (skipping column 0 which is Time)
    sensor_columns = df.columns[1:]
    for idx, col_name in enumerate(sensor_columns):
        sensor_num = idx + 1
        if sensor_num in calib_map:
            t_ref0, t_ref100 = calib_map[sensor_num]
            
            # Apply linear rescaling formula
            df_calibrated[col_name] = (df[col_name] - t_ref0) / (t_ref100 - t_ref0) * 100.0

    # Save calibrated data to new CSV
    df_calibrated.to_csv(output_csv_path, index=False)
    print(f"Successfully calibrated dataset saved to: {output_csv_path}")
    return df_calibrated

# Run calibration on your input CSV file
#if __name__ == "__main__":
#    calibrate_heat_csv("your_input_raw_data.csv", "heat_measurement_calibrated_6sensors.csv")



# %%
calibrate_heat_csv("NOCALI.csv")

# %%
import pandas as pd
import re

def process_and_clean_heat_csv(
    input_csv_path, 
    output_csv_path="cleaned_heat_data.csv",
    # Sensor positions corresponding to T1 through T6:
    sensor_positions=['2.5', '7.5', '12.5', '17.5', '22.5', '27.5']
):
    """
    Cleans heat experiment CSV data:
    1. Removes the 1st column (reading number) and the 'File Name' column.
    2. Extracts numbers enclosed in single quotes ('number') from the 'Notes' column.
    3. Fills missing cells in T1-T6 sequentially with the extracted numbers for each row.
    4. Renames T1-T6 columns to spatial positions.
    5. Removes the 'Notes' column and drops any rows that still contain empty cells.
    """
    # 1. Load CSV file
    df = pd.read_csv(input_csv_path)

    # 2. Remove 1st column (Reading number)
    df = df.iloc[:, 1:]

    # Identify File Name and Notes column names (case-insensitive search)
    file_name_col = [col for col in df.columns if 'photo' in col.lower()]
    notes_col = [col for col in df.columns if 'note' in col.lower()]

    if file_name_col:
        df = df.drop(columns=[file_name_col[0]])

    notes_col_name = notes_col[0] if notes_col else df.columns[-1]

    # Sensor columns are T1 through T6
    sensor_cols = [c for c in df.columns if re.match(r'^T[1-6]$', str(c).strip())]
    if not sensor_cols:
        # Fallback to columns between Time and Notes if T1-T6 aren't named explicitly
        sensor_cols = df.columns[1:-1].tolist()

    # 3. Process 'Notes' column and fill holes in T1-T6
    def fill_missing_from_notes(row):
        note_text = str(row[notes_col_name]) if pd.notna(row[notes_col_name]) else ""
        # Extract all numbers enclosed in single quotes (e.g., '21.5')
        extracted_numbers = re.findall(r"'([-+]?\d*\.?\d+)'", note_text)

        if extracted_numbers:
            num_idx = 0
            for col in sensor_cols:
                # Check if cell is NaN or empty
                if pd.isna(row[col]) or str(row[col]).strip() == "":
                    if num_idx < len(extracted_numbers):
                        row[col] = float(extracted_numbers[num_idx])
                        num_idx += 1
        return row

    df = df.apply(fill_missing_from_notes, axis=1)

    # 4. Rename T1-T6 to sensor positions
    rename_mapping = {sensor_cols[i]: sensor_positions[i] for i in range(min(len(sensor_cols), len(sensor_positions)))}
    df = df.rename(columns=rename_mapping)

    # 5. Remove 'Notes' column
    df = df.drop(columns=[notes_col_name])

    # 6. Drop any rows that still have empty cells (NaNs)
    df_cleaned = df.dropna()

    # Save to output CSV
    df_cleaned.to_csv(output_csv_path, index=False)
    print(f"Cleaned dataset saved successfully to: {output_csv_path}")
    return df_cleaned

# ---------------------------------------------------------
# Example Usage:
# ---------------------------------------------------------



# %%
process_and_clean_heat_csv("Series_004.csv", "cleaned_data.csv")


# %%
#Splitting csv into 3
import pandas as pd

def split_csv_into_three(input_csv_path, output_prefix="output_part"):
    """
    Splits a CSV file into 3 separate files using pure pandas slicing.
    The header (first row) is automatically included in all 3 output files.
    """
    df = pd.read_csv(input_csv_path)
    total_rows = len(df)
    
    # Calculate rows per part and distribute any remainder
    chunk_size = total_rows // 3
    remainder = total_rows % 3

    start = 0
    for i in range(1, 4):
        end = start + chunk_size + (1 if i <= remainder else 0)
        part_df = df.iloc[start:end]
        start = end

        output_filename = f"{output_prefix}_{i}.csv"
        part_df.to_csv(output_filename, index=False)
        print(f"Saved Part {i} ({len(part_df)} rows) -> {output_filename}")



# %%
split_csv_into_three("cleaned_data.csv", "output_part")

