print("SCRIPT STARTED")
input("Press Enter to continue...")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# Variable Star Light Curve Analysis Project
# ============================================================================
# This script analyzes observational data from a variable star to study
# how its brightness (magnitude) changes over time. Variable stars undergo
# periodic or irregular brightness variations, and studying these patterns
# helps astronomers classify the star's type and measure its period.
# ============================================================================


def load_star_data(filepath):
    """
    Load observational data from a CSV file.
    
    Expected columns:
    - BJD: Barycentric Julian Day (observation time)
    - raw: Observed brightness/flux
    - err: Measurement error in brightness
    - post_decorr, post_tfa: (ignored)
    
    Returns a pandas DataFrame with cleaned data.
    """
    try:
        # Read the CSV file with proper column names
        # The file has: BJD, raw, ost_decorr, ost_tfa, err
        df = pd.read_csv(filepath, sep=',', names=['BJD', 'raw', 'ost_decorr', 'ost_tfa', 'err'], skiprows=1)
        print(f"✓ Loaded data from {filepath}")
        print(f"  Total observations: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def clean_data(df):
    """
    Clean the data by removing missing values and invalid entries.
    
    Steps:
    1. Drop rows with NaN (missing) values in critical columns
    2. Remove rows where error values are zero or negative (invalid)
    3. Keep only the necessary columns: BJD, raw, err
    
    Returns a cleaned DataFrame.
    """
    # Select only the columns we need
    df_clean = df[['BJD', 'raw', 'err']].copy()
    
    # Convert columns to numeric values
    df_clean['BJD'] = pd.to_numeric(df_clean['BJD'], errors='coerce')
    df_clean['raw'] = pd.to_numeric(df_clean['raw'], errors='coerce')
    df_clean['err'] = pd.to_numeric(df_clean['err'], errors='coerce')
    
    # Remove rows with missing values
    initial_count = len(df_clean)
    df_clean = df_clean.dropna()
    removed = initial_count - len(df_clean)
    if removed > 0:
        print(f"✓ Removed {removed} rows with missing values")
    
    # Remove rows where error is zero or negative (invalid uncertainties)
    df_clean = df_clean[df_clean['err'] > 0]
    
    # Reset the index for cleaner data
    df_clean = df_clean.reset_index(drop=True)
    
    print(f"✓ Data cleaned: {len(df_clean)} valid observations remaining")
    return df_clean


def create_light_curve(df, output_file='light_curve.png'):
    """
    Create a light curve plot: brightness vs. time with error bars.
    
    The light curve shows the star's brightness variation over the observation
    period. Key features to look for:
    
    - PERIOD: The time it takes for the brightness pattern to repeat.
      Regular, sinusoidal curves suggest pulsating variables (RR Lyrae, etc.).
      
    - AMPLITUDE: The magnitude of brightness changes. Large amplitudes suggest
      eclipsing binaries or Mira variables.
      
    - SHAPE: The shape of the light curve helps identify the star's type:
      * Smooth sinusoid → RR Lyrae, Delta Scuti
      * Sawtooth (sharp rise, gradual fall) → Cepheid
      * Deep eclipses → Eclipsing Binary
      * Irregular → Long-period or semi-regular variables
    
    Parameters:
    - df: Cleaned DataFrame with BJD, raw, err columns
    - output_file: Name of the output image file
    """
    # Create a new figure with appropriate size
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot brightness vs. time with error bars
    ax.errorbar(df['BJD'], df['raw'], yerr=df['err'], 
                fmt='o', markersize=4, capsize=3, capthick=1,
                color='steelblue', ecolor='gray', alpha=0.7,
                label='Observations')
    
    # Labels and title
    ax.set_xlabel('Barycentric Julian Day (BJD)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Brightness (relative flux)', fontsize=12, fontweight='bold')
    ax.set_title('Variable Star Light Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Tight layout to prevent label cutoff
    fig.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=150)
    print(f"✓ Light curve saved to '{output_file}'")
    plt.show()


def analyze_light_curve(df):
    """
    Print basic statistics about the light curve.
    
    These statistics help characterize the star's variability.
    """
    print("\n" + "="*60)
    print("LIGHT CURVE STATISTICS")
    print("="*60)
    
    # Time span
    time_span = df['BJD'].max() - df['BJD'].min()
    print(f"Observation span: {time_span:.2f} days ({time_span/365.25:.2f} years)")
    
    # Brightness statistics
    mean_brightness = df['raw'].mean()
    std_brightness = df['raw'].std()
    min_brightness = df['raw'].min()
    max_brightness = df['raw'].max()
    amplitude = max_brightness - min_brightness
    
    print(f"\nBrightness statistics:")
    print(f"  Mean:      {mean_brightness:.4f}")
    print(f"  Std Dev:   {std_brightness:.4f}")
    print(f"  Min:       {min_brightness:.4f}")
    print(f"  Max:       {max_brightness:.4f}")
    print(f"  Amplitude: {amplitude:.4f}")
    
    # Mean error
    mean_error = df['err'].mean()
    print(f"\nMean measurement error: {mean_error:.6f}")
    print("="*60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Specify the path to your CSV data file
    # Example: "variable_star_observations.csv"
    data_file = "data/mmRR2_lc.csv"
  
    print("\n" + "="*60)
    print("VARIABLE STAR LIGHT CURVE ANALYSIS")
    print("="*60 + "\n")
    
    # Step 1: Load the data
    df = load_star_data(data_file)
    if df is None:
        print("Failed to load data. Exiting.")
        exit()
    
    # Step 2: Clean the data
    df_clean = clean_data(df)
    if len(df_clean) == 0:
        print("No valid data after cleaning. Exiting.")
        exit()
    
    # Step 3: Analyze the light curve
    analyze_light_curve(df_clean)
    
    # Step 4: Create and display the light curve plot
    create_light_curve(df_clean)
    
    print("\n✓ Analysis complete!")
    print("\nNext steps for deeper analysis:")
    print("  1. Look for periodicity using Fourier analysis or Lomb-Scargle")
    print("  2. Measure the period by visual inspection or frequency analysis")
    print("  3. Compare light curve shape to known variable star types")
    print("  4. Model the light curve with theoretical predictions")

    print("Reached end of file")
input("Press Enter to exit...")
