import os
import glob
import pandas as pd

def parse_freesurfer_stats(stats_file):
    """
    Parses a FreeSurfer .stats file into a pandas DataFrame.
    """
    if not os.path.exists(stats_file):
        return None
    
    # Read the file to find column headers
    col_names = []
    with open(stats_file, 'r') as f:
        for line in f:
            if line.startswith('# ColHeaders'):
                # Format is usually: # ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd
                col_names = line.strip().split()[2:] 
                break
    
    # Fallback to standard Desikan-Killiany columns if not clearly found
    if not col_names:
        col_names = ['StructName', 'NumVert', 'SurfArea', 'GrayVol', 'ThickAvg', 
                     'ThickStd', 'MeanCurv', 'GausCurv', 'FoldInd', 'CurvInd']
        
    try:
        # Load the data, ignoring comment lines starting with '#'
        df = pd.read_csv(stats_file, sep=r'\s+', comment='#', names=col_names)
        return df
    except Exception as e:
        print(f"Error parsing {stats_file}: {e}")
        return None

def process_subject(subject_dir, output_dir):
    """
    Extracts DK atlas features for a single subject and saves them to a CSV file.
    """
    subject_id = os.path.basename(os.path.normpath(subject_dir))
    stats_dir = os.path.join(subject_dir, 'stats')
    
    # The Desikan-Killiany atlas stats are stored in aparc.stats
    lh_stats_file = os.path.join(stats_dir, 'lh.aparc.stats')
    rh_stats_file = os.path.join(stats_dir, 'rh.aparc.stats')
    
    df_lh = parse_freesurfer_stats(lh_stats_file)
    df_rh = parse_freesurfer_stats(rh_stats_file)
    
    if df_lh is None and df_rh is None:
        print(f"Skipping {subject_id}: No aparc.stats found.")
        return
        
    # Prepend hemisphere prefix to region names to avoid ID collision
    if df_lh is not None:
        df_lh['StructName'] = 'lh_' + df_lh['StructName']
    if df_rh is not None:
        df_rh['StructName'] = 'rh_' + df_rh['StructName']
        
    # Combine hemispheres
    dfs = []
    if df_lh is not None: dfs.append(df_lh)
    if df_rh is not None: dfs.append(df_rh)
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Set the ROI ID ('StructName') as the index
    df_combined.set_index('StructName', inplace=True)
    
    # Save to CSV
    output_file = os.path.join(output_dir, f"{subject_id}_dk_features.csv")
    df_combined.to_csv(output_file)
    print(f"Saved features for {subject_id} to {output_file}")

def main():
    subjects_dir = '/deneb_disk/3T_vs_low_field/freesurfer_processed_subjects/subjects/'
    
    # Create an output directory in the current workspace
    output_dir = os.path.join(os.getcwd(), 'freesurfer_features_csv')
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all subject directories that contain a 'stats' folder
    subjects = [d for d in glob.glob(os.path.join(subjects_dir, '*')) 
                if os.path.isdir(d) and os.path.exists(os.path.join(d, 'stats'))]
    
    print(f"Found {len(subjects)} processed subjects. Extracting features...")
    
    for subj_dir in subjects:
        process_subject(subj_dir, output_dir)
        
    print(f"Done! All CSV files have been saved in: {output_dir}")

if __name__ == '__main__':
    main()
