import pandas as pd
import os

# This function creates average intensity values for selected time frames
# and fills missing values using linear interpolation
def main(time_frame_increment):
    root_folder = os.path.dirname(os.path.abspath(__file__))

    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)

        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                if file.endswith("_heatmap.csv"):
                    # Define output subfolder path (e.g., time_frame_increment_5)
                    output_subfolder = os.path.join(subfolder_path, f"time_frame_increment_{time_frame_increment}")
                    os.makedirs(output_subfolder, exist_ok=True)
                    # Define expected output file name and check if it already exists
                    output_file = file.replace("_heatmap.csv", "_averaged_intensity_heatmap.csv")
                    output_path = os.path.join(output_subfolder, output_file)
                    #If already processed do not recalculate
                    if os.path.exists(output_path):
                        print(f"Skipped {output_file}: already exists")
                        continue

                    heatmap_path = os.path.join(subfolder_path, file)

                    # Read the heatmap file
                    df = pd.read_csv(heatmap_path)

                    # Ensure correct column names
                    if 'timestamp_sec' not in df.columns or 'intensity' not in df.columns:
                        print(f"Skipped: Missing columns in {heatmap_path}")
                        continue

                    # Create bins for averaging
                    df['time_bin'] = (df['timestamp_sec'] // time_frame_increment) * time_frame_increment

                    # Compute average intensity per bin
                    averaged_df = df.groupby('time_bin')['intensity'].mean().reset_index()
                    averaged_df.columns = ['start_time', 'avg_intensity']

                    # Ensure all bins are included
                    min_time = int(df['timestamp_sec'].min() // time_frame_increment) * time_frame_increment
                    max_time = int(df['timestamp_sec'].max() // time_frame_increment) * time_frame_increment
                    all_bins = pd.DataFrame({'start_time': range(min_time, max_time + time_frame_increment, time_frame_increment)})

                    # Merge and interpolate
                    averaged_df = all_bins.merge(averaged_df, on='start_time', how='left')
                    averaged_df['avg_intensity'] = averaged_df['avg_intensity'].interpolate(method='linear', limit_direction='both')

                    # Add stop_time
                    averaged_df['stop_time'] = averaged_df['start_time'] + time_frame_increment
                    averaged_df = averaged_df[['start_time', 'stop_time', 'avg_intensity']]

                    # Save result
                    averaged_df.to_csv(output_path, index=False)
                    print(f"Saved: {output_path}")
