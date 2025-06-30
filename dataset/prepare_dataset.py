from subtitle_time_frame_processor import main as process_subtitles
from time_frame_vs_averarage_intensity_calculator import main as intensity_average_calculator
from feature_extraction import main as feature_extraction_process
from label_features import main as label_dataset
time_frame_increment = 20
context_window_size=4
run_subtitle_process=True
run_average_calculator=True
run_feature_extraction=True
run_labelling_process=True
if __name__ == "__main__":
    if (run_subtitle_process):
        print("Running subtitle processing...")
        process_subtitles(time_frame_increment)
    if (run_average_calculator):
        print("Running intensity average processing...")
        intensity_average_calculator(time_frame_increment)
    if (run_feature_extraction):
        print("Running feature extraction process...")
        feature_extraction_process(time_frame_increment)
    if (run_labelling_process): 
        print("Running labeling process")
        label_dataset(time_frame_increment,context_window_size)