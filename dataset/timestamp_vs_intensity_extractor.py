import re
import os
import csv

def get_timestamp_and_intensity_pairs(d_string, video_duration_sec):
    coords = re.findall(r'[-+]?\d*\.\d+|\d+', d_string)
    coords = list(map(float, coords))
    ##Convert flat list x1,y1,x2,y2 to (x1,y1),(x2,y2) pairs
    xy_pairs = list(zip(coords[::2], coords[1::2]))

    # Filter out points with negative x-values (invalid for time mapping)
    xy_pairs = [(x, y) for x, y in xy_pairs if x >= 0]

    if not xy_pairs:
        raise ValueError("No valid (x, y) pairs found")

    max_x = max(x for x, y in xy_pairs)
    timestamps_and_intensity = []

    for x, y in xy_pairs:
        timestamp = (x / max_x) * video_duration_sec
        intensity = 100 - y
        timestamps_and_intensity.append((timestamp, intensity))
    return timestamps_and_intensity
    
def find_max_replayed_timestamp(timestamps_and_intensity_list):
    most_replayed = max(timestamps_and_intensity_list, key=lambda pair: pair[1])
    return most_replayed

def save_to_csv(video_id, timestamp_and_intensity_list):
    folder = video_id
    os.makedirs(folder, exist_ok=True)
    csv_path = os.path.join(folder, f"{video_id}_heatmap.csv")
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp_sec", "intensity"])
        for timestamp, intensity in timestamp_and_intensity_list:
            writer.writerow([f"{timestamp:.3f}", f"{intensity:.3f}"])
    print(f"✅ Saved heatmap data to: {csv_path}")
# Example usage
video_id = "hTSaweR8qMI"  # YouTube video ID (same as folder name)
d_path="M 0 100 C 1 80 2 7.4 5 0 C 8 -7.4 11 50 15 62.9 C 19 75.7 21 62 25 64.3 C 29 66.7 31 76.4 35 74.4 C 39 72.4 41 58.4 45 54.4 C 49 50.3 51 49.7 55 54.2 C 59 58.7 61 71.1 65 77 C 69 82.9 71 81.9 75 83.9 C 79 85.8 81 85.8 85 86.9 C 89 88 91 88.7 95 89.3 C 99 89.9 101 89.9 105 90 C 109 90.1 111 90 115 90 C 119 90 121 91 125 90 C 129 89 131 87.4 135 85.1 C 139 82.8 141 77.5 145 78.5 C 149 79.5 151 87.7 155 90 C 159 92.3 161 90 165 90 C 169 90 171 90.6 175 90 C 179 89.4 181 87.9 185 86.9 C 189 85.9 191 84.8 195 84.8 C 199 84.8 201 86.6 205 87 C 209 87.4 211 86.2 215 86.8 C 219 87.4 221 89.4 225 90 C 229 90.6 231 91.5 235 90 C 239 88.5 241 88.9 245 82.3 C 249 75.8 251 56.2 255 57.4 C 259 58.6 261 82.4 265 88.5 C 269 94.6 271 87.9 275 88 C 279 88.1 281 88.6 285 89 C 289 89.4 291 89.8 295 90 C 299 90.2 301 90 305 90 C 309 90 311 92.5 315 90 C 319 87.5 321 79.7 325 77.7 C 329 75.8 331 77.7 335 80.2 C 339 82.7 341 88 345 90 C 349 92 351 90 355 90 C 359 90 361 90 365 90 C 369 90 371 90.2 375 90 C 379 89.8 381 89 385 89 C 389 89 391 89.8 395 90 C 399 90.2 401 94.9 405 90 C 409 85.1 411 66.4 415 65.6 C 419 64.8 421 82.8 425 86.2 C 429 89.6 431 82.6 435 82.6 C 439 82.7 441 87.5 445 86.4 C 449 85.3 451 80.8 455 77.1 C 459 73.4 461 69.7 465 67.9 C 469 66 471 67.4 475 68 C 479 68.6 481 70.8 485 71 C 489 71.2 491 68.7 495 69 C 499 69.3 501 71.6 505 72.7 C 509 73.8 511 75.3 515 74.7 C 519 74.2 521 68.3 525 70 C 529 71.6 531 81.3 535 83 C 539 84.7 541 77.4 545 78.3 C 549 79.1 551 85.1 555 87.3 C 559 89.5 561 91.2 565 89.3 C 569 87.4 571 85.4 575 77.7 C 579 70.1 581 48.5 585 50.9 C 589 53.4 591 82.2 595 90 C 599 97.8 601 90 605 90 C 609 90 611 90 615 90 C 619 90 621 90 625 90 C 629 90 631 91.4 635 90 C 639 88.6 641 83.1 645 82.9 C 649 82.7 651 89.2 655 88.9 C 659 88.5 661 84.2 665 81.1 C 669 78 671 72.6 675 73.6 C 679 74.5 681 82.4 685 85.7 C 689 89 691 89.1 695 90 C 699 90.9 701 93.1 705 90 C 709 86.9 711 85.3 715 74.6 C 719 63.9 721 42.2 725 36.3 C 729 30.5 731 42.7 735 45.5 C 739 48.3 741 44.2 745 50.4 C 749 56.6 751 71.3 755 76.3 C 759 81.3 761 73.1 765 75.4 C 769 77.6 771 84.5 775 87.5 C 779 90.4 781 89.5 785 90 C 789 90.5 791 90 795 90 C 799 90 801 90 805 90 C 809 90 811 90 815 90 C 819 90 821 90 825 90 C 829 90 831 90 835 90 C 839 90 841 97 845 90 C 849 83 851 68.9 855 54.8 C 859 40.6 861 15.5 865 19.4 C 869 23.2 871 59.9 875 73.9 C 879 87.9 881 88.5 885 89.3 C 889 90.2 891 82 895 78 C 899 74 901 67.9 905 69.4 C 909 70.9 911 82.9 915 85.5 C 919 88.1 921 83.2 925 82.4 C 929 81.6 931 79.9 935 81.5 C 939 83 941 88.3 945 90 C 949 91.7 951 90 955 90 C 959 90 961 90 965 90 C 969 90 971 90 975 90 C 979 90 981 92.2 985 90 C 989 87.7 992 81.1 995 78.8 C 998 76.6 999 74.6 1000 78.8 C 1001 83.1 1000 95.8 1000 100"
video_duration_sec =1042.001# seconds
timestamp_and_intensity_list=get_timestamp_and_intensity_pairs(d_path, video_duration_sec)
# Print first 10 (timestamp, intensity) pairs
print("First 10 (timestamp, intensity) pairs:")
for i, (timestamp, intensity) in enumerate(timestamp_and_intensity_list[:30], start=1):
    print(f"{i}. Timestamp: {timestamp:.2f} sec, Intensity: {intensity:.2f}")

timestamp, intensity = find_max_replayed_timestamp(timestamp_and_intensity_list)
print(f"🔥 Most Replayed at {timestamp:.2f} seconds (intensity: {intensity:.2f})") 

# Save to CSV
save_to_csv(video_id, timestamp_and_intensity_list)