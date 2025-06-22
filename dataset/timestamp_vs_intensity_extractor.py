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
video_id = "pZz3tfXEFmU"  # YouTube video ID (same as folder name)
d_path="M 0 100 C 1 85.4 2 30.5 5 27 C 8 23.4 11 69.9 15 82.4 C 19 94.9 21 88 25 89.5 C 29 91 31 90.5 35 90 C 39 89.5 41 89 45 87.2 C 49 85.3 51 84.4 55 80.7 C 59 77.1 61 69.5 65 69 C 69 68.4 71 75.3 75 77.8 C 79 80.3 81 80.6 85 81.6 C 89 82.6 91 82.3 95 82.7 C 99 83.2 101 83.3 105 83.8 C 109 84.3 111 90.7 115 85.2 C 119 79.7 121 62.3 125 56.2 C 129 50.2 131 50.2 135 54.9 C 139 59.7 141 73.8 145 80 C 149 86.2 151 84.8 155 86 C 159 87.3 161 87.1 165 86.2 C 169 85.3 171 83.1 175 81.7 C 179 80.2 181 79.4 185 78.9 C 189 78.4 191 78.8 195 79.1 C 199 79.5 201 80 205 80.6 C 209 81.2 211 81.3 215 82.3 C 219 83.3 221 84.6 225 85.6 C 229 86.6 231 86.8 235 87 C 239 87.3 241 86.1 245 86.7 C 249 87.2 251 89.3 255 90 C 259 90.7 261 90.8 265 90 C 269 89.2 271 86.7 275 86 C 279 85.2 281 86.6 285 86.4 C 289 86.1 291 86 295 84.7 C 299 83.4 301 80.5 305 79.8 C 309 79.2 311 79.8 315 81.4 C 319 82.9 321 86.2 325 87.8 C 329 89.4 331 90.8 335 89.3 C 339 87.9 341 83.7 345 80.4 C 349 77 351 73.7 355 72.4 C 359 71.1 361 71.8 365 73.8 C 369 75.7 371 79.6 375 82.2 C 379 84.8 381 85.5 385 86.9 C 389 88.2 391 88.8 395 89 C 399 89.1 401 88.5 405 87.6 C 409 86.6 411 84.8 415 84.1 C 419 83.4 421 83.6 425 84.1 C 429 84.6 431 86.1 435 86.4 C 439 86.7 441 85.4 445 85.4 C 449 85.4 451 86.7 455 86.5 C 459 86.2 461 86.8 465 84.2 C 469 81.5 471 76.3 475 73.3 C 479 70.3 481 71.6 485 69.1 C 489 66.7 491 65.1 495 61 C 499 57 501 57.6 505 48.9 C 509 40.2 511 27.3 515 17.5 C 519 7.7 521 -9.1 525 0 C 529 9.1 531 49.2 535 63.2 C 539 77.2 541 67.8 545 70 C 549 72.1 551 72.2 555 74 C 559 75.8 561 77.1 565 79.1 C 569 81.1 571 82.8 575 84.1 C 579 85.4 581 85.4 585 85.7 C 589 86 591 85.7 595 85.7 C 599 85.6 601 85.5 605 85.5 C 609 85.6 611 85.8 615 86.1 C 619 86.4 621 87.3 625 87.1 C 629 86.9 631 84.6 635 85.1 C 639 85.7 641 89.1 645 90 C 649 90.8 651 89.5 655 89.2 C 659 88.9 661 88.8 665 88.5 C 669 88.2 671 89.6 675 87.8 C 679 86 681 81.3 685 79.5 C 689 77.7 691 78.1 695 78.8 C 699 79.6 701 85 705 83.2 C 709 81.5 711 74.1 715 70 C 719 65.9 721 60.6 725 62.8 C 729 65 731 79.1 735 81 C 739 82.9 741 74.2 745 72.4 C 749 70.6 751 69.4 755 71.8 C 759 74.2 761 83.8 765 84.3 C 769 84.8 771 80.5 775 74.3 C 779 68.1 781 56.3 785 53.4 C 789 50.5 791 54.1 795 59.6 C 799 65.1 801 77.9 805 80.8 C 809 83.8 811 75.1 815 74.3 C 819 73.5 821 74.1 825 76.8 C 829 79.6 831 85.5 835 88.1 C 839 90.6 841 89.3 845 89.4 C 849 89.6 851 88.6 855 88.7 C 859 88.8 861 89.7 865 90 C 869 90.3 871 90 875 90 C 879 90 881 90 885 90 C 889 90 891 90 895 90 C 899 90 901 90 905 90 C 909 90 911 90 915 90 C 919 90 921 90 925 90 C 929 90 931 90 935 90 C 939 90 941 90 945 90 C 949 90 951 90 955 90 C 959 90 961 90 965 90 C 969 90 971 90 975 90 C 979 90 981 90 985 90 C 989 90 992 90 995 90 C 998 90 999 88 1000 90 C 1001 92 1000 98 1000 100"
video_duration_sec =768.661# seconds
timestamp_and_intensity_list=get_timestamp_and_intensity_pairs(d_path, video_duration_sec)
# Print first 10 (timestamp, intensity) pairs
print("First 10 (timestamp, intensity) pairs:")
for i, (timestamp, intensity) in enumerate(timestamp_and_intensity_list[:30], start=1):
    print(f"{i}. Timestamp: {timestamp:.2f} sec, Intensity: {intensity:.2f}")

timestamp, intensity = find_max_replayed_timestamp(timestamp_and_intensity_list)
print(f"🔥 Most Replayed at {timestamp:.2f} seconds (intensity: {intensity:.2f})") 

# Save to CSV
save_to_csv(video_id, timestamp_and_intensity_list)