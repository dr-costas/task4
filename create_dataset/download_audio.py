# Code Contributor - Ankit Shah - ankit.tronix@gmail.com
import datetime
import multiprocessing
import os
import sys
import time
from multiprocessing import Pool

import pafy
from tqdm import tqdm


# Format audio - 16 bit Signed PCM audio sampled at 44.1kHz
def format_audio(input_audio_file, output_audio_file):
    temp_audio_file = output_audio_file.split('.wav')[0] + '_temp.wav'
    cmd_string = "ffmpeg -loglevel panic -i %s -ac 1 -ar 44100 %s" % (input_audio_file, temp_audio_file)
    os.system(cmd_string)
    cmd_string1 = "sox %s -G -b 16 -r 44100 %s" % (temp_audio_file, output_audio_file)
    os.system(cmd_string1)
    cmd_string2 = "rm -rf %s" % temp_audio_file
    os.system(cmd_string2)


# Trim audio based on start time and duration of audio.
def trim_audio(input_audio_file, output_audio_file, start_time, duration):
    # print input_audio_file
    # print output_audio_file
    cmd_string = "sox %s %s trim %s %s" % (input_audio_file, output_audio_file, start_time, duration)
    os.system(cmd_string)


def multi_run_wrapper(args):
    return download_audio_method(*args)


# Method to download audio - Downloads the best audio available for audio id, calls the formatting audio
# function and then segments the audio formatted based on start and end time.
def download_audio_method(line, csv_file):
    query_id = line.split(",")[0]
    start_seconds = line.split(",")[1]
    end_seconds = line.split(",")[2]
    audio_duration = float(end_seconds) - float(start_seconds)
    # positive_labels = ','.join(line.split(",")[3:]);
    print("Query -> " + query_id)
    # print "start_time -> " + start_seconds
    # print "end_time -> " + end_seconds
    # print "positive_labels -> " + positive_labels
    url = "https://www.youtube.com/watch?v=" + query_id
    try:
        video = pafy.new(url)
        best_audio = video.getbestaudio()
        # .csv - split - to get the folder information. As path is also passed for the audio -
        # creating the directory from the path where this audio script is present. THus using second
        # split to get the folder name where output files shall be downloaded
        output_folder = sys.argv[1].split('.csv')[0].split("/")[-1] + "_" + csv_file.split('.csv')[
            0] + "_" + "audio_downloaded"
        # print output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        path_to_download = output_folder + "/Y" + query_id + "." + best_audio.extension
        # print path_to_download
        best_audio.download(path_to_download)
        formatted_folder = sys.argv[1].split('.csv')[0].split("/")[-1] + "_" + csv_file.split('.csv')[
            0] + "_" + "audio_formatted_downloaded"
        if not os.path.exists(formatted_folder):
            os.makedirs(formatted_folder)
        path_to_formatted_audio = formatted_folder + "/Y" + query_id + ".wav"
        format_audio(path_to_download, path_to_formatted_audio)
        # Trimming code
        segmented_folder = sys.argv[1].split('.csv')[0].split("/")[-1] + "_" + csv_file.split('.csv')[
            0] + "_" + "audio_formatted_and_segmented_downloads"
        if not os.path.exists(segmented_folder):
            os.makedirs(segmented_folder)
        path_to_segmented_audio = segmented_folder + "/Y" + query_id + '_' + start_seconds + '_' + end_seconds + ".wav"
        trim_audio(path_to_formatted_audio, path_to_segmented_audio, start_seconds, audio_duration)

        # Remove the original audio and the formatted audio. Comment line to keep both.
        # Delete "output_folder" or "formatted_folder" to keep one.
        cmd_string2 = "rm -rf %s %s" % (output_folder, formatted_folder)
        # os.system(cmd_string2)
        # Remove formatted audio. Comment the line to keep the formatted files as well.
        # Deleting as we have original - thus formatted_files could be generated easily
        cmd_string3 = "rm -rf %s" % formatted_folder
        # os.system(cmd_string3)

        ex1 = ""
    except Exception as ex:
        ex1 = str(ex) + ',' + str(query_id)
        print("Error is ---> " + str(ex))
    return ex1


# Download audio - Reads 3 lines of input csv file at a time and passes them to multi_run
# wrapper which calls download_audio_method to download the file based on id.
# Multiprocessing module spawns 3 process in parallel which runs download_audio_method.
# Multiprocessing, thus allows downloading process to happen in 40 percent of the time
# approximately to downloading sequentially - processing line by line of input csv file.
def download_audio(csv_file, time_stamp):
    error_log = 'error' + time_stamp + '.log'
    with open(csv_file, "r") as segments_info_file:
        with open(error_log, "a") as fo:
            for line in tqdm(segments_info_file):
                line = (line, csv_file)
                lines_list = list()
                lines_list.append(line)
                try:
                    next_line = next(segments_info_file)
                    next_line = (next_line, csv_file)
                    lines_list.append(next_line)
                except StopIteration:
                    print("end of file")
                try:
                    next_line = next(segments_info_file)
                    next_line = (next_line, csv_file)
                    lines_list.append(next_line)
                except StopIteration:
                    print("end of file")
                # print lines_list
                p = multiprocessing.Pool(3)

                exception = p.map(multi_run_wrapper, lines_list)
                for item in exception:
                    if item:
                        fo.writelines(str(item) + '\n')
                p.close()
                p.join()
        fo.close()


def main(arg):
    ts = time.time()
    time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
    download_audio(arg, time_stamp)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('takes arg1 as csv file to downloaded')
    else:
        main(sys.argv[1])
