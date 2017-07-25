#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Code Contributor: Rohan Badlani, Email: rohan.badlani@gmail.com


class FileFormat(object):
    def __init__(self, results_list):
        """

        :param results_list:
        :type results_list: list[list[dict[str, float|str]]]
        """
        self.results_list = results_list
        self.labels_dict = {}
        self.read_labels()

    def read_labels(self):
        # Filename will act as key and labels list will be the value
        for dict_entries in self.results_list:
            for entry in dict_entries:
                audio_file = entry['audio_file']
                start_time = entry['event_onset']
                end_time = entry['event_offset']
                label = entry['event_label']

                if audio_file not in self.labels_dict.keys():
                    # does not exist
                    if label is not "":
                        self.labels_dict[audio_file] = [label]
                    else:
                        self.labels_dict[audio_file] = []
                else:
                    # exists
                    if label is not "":
                        self.labels_dict[audio_file].append(label)

    def validate_predicted_ds(self, predicted_ds):
        # iterate over predicted list

        # check bothways
        for audio_file in predicted_ds.labels_dict.keys():
            if audio_file not in self.labels_dict.keys():
                return False

        for audio_file in self.labels_dict.keys():
            if audio_file not in predicted_ds.labels_dict.keys():
                return False

        # check complete. One-One mapping
        return True

    def compute_metrics(self, predicted_ds):
        t_p = 0
        f_p = 0
        f_n = 0

        class_wise_metrics = {}

        # iterate over predicted list
        for audioFile in predicted_ds.labels_dict.keys():
            marker_list = [0] * len(self.labels_dict[audioFile])
            for predicted_label in predicted_ds.labels_dict[audioFile]:
                # for a predicted label

                # 1. Check if it is present inside groundTruth, if yes push to TP,
                # mark the existance of that groundtruth label
                index = 0
                for ground_truth_label in self.labels_dict[audioFile]:
                    if predicted_label == ground_truth_label:
                        t_p += 1
                        marker_list[index] = 1
                        break
                    index += 1

                if index == len(self.labels_dict[audioFile]):
                    # not found. Add as FP
                    f_p += 1

            # check markerList, add all FN
            for marker in marker_list:
                if marker == 0:
                    f_n += 1

            for ground_truth_label in self.labels_dict[audioFile]:
                if ground_truth_label in predicted_ds.labels_dict[audioFile]:
                    # the class was predicted correctly
                    if ground_truth_label in class_wise_metrics.keys():
                        class_wise_metrics[ground_truth_label][0] += 1
                    else:
                        # Format: TP, FP, FN
                        class_wise_metrics[ground_truth_label] = [1, 0, 0]
                else:
                    # Not predicted --> FN
                    if ground_truth_label in class_wise_metrics.keys():
                        class_wise_metrics[ground_truth_label][2] += 1
                    else:
                        class_wise_metrics[ground_truth_label] = [0, 0, 1]

            for predicted_label in predicted_ds.labels_dict[audioFile]:
                if predicted_label not in self.labels_dict[audioFile]:
                    # Predicted but not in Groundtruth --> FP
                    if predicted_label in class_wise_metrics.keys():
                        class_wise_metrics[predicted_label][1] += 1
                    else:
                        class_wise_metrics[predicted_label] = [0, 1, 0]

        if (t_p + f_p) != 0:
            precision = float(t_p) / float(t_p + f_p)
        else:
            precision = 0.0
        if (t_p + f_n) != 0:
            recall = float(t_p) / float(t_p + f_n)
        else:
            recall = 0.0
        if (precision + recall) != 0.0:
            f1 = 2 * precision * recall / float(precision + recall)
        else:
            f1 = 0.0

        for classLabel in class_wise_metrics.keys():
            precision = 0.0
            recall = 0.0
            f1 = 0.0

            tp = class_wise_metrics[classLabel][0]
            fp = class_wise_metrics[classLabel][1]
            fn = class_wise_metrics[classLabel][2]
            if (tp + fp) != 0:
                precision = float(tp) / float(tp + fp)
            if (tp + fn) != 0:
                recall = float(tp) / float(tp + fn)
            if (precision + recall) != 0.0:
                f1 = 2 * precision * recall / float(precision + recall)

        return {
            'precision': precision * 100.0,
            'recall': recall * 100.0,
            'f1': f1 * 100.0
        }

    def compute_metrics_string(self, predicted_ds):
        t_p = 0
        f_p = 0
        f_n = 0

        class_wise_metrics = {}

        # iterate over predicted list
        for audio_file in predicted_ds.labels_dict.keys():
            if audio_file in self.labels_dict:
                marker_list = [0] * len(self.labels_dict[audio_file])
                for predicted_label in predicted_ds.labels_dict[audio_file]:
                    # for a predicted label

                    # 1. Check if it is present inside groundTruth, if yes push to TP,
                    # mark the existance of that groundtruth label
                    index = 0
                    for ground_truth_label in self.labels_dict[audio_file]:
                        if predicted_label == ground_truth_label and marker_list[index] != 1:
                            t_p += 1
                            marker_list[index] = 1
                            break
                        index += 1

                    if index == len(self.labels_dict[audio_file]):
                        # not found. Add as FP
                        f_p += 1

                # check markerList, add all FN
                for marker in marker_list:
                    if marker == 0:
                        f_n += 1

                for ground_truth_label in self.labels_dict[audio_file]:
                    if ground_truth_label in predicted_ds.labels_dict[audio_file]:
                        # the class was predicted correctly
                        if ground_truth_label in class_wise_metrics.keys():
                            class_wise_metrics[ground_truth_label][0] += 1
                        else:
                            # Format: TP, FP, FN
                            class_wise_metrics[ground_truth_label] = [1, 0, 0]
                    else:
                        # Not predicted --> FN
                        if ground_truth_label in class_wise_metrics.keys():
                            class_wise_metrics[ground_truth_label][2] += 1
                        else:
                            class_wise_metrics[ground_truth_label] = [0, 0, 1]

                for predicted_label in predicted_ds.labels_dict[audio_file]:
                    if predicted_label not in self.labels_dict[audio_file]:
                        # Predicted but not in Groundtruth --> FP
                        if predicted_label in class_wise_metrics.keys():
                            class_wise_metrics[predicted_label][1] += 1
                        else:
                            class_wise_metrics[predicted_label] = [0, 1, 0]

        if (t_p + f_p) != 0:
            precision = float(t_p) / float(t_p + f_p)
        else:
            precision = 0.0
        if (t_p + f_n) != 0:
            recall = float(t_p) / float(t_p + f_n)
        else:
            recall = 0.0
        if (precision + recall) != 0.0:
            f1 = 2 * precision * recall / float(precision + recall)
        else:
            f1 = 0.0

        output = ""
        output += "\n\nClass-based Metrics\n\n"

        class_wise_precision = 0.0
        class_wise_recall = 0.0
        class_wise_f1 = 0.0
        class_count = 0

        for classLabel in class_wise_metrics.keys():
            class_count += 1

            precision = 0.0
            recall = 0.0
            f1 = 0.0

            tp = class_wise_metrics[classLabel][0]
            fp = class_wise_metrics[classLabel][1]
            fn = class_wise_metrics[classLabel][2]
            if (tp + fp) != 0:
                precision = float(tp) / float(tp + fp)
                class_wise_precision += precision
            if (tp + fn) != 0:
                recall = float(tp) / float(tp + fn)
                class_wise_recall += recall
            if (precision + recall) != 0.0:
                f1 = 2 * precision * recall / float(precision + recall)

            output += "\tClass = " + str(classLabel.split("\n")[0]) + ", Precision = " + str(
                precision) + ", Recall = " + str(recall) + ", F1 Score = " + str(f1) + "\n"

        class_wise_precision = class_wise_precision / class_count
        class_wise_recall = class_wise_recall / class_count
        if (class_wise_precision + class_wise_recall) != 0.0:
            class_wise_f1 = 2 * class_wise_precision * class_wise_recall / \
                            float(class_wise_precision + class_wise_recall)

        output += "\n\n\tComplete Metrics (Computed Class Wise Average)\n\n"
        output += "\tPrecision = " + str(class_wise_precision * 100.0) + "\n"
        output += "\tRecall = " + str(class_wise_recall * 100.0) + "\n"
        output += "\tF1 Score = " + str(class_wise_f1 * 100.0) + "\n"
        output += "\tNumber of Audio Files = " + str(len(self.labels_dict.keys())) + "\n\n"

        output += "\n\n\tComplete Metrics (Computed Instance Wise) - " \
                  "These metrics will be used for system evaluation.\n\n"
        output += "\tPrecision = " + str(precision * 100.0) + "\n"
        output += "\tRecall = " + str(recall * 100.0) + "\n"
        output += "\tF1 Score = " + str(f1 * 100.0) + "\n"
        output += "\tNumber of Audio Files = " + str(len(self.labels_dict.keys())) + "\n\n"

        return output
