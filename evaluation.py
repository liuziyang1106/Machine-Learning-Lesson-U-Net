################################################################################
##                          Segmentation Evaluation                           ##
##  Compute specified performance metrics (only support dice coefficient,     ##
##  intersection over union, Matthew's correlation coefficient,accuracy,      ##
##  Hausdorff distance, sensitivity, precision, and F measure) between the    ##
##  computer segmentation results and radiologist segmentation results.       ##
##  Visualize the results on the original CT scan and export both numerical   ##
##  result and overlaid image file to a customized location.                  ##
################################################################################


import SimpleITK as sitk
import numpy as np
import os
import cv2
import pandas as pd

from collections import OrderedDict


class SegEval(object):
    def __init__(self, comp_path, radio_path):
        self.comp_path = comp_path
        self.radio_path = radio_path
        self.comp_img = None
        self.radio_img = None
        self.filename = []
        self.metrics = []
        self.eval_results = []
        self.visual_results = []

    ##################################
    # Preprocessing Utility Functions #
    ##################################

    def __get_filename(self, file1, file2):
        """
        Extract the longest common prefix of two file names as the file name of the 
        new exported image.
        """

        n1 = len(file1)
        n2 = len(file2)
        result = ""
        j = 0
        i = 0
        while (i <= n1 - 1 and j <= n2 - 1):
            if (file1[i] != file2[j]):
                break
            result += (file1[i])
            i += 1
            j += 1
        return result


    def __read_folder(self, folder_path):
        file_paths = []
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if (os.path.isfile(file_path)):
                file_paths.append(file_path)
        list.sort(file_paths)

        images = []
        for path in file_paths:
            images.append(sitk.ReadImage(path, sitk.sitkUInt16))

        return (file_paths, images)


    def __read_img_from_file(self, comp_path, radio_path):
        if (os.path.isfile(comp_path) and os.path.isfile(radio_path)):
            comp_img = sitk.ReadImage(comp_path, sitk.sitkUInt16)
            radio_img = sitk.ReadImage(radio_path, sitk.sitkUInt16)
            filename = [self.__get_filename(os.path.split(comp_path)[1], os.path.split(radio_path)[1])]
            return (comp_img, radio_img, filename)
        else:
            exit("Invalid file!")


    def __read_img_from_folder(self, comp_path, radio_path):
        filename = []
        if (os.path.isdir(comp_path) and os.path.isdir(radio_path)):
            (comp_file_paths, comp_img) = self.__read_folder(comp_path)
            (radio_file_paths, radio_img) = self.__read_folder(radio_path)
            if (len(comp_img) != len(radio_img)):
                exit("Numbers of files in folders are unmatched!")
            for i in range(len(comp_file_paths)):
                filename.append(self.__get_filename(os.path.split(comp_file_paths[i])[1], os.path.split(radio_file_paths[i])[1]))
            return (comp_img, radio_img, filename)
        else:
            exit("Invalid directory!")


    ###############################
    # Evaluation Utility Functions #
    ###############################

    def __valid_metrics(self, metrics_input):
        """
        Check the validity of the input for computing performance metrics.
        """

        if (len(metrics_input) == 0):
            return None
        else:
            metrics = {"acc", "dice", "hausdorff", "iou", "mcc", "sensitivity", "precision", "F measure", "volume similarity"}
            metrics_dict = OrderedDict()
            for metric in metrics_input:
                if (metric in metrics):
                    metrics_dict[metric] = 0
                else:
                    return None
            return metrics_dict


    def __valid_eval_img(self, img1, img2, filename):
        if not (np.count_nonzero(sitk.GetArrayFromImage(img1)) and np.count_nonzero(sitk.GetArrayFromImage(img2))):
            print("One or more image is empty!")
            return False
        elif (int(img1.GetSpacing()[0]*1000) != int(img2.GetSpacing()[0]*1000)):
            print(img1.GetSpacing(), img2.GetSpacing())
            print("Spacing of corresponding image files ({}) don't match!".format(filename))
            return False
        else: return True


    def __get_sens_spec(self, img1, img2):
        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
        overlap_measures_filter.Execute(img1, img2)
        Fp = overlap_measures_filter.GetFalsePositiveError()
        Tn = 1 - Fp
        Fn = overlap_measures_filter.GetFalseNegativeError()
        Tp = 1 - Fn
        return (Tp, Fp, Tn, Fn)


    def __get_dice(self, img1, img2, filename):
        if (self.__valid_eval_img(img1, img2, filename)):
            overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
            overlap_measures_filter.Execute(img1, img2)
            return overlap_measures_filter.GetDiceCoefficient()


    def __get_iou(self, img1, img2, filename):
        if (self.__valid_eval_img(img1, img2, filename)):
            overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
            overlap_measures_filter.Execute(img1, img2)
            return overlap_measures_filter.GetJaccardCoefficient()


    def __get_acc(self, img1, img2, filename):
        if (self.__valid_eval_img(img1, img2, filename)):
            (Tp, Fp, Tn, Fn) = self.__get_sens_spec(img1, img2)
            if (Tp + Fp + Tn + Fn == 0): 
                print("Attempt to divide by zero in calculating acc for {}!".format(filename))
                return float("NaN")
            else: 
                return (Tp + Tn) / (Tp + Fp + Tn + Fn)


    def __get_hausdorff(self, img1, img2, filename):
        if (self.__valid_eval_img(img1, img2, filename)):
            hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
            hausdorff_distance_filter.Execute(img1, img2)
            return hausdorff_distance_filter.GetHausdorffDistance()


    def __get_mcc(self, img1, img2, filename):
        if (self.__valid_eval_img(img1, img2, filename)):
            (Tp, Fp, Tn, Fn) = self.__get_sens_spec(img1, img2)
            if ((Tp + Fp) * (Tp + Fn) * (Tn + Fp) * (Tn + Fn) <= 0): 
                print("Attempt to square root of negative number or divide by zero in calculating mcc for {}!".format(filename))
                return float("NaN")
            else: 
                return (Tp * Tn - Fp * Fn) / np.sqrt((Tp + Fp) * (Tp + Fn) * (Tn + Fp) * (Tn + Fn))

    
    def __get_sensitivity(self, img1, img2, filename):
        if (self.__valid_eval_img(img1, img2, filename)):
            (Tp, _, _, Fn) = self.__get_sens_spec(img1, img2)
            if (Tp + Fn == 0):
                print("Attempt to divide by zero in calculating sensitivity for {}!".format(filename))
                return float("NaN")
            else:
                return Tp / (Tp + Fn)


    def __get_precision(self, img1, img2, filename):
        if (self.__valid_eval_img(img1, img2, filename)):
            (Tp, Fp, _, _) = self.__get_sens_spec(img1, img2)
            if (Tp + Fp == 0):
                print("Attempt to divide by zero in calculating precision for {}!".format(filename))
                return float("NaN")
            else:
                return Tp / (Tp + Fp)


    def __get_F_measure(self, img1, img2, filename):
        if (self.__valid_eval_img(img1, img2, filename)):
            (Tp, Fp, _, Fn) = self.__get_sens_spec(img1, img2)
            if (2 * Tp + Fp + Fn == 0):
                print("Attempt to divide by zero in calculating F measure for {}!".format(filename))
                return float("NaN")
            else:
                return (2 * Tp) / (2 * Tp + Fp + Fn)

    def __get_volume_similarity(self, img1, img2, filename):
        if (self.__valid_eval_img(img1, img2, filename)):
            volume_similarity_filter = sitk.LabelOverlapMeasuresImageFilter()
            volume_similarity_filter.Execute(img1, img2)
            return volume_similarity_filter.GetVolumeSimilarity()

    def __evaluation_by_img(self, img1, img2, filename, metrics, metrics_dict):
        results_metric = []
        results_value = []
        # Cannot compute for unmatched image files
        if not (self.__valid_eval_img(img1, img2, filename)):
            results_value = [float("NaN")] * len(metrics)
            for metric in metrics_dict:
                results_metric.append(metric)
        else:
            if ("acc" in metrics_dict):
                metrics_dict["acc"] = self.__get_acc(img1, img2, filename)
            if ("dice" in metrics_dict):
                metrics_dict["dice"] = self.__get_dice(img1, img2, filename)
            if ("hausdorff" in metrics_dict):
                metrics_dict["hausdorff"] = self.__get_hausdorff(
                    img1, img2, filename)
            if ("iou" in metrics_dict):
                metrics_dict["iou"] = self.__get_iou(img1, img2, filename)
            if ("mcc" in metrics_dict):
                metrics_dict["mcc"] = self.__get_mcc(img1, img2, filename)
            if ("sensitivity" in metrics_dict):
                metrics_dict["sensitivity"] = self.__get_sensitivity(img1, img2, filename)
            if ("precision" in metrics_dict):
                metrics_dict["precision"] = self.__get_precision(img1, img2, filename)
            if ("F measure" in metrics_dict):
                metrics_dict["F measure"] = self.__get_F_measure(img1, img2, filename)
            if ("volume similarity" in metrics_dict):
                metrics_dict["volume similarity"] = self.__get_volume_similarity(img1, img2, filename)
            for metric in metrics_dict:
                results_metric.append(metric)
                results_value.append(metrics_dict[metric])
        return (results_metric, results_value)


    ##################################
    # Visualization Utility Functions #
    ##################################

    def __read_CT_from_file(self, CT_path):
        if (os.path.isfile(CT_path)):
            CT_image = sitk.ReadImage(CT_path)
            return CT_image
        else:
            print("Invalid CT image file!")
            return None


    def __read_CT_from_folder(self, CT_path, file_number):
        if (os.path.isdir(CT_path)):
            (_, CT_image) = self.__read_folder(CT_path)
            if (len(CT_image) != file_number):
                print("Unmatched number of CT images!")
                return None
            else:
                return CT_image
        else:
            print("Invalid directory!")
            return None


    def __get_slice(self, filename):
        slice_number = int(input("Enter the slice number for {}: ".format(filename)))
        return slice_number


    def __first_overlay(self, computer, radiologist, CT, output_CT, output_radio, win_min=-1024, win_max=976):
        # Impose computerized contour on the CT slice
        computer_overlay = sitk.LabelMapContourOverlay(sitk.Cast(computer, sitk.sitkLabelUInt8), sitk.Cast(sitk.IntensityWindowing(CT, windowMinimum=win_min, windowMaximum=win_max), sitk.sitkUInt8), opacity=1, contourThickness=[2, 2])
        sitk.WriteImage(computer_overlay, output_CT)
        sitk.WriteImage(radiologist, output_radio)


    def __second_overlay(self, output_CT, output_radio):
        # Impose radiologist contour on the CT slice
        radiologist = cv2.imread(output_radio, cv2.IMREAD_GRAYSCALE)
        result_overlay = cv2.imread(output_CT)
        contours, _ = cv2.findContours(radiologist, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i, c in enumerate(contours):
            mask = np.zeros(radiologist.shape, np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            mean, _, _, _ = cv2.mean(radiologist, mask=mask)
            cv2.drawContours(result_overlay, [c], -1, (0, 0, 255), 1)
        os.remove(output_CT)
        os.remove(output_radio)
        return result_overlay


    def __visualize_img(self, comp_img, radio_img, CT_img, filename, export_path, win_min=-1024, win_max=976):
        slice_number = self.__get_slice(filename)
        computer = comp_img[:, :, slice_number]
        radiologist = radio_img[:, :, slice_number]
        CT = CT_img[:, :, slice_number]

        output_CT = export_path + "/" + filename + "_slice" + str(slice_number) + "_CT.png"
        output_radio = export_path + filename + "_radio.png"
        self.__first_overlay(computer, radiologist, CT, output_CT, output_radio)
        result_overlay = self.__second_overlay(output_CT, output_radio)
        outputName = "/" + filename + "_slice" + str(slice_number) + "_result.png"
        return (result_overlay, outputName)


    ##################
    # User Interface #
    ##################

    def evaluation_by_file(self, metrics):
        """
        Given a list of performance metrics, compute for the image files.
        """

        self.metrics = metrics
        metrics_dict = self.__valid_metrics(self.metrics)
        if (metrics_dict == None):
            print("Invalid metrics!")
            return
        else:
            (self.comp_img, self.radio_img, self.filename) = self.__read_img_from_file(self.comp_path, self.radio_path)
            print("Evaluating {}".format(self.filename[0]), end=" ")
            (results_metric, results_value) = self.__evaluation_by_img(self.comp_img, self.radio_img, self.filename[0], self.metrics, metrics_dict)
            self.eval_results = (results_metric, [results_value])
            print("...... complete!")


    def evaluation_by_folder(self, metrics):
        """
        Given a list of performance metrics, compute for each image files in a 
        folder.
        """

        self.metrics = metrics
        metrics_dict = self.__valid_metrics(metrics)
        if (metrics_dict == None):
            print("Invalid metrics!")
            return
        else:
            results_metric = []
            results_values = []
            (self.comp_img, self.radio_img, self.filename) = self.__read_img_from_folder(self.comp_path, self.radio_path)
            for i in range(len(self.comp_img)):
                print("Evaluating {}".format(self.filename[i]), end=" ")
                (results_metric, results_value) = self.__evaluation_by_img(self.comp_img[i], self.radio_img[i], self.filename[i], self.metrics, metrics_dict)
                results_values.append(results_value)
                print("...... complete!")
            self.eval_results = (results_metric, results_values)


    def export_eval_results(self, export_path, excel_name='/results.xlsx'):
        """
        Export the evaluation results (CSV) to a customized location.
        """

        (results_metric, results_value) = self.eval_results
        results_df = pd.DataFrame(data=results_value, index=self.filename, columns=results_metric)
        results_df.to_excel(export_path + excel_name)
        print("Evaluation results have been saved to {}!".format(export_path))


    def visualize_file(self, CT_path):
        """
        Given a slice number for image comparison, apply a colormap to the 
        contours of both computerized (green) and radiologist (red) 
        segmentation results and superimpose them on the original CT slice. The 
        contours are the edges of the labeled regions.
        """

        (self.comp_img, self.radio_img, self.filename) = self.__read_img_from_file(
            self.comp_path, self.radio_path)
        CT_img = self.__read_CT_from_file(CT_path)
        if (CT_img == None): return
        self.visual_results = [self.__visualize_img(
            self.comp_img, self.radio_img, CT_img, self.filename[0], os.path.split(self.radio_path)[0])]


    def visualize_folder(self, CT_path):
        """
        Visualize each computerized and raidologist segmentation results in
        folders on corresponding CT slices.
        """

        (self.comp_img, self.radio_img, self.filename) = self.__read_img_from_folder(
            self.comp_path, self.radio_path)
        CT_img = self.__read_CT_from_folder(CT_path, len(self.comp_img))
        for i in range(len(CT_img)):
            self.visual_results.append(self.__visualize_img(
                self.comp_img[i], self.radio_img[i], CT_img[i], self.filename[i], self.radio_path))

    
    def export_visual_results(self, export_path):
        """
        Export the visualization results (PNG) to a customized location.
        """

        for i in range(len(self.visual_results)):
            (result_overlay, outputName) = self.visual_results[i]
            cv2.imwrite(export_path + outputName, result_overlay)
        print("Visualization results have been saved to {}!".format(export_path))

if __name__ == "__main__":
    # test1 = SegEval("data/sub-041_predict_epoch15.nii.gz", "data/test/sub-041_dseg.nii.gz")
    # test1.evaluation_by_file(["dice", "acc", "hausdorff", "volume similarity"])
    # test1.export_eval_results("eval_results/")

    
    test2 = SegEval("runs/3DUnet_base_resample_40_dice_loss_64*128*128/pred_test_mask/mask"
                   ,"runs/3DUnet_base_resample_40_dice_loss_64*128*128/pred_test_mask/pred")
    test2.evaluation_by_folder(["dice", "acc", "hausdorff", "volume similarity"])
    test2.export_eval_results("runs/3DUnet_base_resample_40_dice_loss_64*128*128/")

    
