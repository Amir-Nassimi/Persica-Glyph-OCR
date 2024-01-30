import numpy as np


class OCRPostProcessor:
    def __init__(self, id_to_name):
        self.id_to_name = id_to_name

    def handle_close_duplicated_char(self, x1x2_arr, x_arr, confs_arr, preds):
        try:
            dists = np.diff(x1x2_arr)
            indices = np.where(dists <= 1)[0]
            for index in indices:
                if confs_arr[index] > confs_arr[index + 1]:
                    x_arr = np.delete(x_arr, index + 1)
                    confs_arr = np.delete(confs_arr, index + 1)
                    preds = np.delete(preds, index + 1)
                else:
                    x_arr = np.delete(x_arr, index)
                    confs_arr = np.delete(confs_arr, index)
                    preds = np.delete(preds, index)
        except IndexError as e:
            print(f"An IndexError occurred: {e}")

        return x_arr, confs_arr, preds

    def handle_missed_character(self, x_arr, preds):
        try:
            dists = np.diff(x_arr)
            min_x = np.min(dists)
            missing_char = -1
            asterisk_char = '*'

            if len(x_arr) == 7:
                indices = np.where(dists / min_x >= 2)[0]

                if len(indices) != 0:
                    preds = np.insert(preds, indices[0] + 1, missing_char)
                else:
                    if type(preds[0]) == int and type(preds[1]) == str:
                        preds = np.insert(preds, 0, missing_char)
                    else:
                        preds = np.append(preds, missing_char)

            elif len(x_arr) == 6:
                indices_1 = np.where((dists / min_x >= 3))[0]
                indices_2 = np.where((dists / min_x >= 2) & (dists / min_x < 3))[0]

                if len(indices_1) != 0 and len(indices_2) == 0:
                    preds = np.insert(preds, indices_1[0] + 1, missing_char)
                    preds = np.insert(preds, indices_1[0] + 2, missing_char)

                elif len(indices_1) == 0 and len(indices_2) != 0:
                    if len(indices_2) == 2:
                        preds = np.insert(preds, indices_2[0] + 1, missing_char)
                        preds = np.insert(preds, indices_2[1] + 2, missing_char)

                    elif len(indices_2) == 1:
                        preds = np.insert(preds, indices_2[0] + 1, missing_char)

                        if type(preds[0]) == int and preds[1] == missing_char and type(preds[2]) == int or \
                                type(preds[0]) == int and type(preds[1]) == str:
                            preds = np.insert(preds, 0, missing_char)
                        else:
                            preds = np.append(preds, missing_char)

                elif len(indices_1) == 0 and len(indices_2) == 0:
                    if type(preds[0]) == str:
                        preds = np.insert(preds, 0, missing_char)
                        preds = np.insert(preds, 0, missing_char)
                    elif type(preds[0]) == int and type(preds[1]) == str:
                        preds = np.insert(preds, 0, missing_char)
                        preds = np.append(preds, missing_char)
                    else:
                        preds = np.append(preds, missing_char)
                        preds = np.append(preds, missing_char)
                else:
                    preds = np.insert(preds, indices_1[0] + 1, missing_char)
                    preds = np.insert(preds, indices_2[0] + 2, missing_char)

            else:
                preds = np.array([missing_char] * 8, dtype=object)

            star_indices = np.where(preds == missing_char)

            # Continue removing '*' elements while the array length is greater than 8
            while len(preds) > 8 and len(star_indices[0]) > 0:
                preds = np.delete(preds, star_indices[0][0])
                star_indices = np.where(preds == missing_char)

            preds[preds == missing_char] = asterisk_char
            return preds

        except (ValueError, TypeError, IndexError) as e:
            error_msg = f"{type(e).__name__}: An error occurred in handle_missed_character."
            print(error_msg)
            return None

    def working_with_results(self, results):

        """
        Process detection results and organize them into a formatted list.

        Args:
        - results (list): List of detection results, where each result contains information
                        about predicted boxes, classes, and confidences.
        - id_to_name (dict): Dictionary mapping class IDs to corresponding names.

        Returns:
        - detection_list (list): A formatted list representing the organized and sorted
                                detection results. Each element in the list corresponds to
                                a part of a detected sequence.

        The method takes detection results and organizes them into a standardized list of
        character sequences based on their spatial arrangement. It extracts relevant information
        such as bounding box coordinates, confidence scores, and class predictions. The results
        are sorted based on the x-coordinate of the center of the bounding boxes to ensure
        proper sequencing.

        The method also handles cases where characters may be close or missing. If the total
        number of characters is not 8, it attempts to reconstruct the missing characters to
        form a complete sequence.

        The final formatted list, 'detection_list', represents the detected sequence in a
        structured manner, with each element corresponding to a part of the sequence.
        """

        x_coors_list = []
        for r in results:
            preds_tensor = r.boxes.cls
            preds_arr = preds_tensor.detach().cpu().numpy()

            confs_tensor = r.boxes.conf
            confs_arr = confs_tensor.detach().cpu().numpy()

            coordination = r.boxes.xyxy
            for c in coordination:
                x1, y1, x2, y2 = map(int, c)
                x_coors_list.append(x1 + x2)

        x_coors_arr = np.array(x_coors_list)
        x_center_arr = (x_coors_arr / 2).astype(int)

        sorted_indices = np.argsort(x_center_arr)
        sorted_x_center_arr = x_center_arr[sorted_indices]
        sorted_x_coors_arr = x_coors_arr[sorted_indices]
        sorted_preds_arr = preds_arr[sorted_indices]
        sorted_confs_arr = confs_arr[sorted_indices]

        sorted_preds_names = [self.id_to_name[int(key)] for key in sorted_preds_arr]
        sorted_preds_names = np.array(sorted_preds_names, dtype=object)

        sorted_x_center_arr, sorted_confs_arr, sorted_preds_names = self.handle_close_duplicated_char(
            sorted_x_coors_arr, sorted_x_center_arr, sorted_confs_arr, sorted_preds_names)

        if len(sorted_x_center_arr) != 8:
            preds_arr_with_missing_char = self.handle_missed_character(sorted_x_center_arr, sorted_preds_names)
        else:
            preds_arr_with_missing_char = sorted_preds_names

        formatted_result = self.format_result(preds_arr_with_missing_char)
        return formatted_result

    def format_result(self, preds):
        if len(preds) == 8:
            p1 = str(preds[0]) + str(preds[1])
            p2 = str(preds[2])
            p3 = str(preds[3]) + str(preds[4]) + str(preds[5])
            p4 = "-"
            p5 = str(preds[6]) + str(preds[7])
        else:
            # Fallback or error handling
            p1 = p2 = p3 = p4 = p5 = "Error"

        return [p1, p2, p3, p4, p5]
