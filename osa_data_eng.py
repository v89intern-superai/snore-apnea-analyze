import os
import time
import librosa
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from typing import List, Tuple
import pandas as pda
import random
import json


class OSAData:
    def __init__(self, data_path: str):
        self.path_list = glob(os.path.join(data_path, "*"))
        self.spo2 = []
        self.heart_rate = []
        self.flow_dr = []

        self.spo2_avg = []
        self.hr_avg = []
        self.spo2_var = []
        self.hr_var = []

        self.osa = []
        self.csa = []
        self.msa = []
        self.hypo = []
        self.sleep_start_time = None
        self.patient_data = []
        self.patient_audio = []

    @staticmethod
    def str2seconds(time_str: str) -> int:
        """Converts a string in the format hh:mm:ss to seconds. Adjusts hours if less than 13."""
        parts = time_str.split(":")
        h, m = map(int, parts[:2])  # Extract hours, minutes, and seconds

        s_part = parts[2]
        if '.' in s_part:
            s, ms = s_part.split('.')
            s = int(s)
            ms = float(f"0.{ms}")  # Convert fractional part to float
        else:
            s = int(s_part)
            ms = 0.0

        # Adjust hour if it's less than 12(this logic may not be suitable for all cases)
        if h < 12:
            h += 24
        return h * 3600 + m * 60 + s + ms

    @staticmethod
    def is_in_problem_interval(x, intervals):
        """Check if a timestamp x falls within any of the given awake intervals."""
        for interval in intervals:
            if interval[0] <= x <= interval[1]:
                return True
        return False

    @staticmethod
    def load_single_data(files: list) -> Tuple[List, List, List, List, List, List, List]:
        """
        Load a single patient data from a list of files and parse it into multiple lists.

        Parameters:
            files (List[str]): A list of file paths containing various types of patient physiological data.
                The list may include:
                - Mobile phone recordings
                - Dictaphone recordings
                - Event annotations
                - Blood oxygen (SpO2) data
                - Heart rate (HR) data
                - Airflow data (if available)

        Returns:
            Tuple[List, List, List, List, List, List, List]:
                A tuple containing seven lists, each storing different types of data:
                - spo2 (List): Oxygen saturation (SpO2) data.
                - heart_rate (List): Heart rate (HR) data.
                - flow_dr (List): Respiratory flow derivation rate (Flow DR) data.
                - [Other four lists]: Specify what these lists represent based on your context.

        Notes:
            - Ensure that the provided file paths are valid and the file formats are correct.
            - This method assumes all files have the same structure and format.
            - If the file format or content does not match expectations, parsing errors may occur.
        """

        awake = None
        record_start = None

        data_json = {
            "audio_path": None,
            "sr": None,
            'spo2': None,
            'heart_rate': None,
            'flow_dr': None,
            'sleep_stage': None,
            'annotation': None,
        }

        start_time = time.time()
        for file in files:
            if "\\" in file:
                file = file.replace("\\", "/")

            # load annotation and annotate start time
            if file.find('annotation') != -1:
                # Extract annotation
                data_json['annotation'] = OSAData.extract_event_annotations(file)
                awake = data_json['annotation']["awake_intervals"]
                record_start = data_json['annotation']["record_start"]  # "record_start" is used to align the audio timeline

        # get audio, SpO2, Heart Rate, Flow_DR and sleep_stage data
        for file in files:
            if file.find('phone') != -1:
                # Convert stereo to mono
                wav_phone, sr = librosa.load(file, sr=None)
                data_json['audio_path'] = file
                data_json['sr'] = sr
                # print(f"simple rate: {sr} Hz")
            # elif file.find('recorder') != -1:
            #     wav_records, sr = librosa.load(mono_path, sr=None)
            elif file.find('SpO2') != -1:
                data_json['spo2'] = OSAData.extract_spo2(file, record_start, awake)

            elif file.find('HR') != -1:
                data_json['heart_rate'] = OSAData.extract_HR(file, record_start)

            elif file.find('Flow_DR') != -1:
                data_json['flow_dr'] = OSAData.extract_flow_dr(file, record_start)

            elif file.find('sleep_stage') != -1:
                data_json['sleep_stage'] = OSAData.extract_sleep_stage(file, record_start)

        end_time = time.time()
        print(f'cost {end_time - start_time:.2f} seconds')

        return data_json, wav_phone

    @staticmethod
    def extract_event_annotations(file_path: str) -> Tuple[List, List, List, List, List]:
        """Extracts event annotations from the given file path."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # print('check check!')
            return data
        except FileNotFoundError:
            print(f"错误: 文件 {file_path} 不存在")
            return None
        except json.JSONDecodeError:
            print(f"错误: 文件 {file_path} 不是有效的 JSON 格式")
            return None
        except Exception as e:
            print(f"错误: 读取文件时发生未知错误: {e}")
            return None

    @staticmethod
    def extract_spo2(spo_path, r_start, awakes):
        """
            Process SpO2 data from a CSV file and filter out values based on timestamps.

            param spo_path: Path to the CSV file containing SpO2 data.
            param awakes: Intervals during which the subject was awake.
            return: A list of filtered SpO2 values.
        """
        # spo2 = []

        # Read and process the SpO2 column and convert time to seconds
        SpO2_temp = pd.read_csv(spo_path, usecols=[1]).iloc[:, 0]
        SpO2_seconds = SpO2_temp.apply(OSAData.str2seconds) - r_start

        # Read and process the SpO2 value column
        SpO2_list = pd.read_csv(spo_path, usecols=[2]).iloc[:, 0]
        SpO2_value = [0 if x == '-' else float(x) for x in SpO2_list]

        # Create DataFrame with processed data
        # df_spo2 = pd.DataFrame({'timestamp': SpO2_seconds, 'spo2': SpO2_value})

        SpO2_data = {
            'SpO2_temp': SpO2_seconds.tolist(),
            'SpO2_value': SpO2_value
        }

        return SpO2_data

    @staticmethod
    def extract_HR(HR_path, r_start):
        """
        Process heart rate data from a CSV file, filter based on timestamps,
        and check for abnormal heart rates (below 40).

        param HR_path: Path to the CSV file containing heart rate data.
        param awakes: Intervals during which the subject was awake.
        return: A tuple containing a list of filtered heart rate values and a list of abnormal heart rates.
        """
        heart_rate = []

        # Read and process the heart rate column and convert time to seconds
        heart_rate_temp = pd.read_csv(HR_path, usecols=[1]).iloc[:, 0]
        heart_rate_seconds = heart_rate_temp.apply(OSAData.str2seconds) - r_start

        # Read and process the heart rate value column
        heart_rate_list = pd.read_csv(HR_path, usecols=[2]).iloc[:, 0]
        heart_rate_value = [0 if x == '-' else float(x) for x in heart_rate_list]

        # Create DataFrame with processed data
        # df_hr = pd.DataFrame({'timestamp': heart_rate_seconds, 'hr': heart_rate_value})

        HR_data = {
            'HR_temp': heart_rate_seconds.tolist(),
            'HR_value': heart_rate_value
        }

        # Filter out rows where the timestamp is within problem intervals or heart rate value is 0
        # mask_hr = ~df_hr['timestamp'].apply(lambda x: OSAData.is_in_problem_interval(x, awakes))
        # filtered_df_hr = df_hr[mask_hr]
        # filtered_df_hr = filtered_df_hr[filtered_df_hr['hr'] != 0]
        #
        # # Extend the output list with the filtered heart rate values
        # heart_rate.extend(df_hr)
        #
        # # Check for abnormal heart rates
        # abnormal_heart_rate = [num for num in filtered_df_hr['hr'].values if int(num) < 40]
        # if abnormal_heart_rate:
        #     print(
        #         f'Minimum abnormal heart rate is: {min(abnormal_heart_rate)}, all abnormal hr: {abnormal_heart_rate}'
        #     )

        return HR_data

    @staticmethod
    def extract_flow_dr(flow_path, r_start):
        """
        Process flow derivative data from a CSV file and filter out values based on timestamps.

        param flow_path: Path to the CSV file containing flow derivative data.
        param awakes: Intervals during which the subject was awake.
        return: A list of filtered flow derivative values.
        """
        # flow_dr = []

        # Read and process the flow derivative column and convert time to seconds
        flow_dr_temp = pd.read_csv(flow_path, usecols=[1]).iloc[:, 0]
        flow_dr_seconds = flow_dr_temp.apply(OSAData.str2seconds) - r_start

        # Read and process the flow derivative value column
        flow_dr_list = pd.read_csv(flow_path, usecols=[2]).iloc[:, 0]
        flow_dr_value = [0 if x == '-' else float(x) for x in flow_dr_list]

        # Create DataFrame with processed data
        # df_flow_dr = pd.DataFrame({'timestamp': flow_dr_seconds, 'flow_dr': flow_dr_value})

        Flow_data = {
            'Flow_temp': flow_dr_seconds.tolist(),
            'Flow_value': flow_dr_value
        }

        # Filter out rows where the timestamp is within problem intervals or flow derivative value is 0
        # mask_flow_dr = ~df_flow_dr['timestamp'].apply(
        #     lambda x: OSAData.is_in_problem_interval(x, awakes))
        # filtered_df_flow_dr = df_flow_dr[mask_flow_dr]
        # # filtered_df_flow_dr = filtered_df_flow_dr[filtered_df_flow_dr['flow_dr'] != 0]
        #
        # # Extend the output list with the filtered flow derivative values
        # flow_dr.extend(df_flow_dr)

        return Flow_data

    @staticmethod
    def extract_sleep_stage(sleep_stage_path, r_start):
        # Read and process the flow derivative column and convert time to seconds
        sleep_stage_temp = pd.read_csv(sleep_stage_path, usecols=[1]).iloc[:, 0]
        sleep_stage_seconds = sleep_stage_temp.apply(OSAData.str2seconds) - r_start

        # Read and process the flow derivative value column
        sleep_stage_stature = pd.read_csv(sleep_stage_path, usecols=[2]).iloc[:, 0].astype(str).tolist()
        # flow_dr_value = [0 if x == '-' else float(x) for x in sleep_stage_stature]

        # Create DataFrame with processed data
        # df_flow_dr = pd.DataFrame({'timestamp': flow_dr_seconds, 'flow_dr': flow_dr_value})

        Sleep_stage_data = {
            'Sleep_stage_temp': sleep_stage_seconds.tolist(),
            'Sleep_stage_stature': sleep_stage_stature
        }
        return Sleep_stage_data

    def load_data(self):
        """
        The data is loaded in a patient-by-patient manner.
        You can process the data of each patient individually before aggregating it,
        or you can wait until all the data is loaded and then perform the data aggregation on the entire dataset.
        """
        patient_num = 1
        # patient_data = []
        for path in self.path_list:
            print(f'Data for patient {patient_num} saved at: {path}')
            if os.path.isdir(path):
                file_list = glob(os.path.join(path, "*"))

                # load wav, spo2, hr, flow_dr, annotation, and get patient awake duration, every timestamp use s.
                data_jsons, wav_data = self.load_single_data(file_list)
                self.patient_data.append(data_jsons)
                self.patient_audio.append(wav_data)

                # You can add the relevant data processing code in the following places

                # You can create a dataset for model training in the following places

                patient_num += 1
            elif os.path.isfile(path):
                continue

        print("All patient is OK!")
        return self.patient_data

    def generate_boxplots(self, output_path):
        for data in self.patient_data:
            awake_interval = data["annotation"]["awake_intervals"]

            if data["spo2"]:
                # Create DataFrame with processed data
                df_spo2 = pd.DataFrame({'timestamp': data["spo2"]["SpO2_temp"], 'spo2': data["spo2"]["SpO2_value"]})
                # Filter out rows where the timestamp is within problem intervals or SpO2 value is 0
                mask_spo2 = ~df_spo2['timestamp'].apply(lambda x: OSAData.is_in_problem_interval(x, awake_interval))
                filtered_df_spo2 = df_spo2[mask_spo2]
                filtered_df_spo2 = filtered_df_spo2[filtered_df_spo2['spo2'] != 0]
                spo2_values = filtered_df_spo2['spo2'].values
                spo2_avg = spo2_values.mean()
                spo2_var = spo2_values.var()
                self.spo2.append(filtered_df_spo2['spo2'].values)
                self.spo2_avg.append(spo2_avg)
                self.spo2_var.append(spo2_var)

            if data["heart_rate"]:
                # Create DataFrame with processed data
                df_HR = pd.DataFrame({'timestamp': data["heart_rate"]["HR_temp"], 'heart_rate': data["heart_rate"]["HR_value"]})
                # Filter out rows where the timestamp is within problem intervals or Heart Rate value is 0
                mask_HR = ~df_HR['timestamp'].apply(lambda x: OSAData.is_in_problem_interval(x, awake_interval))
                filtered_df_HR = df_HR[mask_HR]
                filtered_df_HR = filtered_df_HR[filtered_df_HR['heart_rate'] != 0]
                hr_values = filtered_df_HR['heart_rate'].values
                hr_avg = hr_values.mean()
                hr_var = hr_values.var()
                self.heart_rate.append(filtered_df_HR['heart_rate'].values)
                self.hr_avg.append(hr_avg)
                self.hr_var.append(hr_var)

            if data["flow_dr"]:
                # Create DataFrame with processed data
                df_flow = pd.DataFrame({'timestamp': data["flow_dr"]["Flow_temp"], 'flow_dr': data["flow_dr"]["Flow_value"]})
                # Filter out rows where the timestamp is within problem intervals or Flow_DR value is 0
                mask_flow = ~df_flow['timestamp'].apply(lambda x: OSAData.is_in_problem_interval(x, awake_interval))
                filtered_df_flow = df_flow[mask_flow]
                self.flow_dr.append(filtered_df_flow['flow_dr'].values)

            print("check point!")

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 18))

        all_spo2_avg = np.array(self.spo2_avg).mean()
        all_spo2_std = np.array(self.spo2_avg).std()
        all_spo2_var_mean = (all_spo2_std / all_spo2_avg) * 100

        # all_hr_avg = np.array(self.hr_avg).mean()

        all_hr_avg = np.array(self.hr_avg).mean()
        all_hr_std = np.array(self.hr_avg).std()
        all_hr_var_mean = (all_hr_std / all_hr_avg) * 100

        # all_hr_avg = np.array(self.hr_avg).mean()

        print("{} patient SpO2 avg value is {}".format(len(self.spo2_avg), all_spo2_avg))

        if self.spo2:
            ax1.boxplot(self.spo2)
            ax1.set_ylabel('Spo2')
        if self.heart_rate:
            ax2.boxplot(self.heart_rate)
            ax2.set_ylabel('Heart Rate')
        if self.flow_dr:
            ax3.boxplot(self.flow_dr)
            ax3.set_ylabel('Flow Dr')

        ax1.set_xlabel('Patient ID')

        # plt.show()

        plt.savefig(output_path, dpi=300)
        plt.close()
        print("Boxplots generated and saved to:", output_path)

    def vis_data(self):
        self.load_data()
        self.generate_boxplots()
        print("check point vis_data!")

    def vis_single_patient(self, save_dir):
        self.load_data()
        r_x = random.randrange(1, 51)
        patient = self.patient_data[35]

        # wav, spo2, hr, flow, sleep_stage
        # Convert stereo to mono
        wav_path = patient["audio_path"]
        wav_data, sr = librosa.load(wav_path, sr=None)

        spo2_temp = patient["spo2"]["SpO2_temp"]
        spo2_value = np.array(patient["spo2"]["SpO2_value"])
        # Outlier filtering
        non_zero_elements = spo2_value[np.nonzero(spo2_value)]

        # Calculate the mean of non-zero elements
        mean_non_zero = non_zero_elements.mean()
        # replace
        spo2_value[spo2_value == 0] = int(mean_non_zero)

        hr_temp = patient["heart_rate"]["HR_temp"]
        hr_value = np.array(patient["heart_rate"]["HR_value"])
        # Outlier filtering
        non_zero_elements_2 = hr_value[np.nonzero(hr_value)]

        # Calculate the mean of non-zero elements
        mean_non_zero_2 = non_zero_elements_2.mean()
        # replace
        hr_value[hr_value == 0] = int(mean_non_zero_2)

        flow_temp = patient["flow_dr"]["Flow_temp"]
        flow_value = patient["flow_dr"]["Flow_value"]

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 18))
        if spo2_temp:
            spo2_time = np.arange(spo2_temp[0], spo2_temp[-1] + 1)
            ax2.plot(spo2_time, spo2_value, linewidth=0.8)
            ax2.set_xlim(0, 40000)
            ax2.set_ylabel("SpO2 (%)")
            ax2.set_xlabel('Time (s)')
        if hr_temp:
            hr_time = np.arange(hr_temp[0], hr_temp[-1] + 1)
            ax3.plot(hr_time, hr_value, linewidth=0.8)
            ax3.set_xlim(0, 40000)
            ax3.set_ylabel("Heart Rate (bpm)")
            ax3.set_xlabel('Time (s)')
        if flow_temp:
            flow_time = np.arange(flow_temp[0], flow_temp[-1] + 0.5, 0.5)
            ax4.plot(flow_time, flow_value, linewidth=0.8)
            ax4.set_xlim(0, 40000)
            ax4.set_ylabel("Flow DR")
            ax4.set_xlabel('Time (s)')

        ds_data = wav_data[::10]
        voice_time = np.arange(len(ds_data)) / 4800
        ax1.set_xlabel('Time (s)')

        ax1.plot(voice_time, ds_data, linewidth=0.8)
        ax1.set_xlim(0, 40000)
        ax1.set_ylim(-1, 1)

        # plt.title("check check!")

        save_path = os.path.join(save_dir, 'vis_single.png')

        # plt.show()
        plt.savefig(save_path, dpi=300)
        plt.close()

        print("check point vis_single_patient")


if __name__ == "__main__":
    processor = OSAData(
        data_path="",
    )
    processor.load_data()
    processor.vis_single_patient(save_dir='')

