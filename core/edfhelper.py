from pyedflib import EdfReader
import numpy as np
import datetime
import os
import re
from functools import reduce
from collections import OrderedDict

class ChbFile:
    """
    Edf reader using pyedflib.
    Using in in preprocess.
    """
    def __init__(self, filename, freq, split_size, patient_id=None):
        self._filename = filename
        self.freq = freq
        self._patient_id = patient_id

        with EdfReader(filename) as file:
            self.signals = self.readSignals(file)
            self.duration = file.getFileDuration()
            self.start_time = file.getStartdatetime()
            
        
            
    def readSignals(self, file):
        using_names = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8']
        signals_names = file.getSignalLabels()
        name_indexes = list(map(lambda x: signals_names.index(x), using_names))
        n = len(name_indexes)
        assert n == len(using_names)
        data = []
        for i in name_indexes:
            data.append(file.readSignal(i))
        sigbufs = np.vstack(data)
        return sigbufs
    
    def getSignals(self):
        return self.signals
    
    def get_lables(self):
        return self.lables
    
    def save_split(self, seconds):
        self.split_signals = self.split(seconds)
    
    def split(self, seconds):
        ostatok = self.signals.shape[1] % (self.freq * seconds)
        if not ostatok == 0:
            self.signals = self.signals[:-ostatok]
        num_of_chunks = self.signals.shape[1] / (self.freq * seconds) 
        return np.hsplit(self.signals, num_of_chunks)
    
    def get_duration(self):
        """
        Returns the file duration in seconds
        """
        return self.duration
    
    def get_start_time(self):
        return self.start_time
    

    def set_labels(self, seizures):
        self.labels = np.zeros(self.signals.shape[1])

        seizures_indecies = []
        #short pre-seizure (1)

        for s in seizures:
            #seizure (-1)
            s_start = s[0]
            s_end = s[1]
            ChbFile._set_time_period(self.labels, self.start_time, s_start, s_end, -1)

        # 1
        for s in seizures:
            s_start = s[0] - datetime.timedelta(hours=1)
            s_end = s[0]
            pre_seizure_indx = ChbFile._set_time_period(self.labels, self.start_time, s_start, s_end, 1)
            seizures_indecies.append(pre_seizure_indx)

        # -1
        for s in seizures:
            #long pre-seizure
            s_start = s[0] - datetime.timedelta(hours=4)
            s_end = s[0] - datetime.timedelta(hours=1)
            ChbFile._set_time_period(self.labels, self.start_time, s_start, s_end, -1)

            #after seizure
            s_start = s[1]
            s_end = s[1] + datetime.timedelta(hours=4)
            ChbFile._set_time_period(self.labels, self.start_time, s_start, s_end, -1)

        

        return seizures_indecies


    def _set_time_period(labels, start_time_labels, start_p, end_p, label):
        s_start_indx = int((start_p - start_time_labels).total_seconds() * 256)
        s_end_indx = int((end_p - start_time_labels).total_seconds() * 256)
        if s_start_indx < 0:
            s_start_indx = 0
        if s_end_indx < 0:
            s_end_indx = 0
        if s_start_indx > labels.shape[0] - 1:
            s_start_indx = labels.shape[0]
        if s_end_indx > labels.shape[0] - 1:
            s_end_indx = labels.shape[0]

        if s_start_indx != s_end_indx:
            if label == -1:
                labels[np.where(labels[s_start_indx:s_end_indx] != 1)[0] + s_start_indx] = label
            if label == 1:
                free_indexes = np.where((labels[s_start_indx:s_end_indx] != 1) & (labels[s_start_indx:s_end_indx] != -1))[0]
                if free_indexes.shape[0] == 0:
                    return None 
                assert np.where(np.diff(free_indexes) != 1)[0].shape[0] == 0
                old_st_indx = s_start_indx
                s_start_indx = old_st_indx + free_indexes[0]
                s_end_indx = old_st_indx + free_indexes[-1] + 1
                labels[s_start_indx:s_end_indx] = label

            return s_start_indx, s_end_indx

        return None


                        
    def delete_not_useful_data(self):
        indeces = np.where(self.lables==-1)
        self.lables = np.delete(self.lables, indeces)
        self.signals = np.delete(self.signals, indeces, 1)

    def get_chunks_paperv(self, seizures_indecies):
        indeces = np.where(np.diff(self.labels) != 0)[0] + 1

        tmp_indeces = [0] + indeces.tolist()
        seizure_ch_indeces = []
        for i in range(len(seizures_indecies)):
            seizure_ch_indeces.append([])
        for i in range(len(tmp_indeces)):
            if self.labels[tmp_indeces[i]] != 1:
                continue
            for j in range(len(seizures_indecies)):
                if seizures_indecies[j] == None:
                    continue
                if tmp_indeces[i] >= seizures_indecies[j][0] and tmp_indeces[i] < seizures_indecies[j][1]:
                    seizure_ch_indeces[j] += [i]
                    break

        

        n_signals = np.split(self.signals, indeces, 1)
        n_labels = np.split(self.labels, indeces)

        seizures_signals = []
        seizures_labels = []
        for id, s in enumerate(seizure_ch_indeces):
            seizures_signals.append([])
            seizures_labels.append([])
            for i in s:
                seizures_signals[id].append(n_signals[i])
                seizures_labels[id].append(n_labels[i])

        for i in sorted(reduce(lambda x,y :x+y ,seizure_ch_indeces), reverse=True):
            del n_signals[i]
            del n_labels[i]



        del_indeces = [i for i, x in enumerate(n_labels) if x[0] == -1]
        for index in sorted(del_indeces, reverse=True):
            del n_signals[index]
            del n_labels[index]
        
        return n_signals, n_labels, seizures_signals, seizures_labels

    def get_5s_chunks(self, signals, labels):
        n_signals = []
        n_labels = []
        for i in range(len(signals)):
            c_signal = signals[i]
            if c_signal.shape[1] < 1280:
                continue
            if c_signal.shape[1] % 1280 != 0:
                c_signal = c_signal[:, :-(c_signal.shape[1] % 1280)]
            n_signals += np.split(c_signal, c_signal.shape[1] / 1280, 1)

            
            n_labels += [labels[i][0]] * (labels[i].shape[0] // 1280)

        return n_signals, n_labels

    def get_5s_chunks_simple(self, signals):
        n_signals = []
        n_labels = []
        for i in range(len(signals)):
            c_signal = signals[i]
            if c_signal.shape[1] < 1280:
                continue
            if c_signal.shape[1] % 1280 != 0:
                c_signal = c_signal[:, :-(c_signal.shape[1] % 1280)]
            n_signals += np.split(c_signal, c_signal.shape[1] / 1280, 1)

            n_labels += [1] * (c_signal.shape[1] // 1280)

        return n_signals, n_labels

    def get_raw_5s_chunks(self):
        if self.signals.shape[1] < 1280:
            return None
        signal = self.signals
        if self.signals.shape[1] % 1280 != 0:
            signal = self.signals[:, :-(self.signals.shape[1] % 1280)]
        n_signals = np.split(signal, signal.shape[1] / 1280, 1)
        return np.asarray(n_signals)


class Patient:
    """
    Class to encapsulate patient information. 
    Using in preprocess.
    """

    def __init__(self, path, patient_id):
        self.patient_id = patient_id
        self.folder_name = os.path.basename(os.path.normpath(path))
        self._filename = os.path.join(path, self.folder_name + '-summary.txt')
        self._file = open(self._filename, 'r')
        self._parse_file(self._file)
        self._file.close()
        self.files = []
        self.seizures = []
        
        for key, file in self._metadata_store.items():
            full_path = os.path.join(path, key)
            
            chb = ChbFile(full_path, self.get_freq(), 5, patient_id)
            self.files.append(chb)
            
            for seizure in file['seizure_intervals']:
                start_time_seizure = chb.get_start_time() + datetime.timedelta(seconds=seizure[0])
                end_time_seizure = chb.get_start_time() + datetime.timedelta(seconds=seizure[1])
                self.seizures.append((start_time_seizure, end_time_seizure))

           
        self.signals_5s = []
        self.labels_5s = []

        self.seizure_signals_5s = []
        self.seizure_labels_5s = []
        for _ in range(len(self.seizures)):
            self.seizure_signals_5s.append([])
            self.seizure_labels_5s.append([])

        i = 0
        for chb in self.files:
            i += 1
            seizures_indeces = chb.set_labels(self.seizures)
            s, l, ss, _ = chb.get_chunks_paperv(seizures_indeces)
            s, l = chb.get_5s_chunks(s, l)
            for i in range(len(ss)):
                tmp_ss, tmp_sl = chb.get_5s_chunks_simple(ss[i])
                self.seizure_signals_5s[i] += tmp_ss
                self.seizure_labels_5s[i] += tmp_sl
            self.signals_5s += s
            self.labels_5s += l

        self.signals_5s = np.asarray(self.signals_5s)
        self.labels_5s = np.asarray(self.labels_5s)

        for i in range(len(self.seizures)):
            self.seizure_signals_5s[i] = np.asarray(self.seizure_signals_5s[i])
            self.seizure_labels_5s[i] = np.asarray(self.seizure_labels_5s[i])

        
            
            
    def get_all_data(self):
        all_signals = []
        for chb in self.files:
            signals = chb.get_raw_5s_chunks()
            all_signals.append(signals)
        return np.vstack(all_signals)

    def get_chunks_5s(self):
        return self.signals_5s, self.labels_5s, self.seizure_signals_5s, self.seizure_labels_5s       
    
    def _parse_file(self, file_obj):
        """
        Parse the file object
        :param file_obj: Opened file object
        """
        # Split file into blocks
        data = file_obj.read()
        blocks = data.split('\n\n')

        # Block zero
        self._frequency = self._parse_frequency(blocks[0])
        
        
        # Block one
        self._channel_names = self._parse_channel_names(blocks[1])

        # Block two-N
        self._metadata_store = self._parse_file_metadata(blocks[2:])
        


    def _parse_frequency(self, frequency_block):
        """
        Frequency block parsing with format
        'Data Sampling Rate: ___ Hz'
        :param frequency_block: Frequency block
        :return: Parses and returns the frequency value in Hz
        """
        pattern = re.compile("Data Sampling Rate: (.*?) Hz")
        result = pattern.search(frequency_block)
        # Check if there is a match or not
        if result is None:
            raise ValueError('Frequency block does not contain the correct string ("Data Sampling Rate: __ Hz")')
        result = int(result.group(1))
        return result
    
    def _parse_channel_names(self, channel_block):
        """
        Get channel names from the blocks
        :param channel_block: List of Channel names
        :return: Returns the channel names as a list of strings
        """
        # Split by line
        lines = channel_block.split('\n')
        pattern = re.compile("Channel [0-9]{1,}: (.*?)$")

        output_channel_list = []
        for line in lines:
            channel_name = pattern.search(line)
            if channel_name is not None:
                channel_name = channel_name.group(1)
                output_channel_list.append(channel_name)

        return output_channel_list
    
    
    def _parse_file_metadata(self, seizure_file_blocks):
        """
        Parse the file metadata list blocks to get the seizure intervals
        Note: These are not necessarily in file order, so always check against the filename before continuing.
        :param seizure_file_blocks: List of seizure file blocks
        """
        output_metadata = OrderedDict()
        for block in seizure_file_blocks:
            lines = block.split('\n')
            meta = self._parse_metadata(lines)
            if (meta is not None):
                output_metadata[meta['filename']] = meta
        return output_metadata
    
    def _parse_metadata(self, metadata_block):
        """
        Parse a single seizure metadata block
        """
        # Search first line for seizure file pattern
        pattern_filename = re.compile("File Name: (.*?)$")
        pattern_start_time = re.compile("File Start Time: (.*?)$")
        pattern_end_time = re.compile("File End Time: (.*?)$")
        pattern_seizures = re.compile("Number of Seizures in File: (.*?)$")
        pattern_seizure_start = re.compile("Seizure [0-9]{0,}[ ]{0,}Start Time: (.*?) seconds")
        pattern_seizure_end = re.compile("Seizure [0-9]{0,}[ ]{0,}End Time: (.*?) seconds")

        if pattern_filename.search(metadata_block[0]) is not None:
            file_metadata = dict()
            filename = pattern_filename.search(metadata_block[0]).group(1)
            file_metadata['filename'] = filename
            file_metadata['start_time'] = pattern_start_time.search(metadata_block[1]).group(1)
            file_metadata['end_time'] = pattern_end_time.search(metadata_block[2]).group(1)
            file_metadata['n_seizures'] = int(pattern_seizures.search(metadata_block[3]).group(1))
#             file_metadata['channel_names'] = self._channel_names
#             file_metadata['sampling_rate'] = self.get_sampling_rate()
            seizure_intervals = []
            for i in range(file_metadata['n_seizures']):
                seizure_start = int(pattern_seizure_start.search(metadata_block[4 + i * 2]).group(1))
                seizure_end = int(pattern_seizure_end.search(metadata_block[4 + i * 2 + 1]).group(1))
                seizure_intervals.append((seizure_start, seizure_end))
            file_metadata['seizure_intervals'] = seizure_intervals
            return file_metadata
        else:
            import warnings
            warnings.warn("Block didn't follow the pattern for a metadata file block", Warning)
            # Check channel names
            try:
                self._channel_names = self._parse_channel_names("\n".join(metadata_block))
            except Exception as e:
                print('Failed to parse block as a channel names block')
                raise e
        return None
        
    def get_file_metadata(self):
        """
        Get the metadata for all of the files
        """
        return self._metadata_store
    
    def get_freq(self):
        return self._frequency
    
    def get_seizures(self):
        return self.seizures
    
    def get_files(self):
        return self.files