import os
import random

class DataSplitter():
    """
    Class for working with data when learning each patient separately.
    Includes splitting data on train, test, equalize classes samples.  
    Loads all in RAM.
    """


    def split_data_p(path_to_data, p, s_val):
        f_0 = DataSplitter.get_files_signals_0(path_to_data, p)
        f_1 = DataSplitter.get_files_signals_1(path_to_data, p)

        totat_1_len = 0
        for e in f_1:
            totat_1_len += len(e)

        random.shuffle(f_0)
        f_0_train = f_0[:(totat_1_len - len(f_1[s_val]))]
        f_0_test = f_0[-len(f_1[s_val]):]

        f_1_test = f_1[s_val]
        del f_1[s_val]
        f_1_train = []
        for s in f_1:
            f_1_train += s

        train = f_0_train + f_1_train
        test = f_0_test + f_1_test

        return train, test
        

    def get_files_signals_0(path_to_data, pat):
        result = []
        for p0 in os.listdir(os.path.join(path_to_data, str(pat), 'signals_0')):
            result.append((os.path.join(str(pat), 'signals_0', p0), 0))

        return result

    def get_files_signals_1(path_to_data, pat):
        pat = str(pat)
        result = []
        for s in sorted(os.listdir(os.path.join(path_to_data, pat, 'signals_1')), key=lambda x: int(x)):
            s_result = []
            for data in os.listdir(os.path.join(path_to_data, pat, 'signals_1', s)):
                s_result.append((os.path.join(pat, 'signals_1', s, data), 1))
            result.append(s_result)

        return result

    def get_seizures_number(path_to_data, p_indx):
        return len(os.listdir(os.path.join(path_to_data, str(p_indx), 'signals_1')))

    
    def save_to_file(path_to_save, p, s, train, test):
        with open(os.path.join(path_to_save, '{0}_{1}_train.txt'.format(p, s)), "w") as f:
            first_s = ''
            for e in train:
                f.write(first_s + e[0] + ' ' + str(e[1]))
                first_s = '\n'

        with open(os.path.join(path_to_save, '{0}_{1}_test.txt'.format(p, s)), "w") as f:
            first_s = ''
            for e in test:
                f.write(first_s + e[0] + ' ' + str(e[1]))
                first_s = '\n'

    def load_files(path_to_labels, path_to_data, p, s):
        train = []
        with open(os.path.join(path_to_labels, '{0}_{1}_train.txt'.format(p, s))) as f:
            for line in f.readlines():
                line = line.rstrip('\n')
                data = line.split(' ')
                train.append((data[0], data[1]))

        test = []
        with open(os.path.join(path_to_labels, '{0}_{1}_test.txt'.format(p, s))) as f:
            for line in f.readlines():
                line = line.rstrip('\n')
                data = line.split(' ')
                test.append((data[0], data[1]))

        
        train = list(map(lambda x: (os.path.join(path_to_data, x[0]), int(x[1])), train))
        test = list(map(lambda x: (os.path.join(path_to_data, x[0]), int(x[1])), test))

        return train, test


class DataAE():
    """
    Class for working with data when learning autoencoder.
    Loads all in RAM.
    """

    def get_files(path_to_data, test_coef=0.2):
        res = []
        for p in os.listdir(path_to_data):
            p_path = os.path.join(path_to_data, p)
            if not os.path.isdir(p_path):
                continue
            p_s_0 = os.path.join(p_path, 'signals_0')
            p_s_1 = os.path.join(p_path, 'signals_1')
            for f in os.listdir(p_s_0):
                rel_path = os.path.join(p, 'signals_0', f)
                res.append((rel_path, 0))
            for s in os.listdir(p_s_1):
                s_path = os.path.join(p_s_1, s)
                for f in os.listdir(s_path):
                    rel_path = os.path.join(p, 'signals_1', s, f)
                    res.append((rel_path, 0))
        
        random.shuffle(res)
        split_indx = int(len(res) * test_coef)
        return res[split_indx:], res[:split_indx]

    def save_files(path_to_labels, train, test):
        with open(os.path.join(path_to_labels, 'autoencoder_train.txt'), "w") as f:
            first_s = ''
            for e in train:
                f.write(first_s + e[0] + ' ' + str(e[1]))
                first_s = '\n'

        with open(os.path.join(path_to_labels, 'autoencoder_test.txt'), "w") as f:
            first_s = ''
            for e in test:
                f.write(first_s + e[0] + ' ' + str(e[1]))
                first_s = '\n'

    def load_files(path_to_labels, path_to_data):
        train = []
        with open(os.path.join(path_to_labels, 'autoencoder_train.txt')) as f:
            for line in f.readlines():
                line = line.rstrip('\n')
                data = line.split(' ')
                train.append((data[0], data[1]))

        test = []
        with open(os.path.join(path_to_labels, 'autoencoder_test.txt')) as f:
            for line in f.readlines():
                line = line.rstrip('\n')
                data = line.split(' ')
                test.append((data[0], data[1]))

        
        train = list(map(lambda x: (os.path.join(path_to_data, x[0]), int(x[1])), train))
        test = list(map(lambda x: (os.path.join(path_to_data, x[0]), int(x[1])), test))

        return train, test