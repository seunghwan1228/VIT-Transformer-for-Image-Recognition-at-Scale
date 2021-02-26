import ruamel.yaml
import os
import sys
import yaml
from shutil import copyfile


class ConfigReader:
    def __init__(self, config_dir):
        self.config_dir = config_dir
        self.data_config = os.path.join(config_dir, 'data_config.yaml')
        self.model_config = os.path.join(config_dir, 'model_config.yaml')

        self.yaml_reader = ruamel.yaml.YAML()

    def _read_config(self, source_config_path):
        with open(source_config_path, 'rb') as rc:
            loaded_config = self.yaml_reader.load(rc)
        return dict(loaded_config)

    def load_config(self):
        full_dict = {}
        data_dict = self._read_config(self.data_config)
        full_dict.update(data_dict)
        model_dict = self._read_config(self.model_config)
        full_dict.update(model_dict)
        return full_dict

class PrintConfig:
    def __init__(self, config_dict):
        self.config_dict = config_dict

    @staticmethod
    def _print_dict_values(values, key_name, level=0, tab_size=2):
        tab = level * tab_size * ' '
        print(tab + '-', key_name, ':', values)

    def _print_dictionary(self, dictionary, recursion_level=0):
        for key in dictionary.keys():
            if isinstance(key, dict):
                recursion_level += 1
                self._print_dict_values(dictionary[key], recursion_level)
            else:
                self._print_dict_values(dictionary[key], key, level=recursion_level)

    def print_config(self):
        self._print_dictionary(self.config_dict, recursion_level=0)


class SaveConfig:
    def __init__(self, config_dir):
        self.config_dir = config_dir

    def save_config(self, save_dir):
        data_config_tar = os.path.join(save_dir, 'data_config.yaml')
        model_config_tar = os.path.join(save_dir, 'model_config.yaml')

        copyfile('config/data_config.yaml', data_config_tar)
        copyfile('config/model_config.yaml', model_config_tar)
        print('\nCopy Config files Complete')




if __name__ == '__main__':
    tmp_reader = ConfigReader('config')
    tmp_full_config = tmp_reader.load_config()

    # print(tmp_full_config)

    config_checker = PrintConfig(tmp_full_config)
    config_checker.print_config()

    config_saver = SaveConfig('config')
    config_saver.save_config(r'C:\Users\Owner\Documents\GitHub')





