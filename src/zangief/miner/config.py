import configparser


class Config:
    def __init__(self, config_file):
        if config_file is None:
            config_file = '../../../env/config.ini'
        self.config = configparser.ConfigParser()
        self.config.read(config_file)


    def get_value(self, option, default=None):
        section = "miner"
        if self.config.has_option(section, option):
            return self.config.get(section, option)
        return default
