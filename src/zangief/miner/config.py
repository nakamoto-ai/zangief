import configparser


class Config:
    model_name: str
    device: str
    max_length: str
    do_sample: str
    temperature: str
    top_k: str
    no_repeat_ngram_size: str
    num_beams: str
    
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
