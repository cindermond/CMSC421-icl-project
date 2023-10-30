import json

class Config:
    def __init__(self, config_file=None, style='json'):
        self._set_default_attribute()
        if config_file:
            self.load_from_file(config_file, style)
    
    def _set_default_attribute(self):
        pass

    def load_from_file(self, config_file, style='json'):
        assert style == 'json', "For now the only supported config style is json"
        
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                
            for key, value in file_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
        except FileNotFoundError:
            print(f"Warning: Configuration file {config_file} not found. Using default value.")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in configuration file. Using default value.")
        
        print(f"Successfully loaded config from {config_file}!")


class DatasetConfig(Config):
    def _set_default_attribute(self):
        self.max_steps = 0 #zero means no max steps
        self.override_limits = False
        self.example_size = 4
        self.shuffle_item = True
        self.shuffle_batch = False
        self.repeat_times = 1


class NetConfig(Config):
    def _set_default_attribute(self):
        self.generation_args = {
            "num_beams": 1,
            "max_new_tokens": 1000,
            "min_new_tokens": 1, 
            "temperature": 1.0
        }
        self.model_name = "meta-llama/Llama-2-13b-hf"
        self.cache_dir = "cache"
        self.token_file = None
        self.chat = True
