import yaml
import sys
from prettytable import PrettyTable


def get_configure(config_file='system.yaml'):
    with open(config_file, encoding='utf-8') as f:
        conf_data = yaml.safe_load(f)
    
    return Configure(conf_data)

class Configure:
    def __init__(self, conf_data):
        self.conf_data = conf_data
        for key, value in conf_data.items():
            if isinstance(value, dict):
                value = Configure(value)
            setattr(self, key, value)



    def show_data_summary(self, logger):
        logger.info('++' * 20 + 'CONFIGURATION SUMMARY' + '++' * 20)

        table = PrettyTable()
        table.field_names = ["Configuration", "Value"]
        table.align["Configuration"] = "l"  
        table.align["Value"] = "l"  
        
        for key, value in self.conf_data.items():
            table.add_row([key, value])

        for row in table.get_string().splitlines():
            logger.info(row)  
        
        logger.info('++' * 20 + 'CONFIGURATION SUMMARY END' + '++' * 20)
        sys.stdout.flush()