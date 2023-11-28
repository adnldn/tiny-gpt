import argparse
import json
from ast import literal_eval

def parse_config(globals_dict):
    # Create the parser
    parser = argparse.ArgumentParser(description='Override global variables.')

    # Add the arguments
    parser.add_argument('--config_file', type=str, help='The name of the config file')

    # Loop over the global variables and add them as arguments
    for key, val in globals_dict.items():
        if not key.startswith("__"):  # Ignore special attributes
            parser.add_argument(f'--{key}', type=type(val))

    # Parse the arguments
    args = parser.parse_args()

    # If a config file is provided, load it and merge with the parsed arguments
    if args.config_file:
        with open(args.config_file) as f:
            file_config = json.load(f)
        # Merge file config and command line arguments, command line arguments take precedence
        config = {**file_config, **{k: v for k, v in vars(args).items() if v is not None}}
    else:
        config = {k: v for k, v in vars(args).items() if v is not None}

    # Exclude the 'config_file' key
    config.pop('config_file', None)

    return config


def override_globals(config, globals_dict):
    # Override the global variables
    for key, val in config.items():
        if key in globals_dict:
            try:
                # Attempt to eval it (e.g. if bool, number, or etc)
                attempt = literal_eval(str(val))
            except (SyntaxError, ValueError):
                # If that goes wrong, just use the string
                attempt = val
            # Ensure the types match ok
            assert type(attempt) == type(globals_dict[key]), f"Type mismatch for config key: {key}"
            # Cross fingers
            # print(f"Overriding: {key} = {attempt}")
            globals_dict[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
