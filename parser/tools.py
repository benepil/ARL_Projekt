from configparser import ConfigParser


def get_value(configuration: ConfigParser, section: str, option: str, required_type: type):
    if configuration.has_section(section):

        if configuration.has_option(section, option):
            value: str = configuration[section][option]

            try:
                value = required_type(value)
                return value

            except ValueError:
                raise ValueError(f"Value for option {option} in section {section} must be of "
                                 f"type {required_type}! Unable to cast value {value} to {required_type}")

        else:
            raise ValueError(f"Configuration file has no option: {option} in section: {section}!")

    else:
        raise ValueError(f"Configuration file has no section: {section}!")
