import os
import getpass
from dotenv import load_dotenv

load_dotenv()


def get_pass_api_key(key: str) -> None:
    value = os.getenv(key)
    if not value:
        value = getpass.getpass(prompt=f'Enter API {key}: ').strip()
        os.environ[key] = value


def reset_api_key(key: str) -> None:
    os.environ[key] = ''
