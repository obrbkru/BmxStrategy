#!C:\Users\obrbkru\Documents\projects\BBot\venv\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'technical-indicators==0.0.16','console_scripts','technical_indicators'
__requires__ = 'technical-indicators==0.0.16'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('technical-indicators==0.0.16', 'console_scripts', 'technical_indicators')()
    )
