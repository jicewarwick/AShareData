import sys

from AShareData import IndexHighlighter, set_global_config

if __name__ == '__main__':
    set_global_config(sys.argv[1])
    IndexHighlighter().summary()
