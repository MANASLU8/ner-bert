import sys
from converters import tagged_text_to_internal_format

tagged_file = sys.argv[1]
internal_format_file = sys.argv[2]

tagged_text_to_internal_format(tagged_file, internal_format_file)