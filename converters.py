from file_operations import write_lines, read, read_lines

NO_ENTITY_MARK = 'O'

def raw_text_to_internal_format(input_file, output_file, tokenizer):
    tokens = list(map(lambda sentence: tokenizer.tokenize(sentence.strip()), read(input_file).split('.')))
    tags = list(map(lambda sentence: ["O" for token in sentence], tokens))
    lines = ['labels\ttext\tclf']
    for sentence_tokens, sentence_tags in zip(tokens, tags):
        if len(sentence_tokens) > 0:
            lines.append(f'{" ".join(sentence_tags)}\t{" ".join(sentence_tokens)}\tTrue')
    write_lines(output_file, lines)
    return tokens

def make_line_from_tagged(sentence_tokens, sentence_tags):
	return f'{" ".join(map(lambda tag: tag.upper().replace("-", "_")[:5], sentence_tags))}\t{" ".join(sentence_tokens)}\t{"True" if len(set(sentence_tags)) == 1 and sentence_tags[0] == NO_ENTITY_MARK else "False"}'

def tagged_text_to_internal_format(input_file, output_file):
	lines = read_lines(input_file)
	output_lines = ['labels\ttext\tclf']
	sentence_tokens = []
	sentence_tags = []
	for line in map(lambda line: line.split(' '), lines):
		if len(line) < 2:
			output_lines.append(make_line_from_tagged(sentence_tokens, sentence_tags))
			sentence_tags = []
			sentence_tokens = []
		else:
			sentence_tokens.append(line[0])
			sentence_tags.append(line[1])
	if len(sentence_tokens) > 0:
		output_lines.append(make_line_from_tagged(sentence_tokens, sentence_tags))
	write_lines(output_file, output_lines)
	return sentence_tokens