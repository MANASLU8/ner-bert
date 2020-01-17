from file_operations import write_lines, read

def raw_text_to_internal_format(input_file, output_file, tokenizer):
    tokens = list(map(lambda sentence: tokenizer.tokenize(sentence.strip()), read(input_file).split('.')))
    tags = list(map(lambda sentence: ["O" for token in sentence], tokens))
    lines = ['labels\ttext\tclf']
    for sentence_tokens, sentence_tags in zip(tokens, tags):
        if len(sentence_tokens) > 0:
            lines.append(f'{" ".join(sentence_tags)}\t{" ".join(sentence_tokens)}\tTrue')
    write_lines(output_file, lines)
    return tokens