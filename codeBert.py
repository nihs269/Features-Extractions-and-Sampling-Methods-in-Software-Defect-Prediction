import glob
import javalang
import pygments
import os
import torch
import numpy as np

from collections import OrderedDict
from pygments.lexers import JavaLexer
from pygments.token import Token
from dataset import DATASET
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")


class SourceFileCodeBert:
    """Class representing each source file"""
    __slots__ = ['src', 'comments']

    def __init__(self, src, comments):
        self.src = src
        self.comments = comments


def src_parser():
    """Parse source code directory of a program and colect its java files"""

    # Gettting the list of source files recursively from the source directory
    src_addresses = glob.glob(str(DATASET.src_file) + '/**/*.java', recursive=True)

    # Creating a java lexer instance for pygments.lex() method
    java_lexer = JavaLexer()
    src_files = OrderedDict()

    # Looping to parse each source file
    for src_file in src_addresses:
        with open(src_file, encoding='latin-1') as file:
            src = file.read()

        # Placeholder for different parts of a source file
        list_comments = []
        comments = ''

        # Source parsing
        parse_tree = None
        try:
            parse_tree = javalang.parse.parse(src)
        except:
            pass

        # Triming the source file
        ind = False
        if parse_tree:
            if parse_tree.imports:
                last_imp_path = parse_tree.imports[-1].path
                src = src[src.index(last_imp_path) + len(last_imp_path) + 1:]
            elif parse_tree.package:
                package_name = parse_tree.package.name
                src = src[src.index(package_name) + len(package_name) + 1:]
            else:  # no import and no package declaration
                ind = True
        # javalang can't parse the source file
        else:
            ind = True

        # Lexically tokenize the source file
        lexed_src = pygments.lex(src, java_lexer)

        for i, token in enumerate(lexed_src):
            if token[0] in Token.Comment:
                if ind and i == 0 and token[0] is Token.Comment.Multiline:
                    src = src[src.index(token[1]) + len(token[1]):]
                    continue
                list_comments.append(token[1])
                comments = comments + token[1]

        for comment in list_comments:
            src = src.replace(comment, '')

        # get the package declaration if exists
        if parse_tree and parse_tree.package:
            package_name = parse_tree.package.name
        else:
            package_name = None

        # If source files has package declaration
        if package_name:
            src_id = (package_name + '.' + os.path.basename(src_file))
        else:
            src_id = os.path.basename(src_file)
        src_files[src_id] = SourceFileCodeBert(src, comments)
    return src_files


def codebert_features():
    codebert_file_names = []
    codebert_dict = {}
    source_files = src_parser()
    for key in source_files:
        src_code = source_files.get(key).src
        src_comment = source_files.get(key).comments
        check = 2

        src_code_embeddings = np.zeros((512, 768))
        cmt_embeddings = np.zeros((512, 768))

        try:
            src_tokens = tokenizer.tokenize(src_code)
            src_ids = tokenizer.convert_tokens_to_ids(src_tokens)
            embeddings = model(torch.tensor(src_ids)[None, :])[0][0].detach().numpy()
            src_code_embeddings[:embeddings.shape[0], :embeddings.shape[1]] = embeddings
        except:
            check -= 1

        try:
            cmt_tokens = tokenizer.tokenize(src_comment)
            cmt_ids = tokenizer.convert_tokens_to_ids(cmt_tokens)
            embeddings = model(torch.tensor(cmt_ids)[None, :])[0][0].detach().numpy()
            cmt_embeddings[:embeddings.shape[0], :embeddings.shape[1]] = embeddings
        except:
            check -= 1

        if check != 0:
            dictKey = key[:-5].replace('.', '/') + '.java'
            codebert_file_names.append(dictKey)
            src_embeddings = np.add(src_code_embeddings, cmt_embeddings) / check
            src_embedding = np.zeros((1, 768))
            for embedding in src_embeddings:
                src_embedding = np.add(src_embedding, embedding)
            src_embedding = src_embedding / 768

            codebert_dict.update({dictKey: src_embedding[0]})
    return codebert_file_names, codebert_dict


if __name__ == '__main__':
    src_parser()
