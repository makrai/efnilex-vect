import logging
import sys
import pickle
import os.path

from gensim.models.word2vec import Word2Vec


class EmbedInterProcessor(): 
    def read_txt(self):
        logging.info('Clean an embedding in txt format without header...')
        with open(self.file_pref+'.tmp', mode='w') as out_file:
            with open(self.in_filen) as in_file:
                vocab_size = 0
                dim = None
                for i, line in enumerate(in_file):
                    try:
                        fields = line.strip().decode('utf-8').split(' ')
                    except UnicodeDecodeError as e:
                        logging.warn('in line {}, {}'.format(i, e))
                        continue
                    if not dim:
                        dim = len(fields) - 1
                        logging.info(
                            'processing {}-dimensional model'.format(dim))
                    if len(fields) != dim + 1:
                        logging.info(
                            'line with white space skipped (#{})'.format(
                                i))
                        continue
                    if not fields[0]:
                        logging.info('empty word in line {}, skipped'.format(
                            i))
                        continue
                    vocab_size += 1
                    out_file.write(' '.join(fields).encode('utf-8')+'\n')
        logging.info(
            'Now you need to add the header to {}.tmp\n{} {}'.format(
                vocab_size, dim, self.file_pref))

    def main(self):
        self.in_filen = sys.argv[1]
        self.file_pref, ext = os.path.splitext(self.in_filen)
        # TODO skip words with control characters and decrease vocab size
        # in header
        if ext == '.pkl':
            # this branch is for embeddings from
            # https://sites.google.com/site/rmyeid/projects/polyglot
            logging.warning(
                'There is a version of this function in the multiwsi repo ' 
                'that writes the embedding with fewer digits (using st.format)')
            with open(self.file_pref+'.w2v', mode='w') as out_file:
                with open(self.in_filen, mode='rb') as in_file:
                    words, vecs = pickle.load(in_file)
                out_file.write('{} {}\n'.format(*vecs.shape))
                for word, vec in zip(words, vecs):
                    out_file.write('{}  {}\n'.format(
                        word.encode('utf8'), 
                        ' '.join(str(coord) for coord in vec.tolist())))
        elif ext == '.w2v':
            m = Word2Vec.load_word2vec_format(self.in_filen)
            m.save(self.file_pref+'.gensim')
        elif ext == '.txt':
            self.read_txt()
        elif ext == '.bin':
            if 'glove' in self.file_pref:
                raise NotImplementedError(
                    'glove binaries are not suppoerted')
            else:
                m = Word2Vec.load_word2vec_format(self.in_filen, binary=True)
                logging.info("Saving {}".format(self.file_pref+'.gensim'))
                m.save(self.file_pref+'.gensim')
                logging.info("Saving {}".format(self.file_pref+'.w2v'))
                m.save_word2vec_format(self.file_pref+'.w2v')
        else:
            raise NotImplementedError('unknown extension')


if __name__ == '__main__':
    format_ = "%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s"
    logging.basicConfig(format=format_, level=logging.DEBUG) 
    EmbedInterProcessor().main()
