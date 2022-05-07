import os.path

import songSimplifier
import logging
import argparse
from pathlib import Path
import random
import time
import numpy as np
from tqdm import tqdm

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

VALIDATION_SPLIT = 0.2
NOTES_IN_SEQUENCE = 25
MAXLEN = 75

START_TOKEN = 'STAR'
END_TOKEN = 'ENDD'

LEARN_FILENAME = "learning.txt"
VAL_FILENAME = "val.txt"
ALL_FILENAME = "all.txt"


def get_timestamp_filename(prefix="file", suffix="ckpt"):
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.{suffix}"


def simplify_all_midi_to_txt(midi_dir, txt_dir):
    midi_files = Path(midi_dir).glob('*.mid')
    song_simplifier = songSimplifier.SongSimplifier()

    for midi_file in midi_files:
        my_txt = Path.joinpath(txt_dir, midi_file.stem + ".txt")
        logging.debug(f"{midi_file}\n")
        song_simplifier.midi_to_txt(str(midi_file), str(my_txt))


def create_all_songs_file(txt_dir, examples_dir):
    # write all txt files into one
    all_songs_file_path = Path.joinpath(examples_dir, ALL_FILENAME)
    learning_txt = Path.joinpath(examples_dir, LEARN_FILENAME)
    validation_txt = Path.joinpath(examples_dir, VAL_FILENAME)
    txt_files = list(Path(txt_dir).glob('*.txt'))
    random.shuffle(txt_files)

    learning_file = open(str(learning_txt), 'w')
    validation_file = open(str(validation_txt), 'w')
    all_file = open(str(all_songs_file_path), 'w')

    for txt_file in txt_files:
        file = learning_file
        t = f"learning: {txt_file}"

        if random.uniform(0, 1) < VALIDATION_SPLIT:
            t = f"val: {txt_file}"
            file = validation_file
        logging.debug(t)

        file.write(START_TOKEN + "\n")
        all_file.write(START_TOKEN + "\n")

        with open(str(txt_file), 'r') as txt:
            t = txt.read()
            file.writelines(t)
            all_file.writelines(t)

        file.write(END_TOKEN + "\n")
        all_file.write(END_TOKEN + "\n")

    learning_file.close()
    validation_file.close()
    all_file.close()


def convert_txt_to_midi(txt_dir, new_midi_dir):
    song_simplifier = songSimplifier.SongSimplifier()
    txt_files = Path(txt_dir).glob('*.txt')
    for txt_file in txt_files:
        my_midi = Path.joinpath(new_midi_dir, txt_file.stem + ".mid")
        logging.info(str(txt_file))
        song_simplifier.txt_to_midi(str(txt_file), str(my_midi))


def text_to_tokens(tokens_indices_dict, special_tokens, text):
    i = 0
    tokenized = []
    while i < len(text):
        next4 = str(text[i: i + 4])
        if next4 in special_tokens:
            token = tokens_indices_dict[next4]
            tokenized.append(token)
            i = i + 4
        else:
            token = tokens_indices_dict[text[i]]
            tokenized.append(token)
            i = i + 1
    return tokenized


def get_random_seed_tokens(all_songs_file, tokens_indices_dict, special_tokens, length):
    chars_to_read = length * 4

    with open(str(all_songs_file), 'r') as txt:
        data = txt.read()

    start_index = random.randrange(0, len(data) - chars_to_read - 30)
    text = data[start_index: start_index + chars_to_read]
    tokenized = []
    i = 0
    while text[i] != '\n':
        i = i + 1
        logging.debug(f"Skipping letter {text[i - 1]}")
    i = i + 1  # starting from new line, new line new note

    while i < len(text) and len(tokenized) < length:
        next4 = str(text[i: i + 4])
        if next4 in special_tokens:
            token = tokens_indices_dict[next4]
            tokenized.append(token)
            i = i + 4
        else:
            token = tokens_indices_dict[text[i]]
            tokenized.append(token)
            i = i + 1
    return tokenized


def tokens_to_text(indices_tokens_dict, tokens):
    text = ""
    for token in tokens:
        text = text + indices_tokens_dict[token]
    return text


def create_model(vocab_size, maxlen):
    model = Sequential()

    model.add(LSTM(128, return_sequences=True, input_shape=(maxlen, vocab_size)))
    model.add(Dropout(0.2))

    model.add(LSTM(128))
    model.add(Dropout(0.2))

    model.add(Dense(vocab_size, activation='softmax'))

    logging.info(model.summary())

    optimizer = Adam()  # RMSprop(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model


def sample_predictions(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype("float64")
    predictions = np.log(predictions) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)


def get_all_instruments():
    simplifier = songSimplifier.SongSimplifier()
    i = simplifier.get_non_drum_instruments()
    d = simplifier.get_drum_instruments()
    i.extend(d)
    return i


def main(midi_dir=None, results_dir=None, checkpoint_file=None, skip_txt_creation=False, skip_learning=False):
    if not midi_dir:
        midi_dir = Path.joinpath(Path.home(), "my_midi")
    else:
        midi_dir = Path(midi_dir)
    if not results_dir:
        results_dir = Path.joinpath(Path.home(), "cacofonix")
    else:
        results_dir = Path(results_dir)

    txt_dir = Path.joinpath(results_dir, "my_txt")
    checkpoint_dir = Path.joinpath(results_dir, "my_checkpoints")
    songs_dir = Path.joinpath(results_dir, "beautiful_songs")
    examples_dir = Path.joinpath(results_dir, "examples")
    for _directory in [txt_dir, checkpoint_dir, songs_dir, examples_dir]:
        if os.path.isdir(_directory):
            logging.debug(f"Directory {_directory} exists")
        else:
            os.mkdir(_directory)
            logging.info(f"Directory {_directory} created")

    special_tokens = [START_TOKEN, END_TOKEN]
    special_tokens.extend(get_all_instruments())
    logging.debug(f"Special tokes: {special_tokens}")

    if not skip_txt_creation:
        simplify_all_midi_to_txt(midi_dir, txt_dir)
        create_all_songs_file(txt_dir, examples_dir)

    all_songs_file = Path.joinpath(examples_dir, ALL_FILENAME)
    val_file = Path.joinpath(examples_dir, VAL_FILENAME)
    learn_file = Path.joinpath(examples_dir, LEARN_FILENAME)
    with open(str(all_songs_file), 'r') as all_file:
        all_data = all_file.read()
    with open(str(val_file), 'r') as _val_file:
        val_data = _val_file.read()
    with open(str(learn_file), 'r') as _learn_file:
        learn_data = _learn_file.read()

    special_tokens_freq = {}
    for _token in special_tokens:
        special_tokens_freq[_token] = str(all_data).count(str(_token))
    logging.debug(f"Special tokens freq: {special_tokens_freq}")

    # tokenize
    indices_tokens_dict = {}
    tokens_indices_dict = {}

    tokens = {}
    data = all_data.splitlines()

    for line in data:
        # start and end
        if line in special_tokens:
            if line not in tokens.keys():
                tokens[line] = 0
            else:
                tokens[line] = tokens[line] + 1
            continue

        instrument = str(line[1:5])
        if instrument not in tokens.keys():
            tokens[instrument] = 0
        else:
            tokens[instrument] = tokens[instrument] + 1
        chars_indexes = line[0]
        chars_indexes = chars_indexes + line[5:len(line)]  # all but the instrument
        for c in chars_indexes:
            if c not in tokens.keys():
                tokens[c] = 0
            else:
                tokens[c] = tokens[c] + 1

    tokens['\n'] = len(data)  # new lines
    i = 0
    for token in sorted(tokens.items(), key=lambda x: x[1], reverse=True):
        indices_tokens_dict[i] = token[0]  # is this key?
        tokens_indices_dict[token[0]] = i
        i = i + 1

    logging.debug(str(indices_tokens_dict))
    vocab_size = len(tokens.keys())
    logging.info(f"Vocab size: {vocab_size}")

    tokenized_val = text_to_tokens(tokens_indices_dict, special_tokens, val_data)
    tokenized_lrn = text_to_tokens(tokens_indices_dict, special_tokens, learn_data)

    logging.info("Tokenized")
    step = 5  # why? idk
    sequences_lrn = []
    sequences_val = []
    next_indecies = []
    next_indecies_val = []

    for i in range(0, len(tokenized_lrn) - MAXLEN, step):
        sequences_lrn.append(tokenized_lrn[i: i + MAXLEN])
        next_indecies.append(tokenized_lrn[i + MAXLEN])
    logging.debug(f"Number learn of sequences: {len(sequences_lrn)}")

    X = np.zeros((len(sequences_lrn), MAXLEN, vocab_size), dtype=np.bool)
    y = np.zeros((len(sequences_lrn), vocab_size), dtype=np.bool)
    logging.debug(f"X shape: {X.shape}")
    logging.debug(f"y shape: {y.shape}")

    # what is this magic part?
    for i, sequence in enumerate(sequences_lrn):
        for t, token in enumerate(sequence):
            X[i, t, token] = 1
        y[i, next_indecies[i]] = 1

    for i in range(0, len(tokenized_val) - MAXLEN, step):
        sequences_val.append(tokenized_val[i: i + MAXLEN])
        next_indecies_val.append(tokenized_val[i + MAXLEN])
    logging.debug(f"Number val of sequences: {len(sequences_val)}")

    X_val = np.zeros((len(sequences_val), MAXLEN, vocab_size), dtype=np.bool)
    y_val = np.zeros((len(sequences_val), vocab_size), dtype=np.bool)
    logging.debug(f"X val shape: {X.shape}")
    logging.debug(f"y val shape: {y.shape}")
    for i, sequence in enumerate(sequences_val):
        for t, token in enumerate(sequence):
            X_val[i, t, token] = 1
        y_val[i, next_indecies_val[i]] = 1

    model = create_model(vocab_size, MAXLEN)
    if checkpoint_file:
        model.load_weights(checkpoint_file)

    epochs = 2000

    batch_size = 128
    step = 5

    if not skip_learning:
        validation_error = []
        learning_error = []
        for epoch in range(0, epochs, step):
            logging.info(f"Starting epoch {epoch}")
            history = model.fit(X, y, epochs=step, batch_size=batch_size, validation_data=(X_val, y_val))
            current_epoch = epoch + step
            output_file = get_timestamp_filename(prefix=f"epoch_{current_epoch}")
            output_file = Path.joinpath(checkpoint_dir, output_file)
            logging.info(f"Saving weights to file: {output_file}")
            model.save_weights(str(output_file), overwrite=True)
            logging.debug(f"History: {history.history}")
            validation_error.extend(history.history['val_loss'])
            learning_error.extend(history.history['loss'])
            # for diversity in [0.5, 1.0, 1.2]:
            """
            for diversity in [1.0]:
                generated = ""
                sequence = seed_tokens
                logging.debug(f"Diversity: {diversity}")

                for i in range(20):
                    my_input = np.zeros((1, maxlen, vocab_size))
                    for j, token in enumerate(sequence):
                        my_input[0, j, token] = 1.0

                    prediction = model.predict(my_input)[0]
                    next_token = sample_predictions(prediction, diversity)
                    next_char  = tokens_to_text(indices_tokens_dict, [next_token])
                    sequence   = sequence[1:]
                    sequence.append(next_token)
                    generated += next_char

                print(f"Epoch {epoch + step} generated: {generated}")
            """

    output_txt_song = get_timestamp_filename(prefix="song", suffix="txt")
    output_midi_song = get_timestamp_filename(prefix="song", suffix="mid")
    output_txt_song = Path.joinpath(songs_dir, output_txt_song)
    output_midi_song = Path.joinpath(songs_dir, output_midi_song)
    new_seed_tokens = get_random_seed_tokens(all_songs_file, tokens_indices_dict, special_tokens, MAXLEN)
    get_some_music(model, MAXLEN, vocab_size, output_txt_song, output_midi_song, 100000, new_seed_tokens,
                   indices_tokens_dict)


def translate_from_numbers(tokenizer, numbers):
    text = ""
    real_useful_dictionary = {}
    for key in tokenizer.word_index.keys():
        new_key = tokenizer.word_index[key]
        real_useful_dictionary[new_key] = key
    for number in numbers:
        if number in real_useful_dictionary.keys():
            text = text + real_useful_dictionary[number]
        else:
            logging.debug(f"Not recognized code: {number}")
    return text


def get_some_music(model, maxlen, vocab_size, output_txt_file, output_midi_file, length, seed_tokens,
                   indices_tokens_dict):
    logging.debug(f"Generating {length} tokens of music")
    sequence = seed_tokens
    this_is_just_a_tribute = []

    for i in tqdm(range(0, length)):
        # logging.debug(i)
        my_input = np.zeros((1, maxlen, vocab_size))
        for j, token in enumerate(sequence):
            my_input[0, j, token] = 1.0

        prediction = model.predict(my_input)[0]
        next_token = sample_predictions(prediction)

        sequence = sequence[1:]
        sequence.append(next_token)
        this_is_just_a_tribute.append(next_token)

    song = tokens_to_text(indices_tokens_dict, this_is_just_a_tribute)
    with open(output_txt_file, 'w') as of:
        logging.info(f"Writing song to file: {output_txt_file}")
        of.write(song)
    simplifier = songSimplifier.SongSimplifier()
    simplifier.txt_to_midi(output_txt_file, output_midi_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Midi music generator")
    parser.add_argument("--midi_dir", help="Directory with source midi files", required=False, default=None)
    parser.add_argument("--results", help="Directory to save created songs", required=False, default=None)
    parser.add_argument("--checkpoint", help="Checkpoint with pretrained saved weights to load and skip learning "
                                             "entirely", required=False, default=None)
    parser.add_argument("--skip_txt_creation", help="Assume all.txt file was already created and no midi parsing is "
                                                    "needed", default=False, required=False)
    parser.add_argument("--skip_learning", help="Skip learning and just generate music.", default=False, required=False)
    args = parser.parse_args()
    main(args.midi_dir, args.results, args.checkpoint, args.skip_txt_creation, args.skip_learning)

