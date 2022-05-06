from collections import namedtuple

import pretty_midi
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

SongStats = namedtuple('SongStats', ['velocity', 'est_tempo', 'instruments'])


class SongSimplifier:

    def __init__(self):
        self.DEFAULT_BPM = 220
        self.DEFAULT_VELOCITY = 100

        self.EPIA = 'EPIA'  # ELECTRIC PIANO
        self.BASS = 'BASS'  # BASS
        self.AGUI = 'AGUI'  # ACOUSTIC GUITAR STEEL
        self.STRI = 'STRI'  # STRINGS VIOLIN
        self.SQUA = 'SQUA'  # LEAD 1 SQUARE
        self.SAWT = 'SAWT'  # LEAD 2 SAWTOOTH
        self.AGPI = 'AGPI'  # ACOUSTIC GRAND PIANO

        # self.CYMB = 'CYMB'  # RIDE CYMBAL 1 - removed bc too rare
        self.ABDR = 'ABDR'  # ACCOUSTIC VASS DRUM

        self.instrument_to_number = {}
        self.number_to_instrument = {}
        self.instrument_ranges = {}

        self.drum_instrument_to_number = {}
        self.drum_instrument_ranges = {}
        self.drum_number_to_instrument = {}

        self.instrument_to_number[self.BASS] = 34
        self.instrument_ranges[self.BASS] = list(range(32, 40))
        self.instrument_to_number[self.EPIA] = 1
        self.instrument_ranges[self.EPIA] = [2, 4, 5]
        self.instrument_to_number[self.AGUI] = 25
        self.instrument_ranges[self.AGUI] = [24, 25]
        self.instrument_to_number[self.STRI] = 40
        self.instrument_ranges[self.STRI] = list(range(40, 55))
        self.instrument_to_number[self.SQUA] = 80
        self.instrument_ranges[self.SQUA] = [80, 82, 83, 84, 85, 86, 87]
        self.instrument_to_number[self.SAWT] = 81
        self.instrument_ranges[self.SAWT] = [81]
        self.instrument_to_number[self.AGPI] = 0
        self.instrument_ranges[self.AGPI] = [0]
        self.default_instrument = self.AGPI

        # self.drum_instrument_to_number[self.CYMB] = 50
        # self.drum_instrument_ranges[self.CYMB] = [41, 43, 45, 48, 50, 51, 53, 54, 58]
        self.drum_instrument_to_number[self.ABDR] = 35
        self.drum_instrument_ranges[self.ABDR] = [35, 41, 43, 45, 48, 50, 51, 53, 54, 58]   # [35]
        self.default_drum_instrument = self.ABDR

        for key in self.instrument_to_number.keys():
            value = self.instrument_to_number[key]
            self.number_to_instrument[str(value)] = key

        for key in self.drum_instrument_to_number.keys():
            value = self.drum_instrument_to_number[key]
            self.number_to_instrument[str(value)] = key

        self.note_to_length = {}
        self.note_to_length['s'] = 1/4   # 1/16
        self.note_to_length['o'] = 1/2   # 1/8
        self.note_to_length['q'] = 1     # 1/4
        self.note_to_length['h'] = 2     # 1/2
        self.note_to_length['w'] = 4     # 1
        self.note_to_length['d'] = 8

    def midi_to_txt(self, input_path: str, output_path: str) -> None:
        midi = self._parse_midi(input_path)
        if midi is None:
            return

        bpm = midi.estimate_tempo()

        logging.info(f"Estimate tempo: {midi.estimate_tempo()}")
        logging.info(f"Beat length: {bpm/60.0}")
        logging.info(f"Total velocity: {sum(sum(midi.get_chroma()))}")
        instruments_log = "Instruments:\n"

        song = []

        for instrument in midi.instruments:
            instruments_log += f"{instrument.program} {pretty_midi.program_to_instrument_name(instrument.program)} " \
                               f"class: {pretty_midi.program_to_instrument_class(instrument.program)}\n"

            if instrument.is_drum:
                instrument_code = self._get_drum_instrument_code(instrument.program)
                n_type = 'r'
            else:
                instrument_code = self._get_instrument_code(instrument.program)
                n_type = 'n'

            for note in instrument.notes:
                n = {}
                n['instrument'] = instrument_code
                n['type'] = n_type
                n['note'] = note.pitch
                n['length'] = self._get_note_length_representation(bpm, note.get_duration())
                n['start'] = note.start
                n['end'] = note.end

                song.append(n)

            for pitch_bend in instrument.pitch_bends:
                n = {}
                n['instrument'] = instrument_code
                n['type'] = 'p'
                n['start'] = pitch_bend.time
                n['pitch'] = pitch_bend.pitch
                song.append(n)
        logging.info(instruments_log)

        song = sorted(song, key=lambda k: k['start'])
        song_txt = []
        for i in range(0, len(song)):
            event = song[i]
            e = event['type']   # remove spaces
            e = e + event['instrument']

            offset = 0
            if i != 0:
                offset_duration = event['start'] - song[i-1]['start']
                offset = self._calculate_offset_length_representation(bpm, offset_duration)

            e = f"{e}{round(offset)};"

            if event['type'] in ['r', 'n']:
                e = e + str(event['note']) + str(event['length'])
            if event['type'] == 'p':
                e = e + str(event['pitch'])
            e = e + "\n"
            song_txt.append(e)

        with open(output_path, 'w') as output_file:
            output_file.writelines(song_txt)

    def get_stats(self, input_path):
        midi = self._parse_midi(input_path)
        instruments = {}
        if midi is None:
            return midi
        for instrument in midi.instruments:
            instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
            number_of_notes = len(instrument.notes)
            instruments[instrument_name] = number_of_notes
        return SongStats(velocity=sum(sum(midi.get_chroma())), est_tempo=midi.estimate_tempo(), instruments=instruments)

    def get_drum_instruments(self):
        return list(self.drum_instrument_ranges.keys())

    def get_non_drum_instruments(self):
        return list(self.instrument_ranges.keys())

    def get_all_instruments(self):
        drums = list(self.drum_instrument_ranges.keys())
        general = list(self.instrument_ranges.keys())
        general.extend(drums)
        return general

    def _get_drum_instrument_code(self, program_code: int):
        """
        for key in self.drum_instrument_ranges.keys():
            if program_code in self.drum_instrument_ranges[key]:
                return key
        """
        return self.default_drum_instrument

    def _get_instrument_code(self, program_code: int):
        for key in self.instrument_ranges.keys():
            if program_code in self.instrument_ranges[key]:
                return key
        return self.default_instrument

    def is_txt_valid_note(self, line):
        all_instruments = self.get_all_instruments()
        try:
            separator_index = line.index(';')
            if line[0] not in ['n', 'r', 'p']:
                return False
            if line[1:5] not in all_instruments:
                return False
            if separator_index is None or separator_index < 6:
                return False
            if line[0] != 'p' and line[len(line) - 2] not in self.note_to_length.keys():
                return False
        except Exception as e:
            return False
        return True

    def txt_to_midi(self, input_path: str, output_path: str) -> None:
        song = pretty_midi.PrettyMIDI()

        offset = 0.0
        bpm = self.DEFAULT_BPM

        used_instruments = {}

        with open(input_path, 'r') as input_file:
            events = input_file.readlines()

        for i, event in enumerate(events):
            if not self.is_txt_valid_note(event):
                logging.info(f"line {event[:-1]} is invalid")
                continue

            try:
                event_type = event[0]
                event_instrument = event[1:5]
                separator = event.index(';')

                duration = int(event[5:separator])
                offset = offset + self._calculate_offset_from_representation(bpm, duration)
                is_drum = False

                # adding new instrument
                if event_instrument not in used_instruments.keys():
                    if event_instrument in self.drum_instrument_to_number.keys():
                        is_drum = True
                        instrument_code = self.drum_instrument_to_number[event_instrument]
                    elif event_instrument in self.instrument_to_number.keys():
                        instrument_code = self.instrument_to_number[event_instrument]
                    else:
                        logging.info(f"UNKNOWN INSTRUMENT: {event_instrument}\nskipping this event")
                        continue
                    used_instruments[event_instrument] = pretty_midi.Instrument(program=instrument_code, is_drum=is_drum,
                                                                                name=event_instrument)
                if event_type == 'p':  # pitch
                    pitch_value = event[separator+1:]
                    if int(pitch_value) > 8191:
                        logging.info(f"Line {i}: Pitch value {pitch_value} too big, setting to max")
                        pitch_value = 8190
                    elif int(pitch_value) < -8192:
                        logging.info(f"Line {i}: Pitch value {pitch_value} too small, setting to min")
                        pitch_value = -8190

                    new_pitch = pretty_midi.PitchBend(pitch=int(pitch_value), time=float(offset))
                    used_instruments[event_instrument].pitch_bends.append(new_pitch)

                if event_type in ['r', 'n']:  # drum or note
                    note_duration = self._get_note_length_in_seconds(bpm, event[len(event)-2])  # przedostatni
                    pitch_value = int(event[separator+1:len(event)-2])  # -2 a nie -1 bo jeszcze enter na końcu?
                    note_end = offset + note_duration
                    new_note = pretty_midi.Note(velocity=self.DEFAULT_VELOCITY, pitch=pitch_value, start=offset, end=note_end)
                    used_instruments[event_instrument].notes.append(new_note)

            except ValueError as e:
                print(f"Line \"{event[:-1]}\" nr {i} contains invalid value. {e}")

        for instrument in used_instruments.values():
            song.instruments.append(instrument)
            print(str(instrument))

        song.write(str(output_path))

    def try_fix_txt_file(self, seed_notes, input_path, output_path):
        pass

    def _parse_midi(self, path):
        midi_data = None
        try:
            midi_data = pretty_midi.PrettyMIDI(path)
            midi_data.remove_invalid_notes()
        except Exception as e:
            print(f"{e}\nerror reading midi file {path}")
        return midi_data

    def _get_note_length_representation(self, bpm, note_duration):
        rounding = 1.1
        beat_length = 60.0 / bpm
        x = note_duration/beat_length

        if x < (1/4) * rounding:
            return 's'  # szesnastka 1/4 #1/16
        if x < (1/2) * rounding:
            return 'o'  # ósemka 1/2 #1/8
        if x < 1 * rounding:
            return 'q'  # ćwierćnuta 1 #1/4
        if x < 2 * rounding:
            return 'h'  # półnuta
        if x < 4 * rounding:
            return 'w'  # whole note
        return 'd'      # double note?

    def _get_note_length_in_seconds(self, bpm, code):
        beat_length = 60.0 / bpm
        if code in self.note_to_length.keys():
            return self.note_to_length[code] * beat_length
        return 1/2 * beat_length

    def _calculate_offset_length_representation(self, bpm, offset_duration):
        beat_length = 60.0 / bpm
        return (offset_duration*100)/beat_length

    def _calculate_offset_from_representation(self, bpm, representation):
        beat_length = 60.0 / bpm
        return representation * beat_length / 100
