import songSimplifier
from pathlib import Path

beautiful_music = "D:\\cacofonix2\\beautiful_songs\\"
txt_files = Path(beautiful_music).glob('*.txt')

song_simplifier = songSimplifier.SongSimplifier()

for txt_file in txt_files:
    my_midi = Path.joinpath(Path(beautiful_music), txt_file.stem + ".mid")
    print(str(txt_file))
    song_simplifier.txt_to_midi(str(txt_file), str(my_midi))
    print('\n')


