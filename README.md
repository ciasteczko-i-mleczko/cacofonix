# Cacofonix
character-level LSTM generating new midi music

## Inspirations and sources
Inspired by [this article](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) I decided to experiment with generating music with LSTM. Used [this sample code](https://github.com/keras-team/keras-io/blob/master/examples/generative/lstm_character_level_text_generation.py) from Keras examples to create character level text generator, but instead of regular text I provided it with my representation of music.

### Source midi files
Scrapped a bunch of sample midi files from various sites, mostly old video games music because it is simple, fewer sounds playing at once. I'm not a composer and some of it is probably copyrighted, so I cannot upload it here.

## Results
I run this on my measly GTX970 many times for many hours. After some time, around epoch 80-110, no matter if I used Adam or RSMProp optimiser, it reached minimum training error of ~0.4. 
And after that error just kept growing and growing. 
### Sample music:
- https://soundcloud.com/user-223074311/song-20210722-105303
- https://soundcloud.com/user-223074311/song-20210720-134100
- https://onlinesequencer.net/2735360

## Converting midi to txt
I created a custom class `SongSimplifier` that uses `pretty-midi` library to read midi files and convert them to my custom txt format.
It reads midi events one by one, categorises them to one of predefined subset of all midi instruments and saves the events sorted by start time to a new file.\
eg: `nAGPI118;96q`
- `n` - type of event (n - note, r - drum, p - pitch shift)
- `AGPI` - instrument token (ex. AGPI means acoustic grand piano)
- `118` - offset from previous sound
- `;` - a semicolon to separate numeric values
- `96` - sound pitch value
- `q` - note length, `q` means quarter-note

Example converted file is provided in `sample_data` directory.

Obviously `SongSimplifier` can also convert this txt representation back to midi.\
With this simplification some information is lost, all generated notes have the same default velocity, duration can only be one of the predefined values and bpm is set to default but after conversion most of the songs still sound very similar to source midi, just playing a slower or faster.

## How I plan to make this project better
- #### Get simpler input midi files
Resulting music sounds much better when only one instrument is playing at the time. Model has trouble with generating polyphonic music, even basic chords. Maybe lowering my expectations and providing simpler midi files could help generate better sounding results.
- #### Experiment with different model topologies
- #### Removing STAR and ENDD tokens
I've added STAR and ENDD tokens hoping it could somehow make the model understand start and end of each song and help grasp general song structure like chorus and verses but it lowered the quality of generated music in general.

### Disclaimer
This code I wrote for myself, for fun, so it might not be the most beautiful python.
May contain some Polish, and may not contain very explanatory comments. 
