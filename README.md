# music_generator

Reorganisation of the code at https://www.analyticsvidhya.com/blog/2020/01/how-to-perform-automatic-music-generation/ as an introduciton to music generation.
Implemented the full pipeline and better organised in file/tasks to facilitate changes and further research.

Objectives: 
- Understand how music data is handled and processed
- Research better models (maybe GANs?)
- Implement multi instruments input-output

# Instructions
 - generate_data.py. This file processes MIDI file audio and produce .mat file containing x_train, x_val, y_train, y_val, along with the unique notes present, whose number determines the number of the dictionary.
- train.py loads in the .mat file produced in generate_data.py and trains the model by calling get_wavenet from wavenet_model.py.
- inference.py loads the model trained in train.py and runs inference to produce new music. It does it by loading some music from x_val in random_music and uses it as input to the model, which outputs a single note. Then the note is appended to random_music, and random_music[0] is removed. Then again, this version of random_music is fed to the model. Repeats the process 10 times to produce 10 notes (adjustable parameter).
- utilities.py contains functions to load and save midi files.

# Resources
Repositories with MIDI files: https://medium.com/@vinitasilaparasetty/list-of-midi-file-datasets-for-music-analysis-a4963360096e
