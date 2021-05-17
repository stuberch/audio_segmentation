# Background
This project is part of a master thesis researching 'automated segmentation of the Ehrenreich opera collection' written at the Bern University of Applied Science. Goal of the script is to automatically identify  structural audio sequences via a reference recording. The method used is called *reference-based audio segmentation* (Prätzlich & Müller 2013) ([see publication on researchgate](https://www.researchgate.net/publication/303667411_Freischutz_Digital_a_case_study_for_reference-based_audio_segmentation_of_operas))

# Data
The dataset consists of 23 different versions of Mozarts opera *Die Zauberflöte* stored in https://ehrenreich.bfh.science/data/ (not open to public). To test the script 1'318 manually annotated segments where used, that can be found in assets.

# Paper
The paper was not published but is available opon request via the author (christian.stuber@students.bfh.ch) and the supervisor (eduard.klein@bfh.ch).

# Requirements
* Python 3.8
* librosa 0.8.0
* matplotlib (only for the notebooks)

# Audio Segmentation Script
The final script is 'audioSegmentation_A4.py' in connection with 'supportClasses.py'. All other files are previous versions or testcases of different functions of the script.

* Version A1 - Basis
* Version A2 - Semitone Tuning
* Version A3 - Constraints
* Version A4 - Parameter Tweaking (final)

The script takes a reference version (.wav) and segmentation (.csv) and searches for the same audio sequences in the unknown versions (.wav) in the assets folder. If test data (.csv) of those versions is provided, the received segments are automatically tested based on their similarity with the *Jaccard Index*. Output of the script are the two files *results.csv*, containing the automatically annotated segments and *log.csv*, measuring the performance of the script.
