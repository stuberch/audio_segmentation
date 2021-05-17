# Background
This project is part of a master thesis researching 'automated segmentation of the Ehrenreich opera collection' written at the Bern University of Applied Science. Goal of the script is to automatically itentify structural audio sequences via a reference recording. The method used is the *reference-based audio segmentation* ([see publication on researchgate](https://www.researchgate.net/publication/303667411_Freischutz_Digital_a_case_study_for_reference-based_audio_segmentation_of_operas))

# Data
The dataset consists of 23 different versions of Mozarts opera *Die Zauberflöte* stored in https://ehrenreich.bfh.science/data/ (not open to public). To test the script 1'318 manually annotated segments where used, that can be found in assets.

# Paper
The paper was not published but can be optained by requesting (christian.stuber@students.bfh.ch) and the supervisor (eduard.klein@bfh.ch).

# Requirements
* Python 3.8
* librosa 0.8.0
* matplotlib (only for the notebooks)

# Files
The final script is 'audioSegmentation_A4.py' in connection to 'supportClasses.py'. All other files are previous versions or testcases of different functions of the script.

##


