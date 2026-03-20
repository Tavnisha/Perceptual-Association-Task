# Perceptual Association Task

A Python (PsychoPy) based cognitive psychology experiment investigating how cultural orientation shapes perceptual associations between geometric shapes and social identity labels.

## Overview

This experiment examines whether participants from collectivistic versus individualistic cultural backgrounds differ in their perceptual responsiveness to socially attached labels. Specifically, it measures reaction times to shape-label pairings involving self-referential labels (You) versus close-other labels (Mother) versus neutral labels (Stranger), under brief stimulus exposure. The core hypothesis is that collectivistic participants show stronger or faster associations with mother-related labels, while individualistic participants respond more readily to self-referential labels.

## Experiment Design

### Stimuli
Shapes: Square, Circle, Triangle
Labels: You (Self), Mother, Stranger
Conditions: Matched vs. Mismatched (shape-label pairing)

### Procedure

| Phase | Stimulus Duration | Trials | Notes |
|-------|------------------|--------|-------|
| Practice | 0.5 seconds | 11 per block | Repeats until accuracy threshold is met |
| Test | 0.2 seconds | 180 trials | Main experimental trials |

Participants first complete practice sessions with feedback. Practice continues in loops until the participant exceeds the accuracy threshold. Once threshold is met, participants proceed to the test phase with no feedback. Responses are recorded via keyboard (`a` = correct association, `l` = incorrect association).

### Key Variables

| Variable | Description |
|----------|-------------|
| `Condition` | Matched or Mismatched |
| `Shape` | Triangle, Circle, or Square |
| `Label` | You, Mother, or Stranger |
| `key_resp.corr` | Accuracy (1 = correct, 0 = incorrect) |
| `key_resp.rt` | Reaction time (seconds) |
| `practice_accuracy` | Cumulative accuracy at end of each practice block |

## Output Data

Data is saved as a `.csv` file per participant, named:
```
[ParticipantID]_Perceptual_association_task_[YYYY-MM-DD_HHhMM.SS.ms].csv
```

Key columns in the output:

| Column | Description |
|--------|-------------|
| `participant` | Participant ID |
| `session` | Session number |
| `Condition` | Matched / Mismatched |
| `Shape` | Geometric shape shown |
| `Label` | Social label shown |
| `key_resp.corr` | Correctness of response |
| `key_resp.rt` | Reaction time in seconds |
| `practice_accuracy` | Accuracy score at end of practice block |

## Requirements

```
psychopy
numpy
pandas
```

Install dependencies:
```bash
pip install psychopy numpy pandas
```

> Note: PsychoPy requires a local machine with a display. It is not compatible with cloud-based environments like Google Colab.

## Running the Experiment

Run locally using PsychoPy:
```bash
python Perceptual_association_task.py
```

Or open directly in the PsychoPy Builder/Coder application.

## Built With

[PsychoPy](https://www.psychopy.org/) (v2024.2.1) and Python 3.

## Notes

Stimulus timing is precise and frame-rate dependent; a monitor refresh rate of approximately 60Hz is assumed. The experiment was developed and tested on Windows. File paths for stimuli images (triangle.png, circle.png, square.png) may need to be updated for your local setup. 
