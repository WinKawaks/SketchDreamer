#!/bin/bash

Home="/mnt/fast/nobackup/users/WinKawaks"
echo $Home

echo 'args:' $@

# with control
$Home/anaconda/bin/python run_object_sketching.py $@ --num_strokes 16 --num_segments 5 --num_sketches 5 --attention_init 0 --text_prompt "a_simple_drawing_of_a_bicycle" --sketches_edit "input/wheel.png" --output_name "bicycle_w" --train_with_diffusion --control

# without control (VectorFusion)
$Home/anaconda/bin/python run_object_sketching.py $@ --num_strokes 16 --num_segments 5 --num_sketches 5 --attention_init 0 --text_prompt "a_simple_drawing_of_a_bicycle" --output_name "bicycle_o" --train_with_diffusion