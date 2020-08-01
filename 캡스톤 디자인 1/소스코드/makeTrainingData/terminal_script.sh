#!/bin/bash
#NPARK snr -20
for clean_file in ./clean_file/t3_audio*.wav; do
    for i in {1..100}; do
        python3 create_noisy.py --clean_file ${clean_file} --noise_file ./noise_file/NPARK_ch02_vol100.wav --output_clean_file ./output_clean_file/clean${i}.wav --output_noise_file ./output_noise_file/noise${i}.wav --output_noisy_file ./output_noisy_file/${clean_file:16:10}_noisy_npark${i}_m20.wav --snr -20;
        done;
    done;

#NFIELD snr -20
for clean_file in ./clean_file/t3_audio*.wav; do
    for i in {1..100}; do
        python3 create_noisy.py --clean_file ${clean_file} --noise_file ./noise_file/NFIELD_ch02_vol100.wav --output_clean_file ./output_clean_file/clean${i}.wav --output_noise_file ./output_noise_file/noise${i}.wav --output_noisy_file ./output_noisy_file/${clean_file:16:10}_noisy_nfield${i}_m20.wav --snr -20;
        done;
    done;

#NRIVER snr -20
for clean_file in ./clean_file/t3_audio*.wav; do
    for i in {1..100}; do
        python3 create_noisy.py --clean_file ${clean_file} --noise_file ./noise_file/NRIVER_ch02_vol100.wav --output_clean_file ./output_clean_file/clean${i}.wav --output_noise_file ./output_noise_file/noise${i}.wav --output_noisy_file ./output_noisy_file/${clean_file:16:10}_noisy_nriver${i}_m20.wav --snr -20;
        done;
    done;
