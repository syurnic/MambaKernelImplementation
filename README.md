# Mamba Kernel study: Pytorch and CUDA Implementation

## Overview
This project is both high-level(Pytorch) and low-level(CUDA) immplementation of Mamba and it's core algorithm Selective Scan.
The goal of this project is to understand sequential dependency bottleneck and CUDA architecture and distinction between SSM with transformer

### Focus
* Manual implementation of the parallel associative scan in C++
* Analyzing memory bottleneck during the recurrent state update.
* Writing high-level pytorch code.

### Current Status
* Core C++ Kernel: Implemented simple integer addition version. Forwarding operation will be implemented later
* Pytorch Binding: In progress
* Kernel Fusion: Not immplemented yet
