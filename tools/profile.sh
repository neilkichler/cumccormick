#!/bin/sh
# --backtrace=lbr 
# --sample=process-tree 
# --trace=cuda,nvtx,osrt
nsys profile --cuda-graph-trace=node --cuda-memory-usage=true --vulkan-gpu-workload=false --opengl-gpu-workload=false --trace='cuda,osrt' $1
# nsys profile --cuda-graph-trace=graph --cuda-memory-usage=true --vulkan-gpu-workload=false --opengl-gpu-workload=false --trace='cuda,osrt' $1
