#!/usr/bin/env bash

# -- Get cpu info --
echo "--- CPU Architecture ---"
CPU_MODEL=$(grep 'model name' /proc/cpuinfo | uniq)
CORES_PER_SOCKET=$(lscpu | grep "Core(s) per socket")
SOCKETS=$(lscpu | grep "Socket(s)")
echo "$CPU_MODEL"
echo "$CORES_PER_SOCKET"
echo "$SOCKETS"
echo "------------------------"