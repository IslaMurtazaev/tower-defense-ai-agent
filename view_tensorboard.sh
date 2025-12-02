#!/bin/bash
# Quick script to launch TensorBoard for Q-Learning training

LOG_DIR="runs/q_learning"

if [ ! -d "$LOG_DIR" ]; then
    echo "No TensorBoard logs found in $LOG_DIR"
    echo "Start training first to generate logs."
    exit 1
fi

echo "Starting TensorBoard..."
echo "View at: http://localhost:6006"
echo "Press Ctrl+C to stop"
echo ""

tensorboard --logdir="$LOG_DIR" --port=6006
