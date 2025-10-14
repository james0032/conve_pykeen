# Mac Terminal Quick Reference for Kubernetes

## ❌ Common Misconception

**"Press Ctrl+P, Ctrl+Q to detach"** - This does NOT work on Mac with kubectl!

This only works with Docker's `attach` command, not Kubernetes.

## ✅ The Right Way on Mac

### Use our updated scripts:

```bash
# This now uses 'kubectl exec' which works on Mac
./k8s-helper-v2.sh -n ml-training attach
```

### Inside the pod, use tmux:

```bash
# Start tmux
tmux new -s training

# Run your training
python train.py ...

# Detach: Ctrl+B, then D
# (That's: Hold Ctrl+B, release, then press D)

# Exit pod
exit

# Later, reconnect
./k8s-helper-v2.sh -n ml-training attach
tmux attach -t training
```

## Quick Cheat Sheet

| Task | Command |
|------|---------|
| Connect to pod | `./k8s-helper-v2.sh -n <ns> attach` |
| Start tmux | `tmux new -s training` |
| Detach from tmux | `Ctrl+B, then D` |
| Reattach tmux | `tmux attach -t training` |
| List tmux sessions | `tmux ls` |
| Background task | `nohup command > log.txt 2>&1 &` |
| Check running | `ps aux \| grep python` |
| Kill process | `pkill -f train.py` |

## tmux Essential Commands

```
Ctrl+B, D       - Detach
Ctrl+B, C       - New window
Ctrl+B, N       - Next window
Ctrl+B, P       - Previous window
Ctrl+B, [       - Scroll mode (Q to exit)
Ctrl+B, ?       - Help
```

## Example: Train Overnight

```bash
# 1. Connect
./k8s-helper-v2.sh attach

# 2. Start tmux
tmux new -s overnight

# 3. Run training
python train.py --num-epochs 100 ...

# 4. Detach (Ctrl+B, then D)

# 5. Exit pod
exit

# 6. Close laptop - training continues!

# 7. Next day - reconnect
./k8s-helper-v2.sh attach
tmux attach -t overnight
```

## What Changed

- ✅ k8s-helper-v2.sh now uses `kubectl exec` (works on Mac)
- ✅ Pod automatically installs tmux
- ✅ Startup banner shows tmux instructions
- ✅ Complete guide in MAC_TERMINAL_GUIDE.md

## Files

- `k8s-helper-v2.sh` - Updated script (uses exec)
- `k8s-pod-simple.yaml` - Updated pod (installs tmux)
- `MAC_TERMINAL_GUIDE.md` - Complete guide
- `MAC_QUICK_REFERENCE.md` - This file

Happy training! 🚀
