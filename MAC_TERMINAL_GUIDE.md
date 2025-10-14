# Mac Terminal Guide for Kubernetes Pods

This guide explains how to work with Kubernetes pods on Mac, including the correct way to run background tasks and detach from sessions.

## ⚠️ Important: kubectl attach vs kubectl exec

### The Problem with `kubectl attach`

The common advice "Press Ctrl+P then Ctrl+Q to detach" **does NOT work on Mac** with kubectl. This only works with Docker's `attach` command.

With `kubectl attach`:
- ❌ Ctrl+P, Ctrl+Q doesn't work
- ❌ Cannot detach without killing the pod
- ❌ Closing terminal stops your processes

### ✅ The Solution: Use `kubectl exec`

Our scripts use `kubectl exec` which opens a proper shell session.

## Mac-Friendly Methods

### Method 1: Use tmux Inside Pod ⭐ **Recommended**

**tmux** lets you create persistent sessions that survive disconnects.

```bash
# 1. Connect to pod
./k8s-helper-v2.sh -n ml-training attach

# 2. Inside pod, start tmux
tmux

# 3. Work normally (run training, etc.)
python train.py --train ... --output-dir /workspace/output

# 4. Detach from tmux (NOT from pod)
# Press: Ctrl+B, then press D

# 5. Exit pod normally
exit

# 6. Later, reconnect and resume
./k8s-helper-v2.sh -n ml-training attach
tmux attach
```

**tmux Key Bindings:**
```
Ctrl+B, then D    - Detach from tmux session
Ctrl+B, then C    - Create new window
Ctrl+B, then N    - Next window
Ctrl+B, then P    - Previous window
Ctrl+B, then [    - Scroll mode (Q to quit)
```

### Method 2: Use screen Inside Pod

Similar to tmux, but simpler:

```bash
# Inside pod
screen

# Work normally
python train.py ...

# Detach: Ctrl+A, then D
# Reattach later: screen -r
```

### Method 3: Run in Background with nohup

For long-running tasks that don't need interaction:

```bash
# Inside pod
nohup python train.py \
  --train /workspace/data/processed/train.txt \
  --valid /workspace/data/processed/valid.txt \
  --test /workspace/data/processed/test.txt \
  --entity-to-id /workspace/data/processed/train_entity_to_id.tsv \
  --relation-to-id /workspace/data/processed/train_relation_to_id.tsv \
  --output-dir /workspace/output \
  > training.log 2>&1 &

# Check it's running
jobs
ps aux | grep python

# Exit pod
exit

# Later, check logs
./k8s-helper-v2.sh -n ml-training exec 'tail -f /workspace/conve_pykeen/training.log'
```

### Method 4: Multiple Terminal Windows

Simple but effective:

```bash
# Terminal 1 - Training
./k8s-helper-v2.sh -n ml-training attach
python train.py ...

# Terminal 2 - Monitoring
./k8s-helper-v2.sh -n ml-training exec 'watch -n 5 nvidia-smi'

# Terminal 3 - Logs
./k8s-helper-v2.sh -n ml-training logs -f
```

## Complete Workflows

### Workflow 1: Long Training Session with tmux

```bash
# 1. Connect to pod
./k8s-helper-v2.sh -n ml-training attach

# 2. Start tmux
tmux new -s training

# 3. Start training in tmux
cd /workspace/conve_pykeen
python train.py \
  --train /workspace/data/processed/train.txt \
  --valid /workspace/data/processed/valid.txt \
  --test /workspace/data/processed/test.txt \
  --entity-to-id /workspace/data/processed/train_entity_to_id.tsv \
  --relation-to-id /workspace/data/processed/train_relation_to_id.tsv \
  --output-dir /workspace/output \
  --num-epochs 100

# 4. Detach from tmux
# Press: Ctrl+B, then D

# 5. Exit pod
exit

# 6. Check progress later
./k8s-helper-v2.sh -n ml-training attach
tmux attach -t training

# 7. To kill tmux session when done
tmux kill-session -t training
```

### Workflow 2: Background Training with nohup

```bash
# 1. Connect and start training
./k8s-helper-v2.sh -n ml-training attach

# 2. Run in background
cd /workspace/conve_pykeen
nohup ./run_pipeline.sh /workspace/data /workspace/output > pipeline.log 2>&1 &

# 3. Get the process ID
echo $!  # Note this down
# Or: ps aux | grep train

# 4. Exit safely
exit

# 5. Monitor from outside
./k8s-helper-v2.sh -n ml-training logs -f

# 6. Check specific log
./k8s-helper-v2.sh -n ml-training exec 'tail -f /workspace/conve_pykeen/pipeline.log'

# 7. Check if still running
./k8s-helper-v2.sh -n ml-training exec 'ps aux | grep python'
```

### Workflow 3: Development with Multiple Windows

```bash
# Terminal 1 - Main work
./k8s-helper-v2.sh -n ml-training attach
cd /workspace/conve_pykeen
# Edit, test, run commands

# Terminal 2 - GPU monitoring
./k8s-helper-v2.sh -n ml-training gpu
# Or: watch -n 5 './k8s-helper-v2.sh -n ml-training gpu'

# Terminal 3 - Log watching
./k8s-helper-v2.sh -n ml-training logs -f

# Terminal 4 - File browsing
./k8s-helper-v2.sh -n ml-training exec 'ls -lah /workspace/output'
```

## Installing tmux/screen in Pod

If tmux/screen aren't available:

```bash
# Inside pod (Debian/Ubuntu based)
apt-get update && apt-get install -y tmux screen

# Or add to Dockerfile/pod spec
```

## Mac Terminal Tips

### Tip 1: Use iTerm2

[iTerm2](https://iterm2.com/) is better than default Terminal.app:
- Split panes (Cmd+D vertical, Cmd+Shift+D horizontal)
- Better tab management
- Search in output (Cmd+F)
- Instant replay (Cmd+Opt+B)

### Tip 2: Create Terminal Profiles

In iTerm2, create a profile for Kubernetes work:
1. Preferences → Profiles → + (new profile)
2. Name: "Kubernetes"
3. Command → Send text at start: `cd /path/to/conve_pykeen`
4. Use this profile for K8s work

### Tip 3: Keyboard Shortcuts

```bash
Cmd+T         - New tab
Cmd+D         - Split vertically (iTerm2)
Cmd+Shift+D   - Split horizontally (iTerm2)
Cmd+]         - Next split pane
Cmd+[         - Previous split pane
Cmd+K         - Clear scrollback
Cmd+F         - Find in output
```

### Tip 4: Use aliases

Add to `~/.zshrc` or `~/.bashrc`:

```bash
# Kubernetes aliases
alias k='kubectl'
alias kgp='kubectl get pods'
alias kl='kubectl logs -f'
alias kx='kubectl exec -it'

# ConvE specific
alias conve-attach='cd /path/to/conve_pykeen && ./k8s-helper-v2.sh -n ml-training attach'
alias conve-logs='./k8s-helper-v2.sh -n ml-training logs -f'
alias conve-gpu='./k8s-helper-v2.sh -n ml-training gpu'
```

## Common Scenarios

### Scenario 1: Training takes 8 hours, I need to close my laptop

**Solution:** Use tmux or nohup

```bash
# Option A: tmux (can interact later)
./k8s-helper-v2.sh attach
tmux new -s training
python train.py ...
# Ctrl+B, then D
exit

# Option B: nohup (fire and forget)
./k8s-helper-v2.sh attach
nohup python train.py ... > training.log 2>&1 &
exit
```

Close laptop safely. Training continues in pod.

### Scenario 2: Want to monitor GPU while training

```bash
# Terminal 1: Training with tmux
./k8s-helper-v2.sh attach
tmux new -s training
python train.py ...
# Ctrl+B, then D
exit

# Terminal 2: GPU monitoring
watch -n 5 './k8s-helper-v2.sh gpu'
```

### Scenario 3: Training failed, need to debug

```bash
# Check logs
./k8s-helper-v2.sh logs | tail -100

# Or enter pod to investigate
./k8s-helper-v2.sh attach

# Check last training run
tail -100 /workspace/conve_pykeen/training.log

# Check GPU status
nvidia-smi

# Check disk space
df -h

# Check running processes
ps aux | grep python
```

### Scenario 4: Want to run multiple experiments in parallel

```bash
# Use tmux windows
./k8s-helper-v2.sh attach
tmux

# Window 1: Experiment 1
python train.py --output-dir /workspace/exp1 ...
# Ctrl+B, then C (new window)

# Window 2: Experiment 2
python train.py --output-dir /workspace/exp2 ...
# Ctrl+B, then C (new window)

# Window 3: Monitoring
watch -n 5 nvidia-smi

# Switch windows: Ctrl+B, then N (next) or P (previous)
# Detach: Ctrl+B, then D
```

## Troubleshooting

### Problem: Accidentally closed terminal, lost my session

**If using tmux:**
```bash
./k8s-helper-v2.sh attach
tmux list-sessions  # Find your session
tmux attach -t training
```

**If using nohup:**
```bash
./k8s-helper-v2.sh attach
ps aux | grep python  # Check if still running
tail -f training.log  # Check progress
```

**If neither:**
😢 Process probably died. Need to restart.

### Problem: tmux not found

```bash
./k8s-helper-v2.sh attach
apt-get update && apt-get install -y tmux
```

### Problem: Can't remember tmux commands

Inside tmux:
```
Ctrl+B, then ?    - Show all key bindings
Ctrl+B, then :    - Command mode (type 'list-keys')
```

### Problem: Process still running after I exited?

Check:
```bash
./k8s-helper-v2.sh exec 'ps aux | grep python'
```

Kill if needed:
```bash
./k8s-helper-v2.sh exec 'pkill -f train.py'
```

## Best Practices

✅ **DO:**
- Use tmux for interactive work
- Use nohup for fire-and-forget tasks
- Save logs to files
- Test with short epochs first
- Document your tmux session names

❌ **DON'T:**
- Rely on terminal connection staying alive
- Run long tasks without tmux/nohup
- Forget to save outputs to files
- Close laptop without detaching from tmux

## Quick Reference Card

```
╔══════════════════════════════════════════════════════════╗
║              Mac Kubernetes Cheat Sheet                  ║
╠══════════════════════════════════════════════════════════╣
║ Connect to pod:                                          ║
║   ./k8s-helper-v2.sh -n <namespace> attach              ║
║                                                          ║
║ Inside pod - Start tmux:                                ║
║   tmux new -s training                                  ║
║                                                          ║
║ tmux - Detach:                                          ║
║   Ctrl+B, then D                                        ║
║                                                          ║
║ tmux - Reattach:                                        ║
║   tmux attach -t training                               ║
║                                                          ║
║ Background process:                                      ║
║   nohup python train.py ... > log.txt 2>&1 &           ║
║                                                          ║
║ Check if running:                                       ║
║   ps aux | grep python                                  ║
║                                                          ║
║ Kill process:                                           ║
║   pkill -f train.py                                     ║
╚══════════════════════════════════════════════════════════╝
```

## Summary

**The Key Takeaway:**

On Mac, you cannot detach from kubectl attach. Instead:

1. ✅ **Use tmux** for interactive sessions you want to resume
2. ✅ **Use nohup** for background tasks
3. ✅ **Use multiple terminal windows** for monitoring
4. ✅ **Save logs to files** for later inspection

The updated `k8s-helper-v2.sh` now uses `kubectl exec` which works correctly on Mac! 🎉
