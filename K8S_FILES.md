# Kubernetes Files Overview

This repository includes several Kubernetes configuration files for different deployment scenarios.

## Files

### 1. k8s-pod-simple.yaml (Recommended)
**Best for: Quick start and most use cases**

- Single YAML file, easy to deploy
- Auto-installs dependencies on startup
- Copies code from NFS automatically
- Includes helpful startup banner
- Self-contained configuration

**Deploy:**
```bash
kubectl apply -f k8s-pod-simple.yaml
kubectl attach -it conve-pykeen-training
```

### 2. k8s-pod-with-configmap.yaml
**Best for: Custom configurations and reusable setups**

- Uses ConfigMaps for requirements and startup script
- Separates configuration from pod definition
- Good for managing multiple pods with shared config
- Requires deploying ConfigMap first

**Deploy:**
```bash
kubectl apply -f k8s-configmap.yaml
kubectl apply -f k8s-pod-with-configmap.yaml
```

### 3. k8s-configmap.yaml
**Configuration data for k8s-pod-with-configmap.yaml**

Contains:
- Python requirements
- Startup script

### 4. k8s-pod.yaml (Basic)
**Best for: Simple deployments without auto-setup**

- Minimal configuration
- No automatic setup
- Requires manual code copying and dependency installation

**Deploy:**
```bash
kubectl apply -f k8s-pod.yaml
```

### 5. k8s-helper.sh (Utility Script)
**Command-line helper for managing the pod**

Features:
- Deploy/delete pod
- Attach to pod shell
- View logs and status
- Copy files to/from pod
- Monitor GPU usage
- Check resource usage

**Usage:**
```bash
./k8s-helper.sh deploy      # Deploy pod
./k8s-helper.sh attach      # Attach to shell
./k8s-helper.sh logs -f     # Follow logs
./k8s-helper.sh gpu         # Check GPU
./k8s-helper.sh monitor     # Monitor resources
./k8s-helper.sh copy-from /workspace/output ./results
```

## Which File to Use?

### Quick Start → Use k8s-pod-simple.yaml
```bash
kubectl apply -f k8s-pod-simple.yaml
kubectl attach -it conve-pykeen-training
```

### With Helper Script → Even Easier
```bash
./k8s-helper.sh deploy
./k8s-helper.sh attach
```

### Custom Configuration → Use k8s-pod-with-configmap.yaml
```bash
# Edit k8s-configmap.yaml to customize requirements/startup
kubectl apply -f k8s-configmap.yaml
kubectl apply -f k8s-pod-with-configmap.yaml
```

### Minimal Setup → Use k8s-pod.yaml
```bash
kubectl apply -f k8s-pod.yaml
# Then manually set up inside pod
```

## Common Configurations

### GPU Selection

All YAML files include GPU options. Edit to select your GPU:

```yaml
resources:
  limits:
    # Uncomment ONE:
    #nvidia.com/mig-1g.5gb: 1      # 5GB
    #nvidia.com/mig-2g.10gb: 1     # 10GB
    nvidia.com/mig-3g.20gb: 1      # 20GB (default)
    #nvidia.com/mig-7g.40gb: 1     # 40GB
```

### Memory and CPU

Adjust based on dataset size:

```yaml
resources:
  limits:
    cpu: '3'
    memory: 248G
```

### NFS Path

Update to match your code location:

```yaml
# In startup script, change:
/projects/aixb/jchung/everycure/git/conve_pykeen
# To your actual path
```

## Documentation

- **K8S_DEPLOYMENT.md**: Comprehensive deployment guide
- **README.md**: Main project documentation
- **QUICKSTART.md**: Quick start guide

## Example Workflow

```bash
# 1. Deploy with helper script
./k8s-helper.sh deploy

# 2. Attach to pod
./k8s-helper.sh attach

# 3. Inside pod - run pipeline
./run_pipeline.sh /projects/aixb/jchung/data /workspace/output

# 4. Monitor from another terminal
./k8s-helper.sh monitor

# 5. Copy results
./k8s-helper.sh copy-from /workspace/output ./results

# 6. Cleanup
./k8s-helper.sh delete
```

## Troubleshooting

### Pod Won't Start
```bash
./k8s-helper.sh status
kubectl describe pod conve-pykeen-training
```

### Check Logs
```bash
./k8s-helper.sh logs -f
```

### GPU Not Working
```bash
./k8s-helper.sh gpu
```

### Dependencies Failed
```bash
# Attach and install manually
./k8s-helper.sh attach
pip install -r requirements.txt
```

## Tips

1. **Use the helper script** - It makes everything easier
2. **Start with k8s-pod-simple.yaml** - Best for beginners
3. **Save results to NFS** - Data in /workspace is temporary
4. **Monitor GPU usage** - Use `./k8s-helper.sh gpu`
5. **Check logs regularly** - Use `./k8s-helper.sh logs -f`

## File Comparison

| Feature | simple | configmap | basic | helper |
|---------|--------|-----------|-------|--------|
| Auto-install deps | ✅ | ✅ | ❌ | N/A |
| Auto-copy code | ✅ | ✅ | ❌ | N/A |
| Startup banner | ✅ | ✅ | ❌ | N/A |
| Easy to edit | ✅ | ⚠️ | ✅ | N/A |
| Reusable config | ❌ | ✅ | ❌ | N/A |
| Management tools | ❌ | ❌ | ❌ | ✅ |
| Recommended | ✅ | ⚠️ | ❌ | ✅ |

Legend: ✅ Yes, ❌ No, ⚠️ Partial/Complex, N/A Not Applicable
