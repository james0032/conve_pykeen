# Kubernetes Deployment Guide

This guide explains how to deploy and use ConvE PyKEEN on Kubernetes.

## Prerequisites

- Access to a Kubernetes cluster with GPU support
- `kubectl` configured to access the cluster
- Code copied to NFS mount: `/projects/aixb/jchung/everycure/git/conve_pykeen`
- Training data available on NFS or PVC

## Quick Start

### Option 1: Simple Pod (Recommended)

The simplest way to get started:

```bash
# Deploy the pod
kubectl apply -f k8s-pod-simple.yaml

# Wait for pod to be ready
kubectl wait --for=condition=Ready pod/conve-pykeen-training --timeout=300s

# Attach to the pod
kubectl attach -it conve-pykeen-training
```

### Option 2: Pod with ConfigMap

For more control over configuration:

```bash
# Create ConfigMaps
kubectl apply -f k8s-configmap.yaml

# Deploy the pod
kubectl apply -f k8s-pod-with-configmap.yaml

# Attach to the pod
kubectl attach -it conve-pykeen-training
```

## Available Configurations

### GPU Options

Edit the YAML file to select the GPU type you need:

```yaml
resources:
  limits:
    # Uncomment ONE of these:
    #nvidia.com/mig-1g.5gb: 1      # Small - 5GB VRAM
    #nvidia.com/mig-2g.10gb: 1     # Medium - 10GB VRAM
    nvidia.com/mig-3g.20gb: 1      # Large - 20GB VRAM (default)
    #nvidia.com/mig-7g.40gb: 1     # XLarge - 40GB VRAM
```

### Memory and CPU

Adjust based on your needs:

```yaml
resources:
  limits:
    cpu: '3'           # Adjust CPU cores
    memory: 248G       # Adjust memory
    ephemeral-storage: 1G
```

## Usage

### Once Pod is Running

```bash
# Attach to pod
kubectl attach -it conve-pykeen-training

# Inside the pod, you'll be in /workspace/conve_pykeen
```

### Running the Pipeline

```bash
# Full pipeline (inside pod)
./run_pipeline.sh /projects/aixb/jchung/data /workspace/output

# Or step by step:
python preprocess.py \
  --train /projects/aixb/jchung/data/train.tsv \
  --valid /projects/aixb/jchung/data/valid.tsv \
  --test /projects/aixb/jchung/data/test.tsv \
  --node-dict /projects/aixb/jchung/data/node_dict \
  --rel-dict /projects/aixb/jchung/data/rel_dict \
  --output-dir /workspace/data/processed

python train.py \
  --train /workspace/data/processed/train.txt \
  --valid /workspace/data/processed/valid.txt \
  --test /workspace/data/processed/test.txt \
  --entity-to-id /workspace/data/processed/train_entity_to_id.tsv \
  --relation-to-id /workspace/data/processed/train_relation_to_id.tsv \
  --output-dir /workspace/output \
  --num-epochs 100
```

### Monitoring Training

From outside the pod:

```bash
# Check pod status
kubectl get pod conve-pykeen-training

# View logs
kubectl logs conve-pykeen-training

# Follow logs in real-time
kubectl logs -f conve-pykeen-training

# Check resource usage
kubectl top pod conve-pykeen-training
```

### Copying Results

```bash
# Copy results from pod to local machine
kubectl cp conve-pykeen-training:/workspace/output ./local-output

# Copy to NFS (from inside pod)
cp -r /workspace/output /projects/aixb/jchung/conve_results
```

## Directory Structure

Inside the pod:

```
/workspace/
├── conve_pykeen/          # Code (copied from NFS)
│   ├── train.py
│   ├── evaluate.py
│   ├── preprocess.py
│   └── ...
├── data/                  # PVC storage (persistent)
│   └── processed/         # Preprocessed data
└── output/                # Training outputs
    └── model/             # Trained model

/projects/aixb/jchung/     # NFS mount
├── everycure/git/conve_pykeen/  # Source code
└── data/                  # Your data
```

## Troubleshooting

### Pod Won't Start

```bash
# Check pod status
kubectl describe pod conve-pykeen-training

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp
```

### Out of Memory

Reduce batch size in training:

```bash
python train.py ... --batch-size 128  # instead of 256
```

Or edit the pod YAML to request more memory.

### GPU Not Available

Check if GPU is properly allocated:

```bash
# Inside pod
nvidia-smi

# If not working, check pod GPU allocation
kubectl describe pod conve-pykeen-training | grep -A 5 "Limits"
```

### Code Not Found

Manually copy code to workspace:

```bash
# Inside pod
cp -r /projects/aixb/jchung/everycure/git/conve_pykeen /workspace/
cd /workspace/conve_pykeen
pip install -r requirements.txt
```

### Dependencies Failed to Install

Install manually:

```bash
# Inside pod
pip install torch pykeen numpy pandas tqdm scikit-learn scipy
```

## Cleanup

```bash
# Delete the pod
kubectl delete pod conve-pykeen-training

# Delete ConfigMaps (if used)
kubectl delete configmap conve-requirements conve-startup
```

## Advanced Usage

### Running Multiple Experiments

Create multiple pods with different names:

```bash
# Edit k8s-pod-simple.yaml and change:
metadata:
  name: conve-experiment-1

# Deploy
kubectl apply -f k8s-pod-simple.yaml
```

### Persistent Storage

Results in `/workspace/data` are stored on the PVC and persist across pod restarts.

To save results permanently:

```bash
# Copy to NFS (inside pod)
cp -r /workspace/output /projects/aixb/jchung/experiments/$(date +%Y%m%d_%H%M%S)
```

### Background Training

To run training in background and detach:

```bash
# Inside pod, start training in background
nohup python train.py ... > training.log 2>&1 &

# Detach from pod (Ctrl+P, Ctrl+Q)

# Later, reattach
kubectl attach -it conve-pykeen-training

# Check training log
tail -f training.log
```

## Resource Recommendations

### For Small Datasets (< 10K triples)

```yaml
resources:
  limits:
    cpu: '2'
    memory: 64G
    nvidia.com/mig-1g.5gb: 1
```

### For Medium Datasets (10K - 100K triples)

```yaml
resources:
  limits:
    cpu: '3'
    memory: 128G
    nvidia.com/mig-2g.10gb: 1
```

### For Large Datasets (> 100K triples)

```yaml
resources:
  limits:
    cpu: '4'
    memory: 248G
    nvidia.com/mig-3g.20gb: 1
```

## Best Practices

1. **Always save results to NFS** for permanent storage
2. **Use PVC** (`/workspace/data`) for temporary working files
3. **Monitor GPU usage** with `nvidia-smi` during training
4. **Save checkpoints** regularly (automatic in training script)
5. **Copy logs** to NFS before deleting pod

## Example Workflow

```bash
# 1. Deploy pod
kubectl apply -f k8s-pod-simple.yaml
kubectl wait --for=condition=Ready pod/conve-pykeen-training
kubectl attach -it conve-pykeen-training

# 2. Inside pod - preprocess data
python preprocess.py \
  --train /projects/aixb/jchung/data/train.tsv \
  --valid /projects/aixb/jchung/data/valid.tsv \
  --test /projects/aixb/jchung/data/test.tsv \
  --node-dict /projects/aixb/jchung/data/node_dict \
  --rel-dict /projects/aixb/jchung/data/rel_dict \
  --output-dir /workspace/data/processed

# 3. Train model
python train.py \
  --train /workspace/data/processed/train.txt \
  --valid /workspace/data/processed/valid.txt \
  --test /workspace/data/processed/test.txt \
  --entity-to-id /workspace/data/processed/train_entity_to_id.tsv \
  --relation-to-id /workspace/data/processed/train_relation_to_id.tsv \
  --output-dir /workspace/output \
  --num-epochs 100 \
  --batch-size 256

# 4. Save results to NFS
EXPERIMENT_DIR=/projects/aixb/jchung/experiments/conve_$(date +%Y%m%d_%H%M%S)
mkdir -p $EXPERIMENT_DIR
cp -r /workspace/output/* $EXPERIMENT_DIR/
echo "Results saved to: $EXPERIMENT_DIR"

# 5. Exit pod
exit

# 6. Cleanup (from outside pod)
kubectl delete pod conve-pykeen-training
```

## Support

For issues with:
- **Kubernetes**: Check with your cluster administrator
- **Code/Training**: See main README.md and QUICKSTART.md
- **GPU**: Verify with `nvidia-smi` and check resource allocation
