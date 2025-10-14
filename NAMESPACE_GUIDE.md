# Kubernetes Namespace Configuration Guide

This guide explains how to properly set up and use Kubernetes namespaces with the ConvE PyKEEN deployment.

## What is a Namespace?

Kubernetes namespaces provide a way to isolate resources within a cluster. They're useful for:
- Separating environments (dev, test, prod)
- Organizing projects or teams
- Managing resource quotas
- Access control

## Namespace Configuration Methods

The helper scripts support **4 methods** for setting namespaces (in priority order):

### Method 1: Command Line Flag (Highest Priority) ⭐ **Recommended**

```bash
# Using short flag
./k8s-helper-v2.sh -n my-namespace deploy

# Using long flag
./k8s-helper-v2.sh --namespace ml-training attach

# With multiple options
./k8s-helper-v2.sh -n prod -p my-pod status
```

**Pros:**
- Explicit and clear
- Override any environment settings
- Easy to change per command
- Self-documenting

**Use when:** You want explicit control per command

### Method 2: Environment Variable (Persistent)

```bash
# Set for current shell session
export NAMESPACE=my-namespace

# Now all commands use this namespace
./k8s-helper-v2.sh deploy
./k8s-helper-v2.sh attach
./k8s-helper-v2.sh logs -f

# Unset when done
unset NAMESPACE
```

**Pros:**
- Set once, use multiple times
- Good for working in same namespace
- Can be added to shell profile

**Use when:** You're working in the same namespace for a while

### Method 3: Inline Environment Variable (One-time)

```bash
# Set only for this single command
NAMESPACE=my-namespace ./k8s-helper-v2.sh deploy

# Next command uses default again
./k8s-helper-v2.sh attach  # Uses 'default'
```

**Pros:**
- Doesn't pollute environment
- One-off namespace changes
- Clean and explicit

**Use when:** You need to run a single command in a different namespace

### Method 4: Default (Fallback)

```bash
# Uses 'default' namespace if nothing else is set
./k8s-helper-v2.sh deploy
```

**Pros:**
- No configuration needed
- Quick for simple setups

**Use when:** You only use the default namespace

## Priority Order

When multiple methods are used, the priority is:

```
Command Line Flag > Environment Variable > Default
```

Example:
```bash
# Environment says 'dev', but command line overrides to 'prod'
export NAMESPACE=dev
./k8s-helper-v2.sh -n prod deploy  # Uses 'prod'
```

## Complete Examples

### Example 1: Development Workflow

```bash
# Set up development namespace for the session
export NAMESPACE=ml-dev

# Deploy
./k8s-helper-v2.sh deploy

# Work with pod
./k8s-helper-v2.sh attach

# Monitor
./k8s-helper-v2.sh monitor

# Cleanup
./k8s-helper-v2.sh delete

# Unset when done
unset NAMESPACE
```

### Example 2: Multi-Environment Management

```bash
# Deploy to dev
./k8s-helper-v2.sh -n dev deploy

# Deploy to test
./k8s-helper-v2.sh -n test deploy

# Deploy to prod
./k8s-helper-v2.sh -n prod deploy

# Monitor prod while working in dev
./k8s-helper-v2.sh -n dev attach &
./k8s-helper-v2.sh -n prod monitor
```

### Example 3: Team Collaboration

```bash
# Each team member uses their own namespace
export NAMESPACE=jchung-experiments

# Deploy personal instance
./k8s-helper-v2.sh deploy

# Share results from production namespace
./k8s-helper-v2.sh -n prod copy-from /workspace/output ./shared-results
```

### Example 4: Shell Profile Setup

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# Default Kubernetes namespace for ConvE
export NAMESPACE=ml-training

# Alias for convenience
alias k8s-conve='cd /path/to/conve_pykeen && ./k8s-helper-v2.sh'

# Now you can use:
# k8s-conve deploy
# k8s-conve -n prod attach
```

## Namespace Management

### Create a Namespace

```bash
# Create manually
kubectl create namespace my-namespace

# Or the script auto-creates when deploying
./k8s-helper-v2.sh -n new-namespace deploy
```

### List Namespaces

```bash
kubectl get namespaces

# Or abbreviated
kubectl get ns
```

### Check Current Context Default

```bash
kubectl config view --minify | grep namespace
```

### Set Default Namespace for kubectl

```bash
# Set default namespace in kubectl config
kubectl config set-context --current --namespace=my-namespace

# Now all kubectl commands use this namespace by default
kubectl get pods  # Uses my-namespace
```

## Best Practices

### 1. Use Descriptive Names

```bash
# Good
-n ml-training-dev
-n project-production
-n user-jchung-experiments

# Avoid
-n ns1
-n temp
-n test
```

### 2. Follow Naming Conventions

```bash
# Pattern: <project>-<environment>-<optional-user>
-n conve-dev
-n conve-prod
-n conve-dev-jchung
```

### 3. Document Your Namespaces

Create a `NAMESPACES.md` file:

```markdown
# Project Namespaces

- `conve-dev`: Development and testing
- `conve-staging`: Pre-production validation
- `conve-prod`: Production models
- `conve-experiments`: Individual experiments
```

### 4. Use Environment Variables for Persistent Work

```bash
# Working on development all day
export NAMESPACE=conve-dev

# But can still check production
./k8s-helper-v2.sh -n conve-prod status
```

### 5. Always Verify Namespace

The improved script shows the namespace at the start:

```
Configuration:
  Namespace: ml-training
  Pod Name: conve-pykeen-training
```

Always check this before running destructive operations!

## Common Patterns

### Pattern 1: Personal Development

```bash
# Everyone gets their own namespace
export NAMESPACE=conve-dev-$(whoami)
./k8s-helper-v2.sh deploy
```

### Pattern 2: Feature Branches

```bash
# Match git branch
BRANCH=$(git branch --show-current | tr '/' '-')
./k8s-helper-v2.sh -n "conve-$BRANCH" deploy
```

### Pattern 3: Experiment Tracking

```bash
# Use timestamp for experiments
EXPERIMENT="conve-exp-$(date +%Y%m%d-%H%M%S)"
./k8s-helper-v2.sh -n "$EXPERIMENT" deploy
```

## Troubleshooting

### Error: Namespace not found

```bash
# Check if namespace exists
kubectl get namespace my-namespace

# Create if missing
kubectl create namespace my-namespace

# Or let script auto-create
./k8s-helper-v2.sh -n my-namespace deploy
```

### Error: Pod not found in namespace

Make sure you're using the same namespace:

```bash
# Wrong
./k8s-helper-v2.sh -n dev deploy
./k8s-helper-v2.sh -n prod attach  # Pod is in 'dev'!

# Correct
./k8s-helper-v2.sh -n dev deploy
./k8s-helper-v2.sh -n dev attach
```

### Forgot which namespace?

```bash
# List all pods across namespaces
kubectl get pods --all-namespaces | grep conve

# Or find your pod
kubectl get pods -A | grep $(whoami)
```

### Access Denied

```bash
# Check your permissions
kubectl auth can-i create pods -n my-namespace

# Request access from cluster admin if needed
```

## Script Comparison

### Original Script (k8s-helper.sh)

```bash
# Only environment variable
export NAMESPACE=my-namespace
./k8s-helper.sh deploy
```

### Improved Script (k8s-helper-v2.sh) ⭐

```bash
# Method 1: Command line
./k8s-helper-v2.sh -n my-namespace deploy

# Method 2: Environment variable
export NAMESPACE=my-namespace
./k8s-helper-v2.sh deploy

# Method 3: Inline
NAMESPACE=my-namespace ./k8s-helper-v2.sh deploy

# Shows current configuration
# Auto-creates namespace if needed
# Clearer error messages
```

## Migration Guide

### From Original to Improved Script

```bash
# Old way
export NAMESPACE=my-namespace
./k8s-helper.sh deploy

# New way (backwards compatible)
export NAMESPACE=my-namespace
./k8s-helper-v2.sh deploy

# Or use command line (recommended)
./k8s-helper-v2.sh -n my-namespace deploy
```

Both scripts work, but `k8s-helper-v2.sh` provides better UX.

## Quick Reference

| Task | Command |
|------|---------|
| Use default namespace | `./k8s-helper-v2.sh deploy` |
| Use specific namespace | `./k8s-helper-v2.sh -n prod deploy` |
| Set for session | `export NAMESPACE=dev` |
| One-time override | `NAMESPACE=test ./k8s-helper-v2.sh deploy` |
| Check config | Command shows namespace at start |
| List namespaces | `kubectl get namespaces` |
| Create namespace | Auto-created or `kubectl create ns <name>` |

## Recommendations

✅ **DO:**
- Use command line flags (`-n`) for clarity
- Use descriptive namespace names
- Verify namespace before destructive ops
- Document your namespace strategy

❌ **DON'T:**
- Mix namespaces without realizing it
- Use generic names like "test" or "temp"
- Forget to check which namespace you're in
- Deploy to production by accident

## Summary

**For most users:**
```bash
# Recommended approach
./k8s-helper-v2.sh -n my-namespace deploy
./k8s-helper-v2.sh -n my-namespace attach
```

**For power users:**
```bash
# Set once
export NAMESPACE=my-namespace

# Use many times
./k8s-helper-v2.sh deploy
./k8s-helper-v2.sh attach
./k8s-helper-v2.sh monitor
```

Both approaches work - choose based on your workflow! 🚀
