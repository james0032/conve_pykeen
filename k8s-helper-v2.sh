#!/bin/bash

# Kubernetes helper script for ConvE PyKEEN
# This script provides convenient commands for managing the pod

set -e

# Default values
POD_NAME="conve-pykeen-training"
NAMESPACE="${NAMESPACE:-default}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_usage() {
    echo "Usage: $0 [options] <command> [command-args]"
    echo ""
    echo "Options:"
    echo "  -n, --namespace <name>    Kubernetes namespace (default: $NAMESPACE)"
    echo "  -p, --pod <name>          Pod name (default: $POD_NAME)"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Commands:"
    echo "  deploy        Deploy the pod"
    echo "  delete        Delete the pod"
    echo "  status        Check pod status"
    echo "  logs          View pod logs"
    echo "  attach        Attach to pod shell"
    echo "  exec          Execute command in pod"
    echo "  copy-to       Copy files TO pod"
    echo "  copy-from     Copy files FROM pod"
    echo "  gpu           Check GPU usage in pod"
    echo "  monitor       Monitor pod resources"
    echo ""
    echo "Namespace Configuration:"
    echo "  Method 1 - Command line flag (highest priority):"
    echo "    $0 -n my-namespace deploy"
    echo "    $0 --namespace prod attach"
    echo ""
    echo "  Method 2 - Environment variable:"
    echo "    export NAMESPACE=my-namespace"
    echo "    $0 deploy"
    echo ""
    echo "  Method 3 - Inline environment:"
    echo "    NAMESPACE=my-namespace $0 deploy"
    echo ""
    echo "  Method 4 - Use default (default):"
    echo "    $0 deploy  # Uses 'default' namespace"
    echo ""
    echo "Examples:"
    echo "  # Deploy to default namespace"
    echo "  $0 deploy"
    echo ""
    echo "  # Deploy to specific namespace"
    echo "  $0 -n ml-training deploy"
    echo ""
    echo "  # Attach with custom pod name"
    echo "  $0 -p my-pod-name -n ml-training attach"
    echo ""
    echo "  # View logs from specific namespace"
    echo "  $0 -n prod logs -f"
    echo ""
    echo "  # Execute command"
    echo "  $0 -n dev exec 'python train.py --help'"
    echo ""
    echo "  # Copy files"
    echo "  $0 -n prod copy-from /workspace/output ./results"
    exit 1
}

check_pod_exists() {
    if ! kubectl get pod "$POD_NAME" -n "$NAMESPACE" &> /dev/null; then
        echo -e "${RED}Error: Pod '$POD_NAME' not found in namespace '$NAMESPACE'${NC}"
        echo "Run '$0 -n $NAMESPACE deploy' to create it"
        exit 1
    fi
}

# Parse command line options
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -p|--pod)
            POD_NAME="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            ;;
        -*)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            ;;
        *)
            # First non-option argument is the command
            break
            ;;
    esac
done

# Check if command is provided
if [ $# -eq 0 ]; then
    print_usage
fi

COMMAND="$1"
shift

# Show current configuration
echo -e "${BLUE}Configuration:${NC}"
echo "  Namespace: $NAMESPACE"
echo "  Pod Name: $POD_NAME"
echo ""

# Commands
cmd_deploy() {
    echo -e "${BLUE}Deploying ConvE PyKEEN pod...${NC}"

    if kubectl get pod "$POD_NAME" -n "$NAMESPACE" &> /dev/null; then
        echo -e "${YELLOW}Pod already exists in namespace '$NAMESPACE'${NC}"
        echo "Delete it first with: $0 -n $NAMESPACE delete"
        exit 1
    fi

    if [ ! -f "k8s-pod-simple.yaml" ]; then
        echo -e "${RED}Error: k8s-pod-simple.yaml not found${NC}"
        exit 1
    fi

    # Create namespace if it doesn't exist
    if [ "$NAMESPACE" != "default" ]; then
        if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
            echo -e "${YELLOW}Namespace '$NAMESPACE' doesn't exist. Creating...${NC}"
            kubectl create namespace "$NAMESPACE"
        fi
    fi

    kubectl apply -f k8s-pod-simple.yaml -n "$NAMESPACE"

    echo -e "${GREEN}Pod deployed to namespace '$NAMESPACE'!${NC}"
    echo ""
    echo "Waiting for pod to be ready..."
    kubectl wait --for=condition=Ready pod/"$POD_NAME" -n "$NAMESPACE" --timeout=300s || true

    echo ""
    echo -e "${GREEN}Pod is ready!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Attach to pod: $0 -n $NAMESPACE attach"
    echo "  2. Check status: $0 -n $NAMESPACE status"
    echo "  3. View logs: $0 -n $NAMESPACE logs"
}

cmd_delete() {
    echo -e "${YELLOW}Deleting pod '$POD_NAME' from namespace '$NAMESPACE'...${NC}"
    check_pod_exists

    read -p "Are you sure? This will delete the pod and any data in /workspace (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled"
        exit 0
    fi

    kubectl delete pod "$POD_NAME" -n "$NAMESPACE"
    echo -e "${GREEN}Pod deleted from namespace '$NAMESPACE'${NC}"
}

cmd_status() {
    check_pod_exists

    echo -e "${BLUE}Pod Status in namespace '$NAMESPACE':${NC}"
    kubectl get pod "$POD_NAME" -n "$NAMESPACE"
    echo ""

    echo -e "${BLUE}Pod Details:${NC}"
    kubectl describe pod "$POD_NAME" -n "$NAMESPACE" | grep -A 10 "Status:\|Conditions:\|Events:"
}

cmd_logs() {
    check_pod_exists
    kubectl logs "$@" "$POD_NAME" -n "$NAMESPACE"
}

cmd_attach() {
    check_pod_exists

    echo -e "${GREEN}Connecting to pod in namespace '$NAMESPACE'...${NC}"
    echo ""
    echo -e "${YELLOW}Note:${NC} Using 'exec' for interactive shell (attach doesn't support detach on Mac)"
    echo ""
    echo "Running in background:"
    echo "  1. Inside pod, use: nohup command > output.log 2>&1 &"
    echo "  2. Or use tmux: 'tmux' to start, 'Ctrl+B then D' to detach, 'tmux attach' to resume"
    echo ""
    echo "Exit with 'exit' or Ctrl+D"
    echo ""
    kubectl exec -it "$POD_NAME" -n "$NAMESPACE" -- /bin/bash
}

cmd_exec() {
    check_pod_exists

    if [ -z "$1" ]; then
        echo "Usage: $0 -n $NAMESPACE exec '<command>'"
        echo "Example: $0 -n $NAMESPACE exec 'python train.py --help'"
        exit 1
    fi

    kubectl exec "$POD_NAME" -n "$NAMESPACE" -- bash -c "$1"
}

cmd_copy_to() {
    check_pod_exists

    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "Usage: $0 -n $NAMESPACE copy-to <local-path> <pod-path>"
        echo "Example: $0 -n $NAMESPACE copy-to ./data /workspace/mydata"
        exit 1
    fi

    LOCAL_PATH="$1"
    POD_PATH="$2"

    echo -e "${BLUE}Copying $LOCAL_PATH to $NAMESPACE/$POD_NAME:$POD_PATH${NC}"
    kubectl cp "$LOCAL_PATH" "$NAMESPACE/$POD_NAME:$POD_PATH"
    echo -e "${GREEN}Copy complete${NC}"
}

cmd_copy_from() {
    check_pod_exists

    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "Usage: $0 -n $NAMESPACE copy-from <pod-path> <local-path>"
        echo "Example: $0 -n $NAMESPACE copy-from /workspace/output ./results"
        exit 1
    fi

    POD_PATH="$1"
    LOCAL_PATH="$2"

    echo -e "${BLUE}Copying $NAMESPACE/$POD_NAME:$POD_PATH to $LOCAL_PATH${NC}"
    kubectl cp "$NAMESPACE/$POD_NAME:$POD_PATH" "$LOCAL_PATH"
    echo -e "${GREEN}Copy complete${NC}"
}

cmd_gpu() {
    check_pod_exists

    echo -e "${BLUE}GPU Status in namespace '$NAMESPACE':${NC}"
    kubectl exec "$POD_NAME" -n "$NAMESPACE" -- nvidia-smi
}

cmd_monitor() {
    check_pod_exists

    echo -e "${BLUE}Monitoring pod in namespace '$NAMESPACE' (Ctrl+C to stop)${NC}"
    echo ""

    while true; do
        clear
        echo -e "${BLUE}=== Pod Status (Namespace: $NAMESPACE) ===${NC}"
        kubectl get pod "$POD_NAME" -n "$NAMESPACE"
        echo ""

        echo -e "${BLUE}=== Resource Usage ===${NC}"
        kubectl top pod "$POD_NAME" -n "$NAMESPACE" 2>/dev/null || echo "Metrics not available"
        echo ""

        echo -e "${BLUE}=== GPU Usage ===${NC}"
        kubectl exec "$POD_NAME" -n "$NAMESPACE" -- nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | \
            awk -F', ' '{printf "GPU: %s%%, Memory: %s/%s MB\n", $1, $2, $3}' || echo "GPU info not available"

        echo ""
        echo "Namespace: $NAMESPACE | Pod: $POD_NAME"
        echo "Press Ctrl+C to stop monitoring"
        sleep 5
    done
}

# Execute command
case "$COMMAND" in
    deploy)
        cmd_deploy "$@"
        ;;
    delete)
        cmd_delete "$@"
        ;;
    status)
        cmd_status "$@"
        ;;
    logs)
        cmd_logs "$@"
        ;;
    attach)
        cmd_attach "$@"
        ;;
    exec)
        cmd_exec "$@"
        ;;
    copy-to)
        cmd_copy_to "$@"
        ;;
    copy-from)
        cmd_copy_from "$@"
        ;;
    gpu)
        cmd_gpu "$@"
        ;;
    monitor)
        cmd_monitor "$@"
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        echo ""
        print_usage
        ;;
esac
