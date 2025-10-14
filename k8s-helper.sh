#!/bin/bash

# Kubernetes helper script for ConvE PyKEEN
# This script provides convenient commands for managing the pod

set -e

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
    echo "Usage: $0 <command> [options]"
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
    echo "Examples:"
    echo "  $0 deploy"
    echo "  $0 attach"
    echo "  $0 logs -f"
    echo "  $0 exec 'python train.py --help'"
    echo "  $0 copy-from /workspace/output ./local-output"
    echo "  $0 copy-to ./data /workspace/mydata"
    exit 1
}

check_pod_exists() {
    if ! kubectl get pod "$POD_NAME" -n "$NAMESPACE" &> /dev/null; then
        echo -e "${RED}Error: Pod '$POD_NAME' not found${NC}"
        echo "Run '$0 deploy' to create it"
        exit 1
    fi
}

# Commands
cmd_deploy() {
    echo -e "${BLUE}Deploying ConvE PyKEEN pod...${NC}"

    if kubectl get pod "$POD_NAME" -n "$NAMESPACE" &> /dev/null; then
        echo -e "${YELLOW}Pod already exists. Delete it first with: $0 delete${NC}"
        exit 1
    fi

    if [ ! -f "k8s-pod-simple.yaml" ]; then
        echo -e "${RED}Error: k8s-pod-simple.yaml not found${NC}"
        exit 1
    fi

    kubectl apply -f k8s-pod-simple.yaml -n "$NAMESPACE"

    echo -e "${GREEN}Pod deployed!${NC}"
    echo ""
    echo "Waiting for pod to be ready..."
    kubectl wait --for=condition=Ready pod/"$POD_NAME" -n "$NAMESPACE" --timeout=300s || true

    echo ""
    echo -e "${GREEN}Pod is ready!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Attach to pod: $0 attach"
    echo "  2. Check status: $0 status"
    echo "  3. View logs: $0 logs"
}

cmd_delete() {
    echo -e "${YELLOW}Deleting pod '$POD_NAME'...${NC}"
    check_pod_exists

    read -p "Are you sure? This will delete the pod and any data in /workspace (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled"
        exit 0
    fi

    kubectl delete pod "$POD_NAME" -n "$NAMESPACE"
    echo -e "${GREEN}Pod deleted${NC}"
}

cmd_status() {
    check_pod_exists

    echo -e "${BLUE}Pod Status:${NC}"
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

    echo -e "${GREEN}Attaching to pod...${NC}"
    echo "Press Ctrl+P then Ctrl+Q to detach without stopping the pod"
    echo ""
    kubectl attach -it "$POD_NAME" -n "$NAMESPACE"
}

cmd_exec() {
    check_pod_exists

    if [ -z "$1" ]; then
        echo "Usage: $0 exec '<command>'"
        echo "Example: $0 exec 'python train.py --help'"
        exit 1
    fi

    kubectl exec "$POD_NAME" -n "$NAMESPACE" -- bash -c "$1"
}

cmd_copy_to() {
    check_pod_exists

    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "Usage: $0 copy-to <local-path> <pod-path>"
        echo "Example: $0 copy-to ./data /workspace/mydata"
        exit 1
    fi

    LOCAL_PATH="$1"
    POD_PATH="$2"

    echo -e "${BLUE}Copying $LOCAL_PATH to pod:$POD_PATH${NC}"
    kubectl cp "$LOCAL_PATH" "$NAMESPACE/$POD_NAME:$POD_PATH"
    echo -e "${GREEN}Copy complete${NC}"
}

cmd_copy_from() {
    check_pod_exists

    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "Usage: $0 copy-from <pod-path> <local-path>"
        echo "Example: $0 copy-from /workspace/output ./results"
        exit 1
    fi

    POD_PATH="$1"
    LOCAL_PATH="$2"

    echo -e "${BLUE}Copying pod:$POD_PATH to $LOCAL_PATH${NC}"
    kubectl cp "$NAMESPACE/$POD_NAME:$POD_PATH" "$LOCAL_PATH"
    echo -e "${GREEN}Copy complete${NC}"
}

cmd_gpu() {
    check_pod_exists

    echo -e "${BLUE}GPU Status:${NC}"
    kubectl exec "$POD_NAME" -n "$NAMESPACE" -- nvidia-smi
}

cmd_monitor() {
    check_pod_exists

    echo -e "${BLUE}Monitoring pod resources (Ctrl+C to stop)${NC}"
    echo ""

    while true; do
        clear
        echo -e "${BLUE}=== Pod Status ===${NC}"
        kubectl get pod "$POD_NAME" -n "$NAMESPACE"
        echo ""

        echo -e "${BLUE}=== Resource Usage ===${NC}"
        kubectl top pod "$POD_NAME" -n "$NAMESPACE" 2>/dev/null || echo "Metrics not available"
        echo ""

        echo -e "${BLUE}=== GPU Usage ===${NC}"
        kubectl exec "$POD_NAME" -n "$NAMESPACE" -- nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | \
            awk -F', ' '{printf "GPU: %s%%, Memory: %s/%s MB\n", $1, $2, $3}' || echo "GPU info not available"

        echo ""
        echo "Press Ctrl+C to stop monitoring"
        sleep 5
    done
}

# Main
if [ $# -eq 0 ]; then
    print_usage
fi

COMMAND="$1"
shift

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
