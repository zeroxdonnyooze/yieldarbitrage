#!/bin/bash
# Staging deployment script for yield arbitrage system
set -euo pipefail

# Force staging environment
export ENVIRONMENT=staging

# Source the main deployment script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/deploy.sh"