# How to Add a New Backend via Proxima TUI

This guide explains how a beginner or non-technical user can add a custom backend to Proxima through the Terminal User Interface (TUI).

## Prerequisites

Before you begin, ensure you have:
- ✅ A properly installed and working backend (e.g., cuQuantum, Qiskit, Cirq, QuEST, or your custom backend)
- ✅ Proxima TUI installed and running
- ✅ (Optional) An AI API key configured for AI-assisted guidance

---

## Method 1: Using the Backend Configuration Screen (Easiest)

### Step 1: Launch Proxima TUI
```bash
python run_tui.py
```
Or if installed:
```bash
proxima-tui
```

### Step 2: Navigate to Backends
Press **`4`** or click on **"Backends"** in the sidebar to open the Backend Configuration screen.

### Step 3: View Available Backends
You'll see a list of all installed backends with their status:
- 🟢 **Available** - Ready to use
- 🟡 **Needs Configuration** - Requires setup
- 🔴 **Not Installed** - Backend not found

### Step 4: Configure Your Backend
1. **Scroll** through the list to find your backend
2. **Click** on the backend name or use arrow keys to select it
3. **Press Enter** or click **"Configure"** to open the configuration dialog

### Step 5: Fill in Configuration Details
In the configuration dialog, enter:
- **Backend Name**: Display name (e.g., "My Custom Backend")
- **Backend Path**: Path to your backend installation (if required)
- **GPU Support**: Enable if your backend supports GPU acceleration
- **Device ID**: GPU device ID (0 for first GPU, 1 for second, etc.)
- **Custom Parameters**: Any backend-specific settings

### Step 6: Test the Backend
1. Click **"Test Connection"** to verify the backend works
2. If successful, you'll see a ✅ checkmark
3. If there are issues, the error message will help you troubleshoot

### Step 7: Save Configuration
Click **"Save"** to store your backend configuration.

---

## Method 2: Using AI Assistant for Guided Setup

If you have an AI provider configured, the AI assistant can help you add backends interactively.

### Step 1: Configure AI Assistant First
1. Press **`5`** to go to **Settings**
2. Scroll to **"AI Assistant Settings"**
3. Select your preferred AI provider (OpenAI, Anthropic, Google, etc.)
4. Enter your API key
5. Click **"Verify API Key"** to test
6. Click **"Save Settings"**

### Step 2: Open AI Thinking Panel
1. Go to the **Execution** tab (press **`3`**)
2. The AI Thinking Panel should be visible on the right
3. If hidden, press **Ctrl+T** to show it

### Step 3: Ask AI for Help
Type a message like:
```
Help me add a new quantum backend to Proxima. I have cuQuantum installed at C:\Program Files\NVIDIA\cuQuantum
```

The AI will:
- Guide you through the configuration steps
- Explain what each setting means
- Help troubleshoot any issues
- Suggest optimal settings for your hardware

### Step 4: Follow AI Instructions
The AI will provide step-by-step guidance tailored to your specific backend.

---

## Method 3: Manual Configuration via Settings File

For advanced users who prefer editing configuration files directly:

### Step 1: Open Settings
Press **`5`** to go to Settings

### Step 2: Export Current Config
1. Scroll to the bottom
2. Click **"Export Config"**
3. This saves your current configuration to `~/.proxima/tui_settings.json`

### Step 3: Edit the Configuration
Add your backend to the exported JSON file:

```json
{
  "backends": {
    "my_custom_backend": {
      "name": "My Custom Backend",
      "type": "custom",
      "path": "/path/to/backend",
      "gpu_enabled": true,
      "device_id": 0,
      "parameters": {
        "precision": "double",
        "max_qubits": 30
      }
    }
  }
}
```

### Step 4: Import Configuration
1. Back in Settings, click **"Import Config"**
2. Select your edited JSON file
3. The new backend will be added

---

## Backend-Specific Quick Setup Guides

### Adding cuQuantum Backend
1. Navigate to **Backends** (press `4`)
2. Find **"cuQuantum (GPU)"** in the list
3. Click **"Configure"**
4. Settings:
   - **CUDA Path**: Usually auto-detected
   - **GPU Device**: Select your NVIDIA GPU (0, 1, etc.)
   - **Precision**: Choose `float32` (faster) or `float64` (more accurate)
5. Click **"Test"** then **"Save"**

### Adding Qiskit Aer Backend
1. Navigate to **Backends** (press `4`)
2. Find **"Qiskit Aer"** in the list
3. Click **"Configure"**
4. Settings:
   - **Simulator**: `statevector_simulator`, `qasm_simulator`, or `aer_simulator`
   - **GPU Mode**: Enable if using `aer_simulator_gpu`
5. Click **"Test"** then **"Save"**

### Adding QuEST Backend
1. Navigate to **Backends** (press `4`)
2. Find **"QuEST"** in the list
3. Click **"Configure"**
4. Settings:
   - **Library Path**: Path to QuEST shared library
   - **GPU Support**: Enable if compiled with GPU support
   - **MPI Support**: Enable for distributed computing
5. Click **"Test"** then **"Save"**

### Adding a Custom/Third-Party Backend
1. Navigate to **Backends** (press `4`)
2. Scroll to the bottom and click **"+ Add Custom Backend"**
3. Fill in:
   - **Name**: Your backend's name
   - **Type**: `python`, `c`, `cpp`, or `custom`
   - **Module Path**: Python module path (e.g., `my_backend.simulator`)
   - **Class Name**: The simulator class name (e.g., `QuantumSimulator`)
   - **Initialization Parameters**: JSON object with constructor arguments
4. Click **"Test"** then **"Save"**

---

## Troubleshooting Common Issues

### "Backend not found"
- Verify the backend is properly installed
- Check that the path is correct
- Ensure all dependencies are installed

### "GPU initialization failed"
- Check that GPU drivers are up to date
- Verify CUDA is installed (for NVIDIA backends)
- Try lowering the GPU device ID

### "Module not found"
- Ensure the Python package is installed in the correct environment
- Check that the module path is correct
- Try installing with: `pip install <backend-name>`

### "Permission denied"
- Run Proxima with appropriate permissions
- Check file permissions on backend installation directory

---

## Getting Help

If you encounter issues:

1. **Check the Log**: Press `L` on the Execution tab to view detailed logs
2. **Ask the AI**: Type your error message in the AI Thinking Panel
3. **Export Diagnostics**: Go to Settings → Export Diagnostics
4. **Community Support**: Visit the Proxima documentation or GitHub issues

---

## Quick Reference

| Action | Keyboard Shortcut |
|--------|-------------------|
| Go to Backends | `4` |
| Go to Settings | `5` |
| Toggle AI Panel | `Ctrl+T` |
| Toggle Log | `L` |
| Help | `?` |
| Quit | `Ctrl+Q` |

---

*Last updated: January 2025*
