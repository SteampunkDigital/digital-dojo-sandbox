# WebXR Development Guide — Oculus Rift CV1 + WSL + LAN

This guide documents the complete setup for running a WebXR VR application served from WSL, accessible over a local network, targeting the Oculus Rift CV1 (PC-tethered headset). The lessons here apply broadly to any PC-tethered VR headset with WebXR support.

## Architecture Overview

```
[WSL (Ubuntu)] Flask server :5000
       ↓ (virtual network 172.x.x.x)
[Windows Host] Port forward 0.0.0.0:5000 → WSL:5000
       ↓ (LAN)
[Remote PC] Chrome + Oculus Rift CV1
       → https://10.0.0.x:5000/vr
```

## Prerequisites

- **Oculus PC app** installed and running on the machine with the headset
- **Chrome** (latest) on the same machine
- **OpenXR runtime** set to Oculus (see Headset Setup below)
- **WSL** with Python/Flask serving the application
- **mkcert** for trusted HTTPS certificates

## Step 1: HTTPS (Required for WebXR)

WebXR is restricted to **Secure Contexts**. This means:
- `https://` is required (not `http://`)
- `localhost` and `127.0.0.1` are exempt (always considered secure)
- LAN IP addresses (e.g. `10.0.0.151`) require a valid SSL certificate

### Why self-signed certs are not enough

Browser-generated self-signed certificates trigger security warnings. While you can click through the warning, some browsers may still restrict WebXR functionality on untrusted certificates. Use **mkcert** instead — it installs a local Certificate Authority that browsers fully trust.

### Generate certificates with mkcert

Install mkcert (Windows):
```powershell
choco install mkcert
# or: scoop install mkcert
```

Install the local CA into system trust stores:
```powershell
mkcert -install
```

Generate a certificate covering all access methods:
```powershell
cd /path/to/your/project
mkcert localhost 127.0.0.1 ::1 10.0.0.151
```

This creates:
- `localhost+3.pem` (certificate)
- `localhost+3-key.pem` (private key)

**Important:** Run `mkcert -install` on every machine that will access the site, so the root CA is trusted everywhere.

### Configure Flask to use the certificate

```bash
FLASK_HTTPS=true \
FLASK_SSL_CERT=./localhost+3.pem \
FLASK_SSL_KEY=./localhost+3-key.pem \
python app.py
```

## Step 2: Network Access (WSL → LAN)

WSL uses a virtual network (`172.x.x.x`) that is only visible to the Windows host. Other machines on the LAN cannot reach it directly.

### Find your WSL IP

```bash
# Inside WSL
hostname -I
# Example: 172.18.28.135
```

### Find your Windows LAN IP

```powershell
# In PowerShell on the Windows host
ipconfig
# Look for your Wi-Fi or Ethernet adapter's IPv4 (e.g. 10.0.0.151)
```

### Port forward from Windows to WSL

Run in **PowerShell as Administrator**:
```powershell
# Forward port 5000 from all interfaces to WSL
netsh interface portproxy add v4tov4 `
    listenport=5000 `
    listenaddress=0.0.0.0 `
    connectport=5000 `
    connectaddress=172.18.28.135

# Open Windows Firewall for port 5000
netsh advfirewall firewall add rule `
    name="Digital Dojo VR" `
    dir=in `
    action=allow `
    protocol=TCP `
    localport=5000
```

### Verify access

From the remote machine:
```
https://10.0.0.151:5000/vr
```

### Cleanup (when done)

```powershell
netsh interface portproxy delete v4tov4 listenport=5000 listenaddress=0.0.0.0
netsh advfirewall firewall delete rule name="Digital Dojo VR"
```

## Step 3: Headset Setup (Oculus Rift CV1)

The Rift CV1 is a PC-tethered headset — there is no built-in browser. You run Chrome on the PC and the VR content is sent to the headset.

### Set Oculus as the active OpenXR runtime

Chrome uses OpenXR to communicate with VR headsets. The Oculus runtime must be registered as the active runtime:

**Option A:** Oculus PC app → Settings → General → "Set Oculus as Active" (under OpenXR Runtime)

**Option B:** Registry (if Option A isn't available):
```powershell
reg add "HKLM\SOFTWARE\Khronos\OpenXR\1" `
    /v ActiveRuntime `
    /d "C:\Program Files\Oculus\Support\oculus-runtime\oculus_openxr_64.json" `
    /f
```

Verify:
```powershell
reg query "HKLM\SOFTWARE\Khronos\OpenXR\1" /v ActiveRuntime
```

### Enable Unknown Sources

Oculus PC app → Settings → General → toggle on **"Unknown Sources"**. This allows WebXR content (which doesn't come from the Oculus Store) to access the headset.

### Chrome WebXR flags

Navigate to `chrome://flags` and ensure **WebXR Device API** is set to **Enabled**.

### Test with the official sample

Before testing your own app, verify the setup works:
```
https://immersive-web.github.io/webxr-samples/immersive-vr-session.html
```

If this shows "Enter VR" and works, your setup is correct.

## Step 4: Three.js + WebXR Compatibility

### The XRWebGLBinding problem

Three.js r0.155+ uses the **WebXR Layers API** (`XRWebGLBinding` / `getViewSubImage()`). The Rift CV1's OpenXR runtime does not fully support this API, causing:

```
InvalidStateError: Failed to execute 'getViewSubImage' on 'XRWebGLBinding':
Invalid frame state. There is no shared buffer for layer.
```

### Solution: Manual XR session management

Instead of using Three.js's built-in `renderer.xr` (which tries to use `XRWebGLBinding`), manage the WebXR session directly using the legacy `XRWebGLLayer` API:

```javascript
// DO NOT enable renderer.xr
// renderer.xr.enabled = true;  // ← DON'T DO THIS

// Instead, create the XR session and layer manually:
const session = await navigator.xr.requestSession('immersive-vr', {
    optionalFeatures: ['local-floor']
});

const xrGLLayer = new XRWebGLLayer(session, gl);
session.updateRenderState({ baseLayer: xrGLLayer });

const refSpace = await session.requestReferenceSpace('local-floor');

// Run your own XR frame loop:
session.requestAnimationFrame(function onFrame(time, frame) {
    session.requestAnimationFrame(onFrame);
    const pose = frame.getViewerPose(refSpace);
    if (!pose) return;

    const glLayer = session.renderState.baseLayer;
    gl.bindFramebuffer(gl.FRAMEBUFFER, glLayer.framebuffer);

    for (const view of pose.views) {
        const vp = glLayer.getViewport(view);
        // Set viewport + scissor, then render with Three.js
    }
});
```

### Stereo rendering: the scissor test

When rendering two eyes to a single framebuffer (side by side), you must use `gl.scissor()` to prevent one eye's `clear()` from erasing the other:

```javascript
for (const view of pose.views) {
    const vp = glLayer.getViewport(view);

    gl.viewport(vp.x, vp.y, vp.width, vp.height);
    gl.scissor(vp.x, vp.y, vp.width, vp.height);
    gl.enable(gl.SCISSOR_TEST);

    // Set Three.js camera matrices from XR view
    const cam = view.eye === 'left' ? cameraL : cameraR;
    cam.matrixAutoUpdate = false;
    cam.projectionMatrix.fromArray(view.projectionMatrix);
    cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert();
    cam.matrixWorldInverse.fromArray(view.transform.inverse.matrix);
    cam.matrixWorld.copy(cam.matrixWorldInverse).invert();

    gl.clearColor(0.1, 0.1, 0.18, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    renderer.render(scene, cam);
}

gl.disable(gl.SCISSOR_TEST);
```

Without the scissor test, the second eye's `gl.clear()` wipes the entire framebuffer, leaving the first eye black.

### WebGL context creation

Create the WebGL context with `xrCompatible: true` from the start. This tells Chrome to allocate the context on the GPU connected to the headset, avoiding a context loss during VR entry:

```javascript
const canvas = document.createElement('canvas');
const gl = canvas.getContext('webgl2', {
    xrCompatible: true,
    antialias: false,           // headsets do their own AA
    powerPreference: 'high-performance'
});

// Pass to Three.js
const renderer = new THREE.WebGLRenderer({ canvas, context: gl });
```

If `xrCompatible` is not set, Chrome may create the context on an integrated GPU and then lose it when `makeXRCompatible()` is called during VR entry.

### Performance considerations for VR

- **Disable shadows** — they double GPU work in stereo rendering
- **Skip PMREMGenerator** — large texture allocations stress the GPU during VR transition
- **Set pixel ratio to 1** — the headset manages its own resolution
- **Disable antialiasing** — headsets apply their own distortion/AA
- Use simple ambient + directional lighting instead of environment maps

## Step 5: Debugging WebXR

### Minimal test page

When debugging, strip Three.js out entirely and test with raw WebGL + WebXR:

```javascript
// Render a cycling color to confirm the pipeline works
for (const view of pose.views) {
    const vp = glLayer.getViewport(view);
    gl.viewport(vp.x, vp.y, vp.width, vp.height);
    gl.clearColor(0.1 + 0.1 * Math.sin(time * 0.001), 0.1, 0.3, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);
}
```

This eliminates Three.js as a variable and confirms:
- WebXR session creation works
- `XRWebGLLayer` framebuffer is valid
- Poses are being received
- Both eyes render

### Common failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| "VR Not Available" | OpenXR runtime not set | Set Oculus as active runtime |
| "VR Not Available" | Not HTTPS | Use mkcert certificates |
| Context lost on Enter VR | `xrCompatible` not set | Create context with `xrCompatible: true` |
| `getViewSubImage` error | Three.js Layers API | Use manual XR session (bypass `renderer.xr`) |
| One eye black | `gl.clear()` erasing both eyes | Use `gl.scissor()` per eye |
| Page not responding | Too many assets loading | Load assets lazily, on demand |
| Can't reach from LAN | WSL virtual network | Port forward with `netsh` |

## File Reference

| File | Purpose |
|------|---------|
| `templates/vr.html` | Full VR scene with object placement |
| `templates/vr-test.html` | Minimal WebXR test (raw WebGL, no Three.js) |
| `app.py` | Flask routes (`/vr`, `/vr-test`) |
| `localhost+3.pem` | mkcert SSL certificate |
| `localhost+3-key.pem` | mkcert SSL private key |
