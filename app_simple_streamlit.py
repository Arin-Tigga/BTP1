import streamlit as st
from pathlib import Path
import subprocess
import sys
import time

st.set_page_config(page_title="Inventory Viewer", layout="centered")

st.title("Inventory")

# --- Inventory loader/saver ---
try:
    import inventory_logic as logic
except Exception as e:
    st.error(f"Failed to import inventory logic: {e}")
    raise

with st.sidebar:
    st.header("Settings")
    inventory_path = st.text_input("Inventory JSON path", value="inventory.json")
    if st.button("Reload"):
        st.rerun()

inv_file = Path(inventory_path)
if not inv_file.exists():
    st.warning(f"Inventory file not found at: {inv_file}. A new file will be created when you save.")
    inventory = {}
else:
    try:
        inventory = logic.load_inventory(str(inv_file))
    except Exception as e:
        st.error(f"Failed to load inventory: {e}")
        inventory = {}

st.subheader("Current inventory (editable)")

# Use experimental data editor if available for a nicer edit experience
try:
    rows = [{"item": k, "count": int(v)} for k, v in inventory.items()]
    edited = st.experimental_data_editor(rows, num_rows="dynamic")
    if st.button("Save changes"):
        new_inv = {r['item']: int(r['count']) for r in edited if r.get('item')}
        try:
            logic.save_inventory(str(inv_file), new_inv)
            st.success("Inventory saved")
        except Exception as e:
            st.error(f"Failed to save inventory: {e}")
except Exception:
    # fallback simple editor
    st.write(inventory)
    st.write("\n")
    st.subheader("Manual update")
    item = st.text_input("Item name")
    delta = st.number_input("New count", value=0)
    if st.button("Apply"):
        if item:
            inventory[item] = int(delta)
            try:
                logic.save_inventory(str(inv_file), inventory)
                st.success(f"Updated {item} -> {inventory[item]}")
            except Exception as e:
                st.error(f"Failed to save inventory: {e}")
        else:
            st.warning("Enter item name")

# --- CLI scanner controls (runs action_scanner.py) ---
st.markdown("---")
st.subheader("Run CLI scanner")

if 'scanner_proc' not in st.session_state:
    st.session_state['scanner_proc'] = None

repo_root = Path(__file__).parent
use_mjpeg = st.checkbox("Embed CLI camera output in page (MJPEG)", value=True)
mjpeg_port = st.number_input("MJPEG port", min_value=1024, max_value=65535, value=8082)
col1, col2 = st.columns(2)
with col1:
    if st.button("Start CLI scanner"):
        if st.session_state.get('scanner_proc') is None:
            try:
                cmd = [sys.executable, str(repo_root / 'action_scanner.py')]
                if use_mjpeg:
                    cmd += ['--mjpeg-port', str(mjpeg_port)]
                # open the subprocess with a piped stdin so we can programmatically
                # send the same 's' key that the CLI expects
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(repo_root),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                st.session_state['scanner_proc'] = proc
                st.success(f"Started action_scanner.py (pid={proc.pid})")
            except Exception as e:
                st.error(f"Failed to start action_scanner: {e}")
        else:
            st.info("CLI scanner already running.")
with col2:
    if st.button("Stop CLI scanner"):
        proc = st.session_state.get('scanner_proc')
        if proc is not None:
            try:
                # close stdin to signal EOF to the child (if it cares)
                try:
                    if proc.stdin:
                        proc.stdin.close()
                except Exception:
                    pass

                proc.terminate()
                proc.wait(timeout=5)
                st.success(f"Stopped action_scanner.py (pid={proc.pid})")
            except Exception as e:
                try:
                    proc.kill()
                    st.success(f"Killed action_scanner.py (pid={proc.pid})")
                except Exception as ee:
                    st.error(f"Failed to stop process: {e}; {ee}")
            finally:
                st.session_state['scanner_proc'] = None
        else:
            st.info("No CLI scanner process found.")

# Embed MJPEG stream if requested and process is running
proc = st.session_state.get('scanner_proc')
if proc is not None and use_mjpeg:
    stream_url = f"http://localhost:{mjpeg_port}/stream"
    st.markdown("---")
    st.subheader("CLI camera stream")
    st.markdown(f'<img src="{stream_url}" width="640" />', unsafe_allow_html=True)

# Trigger a scan by sending 's' to the subprocess stdin (if available)
st.markdown("---")
st.subheader("Remote trigger")
if st.button("Trigger scan ('s')"):
    proc = st.session_state.get('scanner_proc')
    if proc is None:
        st.warning("No scanner process running. Start the CLI scanner first.")
    else:
        # If MJPEG is enabled, prefer the HTTP trigger endpoint exposed by the scanner
        if use_mjpeg:
            try:
                import http.client
                conn = http.client.HTTPConnection('localhost', mjpeg_port, timeout=2)
                conn.request('GET', '/trigger')
                resp = conn.getresponse()
                body = resp.read().decode('utf-8', errors='ignore')
                if resp.status == 200:
                    st.success("Triggered scanner via HTTP endpoint.")
                else:
                    st.error(f"Trigger endpoint returned status {resp.status}: {body}")
                conn.close()
            except Exception as e:
                st.warning(f"HTTP trigger failed: {e}. Falling back to stdin.")
                # fallback to stdin write below
                try:
                    if proc.stdin is None:
                        st.error("Scanner stdin is not available. Could not trigger.")
                    else:
                        proc.stdin.write('s\n')
                        proc.stdin.flush()
                        st.info("Sent 's' to scanner process stdin as fallback.")
                except Exception as e2:
                    st.error(f"Fallback stdin trigger failed: {e2}")
        else:
            # MJPEG not enabled; try stdin
            if proc.stdin is None:
                st.error("Scanner stdin is not available. It may not accept programmatic input.")
            else:
                try:
                    proc.stdin.write('s\n')
                    proc.stdin.flush()
                    st.info("Sent 's' to scanner process stdin.")
                except Exception as e:
                    st.error(f"Failed to send input to scanner: {e}")

st.caption("This page displays and edits the inventory JSON. Use the CLI controls below to run the original scanner.")
