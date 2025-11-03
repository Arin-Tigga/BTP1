import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Inventory Viewer", layout="centered")

st.title("Inventory")

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
    # convert to list of dicts for editor
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

st.markdown("---")
st.caption("This page only displays and edits the inventory JSON. Camera and model features removed.")
