import streamlit as st
import pandas as pd
import math
import time
import plotly.graph_objects as go
import plotly.express as px
import ast

st.set_page_config(layout="wide", page_title="Building Construction Calculator")
st.title("Building Construction Calculator")

st.markdown("Calculate STP sizing and pump requirements for building projects.")


# Initialize session state variables
if 'stp_capacity_kld' not in st.session_state:
    st.session_state.stp_capacity_kld = 0.0
if 'sewage_kld' not in st.session_state:
    st.session_state.sewage_kld = 0.0
if 'length' not in st.session_state:
    st.session_state.length = 0.0
if 'width' not in st.session_state:
    st.session_state.width = 0.0
if 'height' not in st.session_state:
    st.session_state.height = 0.0
if 'collection_tank_kld' not in st.session_state:
    st.session_state.collection_tank_kld = 0.0
if 'total_head' not in st.session_state:
    st.session_state.total_head = 0.0
if 'recommended_pump' not in st.session_state:
    st.session_state.recommended_pump = "None"
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame()
if 'show_manual_form' not in st.session_state:
    st.session_state.show_manual_form = False
if 'show_pump_form' not in st.session_state:
    st.session_state.show_pump_form = False

@st.cache_data
def load_pump_data():
    try:
        df = pd.read_csv('pumps.csv')
        if df.empty:
            st.warning("pumps.csv is empty. No pump data available.")
            return pd.DataFrame()
        def parse_head_range(value):
            if pd.isna(value) or not isinstance(value, str) or not value.startswith('['):
                return [0, 0]
            try:
                parsed = ast.literal_eval(value)
                if not isinstance(parsed, list) or len(parsed) < 2:
                    return [0, 0]
                return parsed
            except (ValueError, SyntaxError):
                return [0, 0]
        df['Head m Parsed'] = df['Head m'].apply(parse_head_range)
        df['Min Head m'] = df['Head m Parsed'].apply(lambda x: x[0])
        df['Max Head m'] = df['Head m Parsed'].apply(lambda x: x[1])
        df = df.drop(columns=['Head m Parsed'])
        return df
    except FileNotFoundError:
        st.error("pumps.csv not found. Please ensure it is in the same directory as the script.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading pump data: {e}")
        return pd.DataFrame()

pump_data = load_pump_data()

def process_excel_file(file):
    try:
        df = pd.read_excel(file, sheet_name="STP Calculation", header=[19, 20])
        df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df.columns.values]
        
        results = []
        available_cols = df.columns.tolist()
        pop_col = next((col for col in available_cols if 'Population' in col and 'per flat' in col), None)
        dom_col = next((col for col in available_cols if 'Domestic' in col and 'Water' in col and '(1.0 day)' in col), None)
        flush_col = next((col for col in available_cols if 'Flush' in col and 'Water' in col and '(1.0 day)' in col), None)
        space_col = next((col for col in available_cols if 'Space' in col and 'STP' in col), None)
        building_col = next((col for col in available_cols if 'Building wing or no.' in col), None)
        
        if not all([pop_col, dom_col, flush_col, space_col, building_col]):
            st.warning("Automatic column detection failed. Please map columns manually.")
            st.session_state.column_mapping = {
                'Population': st.selectbox("Map Population Column", ['None'] + available_cols, key="pop_map"),
                'Domestic': st.selectbox("Map Domestic Column", ['None'] + available_cols, key="dom_map"),
                'Flushing': st.selectbox("Map Flushing Column", ['None'] + available_cols, key="flush_map"),
                'Space': st.selectbox("Map Space Column", ['None'] + available_cols, key="space_map"),
                'Building': st.selectbox("Map Building Column", ['None'] + available_cols, key="building_map")
            }
            st.rerun()
        else:
            st.session_state.column_mapping = {'Population': pop_col, 'Domestic': dom_col, 'Flushing': flush_col, 'Space': space_col, 'Building': building_col}
        
        if all(st.session_state.column_mapping.values()) and all(v != 'None' for v in st.session_state.column_mapping.values()):
            for idx, row in df.iterrows():
                building = row.get(st.session_state.column_mapping['Building'], f"Building {idx + 1}")
                if pd.isna(building) or 'Total' in str(building) or 'GRAND' in str(building):
                    continue
                population = float(row[st.session_state.column_mapping['Population']]) if pd.notna(row[st.session_state.column_mapping['Population']]) else 0
                domestic = float(row[st.session_state.column_mapping['Domestic']]) if pd.notna(row[st.session_state.column_mapping['Domestic']]) else 0
                flushing = float(row[st.session_state.column_mapping['Flushing']]) if pd.notna(row[st.session_state.column_mapping['Flushing']]) else 0
                if population == 0 and (domestic == 0 or flushing == 0):
                    continue
                sewage = (domestic * 0.85) + (flushing * 1.0)
                stp_capacity_kld = round((sewage / 1000) * 1.1, 1)
                collection_tank_kld = stp_capacity_kld * 0.3
                space = row[st.session_state.column_mapping['Space']] if pd.notna(row[st.session_state.column_mapping['Space']]) else "34m x 14m x 6m"
                if pd.notna(space) and isinstance(space, str) and 'x' in space:
                    dims = space.split('x')
                    if len(dims) == 3:
                        length = float(dims[0].strip().replace('m', '').strip())
                        width = float(dims[1].strip().replace('m', '').strip())
                        height = float(dims[2].strip().replace('m', '').strip())
                    else:
                        length, width, height = 34.0, 14.0, 6.0
                else:
                    length, width, height = 34.0, 14.0, 6.0
                results.append({
                    'Building': building,
                    'Population': population,
                    'Domestic (L)': domestic,
                    'Flushing (L)': flushing,
                    'Sewage (L)': sewage,
                    'STP Capacity (KLD)': stp_capacity_kld,
                    'Collection Tank (KLD)': collection_tank_kld,
                    'STP Dimensions': f"{length}m x {width}m x {height}m"
                })
            grand_total_idx = df.index[df[st.session_state.column_mapping['Building']] == 'GRAND TOTAL OF ABOVE'].tolist()
            if grand_total_idx:
                row = df.iloc[grand_total_idx[0]]
                population = float(row[st.session_state.column_mapping['Population']]) if pd.notna(row[st.session_state.column_mapping['Population']]) else 0
                domestic = float(row[st.session_state.column_mapping['Domestic']]) if pd.notna(row[st.session_state.column_mapping['Domestic']]) else 0
                flushing = float(row[st.session_state.column_mapping['Flushing']]) if pd.notna(row[st.session_state.column_mapping['Flushing']]) else 0
                sewage = (domestic * 0.85) + (flushing * 1.0)
                stp_capacity_kld = round((sewage / 1000) * 1.1, 1)
                collection_tank_kld = stp_capacity_kld * 0.3
                space = row[st.session_state.column_mapping['Space']] if pd.notna(row[st.session_state.column_mapping['Space']]) else "34m x 14m x 6m"
                if pd.notna(space) and isinstance(space, str) and 'x' in space:
                    dims = space.split('x')
                    if len(dims) == 3:
                        length = float(dims[0].strip().replace('m', '').strip())
                        width = float(dims[1].strip().replace('m', '').strip())
                        height = float(dims[2].strip().replace('m', '').strip())
                    else:
                        length, width, height = 34.0, 14.0, 6.0
                else:
                    length, width, height = 34.0, 14.0, 6.0
                results.append({
                    'Building': 'Grand Total',
                    'Population': population,
                    'Domestic (L)': domestic,
                    'Flushing (L)': flushing,
                    'Sewage (L)': sewage,
                    'STP Capacity (KLD)': stp_capacity_kld,
                    'Collection Tank (KLD)': collection_tank_kld,
                    'STP Dimensions': f"{length}m x {width}m x {height}m"
                })
            return pd.DataFrame(results)
    except Exception as e:
        st.error(f"Error processing Excel file: {e}")
        return pd.DataFrame()

def calculate_stp_params(population, water_usage_lpcd=135, efficiency=0.8):
    if population <= 0 or water_usage_lpcd <= 0 or efficiency <= 0 or efficiency > 1:
        raise ValueError("Invalid input: Population, water usage, and efficiency must be positive; efficiency ≤ 1.")
    domestic = population * (water_usage_lpcd * 0.6667)
    flushing = population * (water_usage_lpcd * 0.3333)
    sewage = (domestic * 0.85) + (flushing * 1.0)
    sewage_kld = sewage / 1000
    stp_capacity_kld = round(sewage_kld * 1.1, 1)
    collection_tank_kld = stp_capacity_kld * 0.3
    length = 0.05 * stp_capacity_kld + 10
    width = 0.03 * stp_capacity_kld + 8
    height = min(0.01 * stp_capacity_kld + 4, 6)
    return sewage_kld, stp_capacity_kld, length, width, height, collection_tank_kld

def calculate_pump_head(vertical_height, horizontal_distance, bends_fittings_loss, pressure_head_kgcm2, stp_capacity_kld):
    if any(x < 0 for x in [vertical_height, horizontal_distance, bends_fittings_loss, pressure_head_kgcm2, stp_capacity_kld]):
        raise ValueError("All inputs must be non-negative.")
    pressure_head_meters = pressure_head_kgcm2 * 10
    flow_rate_lps = (stp_capacity_kld * 1000) / (24 * 60 * 60)
    friction_loss = (vertical_height + horizontal_distance) * 0.083 * 0.8
    total_head_unrounded = vertical_height + friction_loss + bends_fittings_loss + pressure_head_meters
    total_head = math.ceil(total_head_unrounded)
    return flow_rate_lps, friction_loss, total_head, pressure_head_meters

def suggest_pump(total_head, pump_data):
    if pump_data.empty:
        return "No pump data available"
    matching_pumps = pump_data[
        (pump_data['Min Head m'] <= total_head) & 
        (pump_data['Max Head m'] >= total_head)
    ]
    if not matching_pumps.empty:
        return matching_pumps.iloc[0]['Model'].strip()
    else:
        closest_pump = pump_data.iloc[(pump_data['Max Head m'] - total_head).abs().argsort()[:1]]
        return closest_pump.iloc[0]['Model'].strip() + " (Approximate match)"



# Side-by-side options
col1, col2 = st.columns(2)
with col1:
    st.subheader("Upload Excel File")
    uploaded_file = st.file_uploader("Upload STP Calculation Excel", type=["xlsx"], key="excel_upload")
# with col2:
#     st.subheader("Option 2: Manual Input")
#     if st.button("Manual Input"):
#         st.session_state.show_manual_form = not st.session_state.show_manual_form

# Excel Upload Processing
if uploaded_file:
    st.session_state.stp_capacity_kld = 0.0
    with st.spinner("Processing Excel file..."):
        time.sleep(1)
        st.session_state.results_df = process_excel_file(uploaded_file)
        if not st.session_state.results_df.empty:
            selected_buildings = st.multiselect(
                "Select Building(s) for Detailed Calculation",
                st.session_state.results_df['Building'].tolist(),
                key="building_select"
            )
            if selected_buildings:
                selected_df = st.session_state.results_df[st.session_state.results_df['Building'].isin(selected_buildings)]
                # Calculate totals for selected buildings
                total_population = selected_df['Population'].sum()
                total_sewage = selected_df['Sewage (L)'].sum()
                total_stp_capacity = selected_df['STP Capacity (KLD)'].sum()
                total_collection_tank = selected_df['Collection Tank (KLD)'].sum()
                # Append total row to dataframe
                total_row = pd.DataFrame([{
                    'Building': 'Total of Selected',
                    'Population': total_population,
                    'Domestic (L)': selected_df['Domestic (L)'].sum(),
                    'Flushing (L)': selected_df['Flushing (L)'].sum(),
                    'Sewage (L)': total_sewage,
                    'STP Capacity (KLD)': total_stp_capacity,
                    'Collection Tank (KLD)': total_collection_tank,
                    'STP Dimensions': 'N/A (Multiple Buildings)'
                }])
                display_df = pd.concat([selected_df, total_row], ignore_index=True)
                st.dataframe(display_df)
                results_csv = display_df.to_csv(index=False)
                st.download_button("Download STP Results", results_csv, "stp_results.csv")
                # Use total STP capacity for pump calculations
                st.session_state.stp_capacity_kld = total_stp_capacity
                st.session_state.sewage_kld = total_sewage / 1000
                st.session_state.length = 34.0  # Default for multiple buildings
                st.session_state.width = 14.0
                st.session_state.height = 6.0
                st.session_state.collection_tank_kld = total_collection_tank
                # Visualizations side by side
                col1, col2 = st.columns(2)
                with col1:
                    pie_data = pd.DataFrame({
                        'Component': ['Domestic (85%)', 'Flushing (100%)'],
                        'Volume (L)': [selected_df['Domestic (L)'].sum() * 0.85, selected_df['Flushing (L)'].sum()]
                    })
                    fig_pie = go.Figure(data=[go.Pie(labels=pie_data['Component'], values=pie_data['Volume (L)'])])
                    fig_pie.update_layout(title="Sewage Breakdown", width=400, height=400)
                    st.plotly_chart(fig_pie, key="excel_pie")
                with col2:
                    # 3D Scatter Plot using sewage data with multiple bubbles and color range
                    fig_3d = px.scatter_3d(
                        selected_df,
                        x='Domestic (L)',
                        y='Flushing (L)',
                        z='Sewage (L)',
                        color='Sewage (L)',
                        size='Sewage (L)',
                        title='3D STP Sewage Visualization',
                        size_max=60,
                        color_continuous_scale='Viridis'
                    )
                    fig_3d.update_layout(
                        scene_camera=dict(eye=dict(x=1.2, y=1.2, z=0.8)),
                        width=800,
                        height=600,
                        coloraxis_colorbar=dict(title="Sewage (L) Range")
                    )
                    st.plotly_chart(fig_3d, key="excel_3d")
                
                # Button to show pump head form
                if st.button("Calculate Pump Head"):
                    st.session_state.show_pump_form = True
                
                if st.session_state.show_pump_form:
                    with st.form("pump_head_form"):
                        st.markdown("Enter parameters to calculate pump head (uses total STP Capacity from selected buildings).")
                        col1, col2 = st.columns(2)
                        with col1:
                            vertical_height = st.number_input("Vertical Height (m)", min_value=0.0, value=66.0, step=0.1, format="%.2f", key="pump_vh")
                            horizontal_distance = st.number_input("Horizontal Distance (m)", min_value=0.0, value=109.0, step=0.1, format="%.2f", key="pump_hd")
                            pipe_size = st.text_input("Pipe Size", value="6\" GI", key="pump_ps")
                        with col2:
                            bends_fittings_loss = st.number_input("Bends/Fittings Loss (m)", min_value=0.0, value=5.0, step=0.1, format="%.2f", key="pump_bf")
                            pressure_head_kgcm2 = st.number_input("Pressure (Kg/cm²)", min_value=0.0, value=3.5, step=0.1, format="%.2f", key="pump_ph")
                            stp_capacity_kld = st.number_input("STP Capacity (KLD)", min_value=1.0, value=max(1.0, float(st.session_state.stp_capacity_kld)), step=1.0, key="pump_stp")
                        submit_button = st.form_submit_button("Calculate Pump Head")
                        if submit_button:
                            with st.spinner("Calculating..."):
                                time.sleep(1)
                                flow_rate_lps, friction_loss, total_head, pressure_head_meters = calculate_pump_head(
                                    vertical_height, horizontal_distance, bends_fittings_loss, pressure_head_kgcm2, stp_capacity_kld
                                )
                                st.session_state.total_head = total_head
                                st.session_state.recommended_pump = suggest_pump(total_head, pump_data)
                                st.success(f"Total Head Required: {total_head} Meters")
                                st.success(f"Recommended Pump: {st.session_state.recommended_pump}")
                                bar_data = pd.DataFrame({
                                    'Head Component (m)': [vertical_height, friction_loss, bends_fittings_loss, pressure_head_meters]
                                }, index=['Vertical Height', 'Friction Loss', 'Bends/Fittings', 'Pressure Head'])
                                st.bar_chart(bar_data, height=400)
                            st.session_state.show_pump_form = False  # Close form after calculation

# Manual Input Form
if st.session_state.show_manual_form:   
    with st.form("manual_input_form"):
        st.markdown("Enter parameters for STP calculation")
        col1, col2 = st.columns(2)
        with col1:
            population = st.number_input("Population", min_value=1.0, value=500.0, step=1.0)
            water_usage_lpcd = st.number_input("Water Usage (LPCD)", min_value=50.0, value=135.0, step=1.0)
        with col2:
            efficiency = st.number_input("Treatment Efficiency", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
        submit_button = st.form_submit_button("Calculate STP Parameters")
        if submit_button:
            with st.spinner("Calculating..."):
                try:
                    time.sleep(1)
                    sewage_kld, stp_capacity_kld, length, width, height, collection_tank_kld = calculate_stp_params(population, water_usage_lpcd, efficiency)
                    st.session_state.stp_capacity_kld = stp_capacity_kld
                    st.session_state.sewage_kld = sewage_kld
                    st.session_state.length = length
                    st.session_state.width = width
                    st.session_state.height = height
                    st.session_state.collection_tank_kld = collection_tank_kld
                    # Simulate multiple bubbles by creating a range of sewage values
                    base_sewage = sewage_kld * 1000
                    results_df = pd.DataFrame({
                        'Building': ['Manual Input'] * 5,
                        'Population': [population] * 5,
                        'Domestic (L)': [population * (water_usage_lpcd * 0.6667)] * 5,
                        'Flushing (L)': [population * (water_usage_lpcd * 0.3333)] * 5,
                        'Sewage (L)': [base_sewage * 0.8, base_sewage * 0.9, base_sewage, base_sewage * 1.1, base_sewage * 1.2],
                        'STP Capacity (KLD)': [stp_capacity_kld] * 5,
                        'Collection Tank (KLD)': [collection_tank_kld] * 5,
                        'STP Dimensions': [f"{length}m x {width}m x {height}m"] * 5
                    })
                    st.session_state.results_df = results_df
                    st.dataframe(results_df)
                    results_csv = results_df.to_csv(index=False)
                    st.download_button("Download STP Results", results_csv, "stp_results_manual.csv")
                    # Visualizations side by side
                    col1, col2 = st.columns(2)
                    with col1:
                        pie_data = pd.DataFrame({
                            'Component': ['Domestic (85%)', 'Flushing (100%)'],
                            'Volume (L)': [(population * water_usage_lpcd * 0.6667) * 0.85, (population * water_usage_lpcd * 0.3333)]
                        })
                        fig_pie = go.Figure(data=[go.Pie(labels=pie_data['Component'], values=pie_data['Volume (L)'])])
                        fig_pie.update_layout(title="Sewage Breakdown", width=400, height=400)
                        st.plotly_chart(fig_pie, key="manual_pie")
                    with col2:
                        # 3D Scatter Plot using sewage data with multiple bubbles and color range
                        fig_3d = px.scatter_3d(
                            results_df,
                            x='Domestic (L)',
                            y='Flushing (L)',
                            z='Sewage (L)',
                            color='Sewage (L)',
                            size='Sewage (L)',
                            title='3D STP Sewage Visualization',
                            size_max=60,
                            color_continuous_scale='Viridis'
                        )
                        fig_3d.update_layout(
                            scene_camera=dict(eye=dict(x=1.2, y=1.2, z=0.8)),
                            width=800,
                            height=600,
                            coloraxis_colorbar=dict(title="Sewage (L) Range")
                        )
                        st.plotly_chart(fig_3d, key="manual_3d")
                    
                    # Button to show pump head form
                    if st.button("Calculate Pump Head", key="manual_pump_button"):
                        st.session_state.show_pump_form = True
                    
                    if st.session_state.show_pump_form:
                        with st.form("pump_head_form_manual"):
                            st.markdown("Enter parameters to calculate pump head (uses STP Capacity from above).")
                            col1, col2 = st.columns(2)
                            with col1:
                                vertical_height = st.number_input("Vertical Height (m)", min_value=0.0, value=66.0, step=0.1, format="%.2f", key="pump_vh_manual")
                                horizontal_distance = st.number_input("Horizontal Distance (m)", min_value=0.0, value=109.0, step=0.1, format="%.2f", key="pump_hd_manual")
                                pipe_size = st.text_input("Pipe Size", value="6\" GI", key="pump_ps_manual")
                            with col2:
                                bends_fittings_loss = st.number_input("Bends/Fittings Loss (m)", min_value=0.0, value=5.0, step=0.1, format="%.2f", key="pump_bf_manual")
                                pressure_head_kgcm2 = st.number_input("Pressure (Kg/cm²)", min_value=0.0, value=3.5, step=0.1, format="%.2f", key="pump_ph_manual")
                                stp_capacity_kld = st.number_input("STP Capacity (KLD)", min_value=1.0, value=max(1.0, float(st.session_state.stp_capacity_kld)), step=1.0, key="pump_stp_manual")
                            submit_button = st.form_submit_button("Calculate Pump Head")
                            if submit_button:
                                with st.spinner("Calculating..."):
                                    time.sleep(1)
                                    flow_rate_lps, friction_loss, total_head, pressure_head_meters = calculate_pump_head(
                                        vertical_height, horizontal_distance, bends_fittings_loss, pressure_head_kgcm2, stp_capacity_kld
                                    )
                                    st.session_state.total_head = total_head
                                    st.session_state.recommended_pump = suggest_pump(total_head, pump_data)
                                    st.success(f"Total Head Required: {total_head} Meters")
                                    st.success(f"Recommended Pump: {st.session_state.recommended_pump}")
                                    bar_data = pd.DataFrame({
                                        'Head Component (m)': [vertical_height, friction_loss, bends_fittings_loss, pressure_head_meters]
                                    }, index=['Vertical Height', 'Friction Loss', 'Bends/Fittings', 'Pressure Head'])
                                    st.bar_chart(bar_data, height=400)
                                st.session_state.show_pump_form = False  # Close form after calculation
                except ValueError as e:
                    st.error(f"Error: {e}")