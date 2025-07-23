import streamlit as st
import pandas as pd
import math
import time
import ast

st.set_page_config(layout="centered", page_title="Pump Selection Tool")

@st.cache_data
def load_pump_data():
    try:
        df = pd.read_csv('pumps.csv')
        df['Min Head m'] = df['Head m'].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) and x.startswith('[') else x)
        return df
    except FileNotFoundError:
        st.error("Error: pumps.csv not found. Please make sure 'pumps.csv' is in the same directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading pump data: {e}")
        return pd.DataFrame()

pump_data = load_pump_data()

def calculate_pump_head(vertical_height, horizontal_distance, bends_fittings_loss, pressure_head_kgcm2, stp_capacity):
    pressure_head_meters = pressure_head_kgcm2 * 10
    flow_rate_lps = (stp_capacity * 1000) / (24 * 60 * 60)
    flow_friction_loss = 0.1 * (flow_rate_lps ** 2)
    static_friction_loss = (vertical_height + horizontal_distance) * 0.083 * 0.8
    total_friction_loss = static_friction_loss + flow_friction_loss
    total_head_unrounded = vertical_height + total_friction_loss + bends_fittings_loss + pressure_head_meters
    total_head = math.ceil(total_head_unrounded)

    return flow_rate_lps, flow_friction_loss, total_friction_loss, total_head, pressure_head_meters, total_head_unrounded

st.title("Automated Pump Head Calculator")
st.markdown("This tool calculates the required pump head based on building parameters and **STP capacity**, and suggests suitable pump models.")

st.header("Input Building Parameters")
vertical_height = st.number_input("Vertical Height from pump suction (Meters)", min_value=0.0, value=66.0, step=0.1, format="%.2f")
horizontal_distance = st.number_input("Horizontal Distance to farthest outlet (Meters)", min_value=0.0, value=109.0, step=0.1, format="%.2f")
pipe_size_str = st.text_input("Pipe Size (reference only)", value='6" GI')
bends_fittings_loss = st.number_input("Head loss in Bends and Fittings (Meters)", min_value=0.0, value=5.0, step=0.1, format="%.2f")
pressure_head_kgcm2 = st.number_input("Required Pressure (Kg/cm²)", min_value=0.0, value=3.5, step=0.1, format="%.2f")
stp_capacity = st.number_input("STP Capacity (KLD)", min_value=1.0, value=100.0)

st.markdown("---")

if st.button("Calculate Pump Head & Suggest Pump"):
    if pump_data.empty:
        st.error("Pump data not loaded. Please check 'pumps.csv'.")
    else:
        with st.spinner("Calculating..."):
            time.sleep(1)

            flow_rate_lps, flow_friction_loss, total_friction_loss, total_head, pressure_head_meters, total_head_unrounded = calculate_pump_head(
                vertical_height, horizontal_distance, bends_fittings_loss, pressure_head_kgcm2, stp_capacity
            )

            st.markdown("Calculation Results")
            st.success(f"**Total Head Required:** {total_head} Meters")

            suitable_pumps = pump_data[pump_data['Min Head m'] >= total_head].copy()
            if suitable_pumps.empty:
                st.warning("No suitable pumps found for the calculated total head.")
            else:
                st.markdown("Recommended Pumps:")
                recommended_display = suitable_pumps[['Model', 'Manufacturer', 'HP', 'Head m', 'Suitability']]
                recommended_display.columns = ['Pump Model', 'Manufacturer', 'Horsepower (HP)', 'Head Range (m)', 'Suitability']
                st.dataframe(recommended_display)

            st.markdown("---")
            st.header("Head Component Breakdown")
            st.subheader("Flow Rate Derived:")
            st.info(f"**Flow Rate (from STP):** {flow_rate_lps:.2f} LPS")

            st.subheader("Loss Components - Visual Chart")

            bar_data = pd.DataFrame({
                'Head Component (m)': [
                    vertical_height,
                    total_friction_loss - flow_friction_loss,
                    flow_friction_loss,
                    bends_fittings_loss,
                    pressure_head_meters
                ]
            }, index=[
                'Vertical Height',
                'Static Friction Loss',
                'Flow-based Friction Loss',
                'Bends/Fittings Loss',
                'Pressure Head'
            ])

            st.bar_chart(bar_data)

            st.markdown(
                f"""
                <details>
                <summary><strong>Detailed Calculation Steps</strong></summary>
                <ul>
                    <li>STP Capacity: {stp_capacity} KLD</li>
                    <li>Flow Rate = ({stp_capacity} × 1000) / (24 × 60 × 60) = **{flow_rate_lps:.2f} LPS**</li>
                    <li>Flow Friction Loss = 0.1 × Flow² = **{flow_friction_loss:.2f} m**</li>
                    <li>Static Friction Loss = ({vertical_height:.2f} + {horizontal_distance:.2f}) × 0.083 × 0.8 = **{(total_friction_loss - flow_friction_loss):.2f} m**</li>
                    <li>Bends/Fittings Loss = **{bends_fittings_loss:.2f} m**</li>
                    <li>Pressure Head = {pressure_head_kgcm2:.2f} Kg/cm² = **{pressure_head_meters:.2f} m**</li>
                    <li>Total Head (unrounded) = **{total_head_unrounded:.2f} m**</li>
                    <li><strong>Rounded Total Head = {total_head} m</strong></li>
                </ul>
                </details>
                """,
                unsafe_allow_html=True
            )
