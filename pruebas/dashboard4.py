import os
import pandas as pd
import numpy as np
import streamlit as st

# ============ CONFIG BÁSICA ============
st.set_page_config(page_title="Debug Globant Dashboard", layout="wide")

st.title("📊 Debug Dashboard – Globant Motivation")
st.write("Si ves esta pantalla, **Streamlit y el dashboard SÍ están corriendo**.")

# Ruta del CSV (ajusta si lo tienes en otra carpeta)
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data_globant_cnn.csv")


@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    """Carga el CSV y hace un par de columnas básicas para que no truene nada."""
    df = pd.read_csv(path)

    # Intentar parsear una fecha
    if "Date" in df.columns:
        df["Date_dt"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        df["Date_dt"] = np.nan

    return df


# ============ CARGA DE DATOS ============
try:
    df = load_data(DATA_PATH)
    st.success(f"Datos cargados desde: `{DATA_PATH}`")
except Exception as e:
    st.error(f"Error al cargar el CSV: {e}")
    st.stop()

with st.expander("Ver primeras filas del dataset"):
    st.dataframe(df.head())

# ============ EXPLORACIÓN SÚPER SIMPLE ============
st.subheader("Exploración rápida por empleado")

if "EmployeeID" not in df.columns:
    st.error("El CSV no tiene columna `EmployeeID`. No puedo hacer búsqueda por empleado.")
    st.stop()

emp_ids = sorted(df["EmployeeID"].unique().tolist())
selected_emp = st.selectbox("Selecciona un EmployeeID", emp_ids)

df_emp = df[df["EmployeeID"] == selected_emp].copy()

st.write(f"Registros encontrados para empleado `{selected_emp}`: {len(df_emp)}")

if "Date_dt" in df_emp.columns and not df_emp["Date_dt"].isna().all():
    df_emp = df_emp.sort_values("Date_dt")
    st.line_chart(
        df_emp.set_index("Date_dt")[["Engagement_D"]]
        if "Engagement_D" in df_emp.columns
        else df_emp.set_index("Date_dt").iloc[:, :1]
    )
else:
    st.info("No hay columna de fecha válida para graficar.")

st.success("✅ Si ves todo esto, el dashboard ya NO está colgado.")