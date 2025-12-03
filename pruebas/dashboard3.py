import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# Intentar importar Keras de forma segura
try:
    from tensorflow import keras  # type: ignore
    TF_AVAILABLE = True
    TF_ERROR = None
except Exception as e:
    TF_AVAILABLE = False
    TF_ERROR = str(e)


# =========================
#  RUTAS Y CONFIGURACIÓN
# =========================
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data_globant_cnn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "my_model.keras")

CATEGORICAL_COLS = [
    "Position",
    "Location",
    "Studio",
    "Client Tag",
    "Project Tag",
    "Team Name",
]

st.set_page_config(page_title="Globant Engagement NN", layout="wide")


# =========================
#  UTILIDADES DE DATOS
# =========================
def assign_15day_blocks(group: pd.DataFrame) -> pd.DataFrame:
    """
    Asigna bloques de 15 filas por empleado:
    0–14 -> 1, 15–29 -> 2, ...
    """
    group = group.sort_values("Date").reset_index(drop=True)
    group["15_dias"] = (group.index // 15) + 1
    return group


@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    """Carga el CSV y crea columnas de fecha básicas."""
    df = pd.read_csv(path)

    if "Date" not in df.columns:
        raise ValueError("El CSV no tiene columna 'Date'.")

    # Parsear fecha
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Month y Day (si no existen, se generan)
    if "Month" not in df.columns:
        df["Month"] = df["Date"].dt.month
    if "Day" not in df.columns:
        df["Day"] = df["Date"].dt.day

    return df


@st.cache_resource(show_spinner=True)
def build_pipeline(data_path: str, model_path: str):
    """
    Pipeline de entrenamiento replicado:

    - Carga datos
    - Crea 15_dias por EmployeeID
    - One-hot a columnas categóricas
    - Elimina la etiqueta del set de features
    - Normaliza con MinMaxScaler
    - Carga el modelo Keras

    Devuelve:
      df_display: DataFrame con datos originales + 15_dias
      feature_cols: lista de columnas de entrada
      scaler: MinMaxScaler
      X_scaled: np.ndarray (n_samples, n_features)
      model: modelo keras cargado
      class_labels: etiquetas de salida
    """
    if not TF_AVAILABLE:
        raise RuntimeError(
            "TensorFlow/Keras no está disponible en este entorno. "
            f"Detalle: {TF_ERROR}"
        )

    df_raw = load_data(data_path).copy()

    required = {
        "Date",
        "Position",
        "Seniority",
        "Location",
        "Studio",
        "Client Tag",
        "Project Tag",
        "Team Name",
        "EmployeeID",
        "Engagement_D",
        "Month",
        "Day",
    }
    missing = required - set(df_raw.columns)
    if missing:
        raise ValueError(f"Faltan columnas necesarias en el CSV: {missing}")

    # Crear bloques de 15 días por empleado
    df_blocks = (
        df_raw.groupby("EmployeeID", group_keys=False)
        .apply(assign_15day_blocks)
        .reset_index(drop=True)
    )

    # Para mostrar en la UI
    df_display = df_blocks.copy()

    # Preparar datos para el modelo
    df_model = df_blocks.drop(columns=["Date"])

    # One-hot encoding de categóricas
    df_model = pd.get_dummies(df_model, columns=CATEGORICAL_COLS, drop_first=False)

    # Eliminar la etiqueta de las features (si existe numérica, también)
    drop_cols = [c for c in ["Engagement_D", "Engagement_D_num"] if c in df_model.columns]
    features_df = df_model.drop(columns=drop_cols)

    feature_cols = list(features_df.columns)
    X = features_df.astype(float).values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Cargar modelo
    model = keras.models.load_model(model_path)

    # Verificar dimensiones
    model_input_dim = model.input_shape[-1]
    if X_scaled.shape[1] != model_input_dim:
        raise ValueError(
            f"Incompatibilidad entre datos y modelo: "
            f"{X_scaled.shape[1]} features vs {model_input_dim} esperadas por el modelo."
        )

    # Etiquetas de salida en función del número de neuronas finales
    n_outputs = model.output_shape[-1]
    if n_outputs == 3:
        class_labels = ["Bajo", "Medio", "Alto"]
    else:
        class_labels = [f"Clase {i+1}" for i in range(n_outputs)]

    return df_display, feature_cols, scaler, X_scaled, model, class_labels


def predict_from_index(idx: int, X_scaled: np.ndarray, model) -> np.ndarray:
    """Predice usando una fila existente en X_scaled."""
    if idx < 0 or idx >= X_scaled.shape[0]:
        raise IndexError("Índice fuera de rango.")
    x = X_scaled[idx : idx + 1]
    probs = model.predict(x, verbose=0)[0]
    return probs


def predict_from_manual(
    row_dict: dict,
    feature_cols: list[str],
    scaler: MinMaxScaler,
    model,
) -> np.ndarray:
    """
    Aplica las mismas transformaciones a una sola fila manual.
    """
    single_df = pd.DataFrame([row_dict])

    # One-hot a categóricas
    df_model = pd.get_dummies(single_df, columns=CATEGORICAL_COLS, drop_first=False)

    # Eliminar etiqueta si está
    drop_cols = [c for c in ["Engagement_D", "Engagement_D_num"] if c in df_model.columns]
    features_df = df_model.drop(columns=drop_cols)

    # Asegurar todas las columnas esperadas
    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0.0

    features_df = features_df[feature_cols]

    x = features_df.astype(float).values
    x_scaled = scaler.transform(x)
    probs = model.predict(x_scaled, verbose=0)[0]
    return probs


# =========================
#  APP STREAMLIT
# =========================
def main():
    st.title("📊 Globant – Predicción de Engagement (Red Neuronal)")

    # ---- Carga de datos base ----
    try:
        df_raw = load_data(DATA_PATH)
    except Exception as e:
        st.error(f"Error al cargar el CSV `{DATA_PATH}`: {e}")
        return

    with st.expander("Ver muestra de los datos"):
        st.dataframe(df_raw.head())

    # ---- Tabs: Exploración y Predicción ----
    tab_explore, tab_nn = st.tabs(["Exploración básica", "Predicción con NN"])

    # ========== TAB 1: EXPLORACIÓN ==========
    with tab_explore:
        st.subheader("Engagement promedio en el tiempo")

        if "Engagement_D" in df_raw.columns:
            mapping = {"Bajo": 1, "Medio": 2, "Alto": 3}
            df_raw["Engagement_num"] = df_raw["Engagement_D"].map(mapping)
        else:
            df_raw["Engagement_num"] = np.nan
            st.warning("No se encontró 'Engagement_D' para convertir a número.")

        df_plot = df_raw.dropna(subset=["Date"]).copy()
        if df_plot.empty:
            st.info("No hay fechas válidas para graficar.")
        else:
            df_daily = (
                df_plot.groupby("Date")
                .agg(Engagement_prom=("Engagement_num", "mean"))
                .reset_index()
            )
            fig = px.line(
                df_daily,
                x="Date",
                y="Engagement_prom",
                title="Engagement promedio a lo largo del tiempo",
            )
            fig.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Engagement (1=Bajo, 3=Alto)",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ========== TAB 2: PREDICCIÓN NN ==========
    with tab_nn:
        st.subheader("🤖 Predicción con red neuronal")

        if not TF_AVAILABLE:
            st.error(
                "TensorFlow/Keras no está disponible, no puedo cargar el modelo.\n\n"
                f"Detalle técnico: {TF_ERROR}"
            )
            return

        try:
            (
                df_display,
                feature_cols,
                scaler,
                X_scaled,
                model,
                class_labels,
            ) = build_pipeline(DATA_PATH, MODEL_PATH)
        except Exception as e:
            st.error(f"No se pudo inicializar el pipeline de la NN: {e}")
            return

        st.write(
            "Modo por defecto: **Buscar empleado**. "
            "También puedes cambiar a **Llenar manualmente**."
        )

        mode = st.radio(
            "Modo de entrada",
            options=["Buscar empleado", "Llenar manualmente"],
            index=0,
            horizontal=True,
        )

        probs = None

        # ------- MODO 1: BUSCAR EMPLEADO (DEFAULT) -------
        if mode == "Buscar empleado":
            emp_ids = sorted(df_display["EmployeeID"].unique().tolist())
            selected_emp = st.selectbox("EmployeeID", emp_ids)

            df_emp = df_display[df_display["EmployeeID"] == selected_emp].copy()
            df_emp = df_emp.sort_values("Date")

            if df_emp.empty:
                st.warning("No hay registros para este empleado.")
            else:
                # Fechas disponibles
                dates_str = df_emp["Date"].dt.strftime("%Y-%m-%d").tolist()
                default_idx = len(dates_str) - 1 if dates_str else 0

                selected_date_str = st.selectbox(
                    "Selecciona el registro (fecha)",
                    options=dates_str,
                    index=default_idx,
                )

                mask = df_emp["Date"].dt.strftime("%Y-%m-%d") == selected_date_str
                if not mask.any():
                    st.warning("No se encontró el registro seleccionado.")
                else:
                    row_emp = df_emp[mask].iloc[-1]
                    idx = row_emp.name  # índice alineado con X_scaled

                    st.markdown("**Características del registro elegido:**")
                    st.write(
                        {
                            "Date": row_emp["Date"],
                            "Position": row_emp["Position"],
                            "Seniority": row_emp["Seniority"],
                            "Location": row_emp["Location"],
                            "Studio": row_emp["Studio"],
                            "Client Tag": row_emp["Client Tag"],
                            "Project Tag": row_emp["Project Tag"],
                            "Team Name": row_emp["Team Name"],
                            "Month": row_emp["Month"],
                            "Day": row_emp["Day"],
                            "15_dias": row_emp.get("15_dias", np.nan),
                            "Engagement_D (real)": row_emp.get("Engagement_D", None),
                        }
                    )

                    if st.button("🔮 Predecir engagement para este registro"):
                        try:
                            probs = predict_from_index(idx, X_scaled, model)
                        except Exception as e:
                            st.error(f"Error al predecir con el modelo: {e}")

        # ------- MODO 2: LLENAR MANUALMENTE -------
        else:
            st.markdown("### Entrada manual de variables")

            pos = st.selectbox("Position", sorted(df_display["Position"].unique()))
            loc = st.selectbox("Location", sorted(df_display["Location"].unique()))
            studio = st.selectbox("Studio", sorted(df_display["Studio"].unique()))
            client = st.selectbox("Client Tag", sorted(df_display["Client Tag"].unique()))
            project = st.selectbox(
                "Project Tag", sorted(df_display["Project Tag"].unique())
            )
            team = st.selectbox(
                "Team Name", sorted(df_display["Team Name"].unique())
            )

            seniority = st.number_input(
                "Seniority",
                value=int(df_display["Seniority"].median()),
                step=1,
            )

            month = st.slider(
                "Month",
                1,
                12,
                int(df_display["Month"].median()),
            )
            day = st.slider(
                "Day",
                1,
                31,
                int(df_display["Day"].median()),
            )

            emp_manual = st.number_input(
                "EmployeeID",
                value=int(df_display["EmployeeID"].median()),
                step=1,
            )

            max_block = int(df_display.get("15_dias", pd.Series([1])).max())
            block_15 = st.slider("Bloque de 15 días (15_dias)", 1, max_block, 1)

            # Engagement de referencia (no entra al modelo)
            eng_text = st.selectbox(
                "Engagement_D (referencia, no se usa como input)",
                ["Bajo", "Medio", "Alto"],
                index=1,
            )

            if st.button("🔮 Predecir engagement con datos manuales"):
                # Usamos una fecha dummy porque el modelo no la usa (se elimina antes)
                dummy_date = df_display["Date"].min()

                row_dict = {
                    "Date": dummy_date,
                    "Position": pos,
                    "Location": loc,
                    "Studio": studio,
                    "Client Tag": client,
                    "Project Tag": project,
                    "Team Name": team,
                    "Seniority": int(seniority),
                    "Month": int(month),
                    "Day": int(day),
                    "EmployeeID": int(emp_manual),
                    "15_dias": int(block_15),
                    "Engagement_D": eng_text,
                }
                try:
                    probs = predict_from_manual(row_dict, feature_cols, scaler, model)
                except Exception as e:
                    st.error(f"Error al predecir con datos manuales: {e}")

        # ------- MOSTRAR RESULTADO -------
        if probs is not None:
            probs = np.array(probs)
            if probs.ndim != 1:
                st.error("La salida del modelo no tiene la forma esperada.")
                return

            k = min(len(probs), len(class_labels))
            probs = probs[:k]
            idx_max = int(np.argmax(probs))
            pred_label = class_labels[idx_max]

            st.markdown("---")
            st.markdown(
                f"### 🧾 Predicción final: **{pred_label}** "
                f"({probs[idx_max] * 100:.1f}% de probabilidad)"
            )

            df_probs = pd.DataFrame(
                {"Clase": class_labels[:k], "Probabilidad": probs}
            )

            fig_bar = px.bar(
                df_probs,
                x="Clase",
                y="Probabilidad",
                range_y=[0, 1],
                title="Distribución de probabilidad por clase",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            st.dataframe(df_probs.set_index("Clase"))


if __name__ == "__main__":
    main()