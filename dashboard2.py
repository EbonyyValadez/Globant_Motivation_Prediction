import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


def main():
    st.set_page_config(page_title="Globant", layout="wide")

    # Cargar datos
    @st.cache_data
    def load_data(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)

        # Convertir fecha
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Crear columnas temporales si hay fecha
        if "Date" in df.columns:
            df["Week"] = df["Date"].dt.isocalendar().week
            df["DayOfMonth"] = df["Date"].dt.day
            df["DayOfWeek"] = df["Date"].dt.day_name()
        else:
            # Fallback por si acaso
            if "Week" not in df.columns:
                df["Week"] = np.nan
            if "DayOfMonth" not in df.columns:
                df["DayOfMonth"] = np.nan
            if "DayOfWeek" not in df.columns:
                df["DayOfWeek"] = np.nan
        return df

    try:
        # Ajusta la ruta si es necesario
        df = load_data("data_globant_clean.csv")
    except FileNotFoundError:
        st.error(
            "No se encontró el archivo `data_globant_clean.csv`.\n\n"
            "💡 Colócalo en la misma carpeta que este script o cambia la ruta en `load_data()`."
        )
        return

    if "Engagement" not in df.columns:
        st.error("La columna `Engagement` no existe en el CSV. Revisa el nombre exacto.")
        st.stop()

    # Interfaz
    st.title("Engagement Globant")
    st.markdown(
        "Explora el engagement a lo largo del tiempo filtrando los datos por proyecto, estudio, equipo, "
        "posición, seniority y locación."
    )

    st.sidebar.header("Filtros")

    filters = {
        "Project": "Proyecto",
        "Studio": "Estudio",
        "Team Name": "Equipo",
        "Position": "Posición",
        "Seniority": "Seniority",
        "Location": "Locación",
    }

    df_filtered = df.copy()
    for col, label in filters.items():
        if col in df.columns:
            options = ["(Todos)"] + sorted(df[col].dropna().unique().tolist())
            selected = st.sidebar.selectbox(label, options)
            if selected != "(Todos)":
                df_filtered = df_filtered[df_filtered[col].isin([selected])]
        else:
            st.sidebar.warning(f"La columna `{col}` no existe en el CSV.")

    # ===== Tipo de agregación temporal =====
    agg_type = st.sidebar.selectbox(
        "Tipo de agregación temporal:",
        ["Promedio semanal", "Promedio por día de la semana", "Promedio por día del mes"],
    )

    # ===== Tratamiento de ceros =====
    ignore_zero = st.sidebar.checkbox(
        "Excluir engagement <= 0 del promedio",
        value=True,
        help="Los valores <= 0 no cuentan para el promedio, pero sí para el color de la línea.",
    )

    # Preprocesamiento
    if df_filtered.empty:
        st.warning("No hay datos que coincidan con los filtros seleccionados.")
        st.stop()

    df_vis = df_filtered.copy()

    # Marcamos dónde hay 0 o menos
    df_vis["IsZero"] = (df_vis["Engagement"] <= 0).astype(int)

    if ignore_zero:
        df_vis.loc[df_vis["Engagement"] <= 0, "Engagement"] = np.nan

    # Agregación
    def aggregate(df_in: pd.DataFrame, group_col: str) -> pd.DataFrame:
        if group_col not in df_in.columns:
            return pd.DataFrame()
        grouped = df_in.groupby(group_col).agg(
            Engagement_mean=("Engagement", "mean"),
            CountNonNa=("Engagement", "count"),   # registros que sí aportan promedio
            ZeroCount=("IsZero", "sum"),          # cuántos son 0 o menos
        )
        grouped = grouped.reset_index()
        return grouped

    if agg_type == "Promedio semanal":
        group_col = "Week"
        x_title = "Semana del año"
        df_plot = aggregate(df_vis, group_col)

    elif agg_type == "Promedio por día de la semana":
        group_col = "DayOfWeek"
        x_title = "Día de la semana"
        df_plot = aggregate(df_vis, group_col)
        if not df_plot.empty:
            order = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            df_plot[group_col] = pd.Categorical(
                df_plot[group_col], categories=order, ordered=True
            )
            df_plot = df_plot.sort_values(group_col)

    else:  # "Promedio por día del mes"
        group_col = "DayOfMonth"
        x_title = "Día del mes"
        df_plot = aggregate(df_vis, group_col)

    if df_plot.empty:
        st.warning("No hay datos agregados para la combinación de filtros y tipo de agregación.")
        st.stop()

    # Color dinámico
    df_plot["TotalRegistros"] = df_plot["ZeroCount"] + df_plot["CountNonNa"]
    df_plot["PercentZero"] = np.where(
        df_plot["TotalRegistros"] > 0,
        df_plot["ZeroCount"] / df_plot["TotalRegistros"],
        0.0,
    )

    def get_color(p: float) -> str:
        if p < 0.05:
            return "green"
        elif p < 0.15:
            return "yellow"
        else:
            return "red"

    # Gráfica
    st.subheader("Evolución del engagement")

    fig = px.line(
        df_plot,
        x=group_col,
        y="Engagement_mean",
        markers=True,
        title="Engagement promedio según selección",
    )

    # Línea neutra y puntos coloreados según % de ceros
    fig.update_traces(
        line=dict(color="lightgray", width=2),
        marker=dict(size=10),
    )
    # Aplica color punto a punto
    if "PercentZero" in df_plot.columns:
        colors = [get_color(p) for p in df_plot["PercentZero"]]
        fig.update_traces(marker=dict(color=colors))

    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Engagement promedio",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================================
    # Sección: Predicción Cadena de Markov
    # =========================================
    st.subheader("Predicción Cadena de Markov")

    # Copia de datos solo con columnas necesarias
    df_markov = df.copy()

    # Aseguramos columna de estados discretos
    STATE_COL = "Engagement_bin"
    if STATE_COL not in df_markov.columns:
        # Si no existe, creamos 5 estados a partir de la columna Engagement
        # Puedes ajustar los bins manualmente si ya tienes una discretización definida
        n_states = 5
        df_markov[STATE_COL] = pd.cut(
            df_markov["Engagement"],
            bins=n_states,
            labels=[f"S{i+1}" for i in range(n_states)],
            include_lowest=True,
        )

    # Quitamos filas sin estado
    df_markov = df_markov.dropna(subset=[STATE_COL])
    df_markov[STATE_COL] = df_markov[STATE_COL].astype(str)

    @st.cache_data
    def compute_transition_matrix(df_in: pd.DataFrame, state_col: str, id_col: str = "Name"):
        # Orden temporal
        sort_cols = []
        for col in ["Date", "Week", "DayOfMonth"]:
            if col in df_in.columns:
                sort_cols.append(col)

        if not sort_cols:
            # Si no hay columnas temporales, usamos el índice como fallback
            df_local = df_in.reset_index().rename(columns={"index": "_Order"})
            sort_cols_local = ["_Order"]
        else:
            df_local = df_in.copy()
            sort_cols_local = sort_cols

        df_sorted = df_local[[id_col, state_col] + sort_cols_local].dropna(subset=[state_col])
        df_sorted = df_sorted.sort_values([id_col] + sort_cols_local)

        states = np.sort(df_sorted[state_col].unique())
        n = len(states)
        state_to_idx = {s: i for i, s in enumerate(states)}

        counts = np.zeros((n, n), dtype=float)

        for _, group in df_sorted.groupby(id_col):
            s = group[state_col].values
            for i in range(len(s) - 1):
                a = state_to_idx[s[i]]
                b = state_to_idx[s[i + 1]]
                counts[a, b] += 1.0

        row_sums = counts.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            probs = np.where(row_sums > 0, counts / row_sums, 0.0)

        return states, state_to_idx, counts, probs

    states, state_to_idx, counts_mat, P = compute_transition_matrix(df_markov, STATE_COL)

    if len(states) == 0:
        st.info("No hay suficientes datos para construir la cadena de Markov.")
    else:
        # Selector de modo
        modo = st.radio("Filtro", ["Empleado", "Libre"], horizontal=True)

        # Horizonte de predicción
        n_dias = st.slider("Días a futuro:", min_value=1, max_value=21, value=7)

        # Matriz P^n
        from numpy.linalg import matrix_power

        Pn = matrix_power(P, n_dias)

        # Variable donde guardaremos las probabilidades a mostrar
        probs_vector = None
        estado_inicial = None

        if modo == "Libre":
            estado_inicial = st.selectbox("Selecciona el estado inicial", list(states))
            idx = state_to_idx.get(estado_inicial, None)
            if idx is not None:
                probs_vector = Pn[idx, :]
            else:
                st.warning("El estado seleccionado no existe en la matriz de transición.")

        else:  # Empleado
            all_names = sorted(df_markov["Name"].dropna().unique())
            search_text = st.text_input("Buscar empleado")

            if search_text:
                filtered_names = [n for n in all_names if search_text.lower() in n.lower()]
            else:
                filtered_names = all_names

            if not filtered_names:
                st.info("No se encontraron empleados con ese texto de búsqueda.")
            else:
                selected_name = st.selectbox("Seleccionar empleado", filtered_names)

                emp_df = df_markov[df_markov["Name"] == selected_name].copy()
                # Ordenar por fecha (o columnas temporales disponibles)
                if "Date" in emp_df.columns:
                    emp_df = emp_df.sort_values("Date")
                elif "Week" in emp_df.columns:
                    emp_df = emp_df.sort_values("Week")
                elif "DayOfMonth" in emp_df.columns:
                    emp_df = emp_df.sort_values("DayOfMonth")

                if emp_df.empty:
                    st.info("No hay datos para este empleado.")
                else:
                    ultimos = emp_df.tail(10)

                    st.markdown(f"**Últimos {len(ultimos)} registros de {selected_name}**")
                    if "Date" in ultimos.columns:
                        fig_emp = px.line(
                            ultimos,
                            x="Date",
                            y="Engagement",
                            markers=True,
                            title=f"Engagement Historico - {selected_name}",
                        )
                    else:
                        fig_emp = px.line(
                            ultimos.reset_index(),
                            x=ultimos.reset_index().index,
                            y="Engagement",
                            markers=True,
                            title=f"Engagement Historico- {selected_name}",
                        )
                    st.plotly_chart(fig_emp, use_container_width=True)

                    estado_inicial = str(ultimos.iloc[-1][STATE_COL])
                    st.write(f"Estado actual: **{estado_inicial}**")

                    idx = state_to_idx.get(estado_inicial, None)
                    if idx is not None:
                        probs_vector = Pn[idx, :]
                    else:
                        st.warning("El estado del último registro no existe en la matriz de transición.")

        # Mostrar ranking de estados más probables
        if probs_vector is not None:
            ranking = pd.DataFrame(
                {
                    "Estado": states,
                    "Probabilidad": probs_vector,
                }
            ).sort_values("Probabilidad", ascending=False)

            # Gráfica tipo barra con los estados top
            top_k = min(5, len(ranking))
            fig_bar = px.bar(
                ranking.head(top_k),
                x="Estado",
                y="Probabilidad",
                text="Probabilidad",
                title=f"Top {top_k} estados más probables en {n_dias} días",
            )
            fig_bar.update_traces(texttemplate="%{y:.2%}", textposition="outside")
            fig_bar.update_yaxes(title="Probabilidad")
            st.plotly_chart(fig_bar, use_container_width=True)


if __name__ == "__main__":
    main()