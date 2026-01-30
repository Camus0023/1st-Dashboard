"""
Dashboard de Energía Renovable - Streamlit
Carga de datos: archivo por defecto o CSV externo (upload).
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from io import BytesIO
import time

# --- Configuración de la página ---
st.set_page_config(
    page_title="Dashboard Energía Renovable",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Estilos visuales ---
st.markdown("""
<style>
    /* Mensaje de bienvenida */
    .welcome-box {
        background: linear-gradient(135deg, #1a5f4a 0%, #2d8f6f 50%, #1a5f4a 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .welcome-box h1 {
        color: #fff;
        font-size: 2rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .welcome-box p {
        color: rgba(255,255,255,0.95);
        font-size: 1.05rem;
        margin: 0;
    }
    /* Tarjetas KPI */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #f8faf9 0%, #e8f0ee 100%);
        padding: 1rem 1.25rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid rgba(26, 95, 74, 0.15);
    }
    div[data-testid="stMetric"] label {
        color: #1a5f4a !important;
        font-weight: 600 !important;
    }
    /* Títulos de sección */
    h2, h3 {
        color: #1a5f4a !important;
        font-weight: 600 !important;
    }
    /* Sidebar: fondo claro y texto oscuro para buen contraste */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    [data-testid="stSidebar"] > div:first-child {
        background-color: #ffffff !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        color: #1a3d32 !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
        color: #1f2937 !important;
    }
    [data-testid="stSidebar"] .stRadio label, [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #374151 !important;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #1f2937 !important;
    }
    /* Ocultar marca de agua de Streamlit (opcional) */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


def load_csv_with_retry(source, max_retries=3, delay_seconds=0.5):
    """
    Carga y normaliza un CSV con reintentos en caso de fallo.
    source: bytes (de upload) o str (ruta al archivo).
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            if isinstance(source, bytes):
                df = pd.read_csv(BytesIO(source))
            else:
                df = pd.read_csv(source)
            # Normalizar tipos
            if "Fecha_Entrada_Operacion" in df.columns:
                df["Fecha_Entrada_Operacion"] = pd.to_datetime(
                    df["Fecha_Entrada_Operacion"], errors="coerce"
                )
            if "Conectado_SIN" in df.columns:
                df["Conectado_SIN"] = df["Conectado_SIN"].astype(str).str.lower() == "true"
            return df
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(delay_seconds)
    raise last_error


def load_default_data():
    """Carga el CSV por defecto del proyecto con retry."""
    csv_path = Path(__file__).parent / "energia_renovable.csv"
    if not csv_path.exists():
        return None
    return load_csv_with_retry(str(csv_path))


# --- Mensaje de bienvenida ---
st.markdown("""
<div class="welcome-box">
    <h1>⚡ Bienvenido al Dashboard de Energía Renovable</h1>
    <p>Explora proyectos, capacidades e inversiones. Carga tu propio CSV o usa los datos de ejemplo desde el panel lateral.</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar: carga de datos ---
st.sidebar.header("📂 Cargar datos")

use_upload = st.sidebar.radio(
    "Origen de los datos",
    ["Usar datos de ejemplo (energia_renovable.csv)", "Subir mi propio archivo CSV"],
    index=0,
)

df = None
load_error = None

if use_upload == "Subir mi propio archivo CSV":
    uploaded = st.sidebar.file_uploader(
        "Selecciona un archivo CSV",
        type=["csv"],
        help="El CSV debe tener columnas compatibles (ej.: Tecnologia, Operador, Capacidad_Instalada_MW, etc.).",
    )
    if uploaded is not None:
        with st.sidebar.status("Cargando y validando datos...", state="running") as status:
            try:
                df = load_csv_with_retry(uploaded.getvalue())
                status.update(label="Datos cargados correctamente", state="complete")
            except Exception as e:
                load_error = str(e)
                status.update(label="Error al cargar", state="error")
                st.sidebar.error(f"Error (tras reintentos): {load_error}")
else:
    with st.sidebar.status("Cargando datos de ejemplo...", state="running") as status:
        try:
            df = load_default_data()
            if df is not None:
                status.update(label="Datos de ejemplo cargados", state="complete")
            else:
                load_error = "No se encontró energia_renovable.csv."
                status.update(label="Archivo no encontrado", state="error")
                st.sidebar.warning(load_error)
        except Exception as e:
            load_error = str(e)
            status.update(label="Error al cargar", state="error")
            st.sidebar.error(f"Error (tras reintentos): {load_error}")

# --- Contenido principal según haya datos o no ---
if df is None:
    if load_error and use_upload == "Subir mi propio archivo CSV":
        st.error("No se pudieron cargar los datos. Revisa el formato del CSV e inténtalo de nuevo.")
    elif load_error:
        st.warning("No se encontraron datos de ejemplo. Sube un CSV desde el panel lateral.")
    else:
        st.info("👈 Elige **'Usar datos de ejemplo'** o **'Subir mi propio archivo CSV'** en el panel izquierdo para ver el dashboard.")
    st.stop()

# --- Filtros (solo si hay datos) ---
st.sidebar.divider()
st.sidebar.header("🔍 Filtros")

tecnologias = ["Todos"] + sorted(df["Tecnologia"].dropna().unique().tolist())
operadores = ["Todos"] + sorted(df["Operador"].dropna().unique().tolist())
estados = ["Todos"] + sorted(df["Estado_Actual"].dropna().unique().tolist())

filtro_tec = st.sidebar.selectbox("Tecnología", tecnologias)
filtro_op = st.sidebar.selectbox("Operador", operadores)
filtro_est = st.sidebar.selectbox("Estado actual", estados)
solo_sin = st.sidebar.checkbox("Solo conectados al SIN", value=False)

# Aplicar filtros
df_filt = df.copy()
if filtro_tec != "Todos":
    df_filt = df_filt[df_filt["Tecnologia"] == filtro_tec]
if filtro_op != "Todos":
    df_filt = df_filt[df_filt["Operador"] == filtro_op]
if filtro_est != "Todos":
    df_filt = df_filt[df_filt["Estado_Actual"] == filtro_est]
if solo_sin and "Conectado_SIN" in df_filt.columns:
    df_filt = df_filt[df_filt["Conectado_SIN"] == True]

# Datos activos: se inicializan una vez; "Restablecer" en Cualitativo aplica filtros del sidebar
if "edited_df" not in st.session_state:
    st.session_state.edited_df = df_filt.copy()

# Columnas para mostrar en tablas
columnas_esperadas = [
    "ID_Proyecto", "Tecnologia", "Operador", "Capacidad_Instalada_MW",
    "Generacion_Diaria_MWh", "Eficiencia_Planta_Pct", "Estado_Actual",
    "Inversion_Inicial_MUSD", "Fecha_Entrada_Operacion",
]
columnas_tabla = [c for c in columnas_esperadas if c in df_filt.columns]
if not columnas_tabla:
    columnas_tabla = df_filt.columns.tolist()

st.caption(f"📌 Mostrando **{len(df_filt)}** de **{len(df)}** proyectos. Los cambios en **Cualitativo** se reflejan en los demás bloques.")

# --- Tres bloques: Cuantitativo | Cualitativo | Gráfico ---
tab_cuant, tab_cual, tab_graf = st.tabs(["📊 Bloque Cuantitativo", "📋 Bloque Cualitativo", "📈 Bloque Gráfico"])

# ---------- BLOQUE CUANTITATIVO ----------
with tab_cuant:
    st.subheader("Métricas numéricas")
    st.markdown("Indicadores y estadísticas descriptivas de los datos (según filtros y ediciones).")

    col1, col2, col3, col4, col5 = st.columns(5)
    d = st.session_state.edited_df

    with col1:
        st.metric("Proyectos", len(d))
    with col2:
        cap = d["Capacidad_Instalada_MW"].sum() if "Capacidad_Instalada_MW" in d.columns else 0
        st.metric("Capacidad total (MW)", f"{cap:,.0f}")
    with col3:
        gen = d["Generacion_Diaria_MWh"].sum() if "Generacion_Diaria_MWh" in d.columns else 0
        st.metric("Generación diaria (MWh)", f"{gen:,.0f}")
    with col4:
        inv = d["Inversion_Inicial_MUSD"].sum() if "Inversion_Inicial_MUSD" in d.columns else 0
        st.metric("Inversión (M USD)", f"{inv:,.0f}")
    with col5:
        efic = d["Eficiencia_Planta_Pct"].mean() if "Eficiencia_Planta_Pct" in d.columns else 0
        st.metric("Eficiencia media (%)", f"{efic:.1f}")

    st.markdown("---")
    st.markdown("**Estadísticas descriptivas** (variables numéricas)")
    numeric_cols = d.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        st.dataframe(d[numeric_cols].describe().round(2), use_container_width=True)
    else:
        st.info("No hay columnas numéricas para resumir.")

    # Filtros numéricos interactivos opcionales
    with st.expander("🔧 Ajustar rango de valores (filtro extra)"):
        d_cols = st.session_state.edited_df
        if "Capacidad_Instalada_MW" in d_cols.columns:
            min_cap = float(d_cols["Capacidad_Instalada_MW"].min())
            max_cap = float(d_cols["Capacidad_Instalada_MW"].max())
            rango = st.slider("Capacidad instalada (MW)", min_cap, max_cap, (min_cap, max_cap))
            mask = (d_cols["Capacidad_Instalada_MW"] >= rango[0]) & (d_cols["Capacidad_Instalada_MW"] <= rango[1])
            st.session_state.edited_df = d_cols[mask].copy()
            st.caption(f"Quedan {len(st.session_state.edited_df)} proyectos en este rango.")

# ---------- BLOQUE CUALITATIVO ----------
with tab_cual:
    st.subheader("Datos categóricos y tabla editable")
    st.markdown("Resúmenes por categoría y tabla para **ver y modificar** valores. Los cambios se usan en Cuantitativo y Gráfico.")

    d = st.session_state.edited_df

    # Resúmenes cualitativos
    c1, c2, c3 = st.columns(3)
    with c1:
        if "Tecnologia" in d.columns:
            st.markdown("**Por tecnología**")
            st.dataframe(d["Tecnologia"].value_counts(), use_container_width=True, height=180)
    with c2:
        if "Operador" in d.columns:
            st.markdown("**Por operador**")
            st.dataframe(d["Operador"].value_counts(), use_container_width=True, height=180)
    with c3:
        if "Estado_Actual" in d.columns:
            st.markdown("**Por estado**")
            st.dataframe(d["Estado_Actual"].value_counts(), use_container_width=True, height=180)

    st.markdown("---")
    st.markdown("**Tabla interactiva** — edita celdas y los cambios se reflejan en los otros bloques.")

    col_sel = [c for c in columnas_tabla if c in d.columns]
    sort_col = "Capacidad_Instalada_MW" if "Capacidad_Instalada_MW" in d.columns else (col_sel[0] if col_sel else None)
    tabla_ordenada = d[col_sel].sort_values(sort_col, ascending=False) if sort_col else d[col_sel]

    edited = st.data_editor(
        tabla_ordenada,
        use_container_width=True,
        num_rows="dynamic",
        key="cualitativo_editor",
    )
    st.session_state.edited_df = edited

    b1, b2, b3 = st.columns([1, 1, 2])
    with b1:
        if st.button("Restablecer a datos filtrados"):
            st.session_state.edited_df = df_filt.copy()
            st.rerun()
    with b2:
        # Exportar CSV
        csv_export = st.session_state.edited_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar tabla actual (CSV)",
            data=csv_export,
            file_name="energia_renovable_editado.csv",
            mime="text/csv",
        )

# ---------- BLOQUE GRÁFICO ----------
with tab_graf:
    st.subheader("Visualizaciones")
    st.markdown("Gráficos según los datos actuales (filtros y ediciones del bloque Cualitativo).")

    d = st.session_state.edited_df

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        if "Tecnologia" in d.columns:
            st.markdown("**Proyectos por tecnología**")
            count_tec = d["Tecnologia"].value_counts().reset_index()
            count_tec.columns = ["Tecnologia", "Cantidad"]
            fig_tec = px.bar(
                count_tec, x="Tecnologia", y="Cantidad",
                color="Cantidad", color_continuous_scale="Teal",
                labels={"Tecnologia": "Tecnología", "Cantidad": "Nº proyectos"},
            )
            fig_tec.update_layout(showlegend=False, margin=dict(t=10, b=10), font=dict(size=12))
            fig_tec.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,249,0.8)")
            st.plotly_chart(fig_tec, use_container_width=True)

    with row1_col2:
        if "Estado_Actual" in d.columns:
            st.markdown("**Proyectos por estado**")
            count_est = d["Estado_Actual"].value_counts().reset_index()
            count_est.columns = ["Estado_Actual", "Cantidad"]
            fig_est = px.pie(
                count_est, values="Cantidad", names="Estado_Actual",
                color_discrete_sequence=px.colors.sequential.Teal_r,
                hole=0.45,
            )
            fig_est.update_layout(margin=dict(t=10, b=10), font=dict(size=12))
            fig_est.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_est, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        if "Operador" in d.columns and "Capacidad_Instalada_MW" in d.columns:
            st.markdown("**Capacidad instalada por operador (MW)**")
            cap_op = d.groupby("Operador")["Capacidad_Instalada_MW"].sum().reset_index()
            cap_op = cap_op.sort_values("Capacidad_Instalada_MW", ascending=True)
            fig_cap = px.bar(
                cap_op, x="Capacidad_Instalada_MW", y="Operador",
                orientation="h", color="Capacidad_Instalada_MW",
                color_continuous_scale="Teal",
                labels={"Capacidad_Instalada_MW": "MW", "Operador": "Operador"},
            )
            fig_cap.update_layout(showlegend=False, margin=dict(t=10, b=10), font=dict(size=12))
            fig_cap.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,249,0.8)")
            st.plotly_chart(fig_cap, use_container_width=True)

    with row2_col2:
        if "Fecha_Entrada_Operacion" in d.columns:
            st.markdown("**Proyectos por año de entrada en operación**")
            d_ano = d.copy()
            d_ano["Año"] = pd.to_datetime(d_ano["Fecha_Entrada_Operacion"], errors="coerce").dt.year
            count_ano = d_ano.dropna(subset=["Año"]).groupby("Año").size().reset_index(name="Cantidad")
            fig_ano = px.line(
                count_ano, x="Año", y="Cantidad",
                markers=True,
                labels={"Año": "Año", "Cantidad": "Proyectos"},
                color_discrete_sequence=["#1a5f4a"],
            )
            fig_ano.update_layout(margin=dict(t=10, b=10), font=dict(size=12))
            fig_ano.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,249,0.8)")
            st.plotly_chart(fig_ano, use_container_width=True)
