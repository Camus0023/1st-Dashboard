"""
Dashboard de datos - Streamlit
Carga CSV desde la página (cualquier conjunto de datos), a prueba de errores.
Bloques: Cuantitativo, Cualitativo, Gráfico (elaborado y contextual).
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from io import BytesIO
import time

# --- Configuración de la página ---
st.set_page_config(
    page_title="Dashboard de Datos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Estilos visuales ---
st.markdown("""
<style>
    .welcome-box {
        background: linear-gradient(135deg, #1a5f4a 0%, #2d8f6f 50%, #1a5f4a 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .welcome-box h1 { color: #fff; font-size: 2rem; margin-bottom: 0.5rem; font-weight: 700; }
    .welcome-box p { color: rgba(255,255,255,0.95); font-size: 1.05rem; margin: 0; }
    .upload-zone {
        background: #f8faf9;
        border: 2px dashed #2d8f6f;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #f8faf9 0%, #e8f0ee 100%);
        padding: 1rem 1.25rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid rgba(26, 95, 74, 0.15);
    }
    div[data-testid="stMetric"] label { color: #1a5f4a !important; font-weight: 600 !important; }
    h2, h3 { color: #1a5f4a !important; font-weight: 600 !important; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: #1a3d32 !important; font-weight: 700 !important; }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p { color: #1f2937 !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


def load_any_csv(source, max_retries=3, delay_seconds=0.3):
    """
    Carga cualquier CSV de forma robusta: varios encodings, reintentos.
    source: bytes (upload) o str (ruta). Devuelve (df, None) o (None, mensaje_error).
    """
    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    last_error = None

    def _read(data, encoding):
        if isinstance(data, bytes):
            return pd.read_csv(BytesIO(data), encoding=encoding, on_bad_lines="skip")
        return pd.read_csv(data, encoding=encoding, on_bad_lines="skip")

    for attempt in range(1, max_retries + 1):
        for enc in encodings:
            try:
                if isinstance(source, bytes):
                    df = pd.read_csv(BytesIO(source), encoding=enc, on_bad_lines="skip")
                else:
                    df = pd.read_csv(source, encoding=enc, on_bad_lines="skip")
                if df is None or df.empty:
                    return None, "El archivo está vacío o no tiene filas."
                # Limpiar nombres de columnas
                df.columns = df.columns.str.strip()
                if len(df.columns) == 0:
                    return None, "No se detectaron columnas en el CSV."
                # Intentar convertir columnas que parecen fechas
                for col in df.select_dtypes(include=["object"]).columns:
                    try:
                        pd.to_datetime(df[col], errors="coerce")
                        if pd.to_datetime(df[col], errors="coerce").notna().sum() > len(df) * 0.5:
                            df[col] = pd.to_datetime(df[col], errors="coerce")
                    except Exception:
                        pass
                # Booleanos comunes
                for col in df.columns:
                    if df[col].dtype == object and df[col].dropna().astype(str).str.lower().isin(["true", "false", "1", "0", "sí", "si", "no"]).all():
                        try:
                            df[col] = df[col].astype(str).str.lower().replace({"true": True, "false": False, "1": True, "0": False, "sí": True, "si": True, "no": False})
                        except Exception:
                            pass
                return df, None
            except Exception as e:
                last_error = e
        if attempt < max_retries:
            time.sleep(delay_seconds)

    return None, f"No se pudo leer el CSV. Último error: {str(last_error)}. Prueba otro encoding o formato."


def infer_column_types(df):
    """Clasifica columnas en numéricas, categóricas y fecha."""
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    # Fechas: ya convertidas o detectables
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    for col in df.columns:
        if col in numeric or col in date_cols:
            continue
        try:
            s = pd.to_datetime(df[col], errors="coerce")
            if s.notna().sum() > len(df) * 0.3:
                date_cols.append(col)
                df[col] = s
        except Exception:
            pass
    categorical = [c for c in df.columns if c not in numeric and c not in date_cols
                   and df[c].dtype == object and df[c].nunique() < min(200, len(df) // 2)]
    # Si no hay categóricas por tipo, usar object con pocos únicos
    if not categorical:
        for c in df.select_dtypes(include=["object"]).columns:
            if c not in date_cols and df[c].nunique() < min(100, len(df)):
                categorical.append(c)
    return {"numeric": numeric, "categorical": categorical, "datetime": date_cols}


def is_energy_context(df):
    """Detecta si el dataset parece ser de energía renovable (contexto conocido)."""
    cols = set(df.columns)
    energy_hints = {"Tecnologia", "Operador", "Capacidad_Instalada_MW", "Generacion_Diaria_MWh",
                   "Estado_Actual", "Eficiencia_Planta_Pct", "Inversion_Inicial_MUSD", "Fecha_Entrada_Operacion"}
    return len(cols & energy_hints) >= 3


# --- Mensaje de bienvenida ---
st.markdown("""
<div class="welcome-box">
    <h1>📊 Dashboard de Datos</h1>
    <p>Sube cualquier archivo CSV desde aquí. El dashboard se adapta a tus columnas (numéricas, categóricas y fechas).</p>
</div>
""", unsafe_allow_html=True)

# --- Zona de carga en la página principal ---
st.markdown("#### 📂 Cargar datos")
upload_col, demo_col = st.columns([2, 1])

with upload_col:
    uploaded = st.file_uploader(
        "Arrastra o selecciona un archivo CSV",
        type=["csv"],
        help="Cualquier CSV con columnas separadas por coma. Se intentarán varios encodings si falla la lectura.",
        key="main_uploader",
    )

with demo_col:
    use_demo = st.button("📁 Usar datos de ejemplo (energía renovable)")
    default_path = Path(__file__).parent / "energia_renovable.csv"

df = None
load_error = None

if uploaded is not None:
    with st.status("Leyendo y validando CSV...", state="running") as status:
        df, load_error = load_any_csv(uploaded.getvalue())
        if load_error:
            status.update(label="Error al cargar", state="error")
            st.error(load_error)
        else:
            status.update(label="Datos cargados correctamente", state="complete")
            st.success(f"Se cargaron **{len(df)}** filas y **{len(df.columns)}** columnas.")

if use_demo:
    if default_path.exists():
        df, load_error = load_any_csv(str(default_path))
        if load_error:
            st.error(load_error)
        else:
            st.success(f"Datos de ejemplo cargados: **{len(df)}** filas.")
    else:
        st.warning("No se encontró energia_renovable.csv. Sube tu propio CSV arriba.")

# Sin datos: mostrar mensaje y parar
if df is None or df.empty:
    st.info("👆 **Sube un archivo CSV** arriba o pulsa **Usar datos de ejemplo** para empezar.")
    st.stop()

# --- Cantidad de muestras para el análisis ---
st.markdown("#### 📐 Cantidad de muestras para el análisis")
total_filas = len(df)
default_muestras = min(500, total_filas) if total_filas > 500 else total_filas

samples_col1, samples_col2 = st.columns([2, 1])

with samples_col1:
    n_slider = st.slider(
        "Barra: arrastra para elegir cantidad de muestras",
        min_value=1,
        max_value=total_filas,
        value=default_muestras,
        step=10 if total_filas > 100 else 1,
        help="Reduce el número para un análisis más rápido. Se toma una muestra aleatoria reproducible.",
        key="slider_muestras",
    )

with samples_col2:
    n_textbox = st.number_input(
        "O escribe el número exacto",
        min_value=1,
        max_value=total_filas,
        value=n_slider,
        step=10 if total_filas > 100 else 1,
        key="number_muestras",
    )

# Aplicar el valor del textbox (si lo cambió) o el del slider
n_uso = n_textbox
if n_uso < total_filas:
    df = df.sample(n=n_uso, random_state=42)
    st.caption(f"Se usan **{n_uso:,}** de **{total_filas:,}** filas (muestra aleatoria reproducible).")
else:
    st.caption(f"Se usan **todas** las filas (**{total_filas:,}**).")

# Inferir tipos y contexto
col_types = infer_column_types(df)
numeric_cols = col_types["numeric"]
cat_cols = col_types["categorical"]
date_cols = col_types["datetime"]
energy_mode = is_energy_context(df)

# --- Sidebar: filtros dinámicos ---
st.sidebar.header("🔍 Filtros")
df_filt = df.copy()

for col in cat_cols:
    if col not in df_filt.columns or df_filt[col].nunique() > 100:
        continue
    opts = ["Todos"] + sorted(df_filt[col].dropna().astype(str).unique().tolist())
    sel = st.sidebar.selectbox(col, opts, key=f"filt_{col}")
    if sel != "Todos":
        df_filt = df_filt[df_filt[col].astype(str) == sel]

# Filtros numéricos opcionales (primera columna numérica)
if numeric_cols:
    first_num = numeric_cols[0]
    v_min = float(df_filt[first_num].min())
    v_max = float(df_filt[first_num].max())
    if v_min != v_max:
        r = st.sidebar.slider(f"Rango: {first_num}", v_min, v_max, (v_min, v_max), key="slider_num")
        df_filt = df_filt[(df_filt[first_num] >= r[0]) & (df_filt[first_num] <= r[1])]

# Session state para datos editados
if "edited_df" not in st.session_state or len(st.session_state.edited_df) != len(df_filt):
    st.session_state.edited_df = df_filt.copy()

d = st.session_state.edited_df
columnas_tabla = d.columns.tolist()

st.caption(f"📌 **{len(df_filt)}** de **{len(df)}** filas. Ediciones en Cualitativo se reflejan en Cuantitativo y Gráfico.")

# --- Tabs ---
tab_cuant, tab_cual, tab_graf = st.tabs(["📊 Cuantitativo", "📋 Cualitativo", "📈 Gráfico"])

# ---------- BLOQUE CUANTITATIVO ----------
with tab_cuant:
    st.subheader("Métricas numéricas")
    n = len(d)
    st.metric("Filas", n)

    if numeric_cols:
        n_show = min(5, len(numeric_cols))
        cols_ui = st.columns(n_show)
        for i, col in enumerate(numeric_cols[:n_show]):
            with cols_ui[i]:
                if d[col].dtype in ["int64", "float64"]:
                    val = d[col].sum() if d[col].abs().sum() > 1e6 else d[col].mean()
                    label = f"Suma({col})" if d[col].abs().sum() > 1e6 else f"Media({col})"
                    st.metric(label[:25], f"{val:,.2f}")
    st.markdown("---")
    st.markdown("**Estadísticas descriptivas**")
    if numeric_cols:
        st.dataframe(d[numeric_cols].describe().round(2), use_container_width=True)
    else:
        st.info("No hay columnas numéricas en el dataset.")

    with st.expander("🔧 Filtro extra por columna numérica"):
        if numeric_cols:
            col_sel = st.selectbox("Columna", numeric_cols, key="num_filter_col")
            mn = float(d[col_sel].min())
            mx = float(d[col_sel].max())
            rango = st.slider("Rango", mn, mx, (mn, mx), key="num_filter_slider")
            mask = (d[col_sel] >= rango[0]) & (d[col_sel] <= rango[1])
            st.session_state.edited_df = d[mask].copy()
            st.caption(f"Quedan {len(st.session_state.edited_df)} filas.")

# ---------- BLOQUE CUALITATIVO ----------
with tab_cual:
    st.subheader("Datos categóricos y tabla editable")
    d = st.session_state.edited_df

    if cat_cols:
        nc = min(3, len(cat_cols))
        subcols = st.columns(nc)
        for i, col in enumerate(cat_cols[:nc]):
            with subcols[i]:
                st.markdown(f"**Conteo: {col}**")
                st.dataframe(d[col].value_counts(), use_container_width=True, height=160)

    st.markdown("---")
    st.markdown("**Tabla editable** — los cambios se usan en Cuantitativo y Gráfico.")
    sort_col = numeric_cols[0] if numeric_cols else (d.columns[0] if len(d.columns) else None)
    asc = False if numeric_cols else True
    tabla_ordenada = d.sort_values(sort_col, ascending=asc) if sort_col and sort_col in d.columns else d

    edited = st.data_editor(tabla_ordenada, use_container_width=True, num_rows="dynamic", key="cual_editor")
    st.session_state.edited_df = edited

    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        if st.button("Restablecer a datos filtrados"):
            st.session_state.edited_df = df_filt.copy()
            st.rerun()
    with c2:
        st.download_button("Descargar CSV actual", data=st.session_state.edited_df.to_csv(index=False).encode("utf-8"),
                           file_name="datos_editados.csv", mime="text/csv")

# ---------- BLOQUE GRÁFICO (elaborado y contextual) ----------
with tab_graf:
    st.subheader("Visualizaciones")
    d = st.session_state.edited_df

    # Contexto energía: gráficos específicos y más elaborados
    if energy_mode:
        # 1) Sunburst: Tecnología > Operador > conteo (si existen)
        if "Tecnologia" in d.columns and "Operador" in d.columns:
            st.markdown("**Distribución por tecnología y operador** (proyectos de energía renovable)")
            agg = d.groupby(["Tecnologia", "Operador"]).size().reset_index(name="Proyectos")
            fig_sun = px.sunburst(agg, path=["Tecnologia", "Operador"], values="Proyectos",
                                  color="Proyectos", color_continuous_scale="Teal",
                                  title="Proyectos por tecnología y operador")
            fig_sun.update_layout(margin=dict(t=40, b=20), font=dict(size=12))
            fig_sun.update_traces(textinfo="label+value")
            st.plotly_chart(fig_sun, use_container_width=True)

        # 2) Box: Eficiencia por tecnología
        if "Eficiencia_Planta_Pct" in d.columns and "Tecnologia" in d.columns:
            st.markdown("**Eficiencia de planta (%) por tecnología**")
            fig_box = px.box(d, x="Tecnologia", y="Eficiencia_Planta_Pct", color="Tecnologia",
                            points="outliers", title="Distribución de eficiencia por tipo de tecnología")
            fig_box.update_layout(showlegend=False, xaxis_tickangle=-45, margin=dict(t=40, b=80))
            fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,249,0.8)")
            st.plotly_chart(fig_box, use_container_width=True)

        # 3) Scatter: Capacidad vs Generación
        if "Capacidad_Instalada_MW" in d.columns and "Generacion_Diaria_MWh" in d.columns:
            st.markdown("**Capacidad instalada vs generación diaria** (MW vs MWh)")
            try:
                fig_sc = px.scatter(d, x="Capacidad_Instalada_MW", y="Generacion_Diaria_MWh",
                                   color="Tecnologia" if "Tecnologia" in d.columns else None,
                                   size="Eficiencia_Planta_Pct" if "Eficiencia_Planta_Pct" in d.columns else None,
                                   hover_data=d.columns.tolist()[:6],
                                   trendline="ols", trendline_scope="overall")
            except Exception:
                fig_sc = px.scatter(d, x="Capacidad_Instalada_MW", y="Generacion_Diaria_MWh",
                                   color="Tecnologia" if "Tecnologia" in d.columns else None,
                                   size="Eficiencia_Planta_Pct" if "Eficiencia_Planta_Pct" in d.columns else None,
                                   hover_data=d.columns.tolist()[:6])
            fig_sc.update_layout(margin=dict(t=30, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02))
            fig_sc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,249,0.8)")
            st.plotly_chart(fig_sc, use_container_width=True)

        # 4) Barras horizontales: capacidad por operador (con anotaciones)
        if "Operador" in d.columns and "Capacidad_Instalada_MW" in d.columns:
            st.markdown("**Capacidad instalada (MW) por operador**")
            cap_op = d.groupby("Operador")["Capacidad_Instalada_MW"].sum().sort_values(ascending=True)
            fig_bar = go.Figure(go.Bar(x=cap_op.values, y=cap_op.index, orientation="h",
                                       marker_color=px.colors.sequential.Teal_r[:len(cap_op)]))
            fig_bar.update_layout(xaxis_title="Capacidad total (MW)", yaxis_title="Operador",
                                  margin=dict(t=20, b=20), showlegend=False)
            fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,249,0.8)")
            st.plotly_chart(fig_bar, use_container_width=True)

        # 5) Línea temporal: proyectos por año de entrada
        if "Fecha_Entrada_Operacion" in d.columns:
            st.markdown("**Proyectos por año de entrada en operación**")
            d_ano = d.copy()
            d_ano["Año"] = pd.to_datetime(d_ano["Fecha_Entrada_Operacion"], errors="coerce").dt.year
            count_ano = d_ano.dropna(subset=["Año"]).groupby("Año").size().reset_index(name="Proyectos")
            fig_ano = px.line(count_ano, x="Año", y="Proyectos", markers=True,
                              title="Evolución de proyectos por año")
            fig_ano.update_traces(line=dict(width=3), marker=dict(size=10))
            fig_ano.update_layout(margin=dict(t=40, b=20), font=dict(size=12))
            fig_ano.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,249,0.8)")
            st.plotly_chart(fig_ano, use_container_width=True)

    # Gráficos genéricos (siempre o si no es contexto energía)
    st.markdown("---")
    st.markdown("**Vistas genéricas del dataset**")

    g1, g2 = st.columns(2)

    with g1:
        if numeric_cols:
            st.markdown("**Distribución** (histograma)")
            col_hist = st.selectbox("Columna numérica", numeric_cols, key="hist_col")
            fig_hist = px.histogram(d, x=col_hist, nbins=min(50, max(10, d[col_hist].nunique())),
                                    title=f"Distribución de {col_hist}")
            fig_hist.update_layout(margin=dict(t=40, b=20))
            fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,249,0.8)")
            st.plotly_chart(fig_hist, use_container_width=True)

        if cat_cols and not energy_mode:
            st.markdown("**Conteo por categoría**")
            col_cat = st.selectbox("Columna categórica", cat_cols, key="bar_cat")
            cnt = d[col_cat].value_counts().reset_index()
            cnt.columns = [col_cat, "Cantidad"]
            fig_bar = px.bar(cnt, x=col_cat, y="Cantidad", color="Cantidad", color_continuous_scale="Teal")
            fig_bar.update_layout(xaxis_tickangle=-45, margin=dict(t=20, b=80))
            st.plotly_chart(fig_bar, use_container_width=True)

    with g2:
        if len(numeric_cols) >= 2:
            st.markdown("**Correlación entre variables numéricas**")
            num_sub = numeric_cols[:min(8, len(numeric_cols))]
            corr = d[num_sub].corr()
            fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                                 zmin=-1, zmax=1, title="Matriz de correlación")
            fig_corr.update_layout(margin=dict(t=40, b=20))
            st.plotly_chart(fig_corr, use_container_width=True)

        if date_cols and d[date_cols[0]].notna().sum() > 0:
            st.markdown("**Serie temporal** (conteo por fecha)")
            col_date = date_cols[0]
            d_temp = d.copy()
            d_temp["_fecha"] = pd.to_datetime(d_temp[col_date], errors="coerce")
            d_temp["_periodo"] = d_temp["_fecha"].dt.to_period("M").astype(str)
            count_temp = d_temp.dropna(subset=["_fecha"]).groupby("_periodo").size().reset_index(name="Cantidad")
            if len(count_temp) > 0:
                fig_temp = px.line(count_temp, x="_periodo", y="Cantidad", markers=True, title=f"Conteo por mes ({col_date})")
                fig_temp.update_layout(xaxis_tickangle=-45, margin=dict(t=40, b=80))
                st.plotly_chart(fig_temp, use_container_width=True)