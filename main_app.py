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

# --- Estilos visuales: contraste alto (fondos claros, letras oscuras) ---
st.markdown("""
<style>
    /* Fondo general de la app */
    .stApp, [data-testid="stAppViewContainer"] { background-color: #f5f7f6 !important; }
    .main .block-container { background-color: #ffffff !important; padding: 1.5rem; border-radius: 8px; }
    /* Texto principal: siempre oscuro sobre claro */
    .stMarkdown, .stMarkdown p, [data-testid="stMarkdownContainer"], p { color: #111827 !important; }
    .stCaption, [data-testid="stCaption"] { color: #374151 !important; }
    label, [data-testid="stWidgetLabel"], .stWidget label { color: #1f2937 !important; font-weight: 500 !important; }
    .stAlert, [data-testid="stAlert"] { color: #1f2937 !important; background-color: #f9fafb !important; }
    /* Inputs y selectores: fondo blanco, texto oscuro */
    input, textarea, [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
        background-color: #ffffff !important; color: #111827 !important; border: 1px solid #d1d5db !important;
    }
    [data-baseweb="select"] { background-color: #ffffff !important; color: #111827 !important; }
    /* Expander: fondo claro, texto oscuro */
    [data-testid="stExpander"] { background-color: #ffffff !important; border: 1px solid #e5e7eb !important; }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] .stMarkdown { color: #1f2937 !important; }
    /* DataFrames y tablas */
    [data-testid="stDataFrame"], .stDataFrame { background-color: #ffffff !important; }
    [data-testid="stDataFrame"] * { color: #111827 !important; }
    /* Caja de bienvenida: texto claro sobre fondo oscuro */
    .welcome-box {
        background: linear-gradient(135deg, #1a5f4a 0%, #2d8f6f 50%, #1a5f4a 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .welcome-box h1 { color: #ffffff !important; font-size: 2rem; margin-bottom: 0.5rem; font-weight: 700; }
    .welcome-box p { color: rgba(255,255,255,0.98) !important; font-size: 1.05rem; margin: 0; }
    .upload-zone { background: #e8f0ee !important; border: 2px dashed #1a5f4a; border-radius: 12px; padding: 2rem; }
    /* KPIs */
    div[data-testid="stMetric"] {
        background: #ffffff !important;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #d1d5db;
    }
    div[data-testid="stMetric"] label { color: #1a3d32 !important; font-weight: 600 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #111827 !important; }
    h1, h2, h3 { color: #1a3d32 !important; font-weight: 600 !important; }
    /* Sidebar: fondo blanco, texto oscuro */
    [data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e5e7eb; }
    [data-testid="stSidebar"] * { color: #1f2937 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: #1a3d32 !important; font-weight: 700 !important; }
    /* Botones */
    .stButton button { background-color: #1a5f4a !important; color: #ffffff !important; border: none !important; }
    .stButton button:hover { background-color: #2d8f6f !important; color: #ffffff !important; }
    .stDownloadButton button { color: #1f2937 !important; }
    /* Asistente: caja de respuesta */
    .asistente-respuesta { background: #ffffff; color: #111827; padding: 1rem; border-radius: 8px; border: 1px solid #e5e7eb; margin-top: 0.5rem; }
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


def build_all_charts(d, energy_mode, numeric_cols, cat_cols, date_cols):
    """Construye todas las gráficas disponibles según los datos. Devuelve dict { nombre: fig }. """
    charts = {}
    layout_common = dict(paper_bgcolor="rgba(255,255,255,1)", plot_bgcolor="rgba(248,250,249,0.9)", font=dict(color="#1a1a1a"))

    if energy_mode:
        if "Tecnologia" in d.columns and "Operador" in d.columns:
            agg = d.groupby(["Tecnologia", "Operador"]).size().reset_index(name="Proyectos")
            fig = px.sunburst(agg, path=["Tecnologia", "Operador"], values="Proyectos",
                              color="Proyectos", color_continuous_scale="Teal")
            fig.update_layout(**layout_common, margin=dict(t=40, b=20))
            charts["Sunburst: Tecnología y operador"] = fig

        if "Eficiencia_Planta_Pct" in d.columns and "Tecnologia" in d.columns:
            fig = px.box(d, x="Tecnologia", y="Eficiencia_Planta_Pct", color="Tecnologia", points="outliers")
            fig.update_layout(showlegend=False, xaxis_tickangle=-45, margin=dict(t=40, b=80), **layout_common)
            charts["Box: Eficiencia por tecnología"] = fig

        if "Capacidad_Instalada_MW" in d.columns and "Generacion_Diaria_MWh" in d.columns:
            try:
                fig = px.scatter(d, x="Capacidad_Instalada_MW", y="Generacion_Diaria_MWh",
                                 color="Tecnologia" if "Tecnologia" in d.columns else None,
                                 size="Eficiencia_Planta_Pct" if "Eficiencia_Planta_Pct" in d.columns else None,
                                 trendline="ols", trendline_scope="overall")
            except Exception:
                fig = px.scatter(d, x="Capacidad_Instalada_MW", y="Generacion_Diaria_MWh",
                                 color="Tecnologia" if "Tecnologia" in d.columns else None,
                                 size="Eficiencia_Planta_Pct" if "Eficiencia_Planta_Pct" in d.columns else None)
            fig.update_layout(margin=dict(t=30, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02), **layout_common)
            charts["Scatter: Capacidad vs Generación"] = fig

        if "Operador" in d.columns and "Capacidad_Instalada_MW" in d.columns:
            cap_op = d.groupby("Operador")["Capacidad_Instalada_MW"].sum().sort_values(ascending=True)
            fig = go.Figure(go.Bar(x=cap_op.values, y=cap_op.index, orientation="h",
                                   marker_color=px.colors.sequential.Teal_r[:len(cap_op)]))
            fig.update_layout(xaxis_title="Capacidad total (MW)", yaxis_title="Operador", margin=dict(t=20, b=20), showlegend=False, **layout_common)
            charts["Barras: Capacidad por operador"] = fig

        if "Fecha_Entrada_Operacion" in d.columns:
            d_ano = d.copy()
            d_ano["Año"] = pd.to_datetime(d_ano["Fecha_Entrada_Operacion"], errors="coerce").dt.year
            count_ano = d_ano.dropna(subset=["Año"]).groupby("Año").size().reset_index(name="Proyectos")
            fig = px.line(count_ano, x="Año", y="Proyectos", markers=True)
            fig.update_traces(line=dict(width=3), marker=dict(size=10))
            fig.update_layout(margin=dict(t=40, b=20), **layout_common)
            charts["Línea: Proyectos por año"] = fig

    if numeric_cols:
        col_hist = numeric_cols[0]
        fig = px.histogram(d, x=col_hist, nbins=min(50, max(10, d[col_hist].nunique())))
        fig.update_layout(margin=dict(t=40, b=20), **layout_common)
        charts["Histograma: " + col_hist] = fig

    if len(numeric_cols) >= 2:
        num_sub = numeric_cols[:min(8, len(numeric_cols))]
        corr = d[num_sub].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig.update_layout(margin=dict(t=40, b=20), **layout_common)
        charts["Matriz de correlación"] = fig

    if cat_cols and not energy_mode:
        col_cat = cat_cols[0]
        cnt = d[col_cat].value_counts().reset_index()
        cnt.columns = [col_cat, "Cantidad"]
        fig = px.bar(cnt, x=col_cat, y="Cantidad", color="Cantidad", color_continuous_scale="Teal")
        fig.update_layout(xaxis_tickangle=-45, margin=dict(t=20, b=80), **layout_common)
        charts["Barras: " + col_cat] = fig

    if date_cols and d[date_cols[0]].notna().sum() > 0:
        col_date = date_cols[0]
        d_temp = d.copy()
        d_temp["_fecha"] = pd.to_datetime(d_temp[col_date], errors="coerce")
        d_temp["_periodo"] = d_temp["_fecha"].dt.to_period("M").astype(str)
        count_temp = d_temp.dropna(subset=["_fecha"]).groupby("_periodo").size().reset_index(name="Cantidad")
        if len(count_temp) > 0:
            fig = px.line(count_temp, x="_periodo", y="Cantidad", markers=True)
            fig.update_layout(xaxis_tickangle=-45, margin=dict(t=40, b=80), **layout_common)
            charts["Serie temporal: " + col_date] = fig

    return charts


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
tab_cuant, tab_cual, tab_graf, tab_asist = st.tabs(["📊 Cuantitativo", "📋 Cualitativo", "📈 Gráfico", "🤖 Asistente de análisis"])

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

# ---------- BLOQUE GRÁFICO (selector, descarga e informe) ----------
with tab_graf:
    st.subheader("Visualizaciones")
    d = st.session_state.edited_df

    # Construir todas las gráficas disponibles
    all_charts = build_all_charts(d, energy_mode, numeric_cols, cat_cols, date_cols)
    chart_names = list(all_charts.keys())

    if not chart_names:
        st.info("No hay gráficas disponibles con los datos actuales (necesitas columnas numéricas o categóricas).")
    else:
        # Selector: qué gráfica mostrar
        opciones = ["Todas"] + chart_names
        elegida = st.selectbox(
            "Elegir gráfica a mostrar",
            opciones,
            key="sel_grafica",
            help="Selecciona una gráfica concreta o «Todas» para ver todas.",
        )

        # Mostrar gráfica(s)
        if elegida == "Todas":
            for idx, (nombre, fig) in enumerate(all_charts.items()):
                st.markdown(f"**{nombre}**")
                st.plotly_chart(fig, use_container_width=True)
                html_fig = fig.to_html(full_html=False, include_plotlyjs="cdn")
                st.download_button(
                    f"Descargar: {nombre} (HTML)",
                    data=html_fig.encode("utf-8"),
                    file_name=f"grafica_{nombre.replace(' ', '_').replace(':', '')[:30]}.html",
                    mime="text/html",
                    key=f"dl_graf_{idx}",
                )
                st.markdown("---")
        else:
            fig = all_charts[elegida]
            st.plotly_chart(fig, use_container_width=True)
            html_fig = fig.to_html(full_html=False, include_plotlyjs="cdn")
            st.download_button(
                "Descargar esta gráfica (HTML)",
                data=html_fig.encode("utf-8"),
                file_name=f"grafica_{elegida.replace(' ', '_').replace(':', '')[:40]}.html",
                mime="text/html",
                key="dl_grafica_unica",
            )

        # ---------- Generar informe ----------
        st.markdown("---")
        st.markdown("#### Generar informe")
        st.caption("Genera un informe HTML con resumen de datos y todas las gráficas para descargar.")

        from datetime import datetime
        if st.button("Generar informe (HTML)", key="btn_informe"):
            report_title = "Informe de análisis de datos"
            fecha = datetime.now().strftime("%Y-%m-%d %H:%M")
            n_filas = len(d)
            n_cols = len(d.columns)
            num_list = ", ".join(numeric_cols[:10]) if numeric_cols else "Ninguna"
            cat_list = ", ".join(cat_cols[:10]) if cat_cols else "Ninguna"
            html_charts = ""
            for nombre, fig in all_charts.items():
                html_charts += f"<h3>{nombre}</h3>"
                html_charts += fig.to_html(full_html=False, include_plotlyjs="cdn")
                html_charts += "<hr/>"
            report_html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_title}</title>
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 900px; color: #1a1a1a; line-height: 1.5; }}
        h1 {{ color: #1a5f4a; border-bottom: 2px solid #2d8f6f; padding-bottom: 0.5rem; }}
        h2, h3 {{ color: #1a3d32; }}
        .meta {{ color: #6b7280; margin-bottom: 1.5rem; }}
        .resumen {{ background: #f0f4f3; padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
        hr {{ margin: 2rem 0; border: none; border-top: 1px solid #e5e7eb; }}
    </style>
</head>
<body>
    <h1>{report_title}</h1>
    <p class="meta">Generado el {fecha}</p>
    <div class="resumen">
        <h2>Resumen del dataset</h2>
        <ul>
            <li><strong>Filas:</strong> {n_filas:,}</li>
            <li><strong>Columnas:</strong> {n_cols}</li>
            <li><strong>Columnas numéricas:</strong> {num_list}</li>
            <li><strong>Columnas categóricas:</strong> {cat_list}</li>
        </ul>
    </div>
    <h2>Gráficas</h2>
    {html_charts}
</body>
</html>
"""
            st.session_state["report_html"] = report_html
            st.session_state["report_filename"] = f"informe_datos_{datetime.now().strftime('%Y%m%d_%H%M')}.html"

        if st.session_state.get("report_html"):
            st.download_button(
                "Descargar informe (HTML)",
                data=st.session_state["report_html"].encode("utf-8"),
                file_name=st.session_state.get("report_filename", "informe_datos.html"),
                mime="text/html",
                key="dl_informe",
            )

# ---------- ASISTENTE DE ANÁLISIS (Groq - Llama 3.3 70B Versatile) ----------
with tab_asist:
    st.subheader("Asistente de análisis con IA")
    st.markdown("Introduce tu **API Key de Groq** y haz preguntas sobre los datos. El asistente usa el modelo **Llama 3.3 70B Versatile** y tiene contexto del dataset actual (columnas, tipos y resumen).")

    api_key = st.text_input(
        "API Key de Groq",
        type="password",
        placeholder="gsk_...",
        help="Obtén tu clave en https://console.groq.com. No se guarda; solo se usa en esta sesión.",
        key="groq_api_key",
    )

    def get_data_context(d):
        """Resumen del dataframe para el contexto del asistente."""
        lines = [
            f"Dataset: {len(d)} filas, {len(d.columns)} columnas.",
            "Columnas y tipos: " + ", ".join([f"{c} ({d[c].dtype})" for c in d.columns[:30]]),
        ]
        if len(d.columns) > 30:
            lines.append("... (más columnas)")
        lines.append("\nPrimeras filas (muestra):")
        lines.append(d.head(10).to_string())
        if d.select_dtypes(include=["number"]).columns.any():
            lines.append("\nEstadísticas descriptivas (numéricas):")
            lines.append(d.select_dtypes(include=["number"]).describe().round(2).to_string())
        return "\n".join(lines)

    pregunta = st.text_area(
        "Pregunta o solicitud de análisis",
        placeholder="Ej: ¿Qué columnas son numéricas? Resumen de la variable X. ¿Hay valores atípicos?",
        height=100,
        key="asistente_pregunta",
    )

    if st.button("Enviar al asistente", key="btn_asistente"):
        if not api_key or not api_key.strip():
            st.error("Introduce tu API Key de Groq para continuar.")
        elif not pregunta or not pregunta.strip():
            st.warning("Escribe una pregunta o solicitud de análisis.")
        else:
            try:
                from groq import Groq
                client = Groq(api_key=api_key.strip())
                context = get_data_context(d)
                system_msg = (
                    "Eres un asistente de análisis de datos. El usuario te proporciona un resumen de su dataset "
                    "y te hace preguntas. Responde en español, de forma clara y concisa. Si pides más detalle, "
                    "sugiere qué podría hacer el usuario en el dashboard."
                )
                user_msg = (
                    "Contexto del dataset actual (datos con los que está trabajando el usuario):\n\n"
                    f"{context}\n\n"
                    "---\nPregunta del usuario:\n"
                    f"{pregunta.strip()}"
                )
                with st.spinner("Analizando con Llama 3.3 70B Versatile..."):
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        max_tokens=1024,
                        temperature=0.3,
                    )
                respuesta = response.choices[0].message.content
                st.session_state["asistente_ultima_respuesta"] = respuesta

            except Exception as e:
                st.error(f"Error al llamar a la API de Groq: {e}")
                if "api_key" in str(e).lower() or "401" in str(e):
                    st.caption("Comprueba que la API Key sea correcta y esté activa en https://console.groq.com")

    if st.session_state.get("asistente_ultima_respuesta"):
        st.markdown("**Respuesta del asistente**")
        st.markdown(st.session_state["asistente_ultima_respuesta"])