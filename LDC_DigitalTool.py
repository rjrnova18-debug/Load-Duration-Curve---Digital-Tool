import streamlit as st
import pandas as pd
import numpy as np
import re # Para validación de expresiones regulares (Hora)
import plotly.express as px
from color_accessibility import setColorBlindMode, setColorMap
import io
from datetime import datetime # Importar datetime para validación lógica temporal
from fpdf import FPDF
from PDF_download import create_pdf_report

# --- CONFIGURACIÓN DE PÁGINA Y ESTADO ---
st.set_page_config(layout="wide", page_title="Cuadro de Carga LDC", page_icon="⚡️")

# Inicializar estado de sesión para el DataFrame y la validación
if 'df_valid' not in st.session_state:
    st.session_state.df_valid = None
if 'is_validated' not in st.session_state:
    st.session_state.is_validated = False

# --- ESPECIFICACIONES DE VALIDACIÓN ---
COLUMNS_SPECS = {
    'Año': {'dtype': np.int64, 'min': 2000, 'max': 2035},
    'Mes': {'dtype': np.int64, 'min': 1, 'max': 12},
    'Dia': {'dtype': np.int64, 'min': 1, 'max': 31},
    'Hora': {'dtype': str, 'format': r'^\d{2}:\d{2}$'},
    'Potencia (kW)': {'dtype': np.float64, 'min': 0.0, 'max': 1000.0}
}

# --- FUNCIONES DE PROCESAMIENTO ---
## ---------------------------------------------------------------------------------------------------
# Definición para generar Reporte PDF <-- INICIO DEL NUEVO BLOQUE
VERSION = "1.0.0" # Define aquí la versión de tu sistema
# ---------------------------------------------------------------------------------------------------

def process_data(df, hora_inicio_dia, hora_fin_dia):
    """
    Realiza los cálculos de energía, potencia media y segmentación.
    Asume que la Potencia (kW) es la potencia promedio durante el intervalo de 1 hora.
    """
    df_proc = df.copy()

    # 1. Crear columna de fecha/hora completa
    # Se usa el primer día del mes si 'Dia' o 'Mes' no son numéricos
    try:

        # Asegurar tipos correctos antes de construir la fecha
        df_proc['Año'] = df_proc['Año'].astype(float).astype(int)
        df_proc['Mes'] = df_proc['Mes'].astype(float).astype(int)
        df_proc['Dia'] = df_proc['Dia'].astype(float).astype(int)

        # Crear columna de FechaHora
        df_proc['FechaHora'] = pd.to_datetime(
            df_proc[['Año', 'Mes', 'Dia']].astype(str).agg('-'.join, axis=1) + ' ' + df_proc['Hora'].astype(str)
        )
        df_proc = df_proc.set_index('FechaHora').sort_index()
        df_proc['Hora_Solo'] = df_proc.index.hour
        df_proc['Dia_Semana'] = df_proc.index.day_name()

    except Exception as e:
        st.error(f"Error al crear la columna de Fecha/Hora: {e}. Asegúrate de que Año, Mes, Día y Hora son válidos.")
        return None

    # 2. Cálculo de Energía y Potencia Media
    # Asumimos intervalos de 1 hora. Energía (kWh) = Potencia (kW) * 1h
    df_proc['Energia (kWh)'] = df_proc['Potencia (kW)'].fillna(0) # Se usa 0 para faltantes ya validados

    # 3. Segmentación Diurna/Nocturna
    # Convertir a minutos para comparación (más robusto que solo la hora)
    inicio_dia_min = hora_inicio_dia * 60
    fin_dia_min = hora_fin_dia * 60
    
    # Crear una columna de minutos del día
    df_proc['Minuto_Dia'] = df_proc.index.hour * 60 + df_proc.index.minute

    # Lógica de segmentación
    if inicio_dia_min < fin_dia_min:
        # Caso simple: 06:00 a 18:00 (Diurno)
        df_proc['Segmento'] = np.where(
            (df_proc['Minuto_Dia'] >= inicio_dia_min) & (df_proc['Minuto_Dia'] < fin_dia_min), 
            'Diurno', 
            'Nocturno'
        )
    else:
        # Caso cruce de medianoche: 18:00 a 06:00 (Nocturno)
        df_proc['Segmento'] = np.where(
            (df_proc['Minuto_Dia'] >= inicio_dia_min) | (df_proc['Minuto_Dia'] < fin_dia_min), 
            'Diurno', # Diurno es el segmento definido (ej. 18:00-06:00, el restante es Nocturno)
            'Nocturno' # Se invierte la lógica si es cruce de medianoche
        )

    return df_proc
#---------------------------------------------------------------------------------------------------
#Definiciones para guardar como csv o excel
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

@st.cache_data
def convert_df_to_excel(df):
    # Generación robusta de Excel usando io.BytesIO
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Datos Editados')
    processed_data = output.getvalue()
    return processed_data
#---------------------------------------------------------------------------------------------------
def validate_data(df):
    """Realiza la validación de formato, rango, duplicados y coherencia temporal."""
    st.subheader("📊 Resultados de la Validación")
    validation_passed = True
    df_validated = df.copy()

    # 1. Validación de Columnas Faltantes
    expected_cols = list(COLUMNS_SPECS.keys())
    missing_cols = [col for col in expected_cols if col not in df_validated.columns]
    
    if missing_cols:
        st.error(f"❌ Error Crítico: Faltan las siguientes columnas esenciales: {', '.join(missing_cols)}. No se puede continuar.")
        return validation_passed, None

    # Inicializar contadores de errores
    format_errors_count = 0
    range_errors_count = 0
    temporal_errors_count = 0
    
    # 2. Validación de Formato y Rango
    for col, spec in COLUMNS_SPECS.items():
        if spec['dtype'] in [np.int64, np.float64]:
            original_type_errors = df_validated[col].apply(lambda x: pd.to_numeric(x, errors='coerce')).isnull() & df_validated[col].notnull()
            
            if original_type_errors.any():
                st.error(f"❌ Error de Formato en '{col}': {original_type_errors.sum()} valores no son numéricos.")
                validation_passed = False
                format_errors_count += original_type_errors.sum()
                
                # Reporte detallado de errores de formato (no numérico)
                for idx in df_validated[original_type_errors].index:
                    # Usamos idx + 2 (0-base + header + 1-base)
                    row_data = df_validated.loc[idx]
                    error_message = f"❌ Error de Formato en **'{col}'** en la fila **{idx + 2}**: "
                    # Formatear la fila completa para el mensaje
                    details = " | ".join([f"{c}: **{row_data[c]}**" for c in expected_cols])
                    st.markdown(error_message + details)


            df_validated[col] = pd.to_numeric(df_validated[col], errors='coerce')
            
            if 'min' in spec and 'max' in spec:
                range_errors_mask = (df_validated[col] < spec['min']) | (df_validated[col] > spec['max'])
                range_errors = df_validated[range_errors_mask]
                range_errors = range_errors[df_validated[col].notnull()]
                
                if not range_errors.empty:
                    st.error(f"❌ Error de Rango en '{col}': {len(range_errors)} valores fuera del rango ({spec['min']} - {spec['max']}).")
                    range_errors_count += len(range_errors)
                    validation_passed = False
                    
                    # Reporte detallado de errores de rango
                    for idx in range_errors.index:
                        row_data = df_validated.loc[idx]
                        error_message = f"❌ Error de Rango en **'{col}'** en la fila **{idx + 2}** (Valor: **{row_data[col]:.2f}**): "
                        details = " | ".join([f"{c}: **{row_data[c]}**" for c in expected_cols])
                        st.markdown(error_message + details)

        
        elif col == 'Hora' and spec['dtype'] == str and 'format' in spec:
            format_regex = spec['format']
            
            mask_errors = (df_validated[col].astype(str).str.match(format_regex) == False)
            mask_errors = mask_errors & df_validated[col].notnull()
            
            if mask_errors.any():
                error_count = mask_errors.sum()
                #st.error(f"❌ Error de Formato en '{col}': {error_count} valores no cumplen con el formato HH:MM.")
                # format_errors_count += error_count
                # validation_passed = False

                # Reporte detallado de errores de formato (regex)
                for idx in df_validated[mask_errors].index:
                    row_data = df_validated.loc[idx]
                    # Aquí solo reportamos si NO cumple el formato 'HH:MM'.
                    if not re.match(format_regex, str(row_data[col])):
                        st.error(f"❌ Error de Formato en **'{col}'** en la fila **{idx + 2}**: El valor '**{row_data[col]}**' no cumple con el patrón HH:MM.")
                        format_errors_count += 1
                        validation_passed = False
                        details = " | ".join([f"{c}: **{row_data[c]}**" for c in expected_cols])
                        st.markdown(f"**Datos de la fila:** {details}")


    # 3. Validación de Coherencia Temporal (Límites Lógicos)
    # --------------------------------------------------------
    
    temporal_errors_mask = pd.Series(False, index=df_validated.index)

    # 3.a. Validación de Hora Lógica (00:00 a 23:59)
    # Esto captura horas con formato correcto (ej. "25:00") pero ilógicas
    for idx in df_validated.index:
        row = df_validated.loc[idx]
        if pd.notna(row['Hora']):
            hora_str = str(row['Hora']).strip()
            # Ya se validó el formato 'HH:MM' en la sección 2. Aquí validamos el contenido lógico.
            if re.match(r'^\d{2}:\d{2}$', hora_str):
                try:
                    H = int(hora_str[:2])
                    M = int(hora_str[3:])
                    
                    if not (0 <= H <= 23 and 0 <= M <= 59):
                        # Reporte de error temporal (Hora)
                        st.error(f"❌ Error temporal en **'Hora'** en la fila **{idx + 2}**: El valor **{hora_str}** está fuera del rango lógico 00:00-23:59.")
                        temporal_errors_count += 1
                        validation_passed = False
                        
                        details = " | ".join([f"{c}: **{row[c]}**" for c in expected_cols])
                        st.markdown(f"**Datos de la fila:** {details}")
                        temporal_errors_mask.loc[idx] = True

                except ValueError:
                    # Si falla la conversión a int, ya debería haber sido detectado como error de formato
                    # en la sección 2, pero lo marcamos por si acaso.
                    pass 

    # 3.b. Validación de Fecha Lógica (31/02, 13/2024, etc.)
    # Solo intentamos convertir si Año, Mes y Día son numéricos (no NaN ni formato_error)
    valid_date_cols = df_validated[['Año', 'Mes', 'Dia']].notnull().all(axis=1) & ~temporal_errors_mask
    
    for idx in df_validated[valid_date_cols].index:
        row = df_validated.loc[idx]
        try:
            # Intentar crear un objeto datetime con la hora 00:00 (la hora no importa, solo la fecha)
            # Usamos pd.to_datetime con errores='raise'
            datetime(year=int(row['Año']), month=int(row['Mes']), day=int(row['Dia']))

        except Exception as e:
            # Captura errores de fecha imposible (ej. 2024-02-31, 2024-13-01)
            st.error(f"❌ Error temporal en **Fecha** en la fila **{idx + 2}**: La combinación es lógicamente imposible (e.g., día 31 en mes corto, mes > 12).")
            temporal_errors_count += 1
            validation_passed = False
            
            details = " | ".join([f"{c}: **{row[c]}**" for c in expected_cols])
            st.markdown(f"**Datos de la fila:** {details}")
            temporal_errors_mask.loc[idx] = True


    if temporal_errors_count == 0:
        st.success("✅ No se encontraron errores de límites lógicos temporales.")
    else:
        st.warning(f"⚠️ Se encontraron {temporal_errors_count} errores de coherencia temporal.")
        validation_passed = False # Asegurar que la validación falla si hay errores temporales.


    # 4. Validación de Datos Faltantes (NaN)
    # Importante: Esto debe ir DESPUÉS de pd.to_numeric, para contar los valores que eran strings no numéricos como NaN
    missing_data = df_validated.isnull().sum()
    missing_report = missing_data[missing_data > 0]
    
    if not missing_report.empty:
        st.warning("⚠️ Advertencia de Datos Faltantes (NaN):")
        st.dataframe(missing_report.rename("Cantidad de Faltantes"))
        validation_passed = False
    else:
        st.success("✅ No se encontraron datos faltantes (NaN).")

    # 5. Validación de Duplicados
    id_cols = ['Año', 'Mes', 'Dia', 'Hora']
    duplicate_mask = df_validated.duplicated(subset=id_cols, keep=False)
    duplicate_rows = df_validated[duplicate_mask]
    
    if not duplicate_rows.empty:
        st.error(f"❌ Error de Duplicados: Se encontraron {len(duplicate_rows)} filas duplicadas basadas en fecha/hora.")
        st.dataframe(duplicate_rows.sort_values(by=id_cols).head())
        validation_passed = False
        
        # Reporte detallado de errores de duplicados (solo primeras 5 filas para evitar spam)
        st.markdown("⚠️ **Filas con Error de Duplicado (primeras 5)**:")
        for idx in duplicate_rows.sort_values(by=id_cols).head().index:
            row_data = df_validated.loc[idx]
            error_message = f"❌ Error de Duplicado en la fila **{idx + 2}**: "
            details = " | ".join([f"{c}: **{row_data[c]}**" for c in expected_cols])
            st.markdown(error_message + details)


    else:
        st.success("✅ No se encontraron registros duplicados de fecha/hora.")

    # 6. Resumen Final
    st.markdown("---")
    if validation_passed:
        st.balloons()
        st.success("🎉 ¡Validación Terminada! El conjunto de datos es apto para el procesamiento.")
    else:
        st.info(f"⚠️ La validación ha terminado con errores/advertencias. Por favor, corrígelos.")
        
    return validation_passed, df_validated

# --- PESTAÑAS PRINCIPALES ---
tab1, tab2, tab3 = st.tabs(["📂 Carga y Validación de Datos", "⚙️ Procesamiento y Análisis", "Descargar Reporte"])

with tab1:
    st.title("📊 Cuadro de Carga (LDC)")
    st.subheader("Carga y Edición de datos")
    st.info("Sube tu archivo de consumo eléctrico o ingresa datos manualmente para comenzar.")

    df = None # Inicializa df

    # --- OPCIÓN 1: Carga masiva (CSV o Excel) ---
    uploaded_file = st.file_uploader("Cargar archivo CSV o Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_up = pd.read_csv(uploaded_file)
            else:
                df_up = pd.read_excel(uploaded_file)

            st.success("✅ Archivo cargado correctamente.")

            st.subheader("📋 Vista completa de los datos cargados")
            st.info("✏️ Edita tus datos si es necesario")
            
            df = st.data_editor(df_up, num_rows="dynamic", use_container_width=True, height=325, key="editor_uploaded")
            
            st.write("**Filas:**", df.shape[0], "| **Columnas:**", df.shape[1])

            # --- BOTONES DE DESCARGA ---
            col_down_csv, col_down_xlsx = st.columns(2)

            with col_down_csv:
                csv = convert_df_to_csv(df)
                st.download_button(
                    label="⬇️ Descargar como CSV",
                    data=csv,
                    file_name='datos_formato.csv',
                    mime='text/csv',
                    use_container_width=True
                )
        
            with col_down_xlsx:
                st.download_button(
                    label="⬇️ Descargar como Excel (xlsx)",
                    data=convert_df_to_excel(df),
                    file_name='datos_editados.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    use_container_width=True
                )

            if st.button("💾 Guardar y Validar Cambios"):
                is_valid, df_processed = validate_data(df)
                st.session_state.is_validated = is_valid
                st.session_state.df_valid = df_processed
                if is_valid:
                    st.success("¡Datos guardados y listos para el Análisis!")

        except Exception as e:
            st.error(f"Ocurrió un error al leer el archivo. Error: {e}")

    # --- OPCIÓN 2: Carga manual (tabla editable) ---
    else:
        st.info("Puedes cargar un archivo o ingresar datos manualmente a continuación.")

        st.subheader("✍️ Ingreso manual de datos")

        ejemplo = pd.DataFrame({
            "Año": [2024, 2024, 2024],
            "Mes": ["01", "01", "01"],
            "Dia": ["01", "01", "01"],
            "Hora": ["00:00", "01:00", "02:00"],
            "Potencia (kW)": [55.2, 53.3, 52.4]
        })

        filas_visibles = 5
        altura_editor = filas_visibles * 35 + 40

        df = st.data_editor(
            ejemplo,
            num_rows="dynamic",
            use_container_width=True,
            height=altura_editor,
            key="data_editor_manual"
        )

        # --- BOTONES DE DESCARGA ---
        st.markdown("---")
        col_down_csv, col_down_xlsx = st.columns(2)

        with col_down_csv:
            st.download_button(
                label="⬇️ Descargar como CSV",
                data=convert_df_to_csv(df),
                file_name='datos_manuales.csv',
                mime='text/csv',
                use_container_width=True
            )
        
        with col_down_xlsx:
            st.download_button(
                label="⬇️ Descargar como Excel (xlsx)",
                data=convert_df_to_excel(df),
                file_name='datos_manuales.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                use_container_width=True
            )
        
        if st.button("✨ Validar Datos Manuales"):
            is_valid, df_processed = validate_data(df)
            st.session_state.is_validated = is_valid
            st.session_state.df_valid = df_processed
            if is_valid:
                st.success("¡Datos guardados y listos para el Análisis!")

# --- PESTAÑA DE PROCESAMIENTO Y ANÁLISIS ---
with tab2:
    st.title("⚙️ Análisis de Consumo y Carga")
    
    if st.session_state.is_validated and st.session_state.df_valid is not None:
        df_valid = st.session_state.df_valid
        
        st.success(f"Datos válidos cargados. Filas: {df_valid.shape[0]}.")
        st.markdown("---")

        # --- SECCIÓN DE CONFIGURACIÓN DE SEGMENTACIÓN ---
        st.subheader("🛠️ Configuración de Segmentación Diurna/Nocturna")
        col_start, col_end = st.columns(2)
        
        with col_start:
            # Slider para la hora de inicio (0 a 23)
            hora_inicio_dia = st.slider(
                "Hora de Inicio del Período Diurno (HH:00)", 
                min_value=0, 
                max_value=23, 
                value=6, 
                step=1,
                key="slider_start"
            )
        with col_end:
            # Slider para la hora de fin (0 a 23)
            hora_fin_dia = st.slider(
                "Hora de Fin del Período Diurno (HH:00)", 
                min_value=0, 
                max_value=23, 
                value=18, 
                step=1,
                key="slider_end"
            )

        st.info(f"☀️ El período **Diurno** se define entre las **{hora_inicio_dia:02d}:00** y las **{hora_fin_dia:02d}:00**.")

        # Cálculo del período nocturno complementario
        if hora_inicio_dia < hora_fin_dia:
            # Caso normal: el día no cruza medianoche
            st.info(f"🌙 El período **Nocturno** se define entre las **{hora_fin_dia:02d}:00** y las **{hora_inicio_dia:02d}:00** del día siguiente.")
        else:
            # Caso en que el día cruza medianoche
            st.info(f"🌙 El período **Nocturno** se define entre las **{hora_fin_dia:02d}:00** y las **{hora_inicio_dia:02d}:00** del mismo día.")

        st.markdown("---")
        
        # --- PROCESAMIENTO ---
        df_processed = process_data(df_valid, hora_inicio_dia, hora_fin_dia)

        if df_processed is not None:

            # --- AJUSTES ESTACIONALES Y FILTROS ---
            st.subheader("🔎 Filtros Estacionales")
            
            # --- Filtro por Año ---
            all_years = sorted(df_processed.index.year.unique().tolist())
            selected_years = st.multiselect(
                "📅 Filtrar por Año",
                options=all_years,
                default=all_years,
                format_func=lambda x: f"{x}"
            )
            # APLICAR EL FILTRO DE AÑO Y GUARDAR EN df_temp
            df_temp_filtered = df_processed[df_processed.index.year.isin(selected_years)]

            # 🚨 Verificación: si no hay años seleccionados, detener ejecución
            if not selected_years:
                st.warning("⚠️ No se ha seleccionado ningún año. Por favor, selecciona al menos uno para continuar con el análisis.")
                st.stop()  # Detiene la ejecución del bloque actual (todo lo que sigue después)

            # Filtro por Mes
            all_months = df_temp_filtered.index.month.unique().tolist()
            selected_months = st.multiselect(
                "Filtrar por Mes",
                options=all_months,
                default=all_months,
                format_func=lambda x: pd.to_datetime(str(x), format='%m').strftime('%B')
            )
            
            # Filtro por Día del Mes
            all_days = sorted(df_temp_filtered.index.day.unique().tolist())
            day_options = ["Todos los días"] + [f"Día {d}" for d in all_days]

            selected_day_option = st.selectbox(
                "📆 Seleccionar Día del Mes",
                options=day_options,
                index=0
            )

            # Lógica del filtro
            if selected_day_option == "Todos los días":
                selected_days = all_days
            else:
                # Extrae solo el número (por ejemplo, "Día 15" → 15)
                selected_days = [int(selected_day_option.split()[-1])]

            # Filtro por Día de la Semana
            all_weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            selected_weekdays = st.multiselect(
                "Filtrar por Día de la Semana",
                options=all_weekdays,
                default=all_weekdays,
                format_func=lambda x: x.replace('Monday', 'Lunes').replace('Tuesday', 'Martes').replace('Wednesday', 'Miércoles').replace('Thursday', 'Jueves').replace('Friday', 'Viernes').replace('Saturday', 'Sábado').replace('Sunday', 'Domingo')
            )
            
            # Aplicar filtros
            df_filtered = df_temp_filtered[
                (df_temp_filtered.index.month.isin(selected_months)) &
                (df_temp_filtered.index.day.isin(selected_days)) &
                (df_temp_filtered['Dia_Semana'].isin(selected_weekdays))
            ]

            # ---------------------------------------------------------------------------------------------------
            # 🧩 INICIO DE LA FUNCIONALIDAD: AJUSTES ESTACIONALES
            # ---------------------------------------------------------------------------------------------------
            st.markdown("---")
            st.subheader("Aplicar Ajustes Estacionales 🍃") # Título Modificado
            
            col_modo, col_label = st.columns([1, 2])
            with col_modo:
                modo_ajuste = st.radio(
                    "Modo de Ajuste",
                    ("Mensual", "General"),
                    key="radio_ajuste_modo",
                    horizontal=True
                )
            with col_label:
                st.markdown("Ajustar multiplicador (%) para cada mes (50% a 150%)")

            # Mapeo de meses
            nombres_meses = {
                1: "Jan (%)", 2: "Feb (%)", 3: "Mar (%)", 4: "Apr (%)",
                5: "May (%)", 6: "Jun (%)", 7: "Jul (%)", 8: "Aug (%)",
                9: "Sep (%)", 10: "Oct (%)", 11: "Nov (%)", 12: "Dec (%)"
            }
            
            # Inicializar diccionario de multiplicadores (por defecto 1.0)
            multiplicadores = {i: 1.0 for i in range(1, 13)}

            if modo_ajuste == "Mensual":
                # Mostrar 12 sliders en 4 filas de 3 columnas
                for i in range(0, 12, 3):
                    cols = st.columns(3)
                    for j in range(3):
                        mes_num = i + j + 1
                        if mes_num <= 12:
                            with cols[j]:
                                # El valor del slider es el multiplicador, rango [0.5, 1.5]
                                factor = st.slider(
                                    nombres_meses[mes_num],
                                    min_value=0.5,
                                    max_value=1.5,
                                    value=1.0,
                                    step=0.01,
                                    format="x%.2f",
                                    key=f"slider_mes_{mes_num}"
                                )
                                multiplicadores[mes_num] = factor
            else: # Modo General
                # Mostrar un único slider que aplica el factor a todos los meses
                factor_general = st.slider(
                    "Multiplicador General",
                    min_value=0.5,
                    max_value=1.5,
                    value=1.0,
                    step=0.01,
                    format="x%.2f",
                    key="slider_general"
                )
                for i in range(1, 13):
                    multiplicadores[i] = factor_general
            
            # --- Aplicar el ajuste de Potencia (kW) y Energía (kWh) al DataFrame filtrado ---
            if not df_filtered.empty:
                # Obtener el mes de cada fila
                meses_en_df = df_filtered.index.month

                # Crear una serie con el multiplicador correspondiente a cada fila
                multiplicador_serie = meses_en_df.map(multiplicadores)
                
                # Aplicar el ajuste a Potencia (kW) y recalcular Energía (kWh)
                df_filtered['Potencia (kW)_Ajustada'] = df_filtered['Potencia (kW)'] * multiplicador_serie
                df_filtered['Potencia (kW)'] = df_filtered['Potencia (kW)_Ajustada']
                df_filtered['Energia (kWh)'] = df_filtered['Potencia (kW)'] * 1.0 # Asumiendo intervalo de 1 hora
                
                # Eliminar la columna temporal
                df_filtered.drop(columns=['Potencia (kW)_Ajustada'], inplace=True)
            
            st.markdown("---")
            # ---------------------------------------------------------------------------------------------------
            # 🧩 FIN DE LA FUNCIONALIDAD: AJUSTES ESTACIONALES
            # ---------------------------------------------------------------------------------------------------
            
            if df_filtered.empty:
                st.warning("El filtro seleccionado no devuelve datos. Ajusta los parámetros.")
            else:
                
                # --- CÁLCULOS AGREGADOS ---
                
                st.subheader("🔢 Métricas Clave")
                
                # Cálculo de energía diaria/mensual y potencia media
                total_energia_kwh = df_filtered['Energia (kWh)'].sum()
                
                # CONVERSIÓN A GWh: 1 MWh = 1,000 kWh
                total_energia = total_energia_kwh / 1_000 

                promedio_potencia = df_filtered['Potencia (kW)'].mean()
                max_potencia = df_filtered['Potencia (kW)'].max()
                min_potencia = df_filtered['Potencia (kW)'].min()
                factor_carga = (promedio_potencia / max_potencia) * 100 if max_potencia > 0 else 0

                # Fechas de ocurrencia
                hora_max = df_filtered[df_filtered['Potencia (kW)'] == max_potencia].index[0]
                hora_min = df_filtered[df_filtered['Potencia (kW)'] == min_potencia].index[0]

                col_e, col_p, col_max, col_min, col_fc = st.columns(5)
                col_e.metric("Energía Total (MWh)", f"{total_energia:,.2f} MWh")
                col_p.metric("Potencia Media (kW)", f"{promedio_potencia:,.2f} kW")
                col_max.metric("Potencia Máxima (kW)", f"{max_potencia:,.2f} kW")
                col_min.metric("Potencia Mínima (kW)", f"{min_potencia:,.2f} kW")
                col_fc.metric("Factor de Carga", f"{factor_carga:,.2f} %")

                # --- Información adicional ---
                st.info(
                    f"🔺 **Potencia máxima** registrada el **{hora_max.strftime('%Y-%m-%d %H:%M')}** "
                    f"con un valor de **{max_potencia:.2f} kW**.\n\n"
                    f"🔻 **Potencia mínima** registrada el **{hora_min.strftime('%Y-%m-%d %H:%M')}** "
                    f"con un valor de **{min_potencia:.2f} kW**."
                )
                
                #st.markdown("---")

                # --- CÁLCULOS AGREGADOS SEGMENTADOS (NUEVOS PARÁMETROS) ---
                
                df_diurno = df_filtered[df_filtered['Segmento'] == 'Diurno']
                df_nocturno = df_filtered[df_filtered['Segmento'] == 'Nocturno']

                # 1. Diurno -------------------------------------------------------------------------
                if not df_diurno.empty:
                    total_diurno = df_diurno['Potencia (kW)'].sum()/1_000
                    max_diurno = df_diurno['Potencia (kW)'].max() 
                    min_diurno = df_diurno['Potencia (kW)'].min() 
                    mean_diurno = df_diurno['Potencia (kW)'].mean() 
                    fc_diurno = (mean_diurno / max_diurno) * 100 if max_diurno > 0 else 0

                    # Usar idxmax/idxmin para obtener el índice (FechaHora)
                    hora_max_diurno = df_diurno['Potencia (kW)'].idxmax()
                    hora_min_diurno = df_diurno['Potencia (kW)'].idxmin()
                else:
                    total_diurno = max_diurno = min_diurno = mean_diurno = fc_diurno = 0.0
                    hora_max_diurno = hora_min_diurno = None

                # 2. Nocturno
                if not df_nocturno.empty:
                    total_nocturno = df_nocturno['Potencia (kW)'].sum()/1_000
                    max_nocturno = df_nocturno['Potencia (kW)'].max() 
                    min_nocturno = df_nocturno['Potencia (kW)'].min() 
                    mean_nocturno = df_nocturno['Potencia (kW)'].mean()
                    fc_nocturno = (mean_nocturno / max_nocturno) * 100 if max_nocturno > 0 else 0

                    # Usar idxmax/idxmin para obtener el índice (FechaHora)
                    hora_max_nocturno = df_nocturno['Potencia (kW)'].idxmax()
                    hora_min_nocturno = df_nocturno['Potencia (kW)'].idxmin()
                else:
                    total_nocturno = max_nocturno = min_nocturno = mean_nocturno = fc_nocturno = 0.0
                    hora_max_nocturno = hora_min_nocturno = pd.NaT
                
                # Función auxiliar para formatear la hora (maneja NaT)
                def format_time(ts):
                    return ts.strftime('%Y-%m-%d %H:%M') if pd.notna(ts) else 'N/A'

                st.subheader("Métricas por Segmento (☀️Diurno vs. 🌙Nocturno)")

                col_tot_d, col_p_d, col_max_d, col_min_d, col_fc_d = st.columns(5)

                st.info(
                    f"☀️ **Periodo Diurno**\n"
                    f"- 🔺 Potencia máxima: **{max_diurno:.2f} kW** registrada el **{format_time(hora_max_diurno)}**\n"
                    f"- 🔻 Potencia mínima: **{min_diurno:.2f} kW** registrada el **{format_time(hora_min_diurno)}**"
                )

                col_tot_n, col_p_n, col_max_n, col_min_n, col_fc_n = st.columns(5)

                st.info(
                    f"🌙 **Periodo Nocturno**\n"
                    f"- 🔺 Potencia máxima: **{max_nocturno:.2f} kW** registrada el **{format_time(hora_max_nocturno)}**\n"
                    f"- 🔻 Potencia mínima: **{min_nocturno:.2f} kW** registrada el **{format_time(hora_min_nocturno)}**"
                )

                # MOSTRAR MÉTRICAS SEGMENTADAS
                col_tot_d.metric("E. Total Diurna (MWh)", f"{total_diurno:,.2f} MWh")
                col_p_d.metric("P. Media Diurna (kW)", f"{mean_diurno:,.2f} kW")
                col_max_d.metric("P. Máxima Diurna (kW)", f"{max_diurno:,.2f} kW", f"{max_diurno - max_potencia:,.2f} vs Total")
                col_min_d.metric("P. Mínima Diurna (kW)", f"{min_diurno:,.2f} kW", f"{min_diurno - min_potencia:,.2f} vs Total")
                col_fc_d.metric("F. Carga Diurno (%)", f"{fc_diurno:,.2f} %")

                col_tot_n.metric("E. Total Nocturno (MWh)", f"{total_nocturno:,.2f} MWh")
                col_p_n.metric("P. Media Diurna (kW)", f"{mean_nocturno:,.2f} kW")
                col_max_n.metric("P. Máxima Nocturna (kW)", f"{max_nocturno:,.2f} kW", f"{max_nocturno - max_potencia:,.2f} vs Total")
                col_min_n.metric("P. Mínima Nocturna (kW)", f"{min_nocturno:,.2f} kW", f"{min_nocturno - min_potencia:,.2f} vs Total")
                col_fc_n.metric("F. Carga Nocturno (%)", f"{fc_nocturno:,.2f} %")

                #st.markdown("---")

                # -----------------------------------------------------------------------------------------

                # --- TABLA DE RESUMEN DE MÉTRICAS CONSOLIDADAS ---
                # Crear un DataFrame de resumen
                resumen_data = {
                    'Métrica': [
                        'Energía Total (MWh)', 'Potencia Máxima (kW)', 'Potencia Media (kW)', 'Factor de Carga (%)',
                        'E. Total Diurna (MWh)', 'P. Máxima Diurna (kW)', 'P. Media Diurna (kW)', 'F. Carga Diurno (%)',
                        'E. Total Nocturno (MWh)', 'P. Máxima Nocturna (kW)', 'P. Media Nocturna (kW)', 'F. Carga Nocturno (%)'
                    ],
                    'Valor': [
                        total_energia, max_potencia, promedio_potencia, factor_carga,
                        total_diurno, max_diurno, mean_diurno, fc_diurno,
                        total_nocturno, max_nocturno, mean_nocturno, fc_nocturno
                    ],
                    'Unidad': [
                        'MWh', 'kW', 'kW', '%',
                        'MWh', 'kW', 'kW', '%',
                        'MWh', 'kW', 'kW', '%'
                    ],
                    'Ocurrencia': [
                        format_time(hora_max), format_time(hora_max), '-', '-',
                        format_time(hora_max_diurno), format_time(hora_max_diurno), '-', '-',
                        format_time(hora_max_nocturno), format_time(hora_max_nocturno), '-', '-'
                    ]
                }

                df_resumen = pd.DataFrame(resumen_data)

                # Aplicar formato para una mejor visualización
                df_resumen['Valor'] = df_resumen['Valor'].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)

                with st.expander("Ver Tabla de Métricas", expanded=False):
                    st.dataframe(df_resumen, hide_index=True, use_container_width=True)

                col_metric_csv, col_metric_xlx = st.columns(2)
                with col_metric_csv:
                    csv_metric = convert_df_to_csv(df_resumen)
                    st.download_button(
                        label="⬇️ Descargar como CSV",
                        data=csv_metric,
                        file_name='tabla_metricas_clave.csv',
                        mime='text/csv',
                        use_container_width=True
                    )

                with col_metric_xlx:
                    st.download_button(
                        label="⬇️ Descargar como Excel",
                        data=convert_df_to_excel(df_resumen),
                        file_name='tabla_metricas_clave.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        use_container_width=True
                    )

                st.markdown("---")

                # -----------------------------------------------------------------------------------------

                # --- VISUALIZACIÓN ---
                st.subheader("📈 Visualización de la Carga")
                
                # 1. Gráfico de Potencia a lo largo del tiempo
                fig_time = px.line(
                    df_filtered.reset_index(), 
                    x='FechaHora', 
                    y='Potencia (kW)', 
                    title='Potencia (kW) a lo largo del Tiempo',
                    color='Segmento',
                    color_discrete_map={'Diurno': 'orange', 'Nocturno': 'darkblue'}
                )
                fig_time.update_layout(height=400)
                st.plotly_chart(fig_time, use_container_width=True)

                # 2. Perfil de Carga Diario Promedio (por hora)
                df_hourly_avg = df_filtered.groupby(['Hora_Solo', 'Segmento'])['Potencia (kW)'].mean().reset_index()
                df_hourly_avg['Hora_Str'] = df_hourly_avg['Hora_Solo'].apply(lambda x: f'{x:02d}:00')
                
                fig_profile = px.bar(
                    df_hourly_avg,
                    x='Hora_Str',
                    y='Potencia (kW)',
                    color='Segmento',
                    barmode='group',
                    title='Perfil de Carga Promedio (por Hora del Día)',
                    labels={'Potencia (kW)': 'Potencia Promedio (kW)', 'Hora_Str': 'Hora del Día'},
                    color_discrete_map={'Diurno': 'orange', 'Nocturno': 'darkblue'}
                )
                fig_profile.update_layout(height=400, xaxis={'categoryorder':'array', 'categoryarray':df_hourly_avg['Hora_Str'].unique()})
                st.plotly_chart(fig_profile, use_container_width=True)

                #df_spanish = df_filtered.copy()
                #df_spanish["Dia_Semana"] = df_spanish["FechaHora"].dt.day_name()
                traduccion_dias = {
                    "Monday": "Lunes",
                    "Tuesday": "Martes",
                    "Wednesday": "Miércoles",
                    "Thursday": "Jueves",
                    "Friday": "Viernes",
                    "Saturday": "Sábado",
                    "Sunday": "Domingo"
                }
                df_filtered["Dia_Semana"] = df_filtered["Dia_Semana"].map(traduccion_dias)

                traduccion_meses = {
                    "January": "Enero",
                    "February": "Febrero",
                    "March": "Marzo",
                    "April": "Abril",
                    "May": "Mayo",
                    "June": "Junio",
                    "July": "Julio",
                    "August": "Agosto",
                    "September": "Septiembre",
                    "October":"Octubre",
                    "November":"Noviembre",
                    "December":"Diciembre"
                }
                
                #st.markdown("---")

                # Se muestra el DataFrame con todas las columnas procesadas y filtradas.
                with st.expander("Ver Datos Procesados y Filtrados", expanded=False):
                    # El índice (FechaHora) es el identificador principal.
                    st.dataframe(df_filtered)
                
                #Boton de descarga CSV y XLXS
                col_filtered_csv, col_filtered_xlx = st.columns(2)
                with col_filtered_csv:
                    csv_filtered = convert_df_to_csv(df_filtered)
                    st.download_button(
                        label="⬇️ Descargar como CSV",
                        data=csv_filtered,
                        file_name='datos_procesados_filtrados.csv',
                        mime='text/csv',
                        use_container_width=True
                    )

                with col_filtered_xlx:
                    st.download_button(
                        label="⬇️ Descargar como Excel",
                        data=convert_df_to_excel(df_filtered),
                        file_name='datos_procesados_filtrados.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        use_container_width=True
                    )

                st.markdown("---")

                st.subheader("⚡️ Análisis de Energía (kWh) por Período")

                # --- SELECTOR DE TIPO DE CURVA ---
                tipo_curva = st.radio(
                    "Tipo de Curva:",
                    ("Mensual", "Diaria"), # <-- OPCIONES DE GRÁFICO
                    horizontal=True,
                    key="energy_curve_type"
                )
                
                # --- LÓGICA DE GRÁFICOS DE ENERGÍA ---
                
                # 1. Energía Total Mensual
                if tipo_curva == "Mensual":
                    
                    # Agrupar la energía total por mes
                    df_monthly = df_filtered.groupby(df_filtered.index.month)['Energia (kWh)'].sum().reset_index()
                    df_monthly.columns = ['Mes_Num', 'Energia (kWh)']
                    # Convertir el número de mes a nombre para el eje X
                    df_monthly['Mes'] = df_monthly['Mes_Num'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%b'))

                    # Calcular energía promedio para la línea de referencia
                    energia_promedio_mensual = df_monthly['Energia (kWh)'].mean()
                    n_meses = df_monthly['Mes'].nunique()
                    
                    fig_monthly = px.bar(
                        df_monthly,
                        x='Mes',
                        y='Energia (kWh)',
                        title='Energía Total Mensual (kWh)',
                        labels={'Energia (kWh)': 'Energía Total (kWh)'},
                        color='Mes', # Colorear por mes
                        color_discrete_sequence=px.colors.sequential.Sunset # Paleta de colores atractiva
                    )
                
                    # Añadir etiquetas de valor sobre las barras
                    fig_monthly.update_traces(
                        texttemplate='%{y:,.0f}', 
                        textposition='outside',
                        hovertemplate='<b>' + tipo_curva + ': %{x}</b><br>Energía: %{y:,.2f} kWh<extra></extra>'
                    )
                    
                    # Asegurar que el layout deja espacio para el texto
                    fig_monthly.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

                    # Añadir línea de energía promedio
                    fig_monthly.add_hline(
                        y=energia_promedio_mensual,
                        line_dash="dot",
                        line_color="red",
                        annotation_text=f"Energía Promedio Mensual: {energia_promedio_mensual:,.2f} kWh",
                        annotation_position="bottom right",
                        annotation=dict(
                            font=dict(color="black", size=11), # Cambiar el color del texto a negro
                            bgcolor="rgba(255, 255, 255, 0.7)", # Fondo semi-transparente blanco
                            bordercolor="red",
                            borderwidth=1
                        )
                    )
                    fig_monthly.update_layout(height=500, xaxis={'categoryorder':'array', 'categoryarray':df_monthly['Mes'].unique()})
                    st.plotly_chart(fig_monthly, use_container_width=True)

                    fig_energy=fig_monthly
                
                    # --- Interpretación ---
                    st.markdown("""
                        ### 📖 Interpretación:
                        - Esta curva muestra la **energía total acumulada por mes**, agrupada por color.
                        - Es ideal para identificar **patrones anuales** (picos en verano o invierno).
                    """)

                # 2. Energía Total Diaria (por día del mes correspondiente)

                if tipo_curva == "Diaria":

                    chart_type = st.radio(
                        "Tipo de Gráfico:",
                        ("Curva (Línea)", "Diagrama de Barras"),
                        horizontal=True,
                        key="energy_chart_type"
                    )

                    df_daily_energy = df_filtered['Energia (kWh)'].resample('D').sum().reset_index()
                    df_daily_energy.columns = ['Fecha', 'Energia Diaria (kWh)']
                    df_daily_energy['Fecha_Eje'] = df_daily_energy['Fecha'].dt.strftime('%d/%m/%y')
                    df_daily_energy['Mes_Nombre'] = df_daily_energy['Fecha'].dt.strftime('%B')
                    df_daily_energy['Mes_Nombre'] = df_daily_energy['Mes_Nombre'].map(traduccion_meses)
                    energia_promedio_diario = df_daily_energy['Energia Diaria (kWh)'].mean()

                    # 2. Generar la gráfica de línea con Plotly Express

                    if chart_type == "Curva (Línea)":
                        
                        fig_day = px.line(
                            df_daily_energy,
                            x='Fecha_Eje', # Usamos la nueva columna formateada para el eje X
                            y='Energia Diaria (kWh)',
                            color='Mes_Nombre', # Colorear por mes para la comparación
                            title='⚡️ Energía Diaria Total (kWh) por Día y Mes (Comparación por Mes)',
                            markers=True,
                            #color_discrete_sequence=px.colors.sequential.Sunset,
                            labels={
                            "Fecha_Eje": "Día/Mes/Año",
                            "Energia Diaria (kWh)": "Energía Diaria (kWh)",
                            "Mes_Nombre": "Mes"
                            },
                            template="plotly_white"
                        )

                    else: # Diagrama de Barras
                        fig_day = px.bar(
                            df_daily_energy,
                            x='Fecha_Eje',
                            y='Energia Diaria (kWh)',
                            color='Mes_Nombre',
                            title='⚡️ Energía Diaria Total (kWh) por Día y Mes (Comparación por Mes)',
                            color_discrete_sequence=px.colors.sequential.Sunset,
                            labels={
                            "Fecha_Eje": "Día/Mes/Año",
                            "Energia Diaria (kWh)": "Energía Diaria (kWh)",
                            "Mes_Nombre": "Mes"
                            }
                        )
                    
                    # Asegurar que el layout deja espacio para el texto
                    fig_day.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                    # 3. Personalización del Eje X
                    fig_day.update_xaxes(
                        # Se usa el formato de fecha para el tooltip, aunque el eje X sea de tipo 'category' (por usar la columna 'Fecha_Eje')
                        hoverformat="%d/%m/%y", 
                        # Asegurar que todas las etiquetas sean visibles (opcional, si hay muchos días)
                        dtick=7, # Mostrar un tick cada 7 días para evitar superposición
                    )
          
                    fig_day.add_hline(
                        y=energia_promedio_diario,
                        line_dash="dot",
                        line_color="red",
                        annotation_text=f"Energía Promedio Diaria {energia_promedio_diario:,.2f} kWh",
                        annotation_position="bottom right",
                        annotation=dict(
                            font=dict(color="black", size=11), # Cambiar el color del texto a negro
                            bgcolor="rgba(255, 255, 255, 0.7)", # Fondo semi-transparente blanco
                            bordercolor="red",
                            borderwidth=1
                        )
                    )
                    # 5. Mostrar la gráfica en Streamlit
                    st.plotly_chart(fig_day, use_container_width=True)
                    fig_energy=fig_day

                    # --- Interpretación ---
                    st.markdown("""
                        ### 📖 Interpretación:
                        - Esta curva muestra la **energía total acumulada en cada día**, agrupada por color según el mes.
                        - Es ideal para identificar **patrones estacionales o anuales** (picos en verano o invierno).
                    """)
                
                st.markdown("---")
                
                # --- CUADRO DE CARGA DETALLADO (Nueva Sección) ---

                # --- Procesar datos base ---
                df_ldc = df_filtered.copy()
                df_ldc = df_ldc.reset_index()
                df_ldc["FechaHora"] = pd.to_datetime(df_ldc["FechaHora"])
                df_ldc['Potencia (kW)'] = pd.to_numeric(df_ldc['Potencia (kW)'], errors='coerce')

                # Ordenar potencias de mayor a menor
                df_ldc_sorted = df_ldc.sort_values(by='Potencia (kW)', ascending=False).reset_index(drop=True)

                # Calcular duración acumulada (en horas)
                df_ldc_sorted['Duración (h)'] = (df_ldc_sorted.index + 1)

                # --- Gráfico LDC ---
                st.subheader("📊 Curva de Duración de Carga (LDC)")

                fig_ldc = px.line(
                    df_ldc_sorted,
                    x='Duración (h)',
                    y='Potencia (kW)',
                    title="Curva de Duración de Carga (Potencia ordenada de mayor a menor)",
                    line_shape='linear'
                )

                # Añadir líneas horizontales para referencia

                fig_ldc.add_hline(y=max_potencia, line_dash="dash", line_color="red", annotation_text=f"Pico: {max_potencia:,.2f} kW", annotation_position="top right")
                fig_ldc.add_hline(y=promedio_potencia, line_dash="dot", line_color="green", annotation_text=f"Media: {promedio_potencia:,.2f} kW", annotation_position="bottom right")
                fig_ldc.update_traces(line=dict(color="#0078FF", width=2))
                fig_ldc.update_layout(height=550, xaxis_title="Duración (h)", yaxis_title="Potencia (kW)")

                st.plotly_chart(fig_ldc, use_container_width=True)

                df_ordenado = df_ldc_sorted[['Duración (h)', 'Potencia (kW)', 'Año', 'Mes', 'Dia', 'Hora']]

               # --- Opción para ver datos ordenados ---
                with st.expander("📋 Ver datos ordenados por potencia (para el LDC)"):
                    # Se incluyen las columnas Año, Mes, Día, Hora
                    st.dataframe(df_ldc_sorted[['Duración (h)', 'Potencia (kW)', 'Año', 'Mes', 'Dia', 'Hora']], use_container_width=True)

                col_ldc_csv, col_ldc_xlx = st.columns(2)
                with col_ldc_csv:
                    csv_ldc = convert_df_to_csv(df_ordenado)
                    st.download_button(
                        label="⬇️ Descargar como CSV",
                        data=csv_ldc,
                        file_name='tabla_datos_ordenados.csv',
                        mime='text/csv',
                        use_container_width=True
                    )

                with col_ldc_xlx:
                    st.download_button(
                        label="⬇️ Descargar como Excel",
                        data=convert_df_to_excel(df_ordenado),
                        file_name='tabla_datos_ordenados.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        use_container_width=True
                    )
                
                # --- Interpretación visual ---
                st.markdown("""
                    ### 📖 Interpretación:
                    - La curva muestra cómo disminuye la **potencia demandada** conforme pasa el tiempo.
                    - El **área bajo la curva** representa la **energía total consumida**.
                    - Un **factor de carga alto (≈1)** indica un consumo estable (casi plano).
                    - Un **factor de carga bajo (<0.6)** indica picos fuertes y baja utilización promedio de la capacidad instalada.
                """)


                # --- Mapa de Calor de Potencia Promedio (kW) ---------------------------------
                st.subheader("🌡️ Mapa de Calor de Potencia Promedio (kW)")

                df_heatmap = df_ldc.copy()
                
                # --- INICIO DE CORRECCIÓN PARA EL EJE X DEL HEATMAP ---
                # 1. Asegurar formato datetime y extraer componentes, eliminando filas inválidas.
                df_heatmap["FechaHora"] = pd.to_datetime(df_heatmap["FechaHora"], errors='coerce')
                df_heatmap.dropna(subset=['FechaHora'], inplace=True)

                # 2. Extraer Hora_Solo y normalizar/validar de forma defensiva (0-23)
                df_heatmap["Hora_Solo"] = df_heatmap["FechaHora"].dt.hour.astype(int)
                # La normalización por módulo 24 es una medida defensiva, aunque con los filtros aplicados 
                # (que validan Hora en 00:00-23:59), esto solo actuaría sobre errores internos.
                df_heatmap['Hora_Solo'] = df_heatmap['Hora_Solo'] % 24 
                
                # 3. Extraer componentes temporales restantes
                df_heatmap["Dia_Semana"] = df_heatmap["FechaHora"].dt.day_name()
                df_heatmap["Mes"] = df_heatmap["FechaHora"].dt.month
                
                # Aplicar traducción a los días de la semana
                df_heatmap["Dia_Semana"] = df_heatmap["Dia_Semana"].map(traduccion_dias)

                # Base colormap selector (viridis por defecto)
                col_pal, col_acc = st.columns([1,1])
                with col_pal:
                    base_palette = st.selectbox(
                        "Paleta base",
                        options=["pvout", "solar_day"],
                        index=0,
                        help="Paleta perceptualmente uniforme. Por defecto 'viridis'.",
                        key="heatmap_base_palette"
                    )

                with col_acc:
                    # Checkbox accesibilidad (con aria-label equivalente vía label)
                    acc_enabled = st.checkbox("Accesibilidad (daltonismo)", value=False, key="cb_accessibility", help="Activa filtros de simulación daltonismo. Navegable con teclado.",)
                    # Menú solo visible si está marcado
                    cb_mode_label = "Modo de daltonismo"
                    mode_human = "Sin filtro"
                    mode_internal = 'none'

                    if acc_enabled:
                        selection = st.selectbox(
                            "Modo de daltonismo",
                            options=[
                                "Sin filtro",
                                "Protanopia / Protanomalía",
                                "Deuteranopia / Deuteranomalía",
                                "Tritanopia / Tritanomalía",
                                "Acromatopsia (monocromático)",
                            ],
                            index=0,
                            key="cb_mode_select",
                            help="Transforma ÚNICAMENTE los colores del mapa (no recalcula datos)."
                        )
                        mapping = {
                            "Sin filtro": "none",
                            "Protanopia / Protanomalía": "protan",
                            "Deuteranopia / Deuteranomalía": "deutan",
                            "Tritanopia / Tritanomalía": "tritan",
                            "Acromatopsia (monocromático)": "achroma",
                        }
                        mode_internal = mapping.get(selection, 'none')
                        mode_human = selection

                # Selector de base agrupación
                tipo_mapa = st.radio(
                    "Selecciona la base del mapa de calor:",
                    ("Día del Mes", "Día de la Semana", "Mes"),
                    horizontal=True,
                    key="heatmap_groupby"
                )

                # --- Agrupación según elección ------------------------------------------------
                if tipo_mapa == "Día de la Semana":
                    orden_dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
                    df_grouped = df_heatmap.groupby(["Dia_Semana", "Hora_Solo"])["Potencia (kW)"].mean().reset_index()
                    df_grouped["Dia_Semana"] = pd.Categorical(df_grouped["Dia_Semana"], categories=orden_dias, ordered=True)
                    # 4. Asegurar el orden de las horas en la agrupación
                    df_grouped = df_grouped.sort_values(by=['Dia_Semana', 'Hora_Solo'])
                    eje_y = "Dia_Semana"
                    titulo = "Promedio horario por día de la semana"
                
                # --- NUEVA LÓGICA PARA DÍA DEL MES ---
                elif tipo_mapa == "Día del Mes":
                    # 1. Extraer el día numérico
                    df_heatmap['Dia_Del_Mes'] = df_heatmap["FechaHora"].dt.day
                    
                    # 2. Agrupar por Día del Mes y Hora
                    df_grouped = df_heatmap.groupby(["Dia_Del_Mes", "Hora_Solo"])["Potencia (kW)"].mean().reset_index()
                    
                    # 3. Asegurar el orden del Día del Mes (como cadena para Plotly, pero ordenado numéricamente)
                    df_grouped = df_grouped.sort_values(by=['Dia_Del_Mes', 'Hora_Solo'])
                    df_grouped["Dia_Del_Mes"] = df_grouped["Dia_Del_Mes"].astype(str) # Convertir a str para el eje Y
                    
                    eje_y = "Dia_Del_Mes"
                    titulo = "Promedio horario por día del mes"

                else:
                    df_grouped = df_heatmap.groupby(["Mes", "Hora_Solo"])["Potencia (kW)"].mean().reset_index()
                    df_grouped["Mes"] = df_grouped["Mes"].apply(lambda x: pd.to_datetime(str(x), format="%m").strftime("%B"))
                    # 4. Asegurar el orden de las horas en la agrupación
                    df_grouped = df_grouped.sort_values(by=['Mes', 'Hora_Solo'])
                    eje_y = "Mes"
                    titulo = "Promedio horario por mes"

                # --- Construcción del heatmap con Plotly -------------------------------------
                # 1) Primero generamos el colorscale filtrado SIN tocar los datos (idempotente)
                state = setColorBlindMode(acc_enabled, mode_internal)   # {'enabled': bool, 'mode': '...'}
                cs = setColorMap({
                    'base': base_palette,
                    'enabled': state['enabled'],
                    'mode': state['mode'],
                })

                fig_heat = px.density_heatmap(
                    df_grouped,
                    x="Hora_Solo",
                    y=eje_y,
                    z="Potencia (kW)",
                    color_continuous_scale=cs['colorscale'],   # <- paleta base + filtro aplicado
                    title=titulo,
                    labels={"Hora_Solo": "Hora del día", "Potencia (kW)": "Potencia promedio (kW)"},
                )

                # 2) Ajustes de contraste / accesibilidad (tooltips, barra)
                fig_heat.update_layout(
                    height=750,
                    xaxis_nticks=24, # <-- Mantiene los 24 ticks, pero no fuerza el rango.
                    xaxis = dict(
                        tickvals=list(range(0, 24, 2)), # Ticks cada 2 horas
                        ticktext=[f'{h:02d}:00' for h in range(0, 24, 2)],
                        range=[-0.5, 23.5], # Fuerza el rango de 0 a 23, centrado en los bins.
                        constrain='domain'
                    ),
                    coloraxis_colorbar=dict(title="Potencia (kW)"),
                    template="plotly_white",  # fondo claro: mejora contraste de etiquetas
                )

                # 3) (Opcional) Mejorar contraste de textos del eje si base = plasma (más luminosa)
                fig_heat.update_xaxes(title_font=dict(size=12))
                fig_heat.update_yaxes(title_font=dict(size=12))

                st.plotly_chart(fig_heat, use_container_width=True)

                # --- Interpretación ---
                st.markdown("""
                    ### 📖 Interpretación:
                    - Las zonas más oscuras indican **horas con mayor potencia promedio**.
                    - El patrón puede revelar **horas pico recurrentes** o **días con alto consumo**.
                    - Ideal para detectar tendencias estacionales o hábitos de operación.
                """)                         
    else:
        st.warning("Por favor, **carga un archivo en la Pestaña de Carga** y asegúrate de que **la validación sea exitosa** para poder ver y configurar el procesamiento.")

with tab3:
    if st.session_state.is_validated and st.session_state.df_valid is not None:
        df_valid = st.session_state.df_valid

        # 📄 EXPORTAR REPORTE FINAL (PDF) <-- BOTÓN Y LÓGICA DE DESCARGA
        # ------------------------------------------------------------------------------------------
        st.markdown("---")
        st.subheader("✅ Generar Reporte Final (PDF)")
                
        # Recopilar todas las figuras y parámetros para el informe
        figures = {
            'fig_time': fig_time,
            'fig_profile': fig_profile,
            'fig_energy': fig_energy, # La figura que se graficó realmente
            'fig_ldc': fig_ldc,
            'fig_heat': fig_heat
        }
                
        # Generar el PDF usando el DataFrame de resumen de métricas y las figuras
        pdf_output = create_pdf_report(
            df_resumen, 
            VERSION, 
            figures, 
            tipo_curva, 
            tipo_mapa, 
            hora_inicio_dia,
            hora_fin_dia # Añadir las horas para el resumen de la portada
            )

        st.download_button(
            label="⬇️ Exportar Reporte Completo (PDF)",
            data=pdf_output,
            file_name=f'Reporte_Carga_LDC_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf',
            mime='application/pdf',
            use_container_width=True
        )
                
        st.markdown(f"**Versión del Sistema:** `{VERSION}`")
                
    else:
        st.warning("Por favor, **carga un archivo en la Pestaña de Carga** y asegúrate de que **la validación sea exitosa** para poder ver y configurar el procesamiento.")







