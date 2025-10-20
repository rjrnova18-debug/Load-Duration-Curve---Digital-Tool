import pandas as pd
import numpy as np
import re # Para validación de expresiones regulares (Hora)
import plotly.express as px
import io
from datetime import datetime # Importar datetime para validación lógica temporal
from fpdf import FPDF # Importar fpdf2 (¡Asegúrate de instalarlo!)


# Definición para generar Reporte PDF COMPLETO
VERSION = "1.0.1" 

def create_pdf_report(df_resumen, version, figures, tipo_curva, tipo_mapa, hora_inicio_dia, hora_fin_dia):
    """
    Genera un reporte PDF completo, incluyendo métricas clave y todas las figuras de Plotly.
    Requiere la librería 'kaleido' para la exportación de imágenes.
    """
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    
    # --------------------------------------------------------------------------------
    # Función auxiliar para añadir figuras de Plotly como imágenes PNG
    # --------------------------------------------------------------------------------
    def add_plotly_figure(pdf, fig, title, width=170, height_mm=90):
        try:
            # Comprobar si hay espacio suficiente en la página actual
            if pdf.get_y() + height_mm + 15 > pdf.h - 10: 
                 pdf.add_page()
            
            # Título de la sección de gráfico
            pdf.set_font("Arial", size=12, style='B')
            pdf.cell(0, 7, txt=title, ln=1, align="L")
            
            # Convertir Plotly figure a static PNG bytes (alta resolución)
            # Usar un tamaño fijo para consistencia en el PDF
            png_bytes = fig.to_image(format="png", width=600, height=400) 
            
            # Embed the image, 'w' especifica el ancho en mm
            pdf.image(name=io.BytesIO(png_bytes), type='PNG', w=width) 
            pdf.ln(5)
            
        except ImportError:
            # Si kaleido no está instalado
            pdf.set_font("Arial", size=10, style='I')
            pdf.cell(0, 7, txt="[Error: No se pudo generar el gráfico. Instalar 'kaleido'.]", ln=1, align="L")
            pdf.ln(5)
        except Exception as e:
            pdf.set_font("Arial", size=10, style='I')
            pdf.cell(0, 7, txt=f"[Error al generar gráfico: {str(e)[:50]}...]", ln=1, align="L")
            pdf.ln(5)

    # --------------------------------------------------------------------------------
    # 1. PORTADA Y METADATOS
    # --------------------------------------------------------------------------------
    pdf.add_page()
    pdf.set_font("Arial", size=18, style='B')
    pdf.cell(0, 20, txt="REPORTE COMPLETO DE ANÁLISIS DE CARGA LDC", ln=1, align="C")
    
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(0, 7, txt="Generado por: Analizador de Carga LDC", ln=1, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", size=10)
    pdf.cell(0, 7, txt=f"Fecha de Generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align="C")
    pdf.cell(0, 7, txt=f"Versión del Sistema: {version}", ln=1, align="C")
    pdf.cell(0, 7, txt=f"Período de Análisis: {df_resumen.iloc[0]['Ocurrencia'].split(' ')[0]} - {df_resumen.iloc[-1]['Ocurrencia'].split(' ')[0]} (Estimado)", ln=1, align="C")

    pdf.ln(20)
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(0, 10, txt="Configuración del Análisis", ln=1, align="L")
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 5, txt=f"- Segmento Diurno: {hora_inicio_dia:02d}:00 a {hora_fin_dia:02d}:00", ln=1, align="L")
    pdf.cell(0, 5, txt=f"- Tipo de Curva de Energía Reportado: {tipo_curva}", ln=1, align="L")
    pdf.cell(0, 5, txt=f"- Base del Mapa de Calor: {tipo_mapa}", ln=1, align="L")

    pdf.add_page()
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(0, 10, txt="2. Métricas Clave y Resumen de Resultados", ln=1, align="L")
    pdf.ln(5)
    
    # Título de tabla
    pdf.set_font("Arial", size=12, style='B')
    pdf.cell(0, 10, txt="Métricas Totales y por Segmento (Periodo Filtrado)", ln=1, align="L")

    # Tabla de Métricas (usando df_resumen)
    pdf.set_font("Arial", size=9)
    col_width = pdf.w / 4.75 # Ancho de columna dinámico
    row_height = 6
    
    # Headers
    pdf.set_fill_color(200, 220, 255)
    headers = ['Métrica', 'Valor', 'Unidad', 'Ocurrencia']
    for col in headers:
        pdf.cell(col_width, row_height, col, border=1, fill=True, align='C')
    pdf.ln(row_height)
    
    # Data rows
    pdf.set_fill_color(240, 240, 240)
    fill = False
    for index, row in df_resumen.iterrows():
        # Usar utf-8 o latin-1 para compatibilidad, aunque fpdf2 maneja bien el texto
        metrica = str(row['Métrica'])
        if metrica.startswith('E. Total'):
            pdf.set_fill_color(200, 220, 255) # Resaltar las filas de Energía
            fill = True
        
        pdf.cell(col_width, row_height, metrica, border=1, fill=fill, align='L')
        pdf.cell(col_width, row_height, str(row['Valor']), border=1, fill=fill, align='R')
        pdf.cell(col_width, row_height, str(row['Unidad']), border=1, fill=fill, align='C')
        pdf.cell(col_width, row_height, str(row['Ocurrencia']), border=1, fill=fill, align='C')
        pdf.ln(row_height)
        
        if metrica.startswith('E. Total'):
            pdf.set_fill_color(240, 240, 240)
        
        fill = not fill # Alternar color de fila (solo si no es fila de energía)

    pdf.ln(10)
    
    # --------------------------------------------------------------------------------
    # 3. VISUALIZACIONES DE ANÁLISIS
    # --------------------------------------------------------------------------------
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(0, 10, txt="3. Visualizaciones de Análisis", ln=1, align="L")
    pdf.ln(5)
    
    # 3.1. Potencia a lo largo del Tiempo
    add_plotly_figure(pdf, figures['fig_time'], "3.1. Potencia (kW) a lo largo del Tiempo", width=180)
    
    # 3.2. Perfil de Carga Promedio
    add_plotly_figure(pdf, figures['fig_profile'], "3.2. Perfil de Carga Promedio (por Hora del Día)", width=180)

    # 3.3. Curva de Energía (Puede requerir nueva página)
    add_plotly_figure(pdf, figures['fig_energy'], f"3.3. Curva de Energía Total ({tipo_curva})", width=180)

    # 3.4. Curva de Duración de Carga (LDC)
    add_plotly_figure(pdf, figures['fig_ldc'], "3.4. Curva de Duración de Carga (LDC)", width=180)

    # 3.5. Mapa de Calor (Normalmente requiere su propia página debido a su tamaño)
    pdf.add_page()
    add_plotly_figure(pdf, figures['fig_heat'], f"3.5. Mapa de Calor de Potencia Promedio (Agrupación: {tipo_mapa})", width=180, height_mm=130)

    # El resultado es el binario (bytes). Lo convertimos a 'bytes' inmutable para Streamlit.
    return bytes(pdf.output(dest='S', encoding='latin-1')) 

