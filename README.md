# ⚡️ Herramienta Digital para el Análisis del Cuadro de Carga (LDC)

Esta aplicación web construida con **Streamlit** permite a los usuarios cargar, validar, procesar y analizar datos de consumo de potencia (kW) a intervalos regulares (asumiendo 1 hora), facilitando la generación de métricas clave, perfiles de carga, análisis de energía segmentada (Diurna/Nocturna) y la Curva de Duración de Carga (LDC).

## ✨ Características Principales

* **Carga de Datos Flexible:** Soporte para archivos `.csv` y `.xlsx`, además de ingreso manual a través de un editor interactivo.
* **Validación Robusta:** Verificación de formato, rango lógico (e.g., Año: 2000-2035, Potencia: $\ge 0$ kW), coherencia temporal (fechas imposibles, horas inválidas) y duplicados.
* **Segmentación Personalizada:** Define y analiza el consumo por periodos **Diurno** y **Nocturno** con horas de inicio y fin configurables.
* **Ajustes Estacionales:** Aplica factores de ajuste (multiplicadores) mensuales o generales a la potencia para proyecciones o escenarios.
* **Métricas de Rendimiento:** Cálculo de Energía Total (MWh), Potencia Máxima/Media, y Factor de Carga global y segmentado.
* **Visualizaciones Clave:**
    * Gráfico de Potencia a lo largo del tiempo.
    * Perfil de Carga Promedio (por hora).
    * **Curva de Duración de Carga (LDC)**.
    * Mapas de Calor de Potencia Promedio (por Día/Semana/Mes).
* **Exportación:** Descarga de datos editados/procesados, métricas y un **Reporte PDF completo** con todas las visualizaciones y resultados.

## ⚙️ Requisitos

pip install streamlit pandas numpy plotly openpyxl xlsxwriter fpdf2 color_accessibility kaleido
